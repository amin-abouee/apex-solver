//! Parallel normal-equation formation: `H = JᵀJ` and `g = Jᵀr`.
//!
//! The sparsity patterns of `Jᵀ`, `JᵀJ` and the value permutation linking `Jᵀ`
//! to `J` are static for the lifetime of a solve, so they are computed once and
//! cached. Each evaluation then runs as:
//!
//! 1. a parallel gather of `Jᵀ` values from `J` values (rayon, O(nnz)),
//! 2. faer's parallel sparse×sparse numeric kernel into the cached product
//!    pattern,
//! 3. faer's parallel sparse×dense kernel for the gradient.
//!
//! This replaces the previous per-call `transpose().to_col_major().mul(j)`
//! chain, which materialized the transposed Jacobian serially and recomputed
//! the symbolic product on every call. The numeric kernels dispatch through
//! `Par::global()`, so they run on the rayon pool (faer's `rayon` feature).
//!
//! # Invariant
//!
//! The cache is keyed on the Jacobian's sparsity pattern, which must not change
//! between `matches` and `compute`. Within one optimization this holds by
//! construction — the linearizer rebuilds `J` from the same symbolic structure
//! every iteration.

use dyn_stack::MemStack;
use faer::{
    Accum, Mat, get_global_parallelism,
    prelude::Reborrow,
    sparse::linalg::matmul::{
        SparseMatMulInfo, sparse_dense_matmul, sparse_sparse_matmul_numeric,
        sparse_sparse_matmul_numeric_scratch, sparse_sparse_matmul_symbolic,
    },
    sparse::{
        SparseColMat, SparseColMatMut, SparseColMatRef, SymbolicSparseColMat,
        SymbolicSparseColMatRef,
    },
};
use rayon::prelude::*;
use std::mem::MaybeUninit;

use crate::error::ErrorLogging;
use crate::linalg::{Damping, LinAlgError, LinAlgResult};

/// Cached symbolic machinery for forming the normal equations of a fixed
/// Jacobian sparsity pattern.
pub struct NormalEquationsCache {
    /// Sparsity of `Jᵀ` in CSC layout.
    jt_pattern: SymbolicSparseColMat<usize>,
    /// `jt_values[k] = j_values[value_perm[k]]`.
    value_perm: Vec<usize>,
    /// Sparsity of the product `JᵀJ`.
    product_symbolic: SymbolicSparseColMat<usize>,
    /// Per-column flop counts used by faer to partition the numeric matmul.
    product_info: SparseMatMulInfo,
    /// Position of the diagonal entry within each column of `JᵀJ`
    /// (`None` for structurally empty diagonals).
    diag_pos: Vec<Option<usize>>,
    /// Pattern of `JᵀJ` widened so every diagonal entry exists, together with
    /// the mapping from the product's values into it.
    ///
    /// `None` when the product already carries a full diagonal, which is the
    /// common case. A column of `J` that is entirely zero — a fully fixed
    /// variable, as gauge fixing produces — leaves `JᵀJ` with no diagonal entry
    /// there, and damping has to *create* it.
    augmented: Option<AugmentedDiagonal>,
    /// Fingerprint of the pattern the cache was built from.
    ///
    /// Keyed on the full [`PatternFingerprint`](super::pattern::PatternFingerprint),
    /// not just dimensions: an equal-`nnz` permutation must rebuild the
    /// symbolic product and permutation, never reuse them.
    pattern: super::pattern::PatternFingerprint,
    /// Reusable `Jᵀ` value buffer.
    jt_values: Vec<f64>,
    /// Reusable `JᵀJ` value buffer.
    hessian_values: Vec<f64>,
    /// Reusable scratch for faer's numeric kernel.
    stack_buf: Vec<MaybeUninit<u8>>,
}

/// The undamped normal equations of one linearization.
pub struct NormalEquations {
    pub hessian: SparseColMat<usize, f64>,
    /// `Jᵀ·r` (positive; consumers negate where their sign convention demands).
    pub gradient: Mat<f64>,
}

/// Holder for an optional [`NormalEquationsCache`] that stays `Clone`/`Debug`
/// for solver structs: cloning resets the cache (it rebuilds lazily from the
/// next solve's sparsity pattern) and Debug prints a placeholder.
#[derive(Default)]
pub struct LazyNormalEquations(Option<NormalEquationsCache>);

impl Clone for LazyNormalEquations {
    fn clone(&self) -> Self {
        Self(None)
    }
}

impl std::fmt::Debug for LazyNormalEquations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LazyNormalEquations(rebuilds on next solve)")
    }
}

impl LazyNormalEquations {
    /// Rebuild the cache if missing or if `jacobians`' pattern changed.
    pub fn ensure(&mut self, jacobians: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        if self.0.as_ref().is_none_or(|c| !c.matches(jacobians)) {
            self.0 = Some(NormalEquationsCache::try_new(jacobians)?);
        }
        Ok(())
    }

    /// Form `H = JᵀJ` and `g = Jᵀr`, rebuilding the cache when needed.
    pub fn compute(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<NormalEquations> {
        self.ensure(jacobians)?;
        match self.0.as_mut() {
            Some(cache) => cache.compute(residuals, jacobians),
            None => Err(LinAlgError::InvalidState(
                "normal equations cache not initialized".to_string(),
            )),
        }
    }

    /// `H + λ·D` on the cached product pattern — see
    /// [`NormalEquationsCache::damped_hessian`].
    pub fn damped_hessian(&self, damping: &Damping) -> LinAlgResult<SparseColMat<usize, f64>> {
        match self.0.as_ref() {
            Some(cache) => cache.damped_hessian(damping),
            None => Err(LinAlgError::InvalidState(
                "normal equations cache not initialized".to_string(),
            )),
        }
    }

    /// Sparsity of `JᵀJ`, for building compatible factorization symbolics.
    pub fn product_symbolic(&self) -> LinAlgResult<SymbolicSparseColMat<usize>> {
        match self.0.as_ref() {
            Some(cache) => Ok(cache.product_symbolic()),
            None => Err(LinAlgError::InvalidState(
                "normal equations cache not initialized".to_string(),
            )),
        }
    }
}

impl NormalEquationsCache {
    /// Build the cache for a Jacobian sparsity pattern.
    pub fn try_new(jacobian: &SparseColMat<usize, f64>) -> LinAlgResult<Self> {
        let sym = jacobian.as_ref().symbolic();
        let (jt_pattern, value_perm) = transpose_symbolic(sym)?;
        let (product_symbolic, product_info) = sparse_sparse_matmul_symbolic(jt_pattern.rb(), sym)
            .map_err(|e| {
                LinAlgError::MatrixConversion(format!("Symbolic JᵀJ product failed: {e:?}"))
                    .log_with_source(e)
            })?;

        let ncols = jacobian.ncols();
        let diag_pos: Vec<Option<usize>> = (0..ncols)
            .map(|col| {
                // Absolute index into the flat CSC value array.
                let col_start = product_symbolic.col_range(col).start;
                product_symbolic
                    .row_idx_of_col_raw(col)
                    .iter()
                    .position(|&r| r == col)
                    .map(|local| col_start + local)
            })
            .collect();

        let scratch_req = sparse_sparse_matmul_numeric_scratch::<usize, f64>(
            product_symbolic.rb(),
            get_global_parallelism(),
        );

        let augmented = if diag_pos.iter().any(Option::is_none) {
            Some(AugmentedDiagonal::build(&product_symbolic)?)
        } else {
            None
        };

        Ok(Self {
            jt_values: vec![0.0; jt_pattern.compute_nnz()],
            hessian_values: vec![0.0; product_symbolic.compute_nnz()],
            stack_buf: vec![MaybeUninit::uninit(); scratch_req.unaligned_bytes_required()],
            jt_pattern,
            value_perm,
            product_symbolic,
            product_info,
            diag_pos,
            augmented,
            pattern: super::pattern::PatternFingerprint::of(jacobian),
        })
    }

    /// Whether the cache still describes `jacobian`'s sparsity pattern.
    ///
    /// Full-pattern equality: dimensions and nonzero count reject fast, and
    /// the embedded hash rejects equal-`nnz` permutations that would
    /// otherwise reuse a stale symbolic product and value permutation.
    pub fn matches(&self, jacobian: &SparseColMat<usize, f64>) -> bool {
        self.pattern == super::pattern::PatternFingerprint::of(jacobian)
    }

    /// Sparsity of `JᵀJ`, for building factorizations compatible with `compute`.
    pub fn product_symbolic(&self) -> SymbolicSparseColMat<usize> {
        self.product_symbolic.clone()
    }

    /// Form `H = JᵀJ` and `g = Jᵀr` with parallel faer kernels.
    pub fn compute(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<NormalEquations> {
        let j_values = jacobian.as_ref().val();
        let perm = &self.value_perm;

        // Gather Jᵀ values from J — a pure permutation, run in parallel.
        self.jt_values
            .par_iter_mut()
            .zip(perm.par_iter())
            .for_each(|(v, &src)| *v = j_values[src]);

        // Parallel sparse×sparse numeric product into the cached pattern.
        let hessian = {
            let mut stack_buf = std::mem::take(&mut self.stack_buf);
            {
                let stack = MemStack::new(&mut stack_buf);
                let dst =
                    SparseColMatMut::new(self.product_symbolic.rb(), &mut self.hessian_values);
                let jt = SparseColMatRef::new(self.jt_pattern.rb(), &self.jt_values);
                sparse_sparse_matmul_numeric(
                    dst,
                    Accum::Replace,
                    jt,
                    jacobian.as_ref(),
                    1.0,
                    &self.product_info,
                    get_global_parallelism(),
                    stack,
                );
            }
            self.stack_buf = stack_buf;
            SparseColMat::new(self.product_symbolic.clone(), self.hessian_values.clone())
        };

        // Parallel sparse×dense product for the gradient: g = Jᵀ·r.
        let mut gradient = Mat::<f64>::zeros(self.pattern.ncols(), 1);
        {
            let jt = SparseColMatRef::new(self.jt_pattern.rb(), &self.jt_values);
            sparse_dense_matmul(
                gradient.as_mut(),
                Accum::Replace,
                jt,
                residuals.as_ref(),
                1.0,
                get_global_parallelism(),
            );
        }

        Ok(NormalEquations { hessian, gradient })
    }

    /// `H + λ·D` as an owned sparse matrix on the cached product pattern.
    ///
    /// `D_jj = clamp(H_jj, min_diagonal, max_diagonal)` — see [`Damping`]. Each
    /// diagonal position is visited exactly once, so `values[*pos]` still holds
    /// the un-damped `H_jj` when the clamp reads it.
    ///
    /// When `JᵀJ` is missing diagonal entries — a fully fixed variable
    /// contributes an all-zero column, so gauge fixing produces exactly that —
    /// the result is built on a widened pattern that carries the whole
    /// diagonal. That pattern is computed once, not per solve.
    ///
    /// Callers holding a cached factorization symbolic must use
    /// [`Self::product_symbolic_for_damping`] to build it, so the two agree.
    pub fn damped_hessian(&self, damping: &Damping) -> LinAlgResult<SparseColMat<usize, f64>> {
        let Some(aug) = &self.augmented else {
            let mut values = self.hessian_values.clone();
            for pos in self.diag_pos.iter().flatten() {
                values[*pos] += damping.diagonal_term(values[*pos]);
            }
            return Ok(SparseColMat::new(self.product_symbolic.clone(), values));
        };

        let mut values = vec![0.0; aug.nnz];
        for (base, &target) in aug.from_base.iter().enumerate() {
            values[target] = self.hessian_values[base];
        }
        for &pos in &aug.diag_pos {
            values[pos] += damping.diagonal_term(values[pos]);
        }
        Ok(SparseColMat::new(aug.symbolic.clone(), values))
    }

    /// The pattern [`Self::damped_hessian`] produces.
    ///
    /// Equal to [`Self::product_symbolic`] unless the diagonal had to be
    /// widened; factorization symbolics for the damped system must be built
    /// from this one.
    pub fn product_symbolic_for_damping(&self) -> SymbolicSparseColMat<usize> {
        match &self.augmented {
            Some(aug) => aug.symbolic.clone(),
            None => self.product_symbolic.clone(),
        }
    }
}

/// `JᵀJ`'s pattern widened so every diagonal entry is present.
///
/// Damping adds `λ·D` to the diagonal, so a structurally absent diagonal has to
/// be created. Rebuilding the pattern every solve would be wasteful, so it is
/// built once alongside the value mapping that scatters the product's values
/// into it.
struct AugmentedDiagonal {
    symbolic: SymbolicSparseColMat<usize>,
    /// `from_base[k]` is where product value `k` lands in the widened array.
    from_base: Vec<usize>,
    /// Offset of each column's diagonal entry in the widened array.
    diag_pos: Vec<usize>,
    nnz: usize,
}

impl AugmentedDiagonal {
    fn build(product: &SymbolicSparseColMat<usize>) -> LinAlgResult<Self> {
        let ncols = product.ncols();
        let nrows = product.nrows();

        // Union of the product's entries with the full diagonal, per column.
        let mut col_ptr = vec![0usize; ncols + 1];
        let mut row_idx = Vec::with_capacity(product.compute_nnz() + ncols);
        let mut from_base = vec![0usize; product.compute_nnz()];
        let mut diag_pos = Vec::with_capacity(ncols);

        for col in 0..ncols {
            let base_start = product.col_range(col).start;
            let rows = product.row_idx_of_col_raw(col);
            let mut inserted_diag = false;

            for (local, &row) in rows.iter().enumerate() {
                if !inserted_diag && row > col {
                    diag_pos.push(row_idx.len());
                    row_idx.push(col);
                    inserted_diag = true;
                }
                if row == col {
                    diag_pos.push(row_idx.len());
                    inserted_diag = true;
                }
                from_base[base_start + local] = row_idx.len();
                row_idx.push(row);
            }
            if !inserted_diag {
                diag_pos.push(row_idx.len());
                row_idx.push(col);
            }
            col_ptr[col + 1] = row_idx.len();
        }

        let nnz = row_idx.len();
        let symbolic = SymbolicSparseColMat::new_checked(nrows, ncols, col_ptr, None, row_idx);

        Ok(Self {
            symbolic,
            from_base,
            diag_pos,
            nnz,
        })
    }
}

/// Transpose a CSC pattern, returning the transposed pattern together with the
/// value permutation: `Jᵀ.values[k] == J.values[perm[k]]`.
///
/// Rows within each `Jᵀ` column come out sorted because the source columns of
/// `J` are visited in ascending order; `J`'s own row indices need not be
/// sorted.
fn transpose_symbolic(
    j: SymbolicSparseColMatRef<'_, usize>,
) -> LinAlgResult<(SymbolicSparseColMat<usize>, Vec<usize>)> {
    let (m, n) = (j.nrows(), j.ncols());
    let col_ptr = j.col_ptr();
    let row_idx = j.row_idx();
    let nnz = row_idx.len();

    // Column counts of Jᵀ are the row counts of J.
    let mut t_col_ptr = vec![0usize; m + 1];
    for &row in row_idx {
        t_col_ptr[row + 1] += 1;
    }
    for i in 0..m {
        t_col_ptr[i + 1] += t_col_ptr[i];
    }

    let mut t_row_idx = vec![0usize; nnz];
    let mut perm = vec![0usize; nnz];
    let mut next = t_col_ptr[..m].to_vec();
    for col in 0..n {
        let col_start = col_ptr[col];
        let col_end = col_ptr[col + 1];
        for (k, &row) in row_idx[col_start..col_end].iter().enumerate() {
            let pos = next[row];
            t_row_idx[pos] = col;
            perm[pos] = col_start + k;
            next[row] = pos + 1;
        }
    }

    let pattern = SymbolicSparseColMat::new_checked(n, m, t_col_ptr, None, t_row_idx);
    Ok((pattern, perm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    /// Deterministic pseudo-random sparse matrix with ~40% fill.
    fn sample_jacobian(m: usize, n: usize, seed: u64) -> TestResult<SparseColMat<usize, f64>> {
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut state = seed | 1;
        for col in 0..n {
            for row in 0..m {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if (state >> 33) % 5 < 2 {
                    let value = ((state >> 34) % 1000) as f64 / 250.0 - 2.0;
                    triplets.push(Triplet::new(row, col, value));
                }
            }
        }
        Ok(SparseColMat::try_new_from_triplets(m, n, &triplets)?)
    }

    fn naive_hessian(j: &SparseColMat<usize, f64>) -> TestResult<SparseColMat<usize, f64>> {
        use std::ops::Mul;
        let jt = j
            .as_ref()
            .transpose()
            .to_col_major()
            .map_err(|e| format!("transpose to_col_major failed: {e:?}"))?;
        Ok(jt.mul(j.as_ref()))
    }

    fn assert_hessians_close(a: &SparseColMat<usize, f64>, b: &SparseColMat<usize, f64>) {
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        for col in 0..a.ncols() {
            let rows_a = a.symbolic().row_idx_of_col_raw(col);
            let rows_b = b.symbolic().row_idx_of_col_raw(col);
            assert_eq!(rows_a, rows_b, "col {col}: pattern mismatch");
            for (va, vb) in a.val_of_col(col).iter().zip(b.val_of_col(col)) {
                assert!(
                    (va - vb).abs() < 1e-10 * (1.0 + va.abs()),
                    "col {col}: {va} vs {vb}"
                );
            }
        }
    }

    #[test]
    fn test_hessian_and_gradient_match_naive() -> TestResult<()> {
        let j = sample_jacobian(40, 25, 7)?;
        let residuals = Mat::from_fn(40, 1, |i, _| ((i * 13) % 17) as f64 - 8.0);

        let mut cache = NormalEquationsCache::try_new(&j)?;
        let ne = cache.compute(&residuals, &j)?;

        assert_hessians_close(&ne.hessian, &naive_hessian(&j)?);

        use std::ops::Mul;
        let jt = j
            .as_ref()
            .transpose()
            .to_col_major()
            .map_err(|e| format!("{e:?}"))?;
        let expected_g = jt.mul(residuals.as_ref());
        for i in 0..ne.gradient.nrows() {
            assert!((ne.gradient[(i, 0)] - expected_g[(i, 0)]).abs() < 1e-10);
        }
        Ok(())
    }

    #[test]
    fn test_compute_reusable_across_calls() -> TestResult<()> {
        let j = sample_jacobian(30, 20, 42)?;
        let residuals = Mat::from_fn(30, 1, |i, _| (i as f64) * 0.25);

        let mut cache = NormalEquationsCache::try_new(&j)?;
        let first = cache.compute(&residuals, &j)?;
        // Second call overwrites every buffer — must reproduce the same H and g.
        let second = cache.compute(&residuals, &j)?;
        assert_hessians_close(&first.hessian, &second.hessian);
        for i in 0..first.gradient.nrows() {
            assert!((first.gradient[(i, 0)] - second.gradient[(i, 0)]).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn test_matches_detects_shape_change() -> TestResult<()> {
        let j = sample_jacobian(30, 20, 1)?;
        let cache = NormalEquationsCache::try_new(&j)?;
        assert!(cache.matches(&j));

        let other = sample_jacobian(30, 21, 1)?;
        assert!(!cache.matches(&other));
        Ok(())
    }

    #[test]
    fn test_matches_rejects_equal_nnz_permutation() -> TestResult<()> {
        // Same shape and same nnz, one entry relocated within its column: the
        // old `(nrows, ncols, nnz)` triple reused the stale symbolic product
        // and value permutation for this pattern.
        let t = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 1, 4.0),
        ];
        let j = SparseColMat::try_new_from_triplets(3, 2, &t)?;
        let cache = NormalEquationsCache::try_new(&j)?;
        assert!(cache.matches(&j));

        let moved = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 2.0),
            Triplet::new(0, 1, 3.0),
            Triplet::new(2, 1, 4.0),
        ];
        let permuted = SparseColMat::try_new_from_triplets(3, 2, &moved)?;
        assert_eq!(
            permuted.as_ref().compute_nnz(),
            j.as_ref().compute_nnz(),
            "relocation must preserve nnz for the regression to be meaningful"
        );
        assert!(
            !cache.matches(&permuted),
            "an equal-nnz permutation must invalidate the cache"
        );
        Ok(())
    }

    #[test]
    fn test_damped_hessian_adds_lambda_on_diagonal() -> TestResult<()> {
        let j = sample_jacobian(20, 15, 99)?;
        let residuals = Mat::zeros(20, 1);
        let mut cache = NormalEquationsCache::try_new(&j)?;
        let ne = cache.compute(&residuals, &j)?;

        let damped = cache.damped_hessian(&Damping::identity(0.5))?;
        for col in 0..damped.ncols() {
            let rows = damped.symbolic().row_idx_of_col_raw(col);
            let pos = rows
                .iter()
                .position(|&r| r == col)
                .ok_or("missing diagonal in damped hessian")?;
            let expected = ne.hessian.val_of_col(col)[pos] + 0.5;
            assert!((damped.val_of_col(col)[pos] - expected).abs() < 1e-12);
        }
        Ok(())
    }
}
