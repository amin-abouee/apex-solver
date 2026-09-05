//! # Implicit Sparse Schur Complement Solver
//!
//! Matrix-free Schur complement solved with Preconditioned Conjugate
//! Gradients — Ceres's `ITERATIVE_SCHUR`, and matrix-free in the same sense
//! Ceres means it: **neither `S` nor `JᵀJ` is ever formed**.
//!
//! ## What "implicit" has to mean
//!
//! Writing `J = [E | F]` for the eliminated and retained column sets, the
//! reduced system is
//!
//! ```text
//! S = FᵀF − FᵀE·(EᵀE)⁻¹·EᵀF
//! ```
//!
//! and PCG only ever needs its action on a vector. Expanding that action so it
//! reads `J` and nothing else gives the whole solver:
//!
//! ```text
//! y = F·v                       scatter over the retained columns
//! t = Eᵀ·y                      gather over the eliminated columns
//! u = (EᵀE + λD_e)⁻¹·t          block-diagonal, one small solve per block
//! y ← y − E·u                   scatter over the eliminated columns
//! S·v = Fᵀ·y + λD_k·v           gather over the retained columns
//! ```
//!
//! Four passes over `J`'s nonzeros, no intermediate matrix. Ceres's own
//! documentation states the same cost model: "the cost of this evaluation
//! scales with the number of non-zeros in the Jacobian".
//!
//! That distinction is the reason this solver exists. `JᵀJ` for a bundle
//! adjustment problem carries a dense block for **every pair of cameras
//! sharing a landmark** — fill-in `J` does not have — so forming it costs far
//! more memory than `J`, and walking it costs far more time than walking `J`.
//! A solver that built `JᵀJ` and only skipped `S` would avoid the smaller of
//! the two costs while paying the larger one.
//!
//! ## When to use it
//!
//! When `JᵀJ` or `S` will not fit, or barely fits. The explicit solvers
//! factorize an exact `S` and are faster whenever they fit in memory, so this
//! is the large-problem fallback rather than the default —
//! `ExplicitSparseSchur` remains what
//! [`LevenbergMarquardtConfig::for_bundle_adjustment`](crate::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::for_bundle_adjustment)
//! selects.
//!
//! ## Preconditioning
//!
//! [`SchurPreconditioner`] selects between the three Ceres offers for
//! `ITERATIVE_SCHUR`, all built from `J`:
//!
//! | This crate | Ceres | Built from |
//! |---|---|---|
//! | [`SchurPreconditioner::None`] | `IDENTITY` | — |
//! | [`SchurPreconditioner::BlockDiagonal`] | `JACOBI` | diagonal blocks of `FᵀF` |
//! | [`SchurPreconditioner::SchurJacobi`] | `SCHUR_JACOBI` | diagonal blocks of `S` |
//!
//! `SchurJacobi` is the default and generally the best of the three; it costs
//! one pass over the observations to build.
//!
//! ## Automatic group recognition
//!
//! Like every other Schur solver, the eliminated ("group 0") and retained
//! ("group 1") variable sets come from [`SchurPartition`] — the same
//! structure [`ExplicitSparseSchur`](super::explicit::ExplicitSparseSchur) and
//! [`ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur)
//! use. This solver places no restriction on the eliminated variables' DOF or
//! column layout: 3-D points, 1-DOF inverse-depth landmarks, or a mix of both
//! in one problem all work identically, and the retained and eliminated
//! columns may interleave arbitrarily.
//!
//! ## Usage Example
//!
//! ```no_run
//! # use apex_solver::linalg::{ImplicitSparseSchur, SchurPreconditioner};
//! # use apex_solver::linalg::StructureAware;
//! # use apex_solver::core::VarKey;
//! # use apex_solver::core::variable::ManifoldVariable;
//! # use slotmap::{SlotMap, SecondaryMap};
//! # use std::collections::HashSet;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
//! # let variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
//! # let landmark_keys: HashSet<VarKey> = HashSet::new();
//! use apex_solver::linalg::{ImplicitSparseSchur, SchurPreconditioner};
//! use apex_solver::linalg::StructureAware;
//!
//! let mut solver = ImplicitSparseSchur::new()
//!     .with_preconditioner(SchurPreconditioner::SchurJacobi)
//!     .with_cg_config(500, 1e-9);
//! solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
//! # Ok(())
//! # }
//! ```

use crate::core::VarKey;
use crate::core::variable::ManifoldVariable;
use crate::error::ErrorLogging;
use crate::linalg::regularization::invert_with_retry_dyn;
use crate::linalg::schur::jacobian_ops::{column_dot, diag_jt_j};
use crate::linalg::schur::{
    BlockSpan, EliminatedBlocks, PcgParams, SchurOrdering, SchurPartition, SchurPreconditioner,
    effective_landmark_keys, jt_j_vec_product, jt_vec, solve_pcg,
};
use crate::linalg::sparse::pattern;
use crate::linalg::{Damping, LinAlgError, LinAlgResult, LinearSolver, SparseMode, StructureAware};
use faer::Mat;
use faer::sparse::SparseColMat;
use nalgebra::DMatrix;
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};

/// Structural facts about `J` that depend only on its sparsity, so they are
/// rebuilt when the pattern changes and reused across every solve in between.
#[derive(Debug, Clone, Default)]
struct StructureCache {
    /// Sorted rows touched by each eliminated block.
    eliminated_rows: Vec<Vec<usize>>,
    /// Eliminated blocks visible from each kept block — they share a row.
    visibility: Vec<Vec<usize>>,
    /// Pattern the cache was built from.
    fingerprint: Option<pattern::PatternFingerprint>,
}

/// Implicit (matrix-free) Schur complement solver using Preconditioned
/// Conjugate Gradients. Equivalent to Ceres's `ITERATIVE_SCHUR`.
///
/// Forms neither `S` nor `JᵀJ`; see the module documentation for the operator.
#[derive(Debug, Clone)]
pub struct ImplicitSparseSchur {
    partition: Option<SchurPartition>,
    /// `(EᵀE + λD_e)⁻¹` per eliminated block, rebuilt once per solve.
    eliminated: EliminatedBlocks,
    /// Largest eliminated block's DOF, for sizing the block-apply scratch.
    max_eliminated_dof: usize,
    /// Automatic eliminated/retained ("group 0"/"group 1") classification,
    /// combined with manual marks in `initialize_structure`.
    ordering: SchurOrdering,

    // CG parameters
    max_cg_iterations: usize,
    cg_tolerance: f64,

    // Preconditioner type
    preconditioner_type: SchurPreconditioner,

    /// `J` of the last successful solve.
    ///
    /// Held so [`LinearSolver::hessian_vec_product`] can evaluate `Jᵀ(J·v)`
    /// for the optimizers' quadratic model — the same arrangement
    /// `ExplicitSparseSchur`'s chunked variant uses. One copy of `J`, against
    /// the `JᵀJ` this solver exists to avoid.
    jacobian: Option<SparseColMat<usize, f64>>,
    /// `+Jᵀr`, published through [`LinearSolver::get_gradient`].
    gradient: Option<Mat<f64>>,

    // Workspace buffers for the Schur operator (avoid repeated allocations).
    workspace_rows: Vec<f64>, // residual-row sized buffer
    workspace_lm: Vec<f64>,   // eliminated-DOF sized buffer
    block_scratch: Vec<f64>,  // max-eliminated-block-DOF sized scratch

    structure: StructureCache,
}

impl ImplicitSparseSchur {
    /// Default: Schur-Jacobi preconditioner, 500 max iterations, 1e-9 relative
    /// tolerance — matching Ceres's `ITERATIVE_SCHUR` defaults.
    pub fn new() -> Self {
        Self {
            partition: None,
            eliminated: EliminatedBlocks::default(),
            max_eliminated_dof: 1,
            ordering: SchurOrdering::default(),
            max_cg_iterations: 500,
            cg_tolerance: 1e-9,
            preconditioner_type: SchurPreconditioner::default(),
            jacobian: None,
            gradient: None,
            workspace_rows: Vec::new(),
            workspace_lm: Vec::new(),
            block_scratch: Vec::new(),
            structure: StructureCache::default(),
        }
    }

    /// Set the PCG iteration cap and relative residual tolerance.
    pub fn with_cg_config(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_cg_iterations = max_iterations;
        self.cg_tolerance = tolerance;
        self
    }

    /// Select the preconditioner — see the table in the module documentation.
    ///
    /// Mirrors
    /// [`ExplicitSparseSchur::with_preconditioner`](super::explicit::ExplicitSparseSchur::with_preconditioner),
    /// so the two PCG-based solvers are configured the same way.
    pub fn with_preconditioner(mut self, preconditioner: SchurPreconditioner) -> Self {
        self.preconditioner_type = preconditioner;
        self
    }

    /// Set how eliminated variables are recognized.
    pub fn with_ordering(mut self, ordering: SchurOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Construct with explicit CG parameters, keeping the default
    /// preconditioner.
    pub fn with_cg_params(max_iterations: usize, tolerance: f64) -> Self {
        Self::new().with_cg_config(max_iterations, tolerance)
    }

    /// Construct with explicit CG parameters and preconditioner.
    pub fn with_config(
        max_iterations: usize,
        tolerance: f64,
        preconditioner: SchurPreconditioner,
    ) -> Self {
        Self::new()
            .with_cg_config(max_iterations, tolerance)
            .with_preconditioner(preconditioner)
    }

    /// The partition, once [`StructureAware::initialize_structure`] has run.
    pub fn partition(&self) -> Option<&SchurPartition> {
        self.partition.as_ref()
    }

    fn require_partition(&self) -> LinAlgResult<&SchurPartition> {
        self.partition.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput(
                "Schur solver used before initialize_structure; the partition is unknown".into(),
            )
            .log()
        })
    }

    /// Rebuild the structural caches if `J`'s sparsity changed.
    ///
    /// Also enforces the elimination precondition against `J` directly: a
    /// residual row touching two eliminated variables means `EᵀE` is not
    /// block-diagonal, so inverting it blockwise would silently produce a wrong
    /// step. `SchurPartition::verify_block_diagonal` makes the same check
    /// against `JᵀJ`, which this solver never forms.
    fn ensure_structure(&mut self, jacobian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let fingerprint = pattern::PatternFingerprint::of(jacobian);
        if self.structure.fingerprint == Some(fingerprint) {
            return Ok(());
        }

        let partition = self.require_partition()?;
        let symbolic = jacobian.symbolic();

        // Row -> owning eliminated block, and each block's sorted row set.
        let mut row_owner = vec![usize::MAX; jacobian.nrows()];
        let mut eliminated_rows = vec![Vec::new(); partition.eliminated_blocks().len()];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            for offset in 0..block.dof {
                for &row in symbolic.row_idx_of_col_raw(block.col_start + offset) {
                    match row_owner[row] {
                        usize::MAX => {
                            row_owner[row] = block_idx;
                            eliminated_rows[block_idx].push(row);
                        }
                        owned if owned == block_idx => {}
                        owned => {
                            let other = partition.eliminated_blocks()[owned];
                            return Err(LinAlgError::InvalidInput(format!(
                                "variables {:?} and {:?} are both marked for elimination but \
                                 share residual row {row}, so EᵀE is not block-diagonal and \
                                 the elimination would give a wrong step; eliminate only \
                                 mutually unconnected variables",
                                block.key, other.key
                            ))
                            .log());
                        }
                    }
                }
            }
        }
        for rows in &mut eliminated_rows {
            rows.sort_unstable();
        }

        // Eliminated blocks visible from each kept block, via shared rows.
        let visibility: Vec<Vec<usize>> = partition
            .kept_blocks()
            .par_iter()
            .map(|block| {
                let mut seen = Vec::new();
                for offset in 0..block.dof {
                    for &row in symbolic.row_idx_of_col_raw(block.col_start + offset) {
                        let owner = row_owner[row];
                        if owner != usize::MAX {
                            seen.push(owner);
                        }
                    }
                }
                seen.sort_unstable();
                seen.dedup();
                seen
            })
            .collect();

        self.structure = StructureCache {
            eliminated_rows,
            visibility,
            fingerprint: Some(fingerprint),
        };
        Ok(())
    }

    /// `S·v = Fᵀ(F·v − E·(EᵀE+λD_e)⁻¹·Eᵀ·F·v) + λD_k·v`, reading only `J`.
    ///
    /// `damp_kept` carries `λ·D_k` per retained local column, already clamped;
    /// it is empty for an undamped solve. The eliminated side's damping is
    /// baked into the inverted blocks before this is ever called.
    #[allow(clippy::too_many_arguments)]
    fn apply_schur_operator(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        damp_kept: &[f64],
        v: &Mat<f64>,
        out: &mut Mat<f64>,
        rows: &mut [f64],
        temp_lm: &mut [f64],
        scratch: &mut [f64],
    ) {
        let symbolic = jacobian.symbolic();

        // y = F·v
        rows.iter_mut().for_each(|r| *r = 0.0);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let x = v[(local, 0)];
                if x == 0.0 {
                    continue;
                }
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] += vals[k] * x;
                }
            }
        }

        // t = Eᵀ·y
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (k, &row) in idx.iter().enumerate() {
                    acc += vals[k] * rows[row];
                }
                temp_lm[base + offset] = acc;
            }
        }

        // u = (EᵀE + λD_e)⁻¹·t, then y ← y − E·u
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let dof = self.eliminated.dof(block_idx);
            if dof == 0 {
                continue;
            }
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            let slot = &mut scratch[..dof];
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * temp_lm[base + c];
                }
                slot[r] = acc;
            }

            for (offset, &u) in slot.iter().enumerate() {
                if u == 0.0 {
                    continue;
                }
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] -= vals[k] * u;
                }
            }
        }

        // S·v = Fᵀ·y + λD_k·v
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (k, &row) in idx.iter().enumerate() {
                    acc += vals[k] * rows[row];
                }
                if let Some(d) = damp_kept.get(local) {
                    acc += d * v[(local, 0)];
                }
                out[(local, 0)] = acc;
            }
        }
    }

    /// Dense `|rows| × dof` strip of `J` over `rows`, for the columns of
    /// `block`.
    ///
    /// `rows` is sorted and each CSC column's row indices are sorted, so each
    /// entry is one binary search. Rows the column does not touch stay zero.
    fn gather_strip(
        jacobian: &SparseColMat<usize, f64>,
        block: &BlockSpan,
        rows: &[usize],
    ) -> DMatrix<f64> {
        let symbolic = jacobian.symbolic();
        let mut strip = DMatrix::zeros(rows.len(), block.dof);
        for offset in 0..block.dof {
            let col = block.col_start + offset;
            let idx = symbolic.row_idx_of_col_raw(col);
            let vals = jacobian.val_of_col(col);
            for (local_row, &row) in rows.iter().enumerate() {
                if let Ok(k) = idx.binary_search(&row) {
                    strip[(local_row, offset)] = vals[k];
                }
            }
        }
        strip
    }

    /// Build the PCG preconditioner blocks from `J`.
    ///
    /// See the table in the module documentation for what each option is. All
    /// of them produce one dense `dof × dof` inverse per retained variable,
    /// or `None` for [`SchurPreconditioner::None`]. A block that will not
    /// invert falls back to identity: a preconditioner is a convergence aid,
    /// so a singular block must not fail the solve.
    fn build_preconditioner(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        damp_kept: &[f64],
    ) -> Option<Vec<DMatrix<f64>>> {
        if self.preconditioner_type == SchurPreconditioner::None {
            return None;
        }

        let blocks = partition
            .kept_blocks()
            .par_iter()
            .enumerate()
            .map(|(kept_idx, block)| {
                let dof = block.dof;
                let base = partition.kept_offset(kept_idx);

                // (FᵀF + λD_k) restricted to this block.
                let mut s_ii = DMatrix::zeros(dof, dof);
                for c in 0..dof {
                    for r in 0..dof {
                        s_ii[(r, c)] =
                            column_dot(jacobian, block.col_start + r, block.col_start + c);
                    }
                    if let Some(d) = damp_kept.get(base + c) {
                        s_ii[(c, c)] += d;
                    }
                }

                // Schur-Jacobi additionally subtracts the elimination
                // correction, which is what makes it a preconditioner for `S`
                // rather than for the unreduced retained block.
                if self.preconditioner_type == SchurPreconditioner::SchurJacobi {
                    let visible = self
                        .structure
                        .visibility
                        .get(kept_idx)
                        .map_or(&[][..], Vec::as_slice);
                    for &elim_idx in visible {
                        let elim_block = partition.eliminated_blocks()[elim_idx];
                        let rows = &self.structure.eliminated_rows[elim_idx];
                        let f_strip = Self::gather_strip(jacobian, block, rows);
                        let e_strip = Self::gather_strip(jacobian, &elim_block, rows);

                        // m = F_iᵀ·E_j, the (i, j) coupling block of `FᵀE`.
                        let m = f_strip.transpose() * e_strip;
                        let elim_dof = elim_block.dof;
                        let inv = DMatrix::from_column_slice(
                            elim_dof,
                            elim_dof,
                            self.eliminated.block(elim_idx),
                        );
                        s_ii -= &m * inv * m.transpose();
                    }
                }

                invert_with_retry_dyn(&s_ii).unwrap_or_else(|| DMatrix::identity(dof, dof))
            })
            .collect();
        Some(blocks)
    }

    /// `z = M⁻¹·r` for the block-diagonal preconditioner, or `z = r` without one.
    fn apply_preconditioner(blocks: Option<&[DMatrix<f64>]>, r: &Mat<f64>, z: &mut Mat<f64>) {
        let Some(blocks) = blocks else {
            faer::zip!(z, r).for_each(|faer::unzip!(z, r)| *z = *r);
            return;
        };
        let mut offset = 0usize;
        for inv in blocks {
            let dof = inv.nrows();
            for row in 0..dof {
                let mut acc = 0.0;
                for col in 0..dof {
                    acc += inv[(row, col)] * r[(offset + col, 0)];
                }
                z[(offset + row, 0)] = acc;
            }
            offset += dof;
        }
    }

    /// The whole solve, from `J` and `r` to the full-length update.
    ///
    /// `damping` is `None` for the plain normal equations.
    fn solve_from_jacobian(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: Option<&Damping>,
    ) -> LinAlgResult<Mat<f64>> {
        self.ensure_structure(jacobian)?;
        let partition = self.require_partition()?.clone();

        if jacobian.ncols() != partition.total_dof() {
            return Err(LinAlgError::InvalidInput(format!(
                "Jacobian has {} columns but the partition covers {}",
                jacobian.ncols(),
                partition.total_dof()
            ))
            .log());
        }

        // g = Jᵀr; the reduced system is built from −g, as in every other
        // Schur solver here.
        let gradient = jt_vec(jacobian, residuals);
        let kept_dof = partition.kept_dof();
        let eliminated_dof = partition.eliminated_dof();

        // λ·D per column, from diag(JᵀJ) — the only part of `JᵀJ` needed.
        let damp_kept: Vec<f64> = match damping {
            None => Vec::new(),
            Some(d) => {
                let diag = diag_jt_j(jacobian);
                let mut per_kept = vec![0.0; kept_dof];
                for block in partition.kept_blocks() {
                    for offset in 0..block.dof {
                        let col = block.col_start + offset;
                        if let Some(local) = partition.kept_local(col) {
                            per_kept[local] = d.diagonal_term(diag[col]);
                        }
                    }
                }
                per_kept
            }
        };

        // (EᵀE + λD_e)⁻¹, blockwise.
        let mut eliminated = std::mem::take(&mut self.eliminated);
        eliminated.gather_from_jacobian(jacobian, &partition);
        if let Some(d) = damping {
            eliminated.damp(d);
        }
        let inversion = eliminated.invert_in_place(&partition);
        self.eliminated = eliminated;
        inversion?;

        // g_reduced = −g_k + H_ke·H_ee⁻¹·g_e, assembled with `neg_gradient`
        // playing the role of `g` (so the reduced right-hand side matches the
        // explicit solvers' `g_k − H_ke·H_ee⁻¹·g_e` convention).
        let mut g_k = Mat::<f64>::zeros(kept_dof, 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                if let Some(local) = partition.kept_local(col) {
                    g_k[(local, 0)] = -gradient[(col, 0)];
                }
            }
        }
        let mut g_e = Mat::<f64>::zeros(eliminated_dof, 1);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                g_e[(base + offset, 0)] = -gradient[(block.col_start + offset, 0)];
            }
        }

        // correction = H_ke·H_ee⁻¹·g_e = Fᵀ·E·H_ee⁻¹·g_e, again through `J`.
        let mut hee_ge = Mat::<f64>::zeros(eliminated_dof, 1);
        self.apply_eliminated_inverse(&partition, &g_e, &mut hee_ge);
        let correction = self.coupling_times(&partition, jacobian, &hee_ge);

        let g_reduced = Mat::from_fn(kept_dof, 1, |i, _| g_k[(i, 0)] - correction[(i, 0)]);

        let precond = self.build_preconditioner(&partition, jacobian, &damp_kept);
        let delta_k = self.solve_pcg_block(
            &partition,
            jacobian,
            &damp_kept,
            &g_reduced,
            precond.as_deref(),
        );

        // δ_e = H_ee⁻¹·(g_e − H_keᵀ·δ_k) = H_ee⁻¹·(g_e − Eᵀ·F·δ_k)
        let coupling_t = self.coupling_transpose_times(&partition, jacobian, &delta_k);
        let rhs_e = Mat::from_fn(eliminated_dof, 1, |i, _| g_e[(i, 0)] - coupling_t[(i, 0)]);
        let mut delta_e = Mat::<f64>::zeros(eliminated_dof, 1);
        self.apply_eliminated_inverse(&partition, &rhs_e, &mut delta_e);

        let delta = self.combine_updates(&partition, &delta_k, &delta_e);

        self.gradient = Some(gradient);
        self.jacobian = Some(jacobian.clone());
        Ok(delta)
    }

    /// `x = H_ee⁻¹·b`, blockwise.
    fn apply_eliminated_inverse(&self, partition: &SchurPartition, b: &Mat<f64>, x: &mut Mat<f64>) {
        for block_idx in 0..partition.eliminated_blocks().len() {
            let dof = self.eliminated.dof(block_idx);
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * b[(base + c, 0)];
                }
                x[(base + r, 0)] = acc;
            }
        }
    }

    /// `H_ke·u = Fᵀ·(E·u)`, from `J`.
    fn coupling_times(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        u: &Mat<f64>,
    ) -> Mat<f64> {
        let symbolic = jacobian.symbolic();
        let mut rows = vec![0.0; jacobian.nrows()];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let x = u[(base + offset, 0)];
                if x == 0.0 {
                    continue;
                }
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] += vals[k] * x;
                }
            }
        }

        let mut out = Mat::<f64>::zeros(partition.kept_dof(), 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (k, &row) in idx.iter().enumerate() {
                    acc += vals[k] * rows[row];
                }
                out[(local, 0)] = acc;
            }
        }
        out
    }

    /// `H_keᵀ·v = Eᵀ·(F·v)`, from `J`.
    fn coupling_transpose_times(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        v: &Mat<f64>,
    ) -> Mat<f64> {
        let symbolic = jacobian.symbolic();
        let mut rows = vec![0.0; jacobian.nrows()];
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let x = v[(local, 0)];
                if x == 0.0 {
                    continue;
                }
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] += vals[k] * x;
                }
            }
        }

        let mut out = Mat::<f64>::zeros(partition.eliminated_dof(), 1);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (k, &row) in idx.iter().enumerate() {
                    acc += vals[k] * rows[row];
                }
                out[(base + offset, 0)] = acc;
            }
        }
        out
    }

    /// Solve `S·x = b` with the shared PCG primitive and the matrix-free operator.
    fn solve_pcg_block(
        &mut self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        damp_kept: &[f64],
        b: &Mat<f64>,
        precond: Option<&[DMatrix<f64>]>,
    ) -> Mat<f64> {
        // Workspaces come out of `self` so the operator closure can borrow
        // `self` immutably alongside them.
        let mut rows = std::mem::take(&mut self.workspace_rows);
        let mut temp_lm = std::mem::take(&mut self.workspace_lm);
        let mut scratch = std::mem::take(&mut self.block_scratch);
        rows.clear();
        rows.resize(jacobian.nrows(), 0.0);
        temp_lm.clear();
        temp_lm.resize(partition.eliminated_dof(), 0.0);
        scratch.clear();
        scratch.resize(self.max_eliminated_dof.max(1), 0.0);

        let result = solve_pcg(
            b,
            &PcgParams::new(self.max_cg_iterations, self.cg_tolerance),
            |p, ap| {
                self.apply_schur_operator(
                    partition,
                    jacobian,
                    damp_kept,
                    p,
                    ap,
                    &mut rows,
                    &mut temp_lm,
                    &mut scratch,
                );
            },
            |r, z| Self::apply_preconditioner(precond, r, z),
        );

        self.workspace_rows = rows;
        self.workspace_lm = temp_lm;
        self.block_scratch = scratch;
        result.x
    }

    /// Scatter the two solution halves back into one full-length update.
    fn combine_updates(
        &self,
        partition: &SchurPartition,
        delta_k: &Mat<f64>,
        delta_e: &Mat<f64>,
    ) -> Mat<f64> {
        let mut delta = Mat::zeros(partition.total_dof(), 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let global = block.col_start + offset;
                if let Some(local) = partition.kept_local(global) {
                    delta[(global, 0)] = delta_k[(local, 0)];
                }
            }
        }
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                delta[(block.col_start + offset, 0)] = delta_e[(base + offset, 0)];
            }
        }
        delta
    }
}

impl Default for ImplicitSparseSchur {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSolver<SparseMode> for ImplicitSparseSchur {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_from_jacobian(residuals, jacobian, None)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_from_jacobian(residuals, jacobian, Some(damping))
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        // `JᵀJ` was never formed, so evaluate its action from `J`.
        Some(jt_j_vec_product(self.jacobian.as_ref()?, v))
    }

    /// Always `None`: this solver never materializes `JᵀJ`.
    ///
    /// That is the documented degradation for a matrix-free backend — callers
    /// use [`LinearSolver::hessian_vec_product`] instead, which is served
    /// exactly from `J`.
    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        None
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

impl StructureAware for ImplicitSparseSchur {
    fn initialize_structure(
        &mut self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        schur_landmark_keys: &std::collections::HashSet<VarKey>,
    ) -> LinAlgResult<()> {
        let effective_keys =
            effective_landmark_keys(variables, schur_landmark_keys, &self.ordering);

        let mut kept = Vec::new();
        let mut eliminated = Vec::new();

        for (key, variable) in variables {
            let col_start = *variable_index_map.get(key).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("VarKey {:?} not in index map", key))
            })?;
            let span = BlockSpan {
                key,
                col_start,
                dof: variable.dof(),
            };
            if effective_keys.contains(&key) {
                eliminated.push(span);
            } else {
                kept.push(span);
            }
        }

        let partition = SchurPartition::new(kept, eliminated)?;
        self.max_eliminated_dof = partition
            .eliminated_blocks()
            .iter()
            .map(|b| b.dof)
            .max()
            .unwrap_or(0)
            .max(1);
        self.eliminated = EliminatedBlocks::new(&partition);
        self.workspace_lm = vec![0.0; partition.eliminated_dof()];
        self.block_scratch = vec![0.0; self.max_eliminated_dof];
        self.workspace_rows.clear();
        self.structure = StructureCache::default();
        self.partition = Some(partition);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::VarKey;
    use crate::core::variable::Variable;
    use apex_manifolds::{LieGroup, rn, se3};
    use faer::sparse::Triplet;
    use nalgebra::DVector;
    use slotmap::{SecondaryMap, SlotMap};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    type TestSetup = (
        SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        SparseColMat<usize, f64>,
        faer::Mat<f64>,
        std::collections::HashSet<VarKey>,
    );

    /// Build the same 2-camera + 3-landmark test setup used in explicit tests.
    /// Jacobian: 36 rows × 21 cols  (H_cc = 3·I₁₂, H_pp = 4·I₃ — positive definite)
    fn create_schur_test_setup() -> Result<TestSetup, Box<dyn std::error::Error>> {
        let se3_id = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let pt_zero = DVector::from_vec(vec![0.0, 0.0, 0.0]);

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let cam0 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_id.as_slice(),
        ))));
        let cam1 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_id.as_slice(),
        ))));
        let pt0 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));
        let pt1 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));
        let pt2 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));

        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(cam0, 0);
        variable_index_map.insert(cam1, 6);
        variable_index_map.insert(pt0, 12);
        variable_index_map.insert(pt1, 15);
        variable_index_map.insert(pt2, 18);

        let cam_cols = [0usize, 6];
        let lm_cols = [12usize, 15, 18];
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for (ci, &cam_col) in cam_cols.iter().enumerate() {
            for (li, &lm_col) in lm_cols.iter().enumerate() {
                let row_base = (ci * 3 + li) * 6;
                for k in 0..6 {
                    triplets.push(Triplet::new(row_base + k, cam_col + k, 1.0));
                    triplets.push(Triplet::new(row_base + k, lm_col + (k % 3), 1.0));
                }
            }
        }

        let jacobian = SparseColMat::try_new_from_triplets(36, 21, &triplets)?;
        let residuals = faer::Mat::from_fn(36, 1, |i, _| (i % 5) as f64 * 0.1);

        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(pt0);
        landmark_keys.insert(pt1);
        landmark_keys.insert(pt2);

        Ok((
            variables,
            variable_index_map,
            jacobian,
            residuals,
            landmark_keys,
        ))
    }

    #[test]
    fn test_implicit_schur_creation() {
        let solver = ImplicitSparseSchur::new();
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_custom_params() {
        let solver = ImplicitSparseSchur::with_cg_params(100, 1e-8);
        assert_eq!(solver.max_cg_iterations, 100);
        assert_eq!(solver.cg_tolerance, 1e-8);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_full_config() {
        let solver =
            ImplicitSparseSchur::with_config(200, 1e-10, SchurPreconditioner::BlockDiagonal);
        assert_eq!(solver.max_cg_iterations, 200);
        assert_eq!(solver.cg_tolerance, 1e-10);
        assert_eq!(
            solver.preconditioner_type,
            SchurPreconditioner::BlockDiagonal
        );
    }

    #[test]
    fn test_implicit_schur_default() {
        let solver = ImplicitSparseSchur::default();
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert!(solver.partition.is_none());
        assert!(solver.jacobian.is_none());
    }

    #[test]
    fn test_implicit_schur_initialize_structure() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let partition = solver.partition.as_ref().ok_or("partition is None")?;
        assert_eq!(partition.kept_blocks().len(), 2);
        assert_eq!(partition.eliminated_blocks().len(), 3);
        assert_eq!(partition.kept_dof(), 12);
        assert_eq!(partition.eliminated_dof(), 9);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_normal_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        assert_eq!(delta.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_augmented_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(0.1),
        )?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_get_hessian_gradient() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_none());
        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_none());

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // `JᵀJ` is never formed, so `get_hessian` stays `None` by design and
        // the quadratic model is served through `hessian_vec_product`.
        assert!(
            LinearSolver::<SparseMode>::get_hessian(&solver).is_none(),
            "matrix-free solver must not publish a Hessian"
        );
        let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("gradient is None")?;
        assert_eq!(g.nrows(), 21);

        let v = faer::Mat::from_fn(21, 1, |i, _| ((i % 3) as f64) - 1.0);
        let hv = LinearSolver::<SparseMode>::hessian_vec_product(&solver, &v)
            .ok_or("hessian_vec_product is None")?;
        assert_eq!(hv.nrows(), 21);
        Ok(())
    }

    /// A preconditioner changes how fast PCG converges, never what it
    /// converges to, so all three must produce the same step on the same
    /// system. This is what makes each option's implementation testable: a
    /// preconditioner that was silently ignored, or one built wrongly, shows
    /// up here as a disagreement or a failure to converge.
    #[test]
    fn every_preconditioner_reaches_the_same_step() -> TestResult {
        let mut steps = Vec::new();
        for preconditioner in [
            SchurPreconditioner::None,
            SchurPreconditioner::BlockDiagonal,
            SchurPreconditioner::SchurJacobi,
        ] {
            let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
                create_schur_test_setup()?;
            let mut solver = ImplicitSparseSchur::new()
                .with_cg_config(1000, 1e-12)
                .with_preconditioner(preconditioner);
            solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
            steps.push(LinearSolver::<SparseMode>::solve_normal_equation(
                &mut solver,
                &residuals,
                &jacobian,
            )?);
        }

        let reference = &steps[0];
        for (idx, step) in steps.iter().enumerate().skip(1) {
            for row in 0..reference.nrows() {
                let scale = reference[(row, 0)].abs().max(1.0);
                assert!(
                    (reference[(row, 0)] - step[(row, 0)]).abs() / scale < 1e-6,
                    "preconditioner {idx} diverged at row {row}: {} vs {}",
                    step[(row, 0)],
                    reference[(row, 0)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_implicit_schur_block_diagonal_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver =
            ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::BlockDiagonal);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_schur_jacobi_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver =
            ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::SchurJacobi);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_no_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::None);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_augmented_lambda_effect() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;

        let mut s1 = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        s1.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let d1 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut s1,
            &residuals,
            &jacobian,
            &Damping::identity(0.001),
        )?;

        let mut s2 = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        s2.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let d2 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut s2,
            &residuals,
            &jacobian,
            &Damping::identity(100.0),
        )?;

        let norm_diff: f64 = (0..21).map(|i| (d1[(i, 0)] - d2[(i, 0)]).powi(2)).sum();
        assert!(
            norm_diff > 1e-10,
            "Different λ should yield different updates"
        );
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_without_init_returns_error() -> TestResult {
        let triplets: Vec<Triplet<usize, usize, f64>> = vec![Triplet::new(0, 0, 1.0)];
        let jacobian =
            SparseColMat::try_new_from_triplets(1, 1, &triplets).map_err(|e| format!("{e:?}"))?;
        let residuals = faer::Mat::from_fn(1, 1, |_, _| 1.0);
        let mut solver = ImplicitSparseSchur::new();

        let result =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_implicit_initialize_structure_no_cameras_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::rn;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(rn::Rn::new(DVector::zeros(3)))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(k);

        let mut solver = ImplicitSparseSchur::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no retained variables present"
        );
    }

    #[test]
    fn test_implicit_initialize_structure_no_landmarks_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::se3;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let landmark_keys = std::collections::HashSet::<VarKey>::new();

        let mut solver = ImplicitSparseSchur::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no eliminated variables present"
        );
    }

    /// A mixed-DOF, non-contiguous partition (1-DOF inverse depth interleaved
    /// with retained poses) must work end to end — the whole point of moving
    /// this solver onto [`SchurPartition`].
    #[test]
    fn test_implicit_schur_supports_non_contiguous_mixed_dof_partition() -> TestResult {
        use crate::linalg::schur::BlockSpan;

        // Layout: kept A [0..6), eliminated P [6..7) (1-DOF), kept B [7..13),
        // eliminated Q [13..14) (1-DOF). Neither side is contiguous.
        let a = Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]));
        let b = Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]));
        let p = Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0])));
        let q = Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0])));

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let key_a = variables.insert(Box::new(a));
        let key_p = variables.insert(Box::new(p));
        let key_b = variables.insert(Box::new(b));
        let key_q = variables.insert(Box::new(q));

        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(key_a, 0);
        variable_index_map.insert(key_p, 6);
        variable_index_map.insert(key_b, 7);
        variable_index_map.insert(key_q, 13);

        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(key_p);
        landmark_keys.insert(key_q);

        // Observations: A-P, B-P, A-Q, B-Q, plus a small prior on every
        // column so the system is SPD.
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut row = 0usize;
        for &cam_col in &[0usize, 7] {
            for &lm_col in &[6usize, 13] {
                for k in 0..1 {
                    triplets.push(Triplet::new(row + k, cam_col + k, 1.0));
                    triplets.push(Triplet::new(row + k, lm_col, 1.0));
                }
                row += 1;
            }
        }
        for col in 0..14 {
            triplets.push(Triplet::new(row, col, 0.5));
            row += 1;
        }
        let jacobian = SparseColMat::try_new_from_triplets(row, 14, &triplets)?;
        let residuals = faer::Mat::from_fn(row, 1, |i, _| 0.1 * (i as f64 + 1.0));

        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-10);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 14);
        for i in 0..14 {
            assert!(delta[(i, 0)].is_finite());
        }

        // A block spanning a variable never claimed by anything must still
        // round-trip through BlockSpan without panicking (sanity check on the
        // type used to build the partition above).
        let _ = BlockSpan {
            key: key_a,
            col_start: 0,
            dof: 6,
        };
        Ok(())
    }
}
