//! Jacobian linearization — the bridge between the nonlinear factor graph
//! and the linear system solved each iteration.
//!
//! This is the central module for all linearization concerns:
//! - `linearize_block()`: Shared factor evaluation (loss correction, residual accumulation)
//! - [`cpu::sparse`]: Sparse Jacobian assembly using `SparseColMat` and symbolic structure
//! - [`cpu::dense`]: Dense Jacobian assembly using `Mat<f64>`
//! - [`AssemblyBackend`]: Trait bridging linearization with the optimizer's solver types
//!
//! # Architecture
//!
//! ```text
//! Problem (factor graph)
//!     │  AssemblyBackend::assemble()
//!     ▼
//! (r: Mat<f64>, J: M::Jacobian)   ← M: LinearizationMode
//!     │
//!     ▼
//! LinearSolver<M>   (linalg/)
//!     │
//!     ▼
//! dx: Mat<f64>  → manifold update
//! ```

pub mod cpu;

use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};

use faer::Mat;
use faer::sparse::SparseColMat;
use smallvec::SmallVec;
use thiserror::Error;

use crate::core::problem::Problem;
use crate::error::ErrorLogging;
use crate::core::variable::ManifoldVariable;
use crate::core::{FactorKey, VarKey};
use crate::{
    core::{corrector::Corrector, residual_block::ResidualBlock},
    linearizer::cpu::{DenseMode, LinearizationMode, SparseMode},
};

pub use cpu::sparse::SymbolicStructure;

// ============================================================================
// Linearizer error types
// ============================================================================

/// Linearizer-specific error types for Jacobian assembly and symbolic structure operations.
///
/// These errors occur during the linearization phase of optimization, where
/// the nonlinear factor graph is converted into a linear system (residual vector
/// and Jacobian matrix).
///
/// # Error Hierarchy
///
/// `LinearizerError` is a Layer C (deep/module) error. It propagates up through:
/// - `CoreError::Linearizer(LinearizerError)` → for core module callers
/// - `ApexSolverError::Linearizer(LinearizerError)` → for direct API callers
///

#[derive(Debug, Clone, Error)]
pub enum LinearizerError {
    /// Symbolic structure construction or usage failed
    #[error("Symbolic structure error: {0}")]
    SymbolicStructure(String),

    /// Parallel computation error (thread/mutex failures during assembly)
    #[error("Parallel computation error: {0}")]
    ParallelComputation(String),

    /// Variable key missing in index mapping during Jacobian scatter
    #[error("Variable error: {0}")]
    Variable(String),

    /// Factor linearization returned no Jacobian when expected
    #[error("Factor linearization failed: {0}")]
    FactorLinearization(String),

    /// Invalid input (e.g., SparseMode requires symbolic structure)
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result type for linearizer module operations
pub type LinearizerResult<T> = Result<T, LinearizerError>;

// ============================================================================
// Block linearization (shared by sparse and dense paths)
// ============================================================================

/// Metadata produced by evaluating a single residual block.
///
/// The corrected Jacobian itself lives in a caller-provided buffer (see
/// [`compute_block_into`]) — this struct only describes how to scatter it
/// into the global Jacobian.
pub(crate) struct BlockLinearization {
    /// Maps each variable to (local_col_offset, dof_size) within the block Jacobian.
    /// SmallVec inline storage covers all current factor types (≤ 4 variables per factor).
    pub variable_local_idx_size_list: SmallVec<[(usize, usize); 8]>,
    /// Starting row index in the global residual/Jacobian
    pub residual_row_start_idx: usize,
    /// Residual dimension for this block
    pub residual_dim: usize,
}

/// Split a mutable buffer into non-overlapping slices at the given (start, len) offsets.
///
/// `sorted_offsets_lens` must be sorted ascending by `start` and non-overlapping.
/// Uses chained `split_at_mut` calls — no unsafe code needed.
pub(crate) fn split_by_row_offsets_mut<'a>(
    buf: &'a mut [f64],
    sorted_offsets_lens: &[(usize, usize)],
) -> Vec<&'a mut [f64]> {
    let mut remaining = buf;
    let mut result = Vec::with_capacity(sorted_offsets_lens.len());
    let mut current = 0usize;
    for &(start, len) in sorted_offsets_lens {
        let gap = start - current;
        let (_, rest) = remaining.split_at_mut(gap);
        let (slice, rest2) = rest.split_at_mut(len);
        result.push(slice);
        remaining = rest2;
        current = start + len;
    }
    result
}

/// Static per-solve assembly data, built once and reused by every iteration.
///
/// The set of residual blocks and their layout is fixed for the lifetime of a
/// solve, so the block ordering, slice offsets and scratch buffers are computed
/// a single time instead of being rebuilt (collected, sorted, allocated) on
/// each linearization.
pub struct AssemblyWorkspace {
    /// Residual block keys ordered by `residual_row_start_idx`.
    pub(crate) block_order: Vec<FactorKey>,
    /// `(row_start, len)` per ordered block — drives residual slice splitting.
    pub(crate) offsets_lens: Vec<(usize, usize)>,
    /// Flat per-block Jacobian scratch arena, sized `Σ rows·cols`.
    pub(crate) jac_arena: Vec<f64>,
    /// `(start, len)` per ordered block into `jac_arena` — contiguous.
    pub(crate) jac_offsets: Vec<(usize, usize)>,
    /// Reusable global residual buffer.
    pub(crate) residual_buf: Vec<f64>,
    /// Reusable CSC value array the Jacobian blocks scatter into.
    pub(crate) jacobian_values: Vec<f64>,
}

impl AssemblyWorkspace {
    /// Build the workspace for `problem`. Call once per solve.
    pub(crate) fn build(problem: &Problem) -> Self {
        let mut blocks: Vec<(FactorKey, &ResidualBlock)> =
            problem.residual_blocks().iter().collect();
        blocks.sort_by_key(|(_, b)| b.residual_row_start_idx);

        let mut offsets_lens = Vec::with_capacity(blocks.len());
        let mut jac_offsets = Vec::with_capacity(blocks.len());
        let mut total_jac_len = 0usize;
        for (_, block) in &blocks {
            offsets_lens.push((block.residual_row_start_idx, block.factor.residual_dim()));
            let (r, c) = block.factor.jacobian_shape();
            jac_offsets.push((total_jac_len, r * c));
            total_jac_len += r * c;
        }

        AssemblyWorkspace {
            block_order: blocks.into_iter().map(|(k, _)| k).collect(),
            offsets_lens,
            jac_arena: vec![0.0; total_jac_len],
            jac_offsets,
            residual_buf: vec![0.0; problem.total_residual_dimension],
            // Sized on first use: nnz comes from the symbolic structure, which
            // the workspace does not carry.
            jacobian_values: Vec::new(),
        }
    }

    /// Empty workspace for degenerate problems and tests.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        AssemblyWorkspace {
            block_order: Vec::new(),
            offsets_lens: Vec::new(),
            jac_arena: Vec::new(),
            jac_offsets: Vec::new(),
            residual_buf: Vec::new(),
            jacobian_values: Vec::new(),
        }
    }
}

/// Evaluate a single residual block: call `factor.linearize()`, apply loss correction,
/// write the corrected residual into the provided slice, return the Jacobian buffer.
///
/// This is the shared core used by both sparse and dense assembly. It:
/// 1. Gathers `&[f64]` parameter slices for each variable (zero copy from manifold storage)
/// 2. Calls `factor.linearize()` — writes into `residual_slice` and a local Jacobian buffer
/// 3. Applies the robust loss function correction (if any)
/// 4. Returns the corrected Jacobian buffer and metadata for the caller to scatter
pub(crate) fn compute_block_into(
    residual_block: &ResidualBlock,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    residual_slice: &mut [f64],
    mut jacobian_buf: Option<&mut [f64]>,
) -> LinearizerResult<(BlockLinearization, f64)> {
    let mut param_slices: SmallVec<[&[f64]; 8]> = SmallVec::new();
    let mut variable_local_idx_size_list: SmallVec<[(usize, usize); 8]> = SmallVec::new();
    let mut count_variable_local_idx: usize = 0;

    for &var_key in &residual_block.variable_keys {
        // Skipping an unresolved key would silently shorten `param_slices`, and
        // the factor then indexes past its end deep inside `linearize`. The
        // scatter phase already reports this condition as an error; report it
        // here too rather than turning it into a panic downstream.
        let variable = variables.get(var_key).ok_or_else(|| {
            LinearizerError::Variable(format!(
                "residual block references variable key {var_key:?}, which is not in the \
                 variable map"
            ))
            .log()
        })?;
        param_slices.push(variable.as_param_slice());
        let var_size = variable.dof();
        variable_local_idx_size_list.push((count_variable_local_idx, var_size));
        count_variable_local_idx += var_size;
    }

    let (rows, cols) = residual_block.factor.jacobian_shape();
    match jacobian_buf.as_deref_mut() {
        Some(buf) => {
            debug_assert_eq!(buf.len(), rows * cols);
            let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(buf, rows, cols);
            residual_block
                .factor
                .linearize(&param_slices, residual_slice, Some(jac_mut));
        }
        None => residual_block
            .factor
            .linearize(&param_slices, residual_slice, None),
    }

    // Whiten by the block's noise model (r̃ = S·r, J̃ = S·J) BEFORE the robust
    // corrector, so Triggs math, cost accounting and covariance all see the
    // weighted problem. Null is a zero-cost no-op.
    residual_block.noise.whiten_residual(residual_slice);
    if let Some(buf) = jacobian_buf.as_deref_mut() {
        residual_block.noise.whiten_jacobian(buf, rows, cols);
    }

    // Apply robust-loss correction in-place: no heap allocation.
    //
    // The corrected residual/Jacobian drive the linear system; the returned
    // cost is the true robust cost `0.5·ρ(‖S·r‖²)`, not the corrected norm.
    let squared_norm: f64 = residual_slice.iter().map(|x| x * x).sum();
    let cost = if let Some(loss_func) = &residual_block.loss_func {
        let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
        // Jacobian correction must read the original (un-corrected) residual.
        if let Some(buf) = &mut jacobian_buf {
            corrector.correct_jacobian_in_place(residual_slice, buf, rows, cols);
        }
        corrector.correct_residual_in_place(residual_slice);
        corrector.robust_cost()
    } else {
        0.5 * squared_norm
    };

    Ok((
        BlockLinearization {
            variable_local_idx_size_list,
            residual_row_start_idx: residual_block.residual_row_start_idx,
            residual_dim: rows,
        },
        cost,
    ))
}

// ============================================================================
// AssemblyBackend trait (bridges linearizer output with optimizer solver types)
// ============================================================================

/// Type-level backend for assembling (residuals, Jacobian) and performing
/// matrix operations. Implemented by [`SparseMode`] and [`DenseMode`].
///
/// All methods are static — this trait is used as a compile-time strategy
/// selector, not as an object interface. Extends [`LinearizationMode`] with
/// the five operations an optimizer needs each iteration: building `(r, J)`,
/// scaling `J`, unscaling `dx`, and `H·v`.
///
/// All three optimizers (LM, GN, DogLeg) are generic over `M: AssemblyBackend`,
/// giving zero-cost static dispatch through the entire pipeline.
pub trait AssemblyBackend: LinearizationMode {
    /// Assemble residuals and Jacobian from the problem.
    ///
    /// Reuses the per-solve [`AssemblyWorkspace`] scratch buffers; the workspace
    /// must have been built from the same `problem`.
    fn assemble(
        problem: &Problem,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        symbolic_structure: Option<&SymbolicStructure>,
        total_dof: usize,
        workspace: &mut AssemblyWorkspace,
    ) -> LinearizerResult<(Mat<f64>, Self::Jacobian)>;

    /// Compute column norms of the Jacobian (for Jacobi scaling).
    fn compute_column_norms(jacobian: &Self::Jacobian) -> Vec<f64>;

    /// Apply diagonal column scaling to the Jacobian.
    /// Returns a new Jacobian with columns scaled by `1 / (1 + norm)`.
    fn apply_column_scaling(jacobian: &Self::Jacobian, scaling: &[f64]) -> Self::Jacobian;

    /// Apply inverse scaling to a step vector: step_i *= scaling_i
    fn apply_inverse_scaling(step: &Mat<f64>, scaling: &[f64]) -> Mat<f64>;

    /// Hessian-vector product: H * v (needed by DogLeg for Cauchy point)
    fn hessian_vec_product(hessian: &Self::Hessian, vec: &Mat<f64>) -> Mat<f64>;
}

impl AssemblyBackend for SparseMode {
    fn assemble(
        problem: &Problem,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        symbolic_structure: Option<&SymbolicStructure>,
        _total_dof: usize,
        workspace: &mut AssemblyWorkspace,
    ) -> LinearizerResult<(Mat<f64>, SparseColMat<usize, f64>)> {
        let sym = symbolic_structure.ok_or_else(|| {
            LinearizerError::InvalidInput("SparseMode requires symbolic structure".to_string())
        })?;
        crate::linearizer::cpu::sparse::assemble_sparse(
            problem,
            variables,
            variable_index_map,
            sym,
            workspace,
        )
    }

    fn compute_column_norms(jacobian: &SparseColMat<usize, f64>) -> Vec<f64> {
        let ncols = jacobian.ncols();
        let sparse_ref = jacobian.as_ref();
        (0..ncols)
            .into_par_iter()
            .map(|c| {
                let col_norm_squared: f64 =
                    sparse_ref.val_of_col(c).iter().map(|&val| val * val).sum();
                col_norm_squared.sqrt()
            })
            .collect()
    }

    fn apply_column_scaling(
        jacobian: &SparseColMat<usize, f64>,
        scaling: &[f64],
    ) -> SparseColMat<usize, f64> {
        // Scale the value array column-by-column in parallel. The sparsity
        // pattern is unchanged, so no triplet build, sort or sparse product is
        // needed — the previous diagonal-matrix product did exactly that.
        let ncols = jacobian.ncols();
        let col_ptr = jacobian.symbolic().col_ptr();
        let mut values = jacobian.as_ref().val().to_vec();
        let offsets_lens: Vec<(usize, usize)> = (0..ncols)
            .map(|c| (col_ptr[c], col_ptr[c + 1] - col_ptr[c]))
            .collect();

        let columns = split_by_row_offsets_mut(&mut values, &offsets_lens);
        columns
            .into_par_iter()
            .zip((0..ncols).into_par_iter())
            .for_each(|(column, c)| {
                let s = scaling[c];
                for v in column {
                    *v *= s;
                }
            });

        let symbolic = match jacobian.symbolic().to_owned() {
            Ok(symbolic) => symbolic,
            // Fall back to the unscaled Jacobian — mirrors the old behavior
            // when the diagonal scaling matrix could not be built.
            Err(_) => return jacobian.clone(),
        };
        SparseColMat::new(symbolic, values)
    }

    fn apply_inverse_scaling(step: &Mat<f64>, scaling: &[f64]) -> Mat<f64> {
        let mut result = step.clone();
        for i in 0..step.nrows() {
            result[(i, 0)] *= scaling[i];
        }
        result
    }

    fn hessian_vec_product(hessian: &SparseColMat<usize, f64>, vec: &Mat<f64>) -> Mat<f64> {
        use std::ops::Mul;
        hessian.as_ref().mul(vec)
    }
}

impl AssemblyBackend for DenseMode {
    fn assemble(
        problem: &Problem,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        _symbolic_structure: Option<&SymbolicStructure>,
        total_dof: usize,
        workspace: &mut AssemblyWorkspace,
    ) -> LinearizerResult<(Mat<f64>, Mat<f64>)> {
        crate::linearizer::cpu::dense::assemble_dense(
            problem,
            variables,
            variable_index_map,
            total_dof,
            workspace,
        )
    }

    fn compute_column_norms(jacobian: &Mat<f64>) -> Vec<f64> {
        let ncols = jacobian.ncols();
        (0..ncols)
            .map(|c| {
                let mut norm_sq = 0.0;
                for r in 0..jacobian.nrows() {
                    let v = jacobian[(r, c)];
                    norm_sq += v * v;
                }
                norm_sq.sqrt()
            })
            .collect()
    }

    fn apply_column_scaling(jacobian: &Mat<f64>, scaling: &[f64]) -> Mat<f64> {
        let mut result = jacobian.clone();
        for c in 0..jacobian.ncols() {
            for r in 0..jacobian.nrows() {
                result[(r, c)] *= scaling[c];
            }
        }
        result
    }

    fn apply_inverse_scaling(step: &Mat<f64>, scaling: &[f64]) -> Mat<f64> {
        let mut result = step.clone();
        for i in 0..step.nrows() {
            result[(i, 0)] *= scaling[i];
        }
        result
    }

    fn hessian_vec_product(hessian: &Mat<f64>, vec: &Mat<f64>) -> Mat<f64> {
        hessian * vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{VarKey, problem::Problem},
        factors,
        linalg::JacobianMode,
    };
    use apex_manifolds::ManifoldType;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;
    use slotmap::{SecondaryMap, SlotMap};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct LinearFactor {
        target: f64,
    }

    impl factors::Factor for LinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    fn one_var_problem() -> (Problem, VarKey) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None);
        (problem, k)
    }

    #[allow(clippy::type_complexity)]
    fn make_index_map(
        problem: &Problem,
    ) -> (
        SlotMap<VarKey, Box<dyn crate::core::variable::ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        usize,
    ) {
        let variables = problem.variables.clone();
        let mut index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        let mut offset = 0;
        for (k, v) in &variables {
            index_map.insert(k, offset);
            offset += v.dof();
        }
        (variables, index_map, offset)
    }

    // -------------------------------------------------------------------------
    // compute_block_into
    // -------------------------------------------------------------------------

    /// An unresolved variable key must surface as a typed error.
    ///
    /// Skipping it silently shortens `param_slices`, and the factor then indexes
    /// past its end inside `linearize` — a panic deep in parallel assembly
    /// instead of an error at the boundary. `Problem` rejects unknown keys at
    /// registration today, so this is defence for the paths that will remove
    /// variables (sliding-window marginalization).
    #[test]
    fn test_compute_block_into_rejects_missing_variable() -> TestResult {
        let (problem, _k) = one_var_problem();
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no blocks")?;

        // An empty variable map cannot resolve the block's key.
        let variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let mut residual_slice = vec![0.0f64; 1];
        let mut jac_buf = vec![0.0f64; 1];

        let Err(err) =
            compute_block_into(block, &variables, &mut residual_slice, Some(&mut jac_buf))
        else {
            panic!("missing variable key must be an error, not a silent skip");
        };
        assert!(
            matches!(err, LinearizerError::Variable(_)),
            "expected a Variable error, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn test_compute_block_into_residual_value() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, _, _) = make_index_map(&problem);
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no blocks")?;
        let mut residual_slice = vec![0.0f64; 1];
        let mut jac_buf = vec![0.0f64; 1];
        compute_block_into(block, &variables, &mut residual_slice, Some(&mut jac_buf))?;
        assert!((residual_slice[0] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_compute_block_into_jacobian_shape() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, _, _) = make_index_map(&problem);
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no blocks")?;
        let mut residual_slice = vec![0.0f64; 1];
        let mut jac_buf = vec![0.0f64; 1];
        let (result, _cost) =
            compute_block_into(block, &variables, &mut residual_slice, Some(&mut jac_buf))?;
        assert_eq!(result.residual_dim, 1);
        assert_eq!(jac_buf.len(), 1); // 1×1
        Ok(())
    }

    #[test]
    fn test_compute_block_into_variable_local_idx() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, _, _) = make_index_map(&problem);
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no blocks")?;
        let mut residual_slice = vec![0.0f64; 1];
        let mut jac_buf = vec![0.0f64; 1];
        let (result, _cost) =
            compute_block_into(block, &variables, &mut residual_slice, Some(&mut jac_buf))?;
        assert_eq!(result.variable_local_idx_size_list.len(), 1);
        let (local_idx, size) = result.variable_local_idx_size_list[0];
        assert_eq!(local_idx, 0);
        assert_eq!(size, 1);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // split_by_row_offsets_mut
    // -------------------------------------------------------------------------

    #[test]
    fn test_split_by_row_offsets_mut_basic() {
        let mut buf = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let offsets = vec![(0, 2), (3, 2)];
        let slices = split_by_row_offsets_mut(&mut buf, &offsets);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0], &[1.0, 2.0]);
        assert_eq!(slices[1], &[4.0, 5.0]);
    }

    #[test]
    fn test_split_by_row_offsets_mut_write() {
        let mut buf = vec![0.0f64; 4];
        let offsets = vec![(0, 2), (2, 2)];
        {
            let mut slices = split_by_row_offsets_mut(&mut buf, &offsets);
            slices[0][0] = 1.0;
            slices[0][1] = 2.0;
            slices[1][0] = 3.0;
            slices[1][1] = 4.0;
        }
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // -------------------------------------------------------------------------
    // SparseMode AssemblyBackend
    // -------------------------------------------------------------------------

    #[test]
    fn test_sparse_backend_assemble() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, index_map, total_dof) = make_index_map(&problem);
        let sym = crate::linearizer::cpu::sparse::build_symbolic_structure(
            &problem, &variables, &index_map, total_dof,
        )?;
        let (residual, _) = SparseMode::assemble(
            &problem,
            &variables,
            &index_map,
            Some(&sym),
            total_dof,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_sparse_backend_assemble_no_symbolic_returns_error() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, index_map, total_dof) = make_index_map(&problem);
        let result = SparseMode::assemble(
            &problem,
            &variables,
            &index_map,
            None,
            total_dof,
            &mut AssemblyWorkspace::build(&problem),
        );
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_sparse_backend_compute_column_norms() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, index_map, total_dof) = make_index_map(&problem);
        let sym = crate::linearizer::cpu::sparse::build_symbolic_structure(
            &problem, &variables, &index_map, total_dof,
        )?;
        let (_, jacobian) = SparseMode::assemble(
            &problem,
            &variables,
            &index_map,
            Some(&sym),
            total_dof,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        let norms = SparseMode::compute_column_norms(&jacobian);
        assert_eq!(norms.len(), 1);
        assert!((norms[0] - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_sparse_backend_apply_column_scaling() -> TestResult {
        let (problem, _k) = one_var_problem();
        let (variables, index_map, total_dof) = make_index_map(&problem);
        let sym = crate::linearizer::cpu::sparse::build_symbolic_structure(
            &problem, &variables, &index_map, total_dof,
        )?;
        let (_, jacobian) = SparseMode::assemble(
            &problem,
            &variables,
            &index_map,
            Some(&sym),
            total_dof,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        let scaling = vec![0.5_f64];
        let scaled = SparseMode::apply_column_scaling(&jacobian, &scaling);
        let val = scaled.as_ref().val_of_col(0)[0];
        assert!((val - 0.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_sparse_backend_apply_inverse_scaling() {
        let step = Mat::from_fn(1, 1, |_, _| 1.0_f64);
        let scaling = vec![2.0_f64];
        let result = SparseMode::apply_inverse_scaling(&step, &scaling);
        assert!((result[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_backend_hessian_vec_product() -> TestResult {
        let triplets = vec![faer::sparse::Triplet::new(0usize, 0usize, 4.0_f64)];
        let h = SparseColMat::try_new_from_triplets(1, 1, &triplets)?;
        let v = Mat::from_fn(1, 1, |_, _| 2.0_f64);
        let result = SparseMode::hessian_vec_product(&h, &v);
        assert!((result[(0, 0)] - 8.0).abs() < 1e-12);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // DenseMode AssemblyBackend
    // -------------------------------------------------------------------------

    #[test]
    fn test_dense_backend_assemble() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None);
        let (variables, index_map, total_dof) = make_index_map(&problem);
        let (residual, _) = DenseMode::assemble(
            &problem,
            &variables,
            &index_map,
            None,
            total_dof,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_dense_backend_compute_column_norms() {
        let jacobian = Mat::from_fn(1, 1, |_, _| 1.0_f64);
        let norms = DenseMode::compute_column_norms(&jacobian);
        assert_eq!(norms.len(), 1);
        assert!((norms[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dense_backend_apply_column_scaling() {
        let jacobian = Mat::from_fn(1, 1, |_, _| 1.0_f64);
        let scaling = vec![0.5_f64];
        let scaled = DenseMode::apply_column_scaling(&jacobian, &scaling);
        assert!((scaled[(0, 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_dense_backend_apply_inverse_scaling() {
        let step = Mat::from_fn(1, 1, |_, _| 1.0_f64);
        let scaling = vec![2.0_f64];
        let result = DenseMode::apply_inverse_scaling(&step, &scaling);
        assert!((result[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_dense_backend_hessian_vec_product() {
        let h = Mat::from_fn(1, 1, |_, _| 4.0_f64);
        let v = Mat::from_fn(1, 1, |_, _| 2.0_f64);
        let result = DenseMode::hessian_vec_product(&h, &v);
        assert!((result[(0, 0)] - 8.0).abs() < 1e-12);
    }
}
