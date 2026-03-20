//! Jacobian linearization — the bridge between the nonlinear factor graph
//! and the linear system solved each iteration.
//!
//! This is the central module for all linearization concerns:
//! - [`linearize_block()`]: Shared factor evaluation (loss correction, residual accumulation)
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
pub mod gpu;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use faer::sparse::{SparseColMat, Triplet};
use faer::{Col, Mat};
use nalgebra::{DMatrix, DVector};

use crate::core::problem::{Problem, VariableEnum};
use crate::{
    core::{CoreError, corrector::Corrector, residual_block::ResidualBlock},
    error::{ApexSolverError, ApexSolverResult},
    linearizer::cpu::{DenseMode, LinearizationMode, SparseMode},
};

pub use cpu::sparse::SymbolicStructure;

// ============================================================================
// Block linearization (shared by sparse and dense paths)
// ============================================================================

/// Result of linearizing a single residual block.
///
/// Contains the corrected residual, Jacobian, and metadata needed by both
/// sparse and dense assembly strategies to scatter values into the global matrix.
pub(crate) struct BlockLinearization {
    /// Corrected full Jacobian matrix for this block (rows = residual_dim, cols = sum of variable DOFs)
    pub jacobian: DMatrix<f64>,
    /// Maps each variable to (local_col_offset, dof_size) within the block Jacobian
    pub variable_local_idx_size_list: Vec<(usize, usize)>,
    /// Starting row index in the global residual/Jacobian
    pub residual_row_start_idx: usize,
    /// Residual dimension for this block
    pub residual_dim: usize,
}

/// Linearize a single residual block: evaluate factor, apply loss correction, accumulate residual.
///
/// This is the shared core used by both sparse and dense assembly. It:
/// 1. Gathers parameter vectors for each variable referenced by the block
/// 2. Calls `factor.linearize()` to get the local residual and Jacobian
/// 3. Applies the robust loss function correction (if any)
/// 4. Writes the corrected residual into the shared `total_residual` vector
/// 5. Returns the corrected Jacobian and metadata for the caller to scatter
pub(crate) fn linearize_block(
    residual_block: &ResidualBlock,
    variables: &HashMap<String, VariableEnum>,
    total_residual: &Arc<Mutex<Col<f64>>>,
) -> ApexSolverResult<BlockLinearization> {
    let mut param_vectors: Vec<DVector<f64>> = Vec::new();
    let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
    let mut count_variable_local_idx: usize = 0;

    for var_key in &residual_block.variable_key_list {
        if let Some(variable) = variables.get(var_key) {
            param_vectors.push(variable.to_vector());
            let var_size = variable.get_size();
            variable_local_idx_size_list.push((count_variable_local_idx, var_size));
            count_variable_local_idx += var_size;
        }
    }

    let (mut res, jac_opt) = residual_block.factor.linearize(&param_vectors, true);
    let mut jac = jac_opt.ok_or_else(|| {
        CoreError::FactorLinearization(
            "Factor returned None for Jacobian when compute_jacobian=true".to_string(),
        )
        .log()
    })?;

    // Apply loss function if present (critical for robust optimization)
    if let Some(loss_func) = &residual_block.loss_func {
        let squared_norm = res.dot(&res);
        let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
        corrector.correct_jacobian(&res, &mut jac);
        corrector.correct_residuals(&mut res);
    }

    let row_start = residual_block.residual_row_start_idx;
    let dim = residual_block.factor.get_dimension();

    // Write residual into shared accumulator
    {
        let mut total_residual = total_residual.lock().map_err(|e| {
            CoreError::ParallelComputation("Failed to acquire lock on total residual".to_string())
                .log_with_source(e)
        })?;

        let mut total_residual_mut = total_residual.as_mut();
        for i in 0..dim {
            total_residual_mut[row_start + i] = res[i];
        }
    }

    Ok(BlockLinearization {
        jacobian: jac,
        variable_local_idx_size_list,
        residual_row_start_idx: row_start,
        residual_dim: dim,
    })
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
    fn assemble(
        problem: &Problem,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
        symbolic_structure: Option<&SymbolicStructure>,
        total_dof: usize,
    ) -> ApexSolverResult<(Mat<f64>, Self::Jacobian)>;

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
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
        symbolic_structure: Option<&SymbolicStructure>,
        _total_dof: usize,
    ) -> ApexSolverResult<(Mat<f64>, SparseColMat<usize, f64>)> {
        let sym = symbolic_structure.ok_or_else(|| {
            ApexSolverError::from(CoreError::InvalidInput(
                "SparseMode requires symbolic structure".to_string(),
            ))
        })?;
        crate::linearizer::cpu::sparse::assemble_sparse(problem, variables, variable_index_map, sym)
    }

    fn compute_column_norms(jacobian: &SparseColMat<usize, f64>) -> Vec<f64> {
        let ncols = jacobian.ncols();
        let sparse_ref = jacobian.as_ref();
        (0..ncols)
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
        let ncols = jacobian.ncols();
        let triplets: Vec<Triplet<usize, usize, f64>> =
            (0..ncols).map(|c| Triplet::new(c, c, scaling[c])).collect();
        let scaling_mat = match SparseColMat::try_new_from_triplets(ncols, ncols, &triplets) {
            Ok(mat) => mat,
            Err(_) => return jacobian.clone(),
        };
        jacobian * &scaling_mat
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
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
        _symbolic_structure: Option<&SymbolicStructure>,
        total_dof: usize,
    ) -> ApexSolverResult<(Mat<f64>, Mat<f64>)> {
        crate::linearizer::cpu::dense::assemble_dense(
            problem,
            variables,
            variable_index_map,
            total_dof,
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
    use crate::{core::problem::Problem, factors, linalg::JacobianMode, optimizer};
    use apex_manifolds::ManifoldType;
    use faer::Col;
    use nalgebra::{DMatrix, DVector, dvector};
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    struct LinearFactor {
        target: f64,
    }

    impl factors::Factor for LinearFactor {
        fn linearize(
            &self,
            params: &[DVector<f64>],
            compute_jacobian: bool,
        ) -> (DVector<f64>, Option<DMatrix<f64>>) {
            let residual = dvector![params[0][0] - self.target];
            let jacobian = if compute_jacobian {
                Some(DMatrix::from_element(1, 1, 1.0))
            } else {
                None
            };
            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
    }

    fn one_var_problem() -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![5.0]));
        (problem, init)
    }

    // -------------------------------------------------------------------------
    // linearize_block
    // -------------------------------------------------------------------------

    #[test]
    fn test_linearize_block_residual_value() -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(1)));
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no residual blocks")?;
        linearize_block(block, &state.variables, &total_residual)?;
        let r = total_residual.lock().map_err(|e| format!("{e:?}"))?;
        assert!((r[0] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_linearize_block_jacobian_shape() -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(1)));
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no residual blocks")?;
        let result = linearize_block(block, &state.variables, &total_residual)?;
        assert_eq!(result.jacobian.nrows(), 1);
        assert_eq!(result.jacobian.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_linearize_block_variable_local_idx() -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(1)));
        let block = problem
            .residual_blocks()
            .values()
            .next()
            .ok_or("no residual blocks")?;
        let result = linearize_block(block, &state.variables, &total_residual)?;
        assert_eq!(result.variable_local_idx_size_list.len(), 1);
        let (local_idx, size) = result.variable_local_idx_size_list[0];
        assert_eq!(local_idx, 0);
        assert_eq!(size, 1);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // SparseMode AssemblyBackend
    // -------------------------------------------------------------------------

    #[test]
    fn test_sparse_backend_assemble() -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, _) = SparseMode::assemble(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.symbolic_structure.as_ref(),
            state.total_dof,
        )?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_sparse_backend_assemble_no_symbolic_returns_error()
    -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let result = SparseMode::assemble(
            &problem,
            &state.variables,
            &state.variable_index_map,
            None, // no symbolic structure
            state.total_dof,
        );
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_sparse_backend_compute_column_norms() -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (_, jacobian) = SparseMode::assemble(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.symbolic_structure.as_ref(),
            state.total_dof,
        )?;
        let norms = SparseMode::compute_column_norms(&jacobian);
        assert_eq!(norms.len(), 1);
        assert!((norms[0] - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_sparse_backend_apply_column_scaling() -> Result<(), Box<dyn std::error::Error>> {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (_, jacobian) = SparseMode::assemble(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.symbolic_structure.as_ref(),
            state.total_dof,
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
    fn test_sparse_backend_hessian_vec_product() -> Result<(), Box<dyn std::error::Error>> {
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
    fn test_dense_backend_assemble() -> Result<(), Box<dyn std::error::Error>> {
        let mut problem = Problem::new(JacobianMode::Dense);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![5.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, _) = DenseMode::assemble(
            &problem,
            &state.variables,
            &state.variable_index_map,
            None,
            state.total_dof,
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
