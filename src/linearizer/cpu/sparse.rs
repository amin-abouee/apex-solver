//! Sparse Jacobian assembly using symbolic sparsity patterns.
//!
//! This module handles building the symbolic structure (sparsity pattern) once,
//! then efficiently filling in numerical values during each optimization iteration.
//! Uses `SparseColMat` from `faer` for the Jacobian representation.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use faer::{
    Col, Mat,
    sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat},
};
use rayon::prelude::*;

use crate::error::ErrorLogging;
use crate::linearizer::{LinearizerError, LinearizerResult};

use super::super::linearize_block;
use crate::core::problem::{Problem, VariableEnum};

/// Symbolic structure for sparse matrix operations.
///
/// Contains the sparsity pattern (which entries are non-zero) and an ordering
/// for efficient numerical computation. This is computed once at the beginning
/// and reused throughout optimization.
///
/// # Fields
///
/// - `pattern`: The symbolic sparse column matrix structure (row/col indices of non-zeros)
/// - `order`: A fill-reducing ordering/permutation for numerical stability
pub struct SymbolicStructure {
    pub pattern: SymbolicSparseColMat<usize>,
    pub order: Argsort<usize>,
}

/// Build the symbolic sparsity structure for the Jacobian matrix.
///
/// This pre-computes which entries in the Jacobian will be non-zero based on the
/// factor graph connectivity. Each factor connecting variables contributes a dense
/// block to the Jacobian; this function records all such (row, col) pairs.
///
/// Called once before optimization begins. The resulting structure is reused for
/// efficient numerical assembly in [`assemble_sparse()`].
///
/// # Arguments
///
/// * `problem` - The optimization problem (provides residual blocks and their structure)
/// * `variables` - Current variable values (provides DOF sizes)
/// * `variable_index_map` - Maps variable names to their column offset in the Jacobian
/// * `total_dof` - Total degrees of freedom (number of columns)
pub fn build_symbolic_structure(
    problem: &Problem,
    variables: &HashMap<String, VariableEnum>,
    variable_index_map: &HashMap<String, usize>,
    total_dof: usize,
) -> LinearizerResult<SymbolicStructure> {
    let mut indices = Vec::<Pair<usize, usize>>::new();

    problem
        .residual_blocks()
        .iter()
        .for_each(|(_, residual_block)| {
            let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
            let mut count_variable_local_idx: usize = 0;

            for var_key in &residual_block.variable_key_list {
                if let Some(variable) = variables.get(var_key) {
                    variable_local_idx_size_list
                        .push((count_variable_local_idx, variable.get_size()));
                    count_variable_local_idx += variable.get_size();
                }
            }

            for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                if let Some(variable_global_idx) = variable_index_map.get(var_key) {
                    let (_, var_size) = variable_local_idx_size_list[i];

                    for row_idx in 0..residual_block.factor.get_dimension() {
                        for col_idx in 0..var_size {
                            let global_row_idx = residual_block.residual_row_start_idx + row_idx;
                            let global_col_idx = variable_global_idx + col_idx;
                            indices.push(Pair::new(global_row_idx, global_col_idx));
                        }
                    }
                }
            }
        });

    let (pattern, order) = SymbolicSparseColMat::try_new_from_indices(
        problem.total_residual_dimension,
        total_dof,
        &indices,
    )
    .map_err(|e| {
        LinearizerError::SymbolicStructure(
            "Failed to build symbolic sparse matrix structure".to_string(),
        )
        .log_with_source(e)
    })?;

    Ok(SymbolicStructure { pattern, order })
}

/// Assemble residuals and sparse Jacobian from the current variable values.
///
/// Evaluates all residual blocks in parallel, collecting per-block Jacobian values
/// in column-major order. Uses the pre-computed symbolic structure to efficiently
/// construct the final `SparseColMat`.
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `variables` - Current variable values
/// * `variable_index_map` - Maps variable names to their column offset in the Jacobian
/// * `symbolic_structure` - Pre-computed sparsity pattern from [`build_symbolic_structure()`]
pub fn assemble_sparse(
    problem: &Problem,
    variables: &HashMap<String, VariableEnum>,
    variable_index_map: &HashMap<String, usize>,
    symbolic_structure: &SymbolicStructure,
) -> LinearizerResult<(Mat<f64>, SparseColMat<usize, f64>)> {
    let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(
        problem.total_residual_dimension,
    )));

    let total_nnz = symbolic_structure.pattern.compute_nnz();

    // Evaluate all blocks in parallel, collecting sparse Jacobian values
    let jacobian_blocks: Result<Vec<(usize, Vec<f64>)>, LinearizerError> = problem
        .residual_blocks()
        .par_iter()
        .map(|(_, residual_block)| {
            let values = scatter_sparse_block(
                residual_block,
                variables,
                variable_index_map,
                &total_residual,
            )?;
            let size = values.len();
            Ok((size, values))
        })
        .collect();

    let jacobian_blocks = jacobian_blocks?;

    // Flatten block values into a single contiguous array
    let mut jacobian_values = Vec::with_capacity(total_nnz);
    for (_size, mut block_values) in jacobian_blocks {
        jacobian_values.append(&mut block_values);
    }

    let total_residual = Arc::try_unwrap(total_residual)
        .map_err(|_| {
            LinearizerError::ParallelComputation(
                "Failed to unwrap Arc for total residual".to_string(),
            )
            .log()
        })?
        .into_inner()
        .map_err(|e| {
            LinearizerError::ParallelComputation(
                "Failed to extract mutex inner value for total residual".to_string(),
            )
            .log_with_source(e)
        })?;

    let residual_faer = total_residual.as_ref().as_mat().to_owned();
    let jacobian_sparse = SparseColMat::new_from_argsort(
        symbolic_structure.pattern.clone(),
        &symbolic_structure.order,
        jacobian_values.as_slice(),
    )
    .map_err(|e| {
        LinearizerError::SymbolicStructure(
            "Failed to create sparse Jacobian from argsort".to_string(),
        )
        .log_with_source(e)
    })?;

    Ok((residual_faer, jacobian_sparse))
}

/// Linearize a single block and extract Jacobian values in column-major order for sparse assembly.
///
/// Uses the shared [`linearize_block()`] helper, then scatters the block Jacobian
/// into a flat `Vec<f64>` matching the symbolic structure's expected ordering.
fn scatter_sparse_block(
    residual_block: &crate::core::residual_block::ResidualBlock,
    variables: &HashMap<String, VariableEnum>,
    variable_index_map: &HashMap<String, usize>,
    total_residual: &Arc<Mutex<Col<f64>>>,
) -> LinearizerResult<Vec<f64>> {
    let block = linearize_block(residual_block, variables, total_residual)?;

    let mut local_jacobian_values = Vec::new();
    for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
        if variable_index_map.contains_key(var_key) {
            let (variable_local_idx, var_size) = block.variable_local_idx_size_list[i];
            let variable_jac = block
                .jacobian
                .view((0, variable_local_idx), (block.residual_dim, var_size));

            for row_idx in 0..block.residual_dim {
                for col_idx in 0..var_size {
                    local_jacobian_values.push(variable_jac[(row_idx, col_idx)]);
                }
            }
        } else {
            return Err(LinearizerError::Variable(format!(
                "Missing key {} in variable-to-column-index mapping",
                var_key
            ))
            .log());
        }
    }

    Ok(local_jacobian_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::problem::Problem, factors, linalg::JacobianMode, optimizer};
    use apex_manifolds::ManifoldType;
    use nalgebra::{DMatrix, DVector, dvector};
    use std::collections::HashMap;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

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
    // build_symbolic_structure
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_symbolic_structure_nnz() -> TestResult {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = build_symbolic_structure(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert_eq!(sym.pattern.compute_nnz(), 1);
        Ok(())
    }

    #[test]
    fn test_build_symbolic_structure_dimensions() -> TestResult {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = build_symbolic_structure(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert_eq!(sym.pattern.nrows(), 1);
        assert_eq!(sym.pattern.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_build_symbolic_structure_two_factors() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 1.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![5.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = build_symbolic_structure(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert_eq!(sym.pattern.compute_nnz(), 2);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // assemble_sparse
    // -------------------------------------------------------------------------

    #[test]
    fn test_assemble_sparse_basic() -> TestResult {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = state
            .symbolic_structure
            .ok_or("symbolic_structure is None")?;
        let (residual, _) =
            assemble_sparse(&problem, &state.variables, &state.variable_index_map, &sym)?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_jacobian_value() -> TestResult {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = state
            .symbolic_structure
            .ok_or("symbolic_structure is None")?;
        let (_, jacobian) =
            assemble_sparse(&problem, &state.variables, &state.variable_index_map, &sym)?;
        let val = jacobian.as_ref().val_of_col(0)[0];
        assert!((val - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_zero_residual() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 3.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![3.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = state
            .symbolic_structure
            .ok_or("symbolic_structure is None")?;
        let (residual, _) =
            assemble_sparse(&problem, &state.variables, &state.variable_index_map, &sym)?;
        assert!(residual[(0, 0)].abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_dimensions() -> TestResult {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = state
            .symbolic_structure
            .ok_or("symbolic_structure is None")?;
        let (residual, jacobian) =
            assemble_sparse(&problem, &state.variables, &state.variable_index_map, &sym)?;
        assert_eq!(residual.nrows(), 1);
        assert_eq!(residual.ncols(), 1);
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_two_variables() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&["y"], Box::new(LinearFactor { target: 0.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![2.0]));
        init.insert("y".to_string(), (ManifoldType::RN, dvector![7.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = state
            .symbolic_structure
            .ok_or("symbolic_structure is None")?;
        let (residual, _) =
            assemble_sparse(&problem, &state.variables, &state.variable_index_map, &sym)?;
        assert_eq!(residual.nrows(), 2);
        let rsum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((rsum - 9.0).abs() < 1e-12);
        Ok(())
    }

    /// Exercises the `CoreError::Variable` error path inside `scatter_sparse_block()`.
    ///
    /// The symbolic structure is built correctly, but we pass an empty
    /// `variable_index_map` to `assemble_sparse` so the variable key lookup
    /// fails, triggering the `Missing key` error branch.
    #[test]
    fn test_assemble_sparse_missing_variable_key_returns_error() -> TestResult {
        let (problem, init) = one_var_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let sym = state
            .symbolic_structure
            .ok_or("symbolic_structure is None")?;

        // Pass an empty variable_index_map: "x" is missing → should trigger CoreError::Variable
        let empty_map = HashMap::new();
        let result = assemble_sparse(&problem, &state.variables, &empty_map, &sym);
        assert!(
            result.is_err(),
            "assemble_sparse with missing variable key should return Err"
        );
        Ok(())
    }
}
