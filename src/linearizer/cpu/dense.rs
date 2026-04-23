//! Dense Jacobian assembly for small-to-medium problems.
//!
//! This module assembles the Jacobian directly into a dense `Mat<f64>`, avoiding
//! sparse data structure overhead. Optimal for problems with < ~500 DOF where
//! the Jacobian is not extremely sparse.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use faer::{Col, Mat};
use rayon::prelude::*;

use crate::linearizer::{LinearizerError, LinearizerResult};
use crate::error::ErrorLogging;

use super::super::linearize_block;
use crate::core::problem::{Problem, VariableEnum};

/// Assemble residuals and dense Jacobian from the current variable values.
///
/// Evaluates all residual blocks in parallel, writing Jacobian blocks directly
/// into a pre-allocated dense matrix. No symbolic structure is needed.
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `variables` - Current variable values
/// * `variable_index_map` - Maps variable names to their column offset in the Jacobian
/// * `total_dof` - Total degrees of freedom (number of columns)
pub fn assemble_dense(
    problem: &Problem,
    variables: &HashMap<String, VariableEnum>,
    variable_index_map: &HashMap<String, usize>,
    total_dof: usize,
) -> LinearizerResult<(Mat<f64>, Mat<f64>)> {
    let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(
        problem.total_residual_dimension,
    )));
    let jacobian_dense = Arc::new(Mutex::new(Mat::<f64>::zeros(
        problem.total_residual_dimension,
        total_dof,
    )));

    // Evaluate all blocks in parallel
    problem.residual_blocks().par_iter().try_for_each(
        |(_, residual_block)| -> Result<(), LinearizerError> {
            let block = linearize_block(residual_block, variables, &total_residual)?;

            // Scatter Jacobian block into the dense matrix
            let mut jac_dense = jacobian_dense.lock().map_err(|e| {
                LinearizerError::ParallelComputation(
                    "Failed to acquire lock on dense Jacobian".to_string(),
                )
                .log_with_source(e)
            })?;

            for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                let col_offset = *variable_index_map.get(var_key).ok_or_else(|| {
                    LinearizerError::Variable(format!(
                        "Missing key {} in variable-to-column-index mapping",
                        var_key
                    ))
                    .log()
                })?;
                let (variable_local_idx, var_size) = block.variable_local_idx_size_list[i];
                let variable_jac = block
                    .jacobian
                    .view((0, variable_local_idx), (block.residual_dim, var_size));

                for row in 0..block.residual_dim {
                    for col in 0..var_size {
                        jac_dense[(block.residual_row_start_idx + row, col_offset + col)] =
                            variable_jac[(row, col)];
                    }
                }
            }

            Ok(())
        },
    )?;

    let total_residual = Arc::try_unwrap(total_residual)
        .map_err(|_| {
            LinearizerError::ParallelComputation("Failed to unwrap Arc for total residual".to_string())
                .log()
        })?
        .into_inner()
        .map_err(|e| {
            LinearizerError::ParallelComputation(
                "Failed to extract mutex inner value for total residual".to_string(),
            )
            .log_with_source(e)
        })?;

    let jacobian_dense = Arc::try_unwrap(jacobian_dense)
        .map_err(|_| {
            LinearizerError::ParallelComputation("Failed to unwrap Arc for dense Jacobian".to_string())
                .log()
        })?
        .into_inner()
        .map_err(|e| {
            LinearizerError::ParallelComputation(
                "Failed to extract mutex inner value for dense Jacobian".to_string(),
            )
            .log_with_source(e)
        })?;

    let residual_faer = total_residual.as_ref().as_mat().to_owned();
    Ok((residual_faer, jacobian_dense))
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

    fn one_var_dense_problem() -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
        let mut problem = Problem::new(JacobianMode::Dense);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![5.0]));
        (problem, init)
    }

    #[test]
    fn test_assemble_dense_basic() -> TestResult {
        let (problem, init) = one_var_dense_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, jacobian) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_jacobian_dimensions() -> TestResult {
        let (problem, init) = one_var_dense_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, jacobian) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert_eq!(residual.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian.ncols(), state.total_dof);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_zero_residual() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 3.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![3.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, _) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert!(residual[(0, 0)].abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_two_variables() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&["y"], Box::new(LinearFactor { target: 0.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![2.0]));
        init.insert("y".to_string(), (ManifoldType::RN, dvector![7.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, jacobian) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert_eq!(jacobian.nrows(), 2);
        assert_eq!(jacobian.ncols(), 2);
        let rsum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((rsum - 9.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_residual_faer_shape() -> TestResult {
        let (problem, init) = one_var_dense_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;
        let (residual, _) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;
        assert_eq!(residual.nrows(), 1);
        assert_eq!(residual.ncols(), 1);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // New tests for previously uncovered code paths
    // -------------------------------------------------------------------------

    /// Factor that connects two variables (verifies column scattering for multi-variable factors).
    struct BinaryLinearFactor {
        target_x: f64,
        target_y: f64,
    }

    impl factors::Factor for BinaryLinearFactor {
        fn linearize(
            &self,
            params: &[DVector<f64>],
            compute_jacobian: bool,
        ) -> (DVector<f64>, Option<DMatrix<f64>>) {
            let residual =
                nalgebra::dvector![params[0][0] - self.target_x, params[1][0] - self.target_y];
            let jacobian = if compute_jacobian {
                // 2 residuals × 2 variables = 2×2 Jacobian: identity blocks side-by-side
                let mut j = DMatrix::zeros(2, 2);
                j[(0, 0)] = 1.0;
                j[(1, 1)] = 1.0;
                Some(j)
            } else {
                None
            };
            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            2
        }
    }

    /// Test that a binary factor correctly scatters its Jacobian blocks into two separate columns.
    #[test]
    fn test_assemble_dense_binary_factor() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        problem.add_residual_block(
            &["x", "y"],
            Box::new(BinaryLinearFactor {
                target_x: 0.0,
                target_y: 0.0,
            }),
            None,
        );
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, nalgebra::dvector![3.0]));
        init.insert("y".to_string(), (ManifoldType::RN, nalgebra::dvector![5.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;

        let (residual, jacobian) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;

        // 2 residuals, 2 DOF columns
        assert_eq!(residual.nrows(), 2);
        assert_eq!(jacobian.nrows(), 2);
        assert_eq!(jacobian.ncols(), 2);

        // Residuals should be the variable values (since targets are 0)
        let r_sum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((r_sum - 8.0).abs() < 1e-10, "residual sum = {r_sum}");

        // Jacobian: each variable has its own 1×1 identity block
        // col 0 → x block, col 1 → y block (or vice versa depending on sort order)
        let jac_sum: f64 = (0..2)
            .map(|c| (0..2).map(|r| jacobian[(r, c)]).sum::<f64>())
            .sum();
        assert!(
            (jac_sum - 2.0).abs() < 1e-10,
            "sum of jacobian entries = {jac_sum}"
        );
        Ok(())
    }

    /// Test that a missing variable key in variable_index_map returns an Err.
    #[test]
    fn test_assemble_dense_missing_variable_key_returns_error() -> TestResult {
        let (problem, init) = one_var_dense_problem();
        let state = optimizer::initialize_optimization_state(&problem, &init)?;

        // Pass an empty index map — the key "x" will be missing
        let empty_map: HashMap<String, usize> = HashMap::new();
        let result = assemble_dense(&problem, &state.variables, &empty_map, state.total_dof);
        assert!(
            result.is_err(),
            "missing variable key should produce an Err"
        );
        Ok(())
    }

    /// Test that individual residual values in a multi-block problem are correct.
    #[test]
    fn test_assemble_dense_multi_block_residual_values() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 1.0 }), None);
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 4.0 }), None);
        let mut init = HashMap::new();
        init.insert("x".to_string(), (ManifoldType::RN, dvector![3.0]));
        let state = optimizer::initialize_optimization_state(&problem, &init)?;

        let (residual, jacobian) = assemble_dense(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.total_dof,
        )?;

        // 2 residual blocks of dimension 1 each → 2 rows
        assert_eq!(residual.nrows(), 2);

        // residuals: [3-1, 3-4] = [2, -1] in some order
        let vals: std::collections::HashSet<i64> =
            (0..2).map(|i| (residual[(i, 0)] * 1e6) as i64).collect();
        assert!(vals.contains(&2_000_000), "Missing residual entry 2.0");
        assert!(vals.contains(&-1_000_000), "Missing residual entry -1.0");

        // Jacobian: both rows should have a 1 in column 0 (single variable "x")
        assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((jacobian[(1, 0)] - 1.0).abs() < 1e-10);
        Ok(())
    }
}
