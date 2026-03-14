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

use crate::{
    core::CoreError,
    error::{ApexSolverError, ApexSolverResult},
};

use super::super::problem::{Problem, VariableEnum};
use super::linearize_block;

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
) -> ApexSolverResult<(Mat<f64>, Mat<f64>)> {
    let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(
        problem.total_residual_dimension,
    )));
    let jacobian_dense = Arc::new(Mutex::new(Mat::<f64>::zeros(
        problem.total_residual_dimension,
        total_dof,
    )));

    // Evaluate all blocks in parallel
    problem.residual_blocks().par_iter().try_for_each(
        |(_, residual_block)| -> Result<(), ApexSolverError> {
            let block = linearize_block(residual_block, variables, &total_residual)?;

            // Scatter Jacobian block into the dense matrix
            let mut jac_dense = jacobian_dense.lock().map_err(|e| {
                CoreError::ParallelComputation(
                    "Failed to acquire lock on dense Jacobian".to_string(),
                )
                .log_with_source(e)
            })?;

            for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                let col_offset = *variable_index_map.get(var_key).ok_or_else(|| {
                    CoreError::Variable(format!(
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
            CoreError::ParallelComputation("Failed to unwrap Arc for total residual".to_string())
                .log()
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::ParallelComputation(
                "Failed to extract mutex inner value for total residual".to_string(),
            )
            .log_with_source(e)
        })?;

    let jacobian_dense = Arc::try_unwrap(jacobian_dense)
        .map_err(|_| {
            CoreError::ParallelComputation("Failed to unwrap Arc for dense Jacobian".to_string())
                .log()
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::ParallelComputation(
                "Failed to extract mutex inner value for dense Jacobian".to_string(),
            )
            .log_with_source(e)
        })?;

    let residual_faer = total_residual.as_ref().as_mat().to_owned();
    Ok((residual_faer, jacobian_dense))
}
