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

use crate::{
    core::CoreError,
    error::{ApexSolverError, ApexSolverResult},
};

use super::super::problem::{Problem, VariableEnum};
use super::linearize_block;

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
) -> ApexSolverResult<SymbolicStructure> {
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
        CoreError::SymbolicStructure("Failed to build symbolic sparse matrix structure".to_string())
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
) -> ApexSolverResult<(Mat<f64>, SparseColMat<usize, f64>)> {
    let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(
        problem.total_residual_dimension,
    )));

    let total_nnz = symbolic_structure.pattern.compute_nnz();

    // Evaluate all blocks in parallel, collecting sparse Jacobian values
    let jacobian_blocks: Result<Vec<(usize, Vec<f64>)>, ApexSolverError> = problem
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

    let residual_faer = total_residual.as_ref().as_mat().to_owned();
    let jacobian_sparse = SparseColMat::new_from_argsort(
        symbolic_structure.pattern.clone(),
        &symbolic_structure.order,
        jacobian_values.as_slice(),
    )
    .map_err(|e| {
        CoreError::SymbolicStructure("Failed to create sparse Jacobian from argsort".to_string())
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
) -> ApexSolverResult<Vec<f64>> {
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
            return Err(CoreError::Variable(format!(
                "Missing key {} in variable-to-column-index mapping",
                var_key
            ))
            .log()
            .into());
        }
    }

    Ok(local_jacobian_values)
}
