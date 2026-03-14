//! Jacobian assembly strategies for different matrix formats.
//!
//! This module separates assembly logic from the [`Problem`](super::problem::Problem) struct,
//! enabling clean support for sparse, dense, and future GPU backends. Each submodule handles
//! a specific matrix format:
//!
//! - [`sparse`]: Sparse Jacobian assembly using `SparseColMat` and symbolic structure
//! - [`dense`]: Dense Jacobian assembly using `Mat<f64>`
//!
//! Shared linearization logic (factor evaluation, loss correction, residual accumulation)
//! is provided by [`linearize_block()`] to avoid code duplication between assembly strategies.

pub mod dense;
pub mod sparse;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use faer::Col;
use nalgebra::{DMatrix, DVector};

use crate::{
    core::{CoreError, corrector::Corrector, residual_block::ResidualBlock},
    error::ApexSolverResult,
};

use super::problem::VariableEnum;

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
