//! Sparse Jacobian assembly using symbolic sparsity patterns.

use faer::{
    Mat,
    sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat},
};
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};

use crate::core::VarKey;
use crate::error::ErrorLogging;
use crate::linearizer::{
    BlockLinearization, LinearizerError, LinearizerResult, compute_block_into,
    split_by_row_offsets_mut,
};

use crate::core::problem::Problem;
use crate::core::variable::ManifoldVariable;

/// Symbolic structure for sparse matrix operations.
pub struct SymbolicStructure {
    pub pattern: SymbolicSparseColMat<usize>,
    pub order: Argsort<usize>,
}

/// Build the symbolic sparsity structure for the Jacobian matrix.
pub fn build_symbolic_structure(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    total_dof: usize,
) -> LinearizerResult<SymbolicStructure> {
    let mut indices = Vec::<Pair<usize, usize>>::new();

    problem.residual_blocks().iter().for_each(|(_, block)| {
        let mut var_local_sizes = Vec::<(usize, usize)>::new();
        let mut local_offset = 0;

        for &var_key in &block.variable_keys {
            if let Some(variable) = variables.get(var_key) {
                var_local_sizes.push((local_offset, variable.dof()));
                local_offset += variable.dof();
            }
        }

        for (i, &var_key) in block.variable_keys.iter().enumerate() {
            if let Some(&global_col) = variable_index_map.get(var_key) {
                if let Some((_, var_size)) = var_local_sizes.get(i) {
                    for row in 0..block.factor.residual_dim() {
                        for col in 0..*var_size {
                            indices.push(Pair::new(
                                block.residual_row_start_idx + row,
                                global_col + col,
                            ));
                        }
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
pub fn assemble_sparse(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    symbolic_structure: &SymbolicStructure,
) -> LinearizerResult<(Mat<f64>, SparseColMat<usize, f64>)> {
    let total_nnz = symbolic_structure.pattern.compute_nnz();

    // Collect and sort blocks by residual_row_start_idx so pre-split is contiguous
    let mut blocks: Vec<&crate::core::residual_block::ResidualBlock> =
        problem.residual_blocks().values().collect();
    blocks.sort_by_key(|b| b.residual_row_start_idx);

    // Pre-allocate the global residual buffer and split into non-overlapping slices
    let mut residual_buf = vec![0.0f64; problem.total_residual_dimension];
    let offsets_lens: Vec<(usize, usize)> = blocks
        .iter()
        .map(|b| (b.residual_row_start_idx, b.factor.residual_dim()))
        .collect();
    let residual_slices = split_by_row_offsets_mut(&mut residual_buf, &offsets_lens);

    // Pre-allocate one Jacobian buffer per block (sized to factor.jacobian_shape).
    // These buffers are reused as scratch for the in-place factor write and the
    // in-place loss correction — no per-block allocation inside the parallel loop.
    let mut jacobian_buffers: Vec<Vec<f64>> = blocks
        .iter()
        .map(|b| {
            let (r, c) = b.factor.jacobian_shape();
            vec![0.0f64; r * c]
        })
        .collect();

    // Parallel evaluation: each task gets a unique residual slice and a unique
    // Jacobian buffer (mutable, non-aliasing) — pure zero-copy through factor.linearize.
    let block_results: Vec<LinearizerResult<BlockLinearization>> = jacobian_buffers
        .par_iter_mut()
        .zip(residual_slices.into_par_iter())
        .zip(blocks.par_iter())
        .map(|((jac_buf, res_slice), block)| {
            jac_buf.fill(0.0);
            compute_block_into(block, variables, res_slice, jac_buf.as_mut_slice())
        })
        .collect();

    let block_results = block_results
        .into_iter()
        .collect::<LinearizerResult<Vec<_>>>()?;

    // Scatter Jacobian blocks into CSC value array (serial, pre-computed positions).
    let mut jacobian_values = Vec::with_capacity(total_nnz);
    for ((bl, block), jac_buf) in block_results
        .iter()
        .zip(blocks.iter())
        .zip(jacobian_buffers.iter())
    {
        scatter_sparse_block(bl, block, variable_index_map, jac_buf, &mut jacobian_values)?;
    }

    // Convert residual buffer to faer Mat
    let n = problem.total_residual_dimension;
    let residual_faer = faer::Mat::from_fn(n, 1, |i, _| residual_buf[i]);

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

fn scatter_sparse_block(
    bl: &BlockLinearization,
    residual_block: &crate::core::residual_block::ResidualBlock,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    jacobian_buf: &[f64],
    jacobian_values: &mut Vec<f64>,
) -> LinearizerResult<()> {
    for (i, &var_key) in residual_block.variable_keys.iter().enumerate() {
        if variable_index_map.contains_key(var_key) {
            let (local_col, var_size) = bl.variable_local_idx_size_list[i];
            // symbolic indices are pushed row-major: (row outer, col inner)
            // jacobian_buf is column-major: buf[(local_col + col) * residual_dim + row]
            for row in 0..bl.residual_dim {
                for col in 0..var_size {
                    jacobian_values.push(jacobian_buf[(local_col + col) * bl.residual_dim + row]);
                }
            }
        } else {
            return Err(LinearizerError::Variable(format!(
                "VarKey {:?} missing in variable-to-column-index mapping",
                var_key
            ))
            .log());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::problem::Problem, factors, linalg::JacobianMode};
    use apex_manifolds::ManifoldType;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;

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

    fn build_index_map(problem: &Problem) -> (SecondaryMap<VarKey, usize>, usize) {
        let mut map = SecondaryMap::new();
        let mut offset = 0;
        for (k, v) in &problem.variables {
            map.insert(k, offset);
            offset += v.dof();
        }
        (map, offset)
    }

    #[test]
    fn test_build_symbolic_structure_nnz() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(sym.pattern.compute_nnz(), 1);
        Ok(())
    }

    #[test]
    fn test_build_symbolic_structure_dimensions() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(sym.pattern.nrows(), 1);
        assert_eq!(sym.pattern.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_build_symbolic_structure_two_factors() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 1.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(sym.pattern.compute_nnz(), 2);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_basic() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(&problem, &problem.variables, &index_map, &sym)?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_jacobian_value() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (_, jacobian) = assemble_sparse(&problem, &problem.variables, &index_map, &sym)?;
        let val = jacobian.as_ref().val_of_col(0)[0];
        assert!((val - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_zero_residual() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 3.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(&problem, &problem.variables, &index_map, &sym)?;
        assert!(residual[(0, 0)].abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_dimensions() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, jacobian) = assemble_sparse(&problem, &problem.variables, &index_map, &sym)?;
        assert_eq!(residual.nrows(), 1);
        assert_eq!(residual.ncols(), 1);
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_two_variables() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let kx = problem.add_variable(ManifoldType::RN, dvector![2.0]);
        let ky = problem.add_variable(ManifoldType::RN, dvector![7.0]);
        problem.add_residual_block(&[kx], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&[ky], Box::new(LinearFactor { target: 0.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(&problem, &problem.variables, &index_map, &sym)?;
        assert_eq!(residual.nrows(), 2);
        let rsum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((rsum - 9.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_missing_variable_key_returns_error() -> TestResult {
        let (problem, _) = one_var_problem();
        let (_, total_dof) = build_index_map(&problem);
        let (index_map_full, _) = build_index_map(&problem);
        let sym =
            build_symbolic_structure(&problem, &problem.variables, &index_map_full, total_dof)?;

        let empty: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        let result = assemble_sparse(&problem, &problem.variables, &empty, &sym);
        assert!(result.is_err(), "expected Err for missing variable key");
        Ok(())
    }
}
