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
    AssemblyWorkspace, BlockLinearization, LinearizerError, LinearizerResult, compute_block_into,
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
///
/// Blocks are visited in `residual_row_start_idx` order, matching
/// [`AssemblyWorkspace`](crate::linearizer::AssemblyWorkspace) `block_order`
/// and therefore the order [`assemble_sparse`] pushes Jacobian values in.
/// That shared order is load-bearing: `new_from_argsort` pairs the pushed
/// values with these pairs positionally, so any disagreement silently
/// permutes `J`'s rows against `r`. Slotmap iteration order coincides with
/// row order only until rows are regrouped
/// ([`Problem::group_rows_for_elimination`](crate::core::problem::Problem::group_rows_for_elimination)),
/// which is exactly when the Schur path needs assembly to be right.
pub fn build_symbolic_structure(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    total_dof: usize,
) -> LinearizerResult<SymbolicStructure> {
    let mut indices = Vec::<Pair<usize, usize>>::new();

    let mut blocks: Vec<_> = problem.residual_blocks().iter().collect();
    blocks.sort_by_key(|(_, block)| block.residual_row_start_idx);
    blocks.iter().for_each(|(_, block)| {
        let mut var_local_sizes = Vec::<(usize, usize)>::new();
        let mut local_offset = 0;

        for &var_key in &block.variable_keys {
            if let Some(variable) = variables.get(var_key) {
                var_local_sizes.push((local_offset, variable.dof()));
                local_offset += variable.dof();
            }
        }

        for (i, &var_key) in block.variable_keys.iter().enumerate() {
            if let Some(&global_col) = variable_index_map.get(var_key)
                && let Some((_, var_size)) = var_local_sizes.get(i)
            {
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
/// Reuses the block ordering, slice offsets and scratch buffers cached in
/// `workspace` — nothing static is rebuilt or reallocated per call.
pub fn assemble_sparse(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    symbolic_structure: &SymbolicStructure,
    workspace: &mut AssemblyWorkspace,
) -> LinearizerResult<(Mat<f64>, SparseColMat<usize, f64>)> {
    let total_nnz = symbolic_structure.pattern.compute_nnz();

    // Reset the residual buffer, then split it (and the Jacobian arena) into
    // non-overlapping slices in the pre-computed block order.
    workspace.residual_buf.fill(0.0);
    let residual_slices =
        split_by_row_offsets_mut(&mut workspace.residual_buf, &workspace.offsets_lens);
    let jac_slices = split_by_row_offsets_mut(&mut workspace.jac_arena, &workspace.jac_offsets);
    let residual_blocks = problem.residual_blocks();

    // Parallel evaluation: each task gets a unique residual slice and a unique
    // Jacobian buffer (mutable, non-aliasing) — pure zero-copy through factor.linearize.
    let block_results: Vec<LinearizerResult<BlockLinearization>> = residual_slices
        .into_par_iter()
        .zip(jac_slices)
        .zip(workspace.block_order.par_iter())
        .map(|((res_slice, jac_buf), key)| {
            let block = &residual_blocks[*key];
            jac_buf.fill(0.0);
            compute_block_into(block, variables, res_slice, Some(jac_buf)).map(|(bl, _)| bl)
        })
        .collect();

    let block_results = block_results
        .into_iter()
        .collect::<LinearizerResult<Vec<_>>>()?;

    // Re-split the arena to read the corrected Jacobian blocks back for scattering.
    let jac_slices = split_by_row_offsets_mut(&mut workspace.jac_arena, &workspace.jac_offsets);

    // Scatter Jacobian blocks into CSC value array (serial, pre-computed positions).
    // Reuse the workspace buffer; `scatter_sparse_block` pushes into it, so it
    // must start empty while keeping its capacity across calls.
    let mut jacobian_values = std::mem::take(&mut workspace.jacobian_values);
    jacobian_values.clear();
    jacobian_values.reserve(total_nnz);
    for ((bl, key), jac_buf) in block_results
        .iter()
        .zip(workspace.block_order.iter())
        .zip(jac_slices.iter())
    {
        let block = &residual_blocks[*key];
        scatter_sparse_block(bl, block, variable_index_map, jac_buf, &mut jacobian_values)?;
    }

    // Convert residual buffer to faer Mat
    let n = problem.total_residual_dimension;
    let residual_faer = faer::Mat::from_fn(n, 1, |i, _| workspace.residual_buf[i]);

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
    });

    // Return the buffer to the workspace before propagating any error.
    workspace.jacobian_values = jacobian_values;

    Ok((residual_faer, jacobian_sparse?))
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

    /// Unary factor with a configurable Jacobian gain, so a row permutation
    /// of `J` is observable (uniform gains would hide it).
    struct ScaledLinearFactor {
        target: f64,
        gain: f64,
    }

    impl factors::Factor for ScaledLinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = self.gain;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
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

    /// Regrouped rows must not desynchronize the sparse Jacobian from the
    /// residual vector.
    ///
    /// Regression: the symbolic index pairs were pushed in slotmap order while
    /// Jacobian values scatter in row-start order, so after
    /// `group_rows_for_elimination` the two orders disagreed and `J`'s rows no
    /// longer matched `r`'s — silently corrupting every Schur-backed solve
    /// while Cholesky (which never regroups) stayed exact. The dense path
    /// indexes absolutely and is immune, so it serves as the oracle here.
    /// Insertion is deliberately C,B,A so slotmap order and grouped row order
    /// differ.
    #[test]
    fn test_grouped_sparse_assembly_matches_dense() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let c = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        let b = problem.add_variable(ManifoldType::RN, dvector![2.0]);
        let a = problem.add_variable(ManifoldType::RN, dvector![1.0]);
        // Distinct gains per block so a row permutation of J is observable.
        for (key, target, gain) in [(c, 30.0, 3.0), (b, 20.0, 2.0), (a, 10.0, 1.0)] {
            problem.add_residual_block(&[key], Box::new(ScaledLinearFactor { target, gain }), None);
        }
        problem.mark_for_elimination(c);
        assert!(
            problem.group_rows_for_elimination(),
            "insertion order must differ from grouped order for this test"
        );

        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let mut workspace = AssemblyWorkspace::build(&problem);
        let (r_sparse, j_sparse) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut workspace,
        )?;
        let (r_dense, j_dense) = crate::linearizer::cpu::dense::assemble_dense(
            &problem,
            &problem.variables,
            &index_map,
            total_dof,
            &mut workspace,
        )?;

        assert_eq!(r_sparse.nrows(), 3);
        for i in 0..3 {
            assert!(
                (r_sparse[(i, 0)] - r_dense[(i, 0)]).abs() < 1e-12,
                "residual row {i} differs"
            );
        }
        // Each block is unary: exactly one nonzero per row and column, and it
        // must sit in the same (row, col) in both assemblies — with distinct
        // gains per block, any row permutation shows up as a value mismatch.
        assert_eq!(j_sparse.as_ref().compute_nnz(), 3);
        for col in 0..total_dof {
            let rows = j_sparse.symbolic().row_idx_of_col_raw(col);
            let vals = j_sparse.val_of_col(col);
            assert_eq!(rows.len(), 1, "col {col} must hold exactly one entry");
            assert!(
                (vals[0] - j_dense[(rows[0], col)]).abs() < 1e-12,
                "sparse value {} disagrees with dense {} at ({}, {col})",
                vals[0],
                j_dense[(rows[0], col)],
                rows[0]
            );
            for row in 0..3 {
                if row != rows[0] {
                    assert!(
                        j_dense[(row, col)].abs() < 1e-12,
                        "dense has an unexpected entry at ({row}, {col})"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_basic() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_jacobian_value() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (_, jacobian) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
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
        let (residual, _) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert!(residual[(0, 0)].abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_dimensions() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, jacobian) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
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
        let (residual, _) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
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
        let result = assemble_sparse(
            &problem,
            &problem.variables,
            &empty,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        );
        assert!(result.is_err(), "expected Err for missing variable key");
        Ok(())
    }
}
