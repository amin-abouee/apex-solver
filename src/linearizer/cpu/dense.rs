//! Dense Jacobian assembly for small-to-medium problems.
//!
//! This module assembles the Jacobian directly into a dense `Mat<f64>`, avoiding
//! sparse data structure overhead. Optimal for problems with < ~500 DOF where
//! the Jacobian is not extremely sparse.

use faer::Mat;
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

/// Assemble residuals and dense Jacobian from the current variable values.
pub fn assemble_dense(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    total_dof: usize,
) -> LinearizerResult<(Mat<f64>, Mat<f64>)> {
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

    // Pre-allocate one Jacobian buffer per block — see sparse.rs for rationale.
    let mut jacobian_buffers: Vec<Vec<f64>> = blocks
        .iter()
        .map(|b| {
            let (r, c) = b.factor.jacobian_shape();
            vec![0.0f64; r * c]
        })
        .collect();

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

    // Scatter Jacobian blocks into dense matrix (serial)
    let mut jacobian_dense = Mat::<f64>::zeros(problem.total_residual_dimension, total_dof);
    for ((bl, block), jac_buf) in block_results
        .iter()
        .zip(blocks.iter())
        .zip(jacobian_buffers.iter())
    {
        scatter_dense_block(bl, block, variable_index_map, jac_buf, &mut jacobian_dense)?;
    }

    // Convert residual buffer to faer Mat
    let n = problem.total_residual_dimension;
    let residual_faer = Mat::from_fn(n, 1, |i, _| residual_buf[i]);

    Ok((residual_faer, jacobian_dense))
}

fn scatter_dense_block(
    bl: &BlockLinearization,
    residual_block: &crate::core::residual_block::ResidualBlock,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    jacobian_buf: &[f64],
    jacobian_dense: &mut Mat<f64>,
) -> LinearizerResult<()> {
    for (i, &var_key) in residual_block.variable_keys.iter().enumerate() {
        let &global_col = variable_index_map.get(var_key).ok_or_else(|| {
            LinearizerError::Variable(format!(
                "VarKey {:?} missing in variable-to-column-index mapping",
                var_key
            ))
            .log()
        })?;
        let (local_col, var_size) = bl.variable_local_idx_size_list[i];
        for col in 0..var_size {
            let col_start = (local_col + col) * bl.residual_dim;
            for row in 0..bl.residual_dim {
                jacobian_dense[(bl.residual_row_start_idx + row, global_col + col)] =
                    jacobian_buf[col_start + row];
            }
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
        let mut problem = Problem::new(JacobianMode::Dense);
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
    fn test_assemble_dense_basic() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, jacobian) =
            assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_jacobian_dimensions() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, jacobian) =
            assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(residual.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian.ncols(), total_dof);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_zero_residual() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let k = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 3.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, _) = assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;
        assert!(residual[(0, 0)].abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_two_variables() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let kx = problem.add_variable(ManifoldType::RN, dvector![2.0]);
        let ky = problem.add_variable(ManifoldType::RN, dvector![7.0]);
        problem.add_residual_block(&[kx], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&[ky], Box::new(LinearFactor { target: 0.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, jacobian) =
            assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(jacobian.nrows(), 2);
        assert_eq!(jacobian.ncols(), 2);
        let rsum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((rsum - 9.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_dense_residual_faer_shape() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, _) = assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(residual.nrows(), 1);
        assert_eq!(residual.ncols(), 1);
        Ok(())
    }

    struct BinaryLinearFactor {
        target_x: f64,
        target_y: f64,
    }

    impl factors::Factor for BinaryLinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target_x;
            residual[1] = params[1][0] - self.target_y;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
                *jac.rb_mut().get_mut(1, 1) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            2
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (2, 2)
        }
    }

    #[test]
    fn test_assemble_dense_binary_factor() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let kx = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        let ky = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(
            &[kx, ky],
            Box::new(BinaryLinearFactor {
                target_x: 0.0,
                target_y: 0.0,
            }),
            None,
        );
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, jacobian) =
            assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;

        assert_eq!(residual.nrows(), 2);
        assert_eq!(jacobian.nrows(), 2);
        assert_eq!(jacobian.ncols(), 2);

        let r_sum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((r_sum - 8.0).abs() < 1e-10, "residual sum = {r_sum}");

        let jac_sum: f64 = (0..2)
            .map(|c| (0..2).map(|r| jacobian[(r, c)]).sum::<f64>())
            .sum();
        assert!(
            (jac_sum - 2.0).abs() < 1e-10,
            "sum of jacobian entries = {jac_sum}"
        );
        Ok(())
    }

    #[test]
    fn test_assemble_dense_missing_variable_key_returns_error() -> TestResult {
        let (problem, _) = one_var_problem();
        let (_, total_dof) = build_index_map(&problem);

        let empty: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        let result = assemble_dense(&problem, &problem.variables, &empty, total_dof);
        assert!(
            result.is_err(),
            "missing variable key should produce an Err"
        );
        Ok(())
    }

    #[test]
    fn test_assemble_dense_multi_block_residual_values() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let k = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 1.0 }), None);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 4.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let (residual, jacobian) =
            assemble_dense(&problem, &problem.variables, &index_map, total_dof)?;

        assert_eq!(residual.nrows(), 2);

        let vals: std::collections::HashSet<i64> =
            (0..2).map(|i| (residual[(i, 0)] * 1e6) as i64).collect();
        assert!(vals.contains(&2_000_000), "Missing residual entry 2.0");
        assert!(vals.contains(&-1_000_000), "Missing residual entry -1.0");

        assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((jacobian[(1, 0)] - 1.0).abs() < 1e-10);
        Ok(())
    }
}
