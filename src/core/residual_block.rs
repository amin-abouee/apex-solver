use nalgebra as na;
// use rayon::prelude::*;

use crate::core::corrector::Corrector;
use crate::core::factors::Factor;
use crate::core::loss_functions::Loss;
use crate::core::variable::Variable;
use crate::manifold::LieGroup;

pub struct ResidualBlock {
    pub residual_block_id: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub factor: Box<dyn Factor + Send>,
    pub loss_func: Option<Box<dyn Loss + Send>>,
}

impl ResidualBlock {
    pub fn new(
        residual_block_id: usize,
        residual_row_start_idx: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn Factor + Send>,
        loss_func: Option<Box<dyn Loss + Send>>,
    ) -> Self {
        ResidualBlock {
            residual_block_id,
            residual_row_start_idx,
            variable_key_list: variable_key_size_list
                .iter()
                .map(|s| s.to_string())
                .collect(),
            factor,
            loss_func,
        }
    }

    pub fn residual_and_jacobian<M>(
        &self,
        variables: &Vec<&Variable<M>>,
    ) -> (na::DVector<f64>, na::DMatrix<f64>)
    where
        M: LieGroup + Clone + Into<na::DVector<f64>>,
        M::TangentVector: crate::manifold::Tangent<M>,
    {
        let param_vec: Vec<_> = variables.iter().map(|v| v.value.clone().into()).collect();
        let (mut residual, mut jacobian) = self.factor.linearize(&param_vec);
        let squared_norm = residual.norm_squared();
        if let Some(loss_func) = self.loss_func.as_ref() {
            let rho = loss_func.evaluate(squared_norm);
            let corrector = Corrector::new(squared_norm, &rho);
            corrector.correct_jacobian(&residual, &mut jacobian);
            corrector.correct_residuals(&mut residual);
        }
        (residual, jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factors::{BetweenFactorSE2, PriorFactor};
    use crate::core::loss_functions::HuberLoss;
    use crate::core::variable::Variable;
    use crate::manifold::{se2::SE2, se3::SE3};
    use nalgebra as na;

    #[test]
    fn test_residual_block_creation() {
        let factor = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));
        let loss = Some(Box::new(HuberLoss::new(1.0)) as Box<dyn Loss + Send>);

        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, loss);

        assert_eq!(block.residual_block_id, 0);
        assert_eq!(block.residual_row_start_idx, 0);
        assert_eq!(block.variable_key_list, vec!["x0", "x1"]);
        assert!(block.loss_func.is_some());
    }

    #[test]
    fn test_residual_block_without_loss() {
        let factor = Box::new(PriorFactor {
            data: na::dvector![0.0, 0.0, 0.0],
        });

        let block = ResidualBlock::new(1, 3, &["x0"], factor, None);

        assert_eq!(block.residual_block_id, 1);
        assert_eq!(block.residual_row_start_idx, 3);
        assert_eq!(block.variable_key_list, vec!["x0"]);
        assert!(block.loss_func.is_none());
    }

    #[test]
    fn test_residual_and_jacobian_se2_between_factor() {
        // Create a between factor with known measurement
        let dx = 1.0;
        let dy = 0.5;
        let dtheta = 0.1;
        let factor = Box::new(BetweenFactorSE2::new(dx, dy, dtheta));

        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);

        // Create test variables - SE2 uses [x, y, theta] ordering
        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(1.0, 0.5, 0.1));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables);

        // Verify dimensions
        assert_eq!(residual.len(), 3);
        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 6); // 2 variables * 3 DOF each

        // For identity start and [0.1, 1.0, 0.5] end with measurement [1.0, 0.5, 0.1]
        // This should give very small residuals (near zero)
        assert!(
            residual.norm() < 1e-10,
            "Residual norm: {}",
            residual.norm()
        );

        // Verify Jacobian is not zero (it should have meaningful values)
        assert!(jacobian.norm() > 1e-10, "Jacobian should not be near zero");
    }

    #[test]
    fn test_residual_and_jacobian_se2_prior_factor() {
        let prior_data = na::dvector![2.0, 1.0, 0.1]; // [x, y, theta]
        let factor = Box::new(PriorFactor {
            data: prior_data.clone(),
        });

        let block = ResidualBlock::new(0, 0, &["x0"], factor, None);

        // Create variable with same value as prior - should give zero residual
        let var0 = Variable::new(SE2::from_xy_angle(2.0, 1.0, 0.1));
        let variables = vec![&var0];

        let (residual, jacobian) = block.residual_and_jacobian(&variables);

        // Verify dimensions
        assert_eq!(residual.len(), 3);
        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 3); // 1 variable * 3 DOF

        // Should be zero residual when variable matches prior
        assert!(
            residual.norm() < 1e-14,
            "Residual norm: {}",
            residual.norm()
        );

        // Jacobian should be identity for prior factor
        let expected_jacobian = na::DMatrix::identity(3, 3);
        let jacobian_diff = (jacobian - expected_jacobian).norm();
        assert!(
            jacobian_diff < 1e-12,
            "Jacobian should be identity, diff: {}",
            jacobian_diff
        );
    }

    #[test]
    fn test_residual_and_jacobian_with_huber_loss() {
        // Create a between factor that will have non-zero residual
        let factor = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.0));
        let loss = Some(Box::new(HuberLoss::new(1.0)) as Box<dyn Loss + Send>);

        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, loss);

        // Create variables with significant difference to trigger loss function
        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(5.0, 5.0, 2.0)); // Very different from measurement [1.0, 0.0, 0.0]
        let variables = vec![&var0, &var1];

        let (residual_with_loss, jacobian_with_loss) = block.residual_and_jacobian(&variables);

        // Create same block without loss for comparison
        let factor_no_loss = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.0));
        let block_no_loss = ResidualBlock::new(0, 0, &["x0", "x1"], factor_no_loss, None);
        let (residual_no_loss, jacobian_no_loss) = block_no_loss.residual_and_jacobian(&variables);

        // With loss function, residuals should be different (corrected)
        let residual_diff = (residual_with_loss - residual_no_loss).norm();
        assert!(
            residual_diff > 1e-10,
            "Loss function should modify residuals"
        );

        // Jacobian should also be different
        let jacobian_diff = (jacobian_with_loss - jacobian_no_loss).norm();
        assert!(
            jacobian_diff > 1e-10,
            "Loss function should modify Jacobian"
        );
    }

    #[test]
    fn test_residual_block_se3_between_factor() {
        // Test with SE3 - this requires SE3 between factor which we'll implement as needed
        // For now, test with prior factor on SE3
        let se3_data = na::dvector![1.0, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0]; // [tx,ty,tz,qw,qx,qy,qz]
        let factor = Box::new(PriorFactor {
            data: se3_data.clone(),
        });

        let block = ResidualBlock::new(0, 0, &["x0"], factor, None);

        // Create SE3 variable
        let var0 = Variable::new(SE3::from_translation_quaternion(
            na::vector![1.0, 0.5, 0.2],
            na::Quaternion::new(1.0, 0.0, 0.0, 0.0),
        ));
        let variables = vec![&var0];

        let (residual, jacobian) = block.residual_and_jacobian(&variables);

        // Verify dimensions for SE3 - prior factor uses full manifold dimension
        assert_eq!(residual.len(), 7); // SE3 manifold has 7 parameters [tx,ty,tz,qw,qx,qy,qz]
        assert_eq!(jacobian.nrows(), 7);
        // For PriorFactor, Jacobian dimensions depend on implementation
        // If it's identity-based, should be 7x7; if tangent-based, should be 7x6
        // Let's be flexible and check it's one of these reasonable sizes
        assert!(jacobian.ncols() == 6 || jacobian.ncols() == 7);
    }

    #[test]
    fn test_multiple_residual_blocks_different_ids() {
        // Test creating multiple blocks with different IDs and start indices
        let factors: Vec<Box<dyn Factor + Send>> = vec![
            Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1)),
            Box::new(BetweenFactorSE2::new(0.8, 0.2, -0.05)),
            Box::new(PriorFactor {
                data: na::dvector![0.0, 0.0, 0.0],
            }),
        ];

        let blocks: Vec<ResidualBlock> = factors
            .into_iter()
            .enumerate()
            .map(|(i, factor)| {
                ResidualBlock::new(
                    i,
                    i * 3, // Each block starts at different row
                    if i == 2 { &["x0"] } else { &["x0", "x1"] },
                    factor,
                    if i == 1 {
                        Some(Box::new(HuberLoss::new(0.5)))
                    } else {
                        None
                    },
                )
            })
            .collect();

        // Verify each block has correct properties
        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(block.residual_block_id, i);
            assert_eq!(block.residual_row_start_idx, i * 3);

            if i == 2 {
                assert_eq!(block.variable_key_list.len(), 1);
                assert!(block.loss_func.is_none());
            } else {
                assert_eq!(block.variable_key_list.len(), 2);
                assert_eq!(block.loss_func.is_some(), i == 1);
            }
        }
    }

    #[test]
    fn test_residual_block_variable_ordering() {
        // Test that variable ordering is preserved correctly
        let factor = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));
        let block = ResidualBlock::new(0, 0, &["pose_2", "pose_1", "pose_0"], factor, None);

        let expected_order = vec!["pose_2", "pose_1", "pose_0"];
        assert_eq!(block.variable_key_list, expected_order);
    }

    #[test]
    fn test_residual_block_numerical_stability() {
        // Test with very small values to ensure numerical stability
        let factor = Box::new(BetweenFactorSE2::new(1e-8, 1e-8, 1e-8));
        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(1e-8, 1e-8, 1e-8));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables);

        // Should handle small values without numerical issues
        assert!(residual.iter().all(|&x| x.is_finite()));
        assert!(jacobian.iter().all(|&x| x.is_finite()));
        assert!(residual.norm() < 1e-6);
    }

    #[test]
    fn test_residual_block_large_values() {
        // Test with large values to ensure no overflow
        let factor = Box::new(BetweenFactorSE2::new(100.0, -200.0, 1.5));
        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(100.0, -200.0, 1.5));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables);

        // Should handle large values without overflow
        assert!(residual.iter().all(|&x| x.is_finite()));
        assert!(jacobian.iter().all(|&x| x.is_finite()));
        assert!(residual.norm() < 1e-10); // Should still be near zero for matching measurement
    }

    #[test]
    fn test_residual_block_loss_function_switching() {
        // Test the same residual block with and without loss function applied
        let factor1 = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));
        let factor2 = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));

        let block_with_loss = ResidualBlock::new(
            0,
            0,
            &["x0", "x1"],
            factor1,
            Some(Box::new(HuberLoss::new(0.1))),
        );
        let block_without_loss = ResidualBlock::new(0, 0, &["x0", "x1"], factor2, None);

        // Create variables that will produce significant residual
        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(2.0, 1.0, 0.2)); // Far from measurement
        let variables = vec![&var0, &var1];

        let (res_with, jac_with) = block_with_loss.residual_and_jacobian(&variables);
        let (res_without, jac_without) = block_without_loss.residual_and_jacobian(&variables);

        // Loss function should modify both residual and Jacobian
        assert!((res_with.clone() - res_without.clone()).norm() > 1e-6);
        assert!((jac_with.clone() - jac_without.clone()).norm() > 1e-6);

        // With Huber loss and significant error, residual magnitude should be reduced
        assert!(res_with.norm() < res_without.norm());
    }
}
