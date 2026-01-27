//! Residual blocks that connect factors with robust loss functions.
//!
//! A `ResidualBlock` is the fundamental building block of the optimization problem. It wraps
//! a [`Factor`] (which computes residuals and Jacobians) with an optional [`Loss`] function
//! (which provides robustness to outliers). Each residual block corresponds to one measurement
//! or constraint in the factor graph.
//!
//! # Role in Optimization
//!
//! The `ResidualBlock` coordinates three key components:
//!
//! 1. **Factor**: Computes the raw residual `r(x)` and Jacobian `J = ∂r/∂x`
//! 2. **Loss function** (optional): Evaluates `ρ(||r||²)` for robust cost
//! 3. **Corrector**: Applies loss function via residual/Jacobian adjustment
//!
//! During each optimization iteration, the residual block:
//! - Evaluates the factor at current variable values
//! - Computes the squared residual norm
//! - If a loss function is present, creates a `Corrector` and applies corrections
//! - Returns the (possibly corrected) residual and Jacobian
//!
//! # Structure in the Problem
//!
//! The [`Problem`](crate::core::problem::Problem) maintains a collection of residual blocks.
//! Each block is assigned:
//! - A unique ID for identification
//! - A starting row index in the global Jacobian matrix
//! - A list of connected variable keys
//! - The factor implementation
//! - An optional loss function
//!
//! # Example
//!
//! ```
//! use apex_solver::core::residual_block::ResidualBlock;
//! use apex_solver::factors::{Factor, BetweenFactor};
//! use apex_solver::core::loss_functions::{LossFunction, HuberLoss};
//! use apex_solver::core::variable::Variable;
//! use apex_solver::manifold::se2::SE2;
//! # use apex_solver::error::ApexSolverResult;
//! # fn example() -> ApexSolverResult<()> {
//!
//! // Create a between factor (measurement between two poses)
//! let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
//!
//! // Add robust loss function for outlier rejection
//! let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send>);
//!
//! // Create residual block
//! let block = ResidualBlock::new(
//!     0,                      // Block ID
//!     0,                      // Starting row in Jacobian
//!     &["x0", "x1"],          // Connected variables
//!     factor,
//!     loss,
//! );
//!
//! // Later, during optimization:
//! let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
//! let var1 = Variable::new(SE2::from_xy_angle(1.1, 0.05, 0.12));
//! let variables = vec![&var0, &var1];
//!
//! let Ok((residual, jacobian)) = block.residual_and_jacobian(&variables) else { todo!() };
//! // residual and jacobian are now ready for the linear solver
//! # Ok(())
//! # }
//! # example().unwrap();
//! ```

use nalgebra::{DMatrix, DVector};

use crate::core::{
    CoreError, CoreResult, corrector::Corrector, loss_functions::LossFunction, variable::Variable,
};
use crate::factors::Factor;
use crate::manifold::{LieGroup, Tangent};

/// A residual block that wraps a factor with an optional robust loss function.
///
/// Each residual block represents one measurement or constraint in the optimization problem.
/// It connects one or more variables through a factor, and optionally applies a robust loss
/// function for outlier rejection.
///
/// # Fields
///
/// - `residual_block_id`: Unique identifier for this block
/// - `residual_row_start_idx`: Starting row index in the global residual/Jacobian matrix
/// - `variable_key_list`: Names of the variables connected by this block
/// - `factor`: The factor that computes residuals and Jacobians
/// - `loss_func`: Optional robust loss function (e.g., Huber, Cauchy)
///
/// # Thread Safety
///
/// Residual blocks are designed for parallel evaluation. Both the `factor` and `loss_func`
/// must be `Send` to enable parallel processing across multiple residual blocks.
pub struct ResidualBlock {
    /// Unique identifier for this residual block
    pub residual_block_id: usize,

    /// Starting row index in the global residual vector and Jacobian matrix
    ///
    /// This allows the optimizer to place this block's residual and Jacobian contributions
    /// at the correct location in the full problem matrices.
    pub residual_row_start_idx: usize,

    /// List of variable names (keys) that this block connects
    ///
    /// For example, a between factor connecting poses "x0" and "x1" would have
    /// `variable_key_list = ["x0", "x1"]`.
    pub variable_key_list: Vec<String>,

    /// The factor that computes residuals and Jacobians
    ///
    /// Must implement the `Factor` trait and be thread-safe (`Send`).
    pub factor: Box<dyn Factor + Send>,

    /// Optional robust loss function for outlier rejection
    ///
    /// If `None`, standard least squares is used. If `Some`, the corrector algorithm
    /// is applied to downweight outliers.
    pub loss_func: Option<Box<dyn LossFunction + Send>>,
}

impl ResidualBlock {
    /// Create a new residual block.
    ///
    /// # Arguments
    ///
    /// * `residual_block_id` - Unique identifier for this block
    /// * `residual_row_start_idx` - Starting row in the global residual vector
    /// * `variable_key_size_list` - Names of the connected variables (as string slices)
    /// * `factor` - Factor implementation (boxed trait object)
    /// * `loss_func` - Optional robust loss function (boxed trait object)
    ///
    /// # Returns
    ///
    /// A new `ResidualBlock` instance ready for use in optimization
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::residual_block::ResidualBlock;
    /// use apex_solver::factors::{Factor, BetweenFactor};
    /// use apex_solver::core::loss_functions::{LossFunction, HuberLoss};
    /// use apex_solver::manifold::se2::SE2;
    /// # use apex_solver::error::ApexSolverResult;
    /// # fn example() -> ApexSolverResult<()> {
    ///
    /// let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
    /// let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send>);
    ///
    /// let block = ResidualBlock::new(
    ///     0,                  // First block
    ///     0,                  // Starts at row 0
    ///     &["x0", "x1"],      // Connects two variables
    ///     factor,
    ///     loss,
    /// );
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn new(
        residual_block_id: usize,
        residual_row_start_idx: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn Factor + Send>,
        loss_func: Option<Box<dyn LossFunction + Send>>,
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

    /// Compute residual and Jacobian for this block at the given variable values.
    ///
    /// This is the core method called during each optimization iteration. It:
    /// 1. Extracts values from the provided variables
    /// 2. Calls the factor's `linearize` method
    /// 3. If a loss function is present, applies the corrector algorithm
    /// 4. Returns the (possibly corrected) residual and Jacobian
    ///
    /// # Arguments
    ///
    /// * `variables` - References to the variables connected by this block, in order
    ///
    /// # Returns
    ///
    /// Tuple `(residual, jacobian)` where:
    /// - `residual`: N-dimensional error vector (possibly downweighted by loss function)
    /// - `jacobian`: N × M matrix of derivatives (possibly corrected by loss function)
    ///
    /// # Type Parameters
    ///
    /// * `M` - The manifold type (e.g., SE2, SE3, SO3) that implements `LieGroup`
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::residual_block::ResidualBlock;
    /// use apex_solver::factors::{Factor, BetweenFactor};
    /// use apex_solver::core::variable::Variable;
    /// use apex_solver::manifold::se2::SE2;
    ///
    /// let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
    /// let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);
    ///
    /// let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
    /// let var1 = Variable::new(SE2::from_xy_angle(1.0, 0.0, 0.1));
    /// let variables = vec![&var0, &var1];
    ///
    /// let Ok((residual, jacobian)) = block.residual_and_jacobian(&variables) else { todo!() };
    /// // Use residual and jacobian in optimization linear system
    /// ```
    ///
    /// # Implementation Details
    ///
    /// When a loss function is present:
    /// - Computes `s = ||r||²` (squared residual norm)
    /// - Creates a `Corrector` using the loss function evaluation at `s`
    /// - Applies corrections to both residual and Jacobian
    /// - This effectively converts the robust problem into weighted least squares
    ///
    /// Without a loss function:
    /// - Returns raw residual and Jacobian from the factor
    /// - Equivalent to standard (non-robust) least squares
    pub fn residual_and_jacobian<M>(
        &self,
        variables: &Vec<&Variable<M>>,
    ) -> CoreResult<(DVector<f64>, DMatrix<f64>)>
    where
        M: LieGroup + Clone + Into<DVector<f64>>,
        M::TangentVector: Tangent<M>,
    {
        // Extract variable values as DVector for the factor
        let param_vec: Vec<_> = variables.iter().map(|v| v.value.clone().into()).collect();

        // Compute raw residual and Jacobian from the factor
        let (mut residual, jacobian_opt) = self.factor.linearize(&param_vec, true);
        let mut jacobian = jacobian_opt.ok_or_else(|| {
            CoreError::FactorLinearization(
                "Factor returned None for Jacobian when compute_jacobian=true".to_string(),
            )
            .log()
        })?;

        // Apply robust loss function if present
        if let Some(loss_func) = self.loss_func.as_ref() {
            // Compute squared norm: s = ||r||²
            let squared_norm = residual.norm_squared();

            // Create corrector and apply to residual and Jacobian
            let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
            corrector.correct_jacobian(&residual, &mut jacobian);
            corrector.correct_residuals(&mut residual);
        }

        Ok((residual, jacobian))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        loss_functions::{HuberLoss, LossFunction},
        variable::Variable,
    };
    use crate::factors::{BetweenFactor, PriorFactor};
    use crate::manifold::{se2::SE2, se3::SE3};
    use nalgebra::{Quaternion, dvector, vector};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_residual_block_creation() -> TestResult {
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
        let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send>);

        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, loss);

        assert_eq!(block.residual_block_id, 0);
        assert_eq!(block.residual_row_start_idx, 0);
        assert_eq!(block.variable_key_list, vec!["x0", "x1"]);
        assert!(block.loss_func.is_some());

        Ok(())
    }

    #[test]
    fn test_residual_block_without_loss() -> TestResult {
        let factor = Box::new(PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        });

        let block = ResidualBlock::new(1, 3, &["x0"], factor, None);

        assert_eq!(block.residual_block_id, 1);
        assert_eq!(block.residual_row_start_idx, 3);
        assert_eq!(block.variable_key_list, vec!["x0"]);
        assert!(block.loss_func.is_none());

        Ok(())
    }

    #[test]
    fn test_residual_and_jacobian_se2_between_factor() -> TestResult {
        // Create a between factor with known measurement
        let dx = 1.0;
        let dy = 0.5;
        let dtheta = 0.1;
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(dx, dy, dtheta)));

        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);

        // Create test variables - SE2 uses [x, y, theta] ordering
        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(1.0, 0.5, 0.1));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

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

        Ok(())
    }

    #[test]
    fn test_residual_and_jacobian_with_huber_loss() -> TestResult {
        // Create a between factor that will have non-zero residual
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0)));
        let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send>);

        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, loss);

        // Create variables with significant difference to trigger loss function
        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(5.0, 5.0, 2.0)); // Very different from measurement [1.0, 0.0, 0.0]
        let variables = vec![&var0, &var1];

        let (residual_with_loss, jacobian_with_loss) = block.residual_and_jacobian(&variables)?;

        // Create same block without loss for comparison
        let factor_no_loss = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0)));
        let block_no_loss = ResidualBlock::new(0, 0, &["x0", "x1"], factor_no_loss, None);
        let (residual_no_loss, jacobian_no_loss) =
            block_no_loss.residual_and_jacobian(&variables)?;

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

        Ok(())
    }

    #[test]
    fn test_residual_block_se3_between_factor() -> TestResult {
        // Test with SE3 - use prior factor on SE3
        let se3_data = dvector![1.0, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0]; // [tx,ty,tz,qw,qx,qy,qz]
        let factor = Box::new(PriorFactor {
            data: se3_data.clone(),
        });

        let block = ResidualBlock::new(0, 0, &["x0"], factor, None);

        // Create SE3 variable
        let var0 = Variable::new(SE3::from_translation_quaternion(
            vector![1.0, 0.5, 0.2],
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
        ));
        let variables = vec![&var0];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        // Verify dimensions for SE3 - prior factor uses full manifold dimension
        assert_eq!(residual.len(), 7); // SE3 manifold has 7 parameters [tx,ty,tz,qw,qx,qy,qz]
        assert_eq!(jacobian.nrows(), 7);
        // For PriorFactor, Jacobian dimensions depend on implementation
        // If it's identity-based, should be 7x7; if tangent-based, should be 7x6
        // Let's be flexible and check it's one of these reasonable sizes
        assert!(jacobian.ncols() == 6 || jacobian.ncols() == 7);

        Ok(())
    }

    #[test]
    fn test_multiple_residual_blocks_different_ids() -> TestResult {
        // Test creating multiple blocks with different IDs and start indices
        let factors: Vec<Box<dyn Factor + Send>> = vec![
            Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1))),
            Box::new(BetweenFactor::new(SE2::from_xy_angle(0.8, 0.2, -0.05))),
            Box::new(PriorFactor {
                data: dvector![0.0, 0.0, 0.0],
            }),
        ];

        let blocks: Vec<ResidualBlock> = factors
            .into_iter()
            .enumerate()
            .map(
                |(i, factor)| -> Result<ResidualBlock, Box<dyn std::error::Error>> {
                    Ok(ResidualBlock::new(
                        i,
                        i * 3, // Each block starts at different row
                        if i == 2 { &["x0"] } else { &["x0", "x1"] },
                        factor,
                        if i == 1 {
                            Some(Box::new(HuberLoss::new(0.5)?))
                        } else {
                            None
                        },
                    ))
                },
            )
            .collect::<Result<Vec<_>, _>>()?;

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

        Ok(())
    }

    #[test]
    fn test_residual_block_variable_ordering() -> TestResult {
        // Test that variable ordering is preserved correctly
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
        let block = ResidualBlock::new(0, 0, &["pose_2", "pose_1", "pose_0"], factor, None);

        let expected_order = vec!["pose_2", "pose_1", "pose_0"];
        assert_eq!(block.variable_key_list, expected_order);

        Ok(())
    }

    #[test]
    fn test_residual_block_numerical_stability() -> TestResult {
        // Test with very small values to ensure numerical stability
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1e-8, 1e-8, 1e-8)));
        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(1e-8, 1e-8, 1e-8));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        // Should handle small values without numerical issues
        assert!(residual.iter().all(|&x| x.is_finite()));
        assert!(jacobian.iter().all(|&x| x.is_finite()));
        assert!(residual.norm() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_residual_block_large_values() -> TestResult {
        // Test with large values to ensure no overflow
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(100.0, -200.0, 1.5)));
        let block = ResidualBlock::new(0, 0, &["x0", "x1"], factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(100.0, -200.0, 1.5));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        // Should handle large values without overflow
        assert!(residual.iter().all(|&x| x.is_finite()));
        assert!(jacobian.iter().all(|&x| x.is_finite()));
        assert!(residual.norm() < 1e-10); // Should still be near zero for matching measurement

        Ok(())
    }

    #[test]
    fn test_residual_block_loss_function_switching() -> TestResult {
        // Test the same residual block with and without loss function applied
        let factor1 = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
        let factor2 = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));

        let block_with_loss = ResidualBlock::new(
            0,
            0,
            &["x0", "x1"],
            factor1,
            Some(Box::new(HuberLoss::new(0.1)?)),
        );
        let block_without_loss = ResidualBlock::new(0, 0, &["x0", "x1"], factor2, None);

        // Create variables that will produce significant residual
        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(2.0, 1.0, 0.2)); // Far from measurement
        let variables = vec![&var0, &var1];

        let (res_with, jac_with) = block_with_loss.residual_and_jacobian(&variables)?;
        let (res_without, jac_without) = block_without_loss.residual_and_jacobian(&variables)?;

        // Loss function should modify both residual and Jacobian
        assert!((res_with.clone() - res_without.clone()).norm() > 1e-6);
        assert!((jac_with.clone() - jac_without.clone()).norm() > 1e-6);

        // With Huber loss and significant error, residual magnitude should be reduced
        assert!(res_with.norm() < res_without.norm());

        Ok(())
    }
}
