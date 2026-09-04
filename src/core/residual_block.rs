//! Residual blocks that connect factors with robust loss functions.
//!
//! A `ResidualBlock` is the fundamental building block of the optimization problem. It wraps
//! a [`Factor`] (which computes residuals and Jacobians) with an optional
//! [`LossFunction`]
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
//! use apex_solver::core::{FactorKey, VarKey};
//! use apex_solver::factors::Factor;
//! use apex_solver::factors::pose::BetweenFactor;
//! use apex_solver::core::loss_functions::{LossFunction, HuberLoss};
//! use apex_solver::core::variable::Variable;
//! use apex_solver::manifold::se2::SE2;
//! use slotmap::SlotMap;
//! # use apex_solver::core::CoreResult;
//! # fn example() -> CoreResult<()> {
//!
//! let mut var_sm: SlotMap<VarKey, ()> = SlotMap::with_key();
//! let k0 = var_sm.insert(());
//! let k1 = var_sm.insert(());
//! let mut fac_sm: SlotMap<FactorKey, ()> = SlotMap::with_key();
//! let fk = fac_sm.insert(());
//!
//! // Create a between factor (measurement between two poses)
//! let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
//!
//! // Add robust loss function for outlier rejection
//! let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send + Sync>);
//!
//! // Create residual block
//! let block = ResidualBlock::new(
//!     fk,                     // Block handle
//!     0,                      // Starting row in Jacobian
//!     &[k0, k1],              // Connected variable handles
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
    CoreResult, FactorKey, VarKey, corrector::Corrector, loss_functions::LossFunction,
    noise::NoiseModel, variable::Variable,
};
use crate::factors::Factor;
use apex_manifolds::{LieGroup, Tangent};

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
    /// Stable generational handle identifying this residual block
    pub residual_block_id: FactorKey,

    /// Starting row index in the global residual vector and Jacobian matrix
    ///
    /// This allows the optimizer to place this block's residual and Jacobian contributions
    /// at the correct location in the full problem matrices.
    pub residual_row_start_idx: usize,

    /// Ordered list of variable handles that this block connects.
    ///
    /// Each `VarKey` is a stable generational index into the Problem's variable slotmap.
    /// For example, a between factor connecting two poses would have `variable_keys = [k0, k1]`.
    pub variable_keys: Vec<VarKey>,

    /// The factor that computes residuals and Jacobians
    ///
    /// Must implement the `Factor` trait and be thread-safe (`Send`).
    pub factor: Box<dyn Factor + Send + Sync>,

    /// Optional robust loss function for outlier rejection
    ///
    /// If `None`, standard least squares is used. If `Some`, the corrector algorithm
    /// is applied to downweight outliers.
    ///
    /// The `Send + Sync` bound mirrors `factor`: residual blocks are evaluated
    /// by rayon tasks that share `&ResidualBlock` references, so both the
    /// factor and the loss must be safe to share across threads — not merely
    /// safe to move between them.
    pub loss_func: Option<Box<dyn LossFunction + Send + Sync>>,

    /// Measurement noise model; whitened before the robust-loss corrector.
    pub noise: NoiseModel,
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
    /// use apex_solver::core::{FactorKey, VarKey};
    /// use apex_solver::factors::Factor;
    /// use apex_solver::factors::pose::BetweenFactor;
    /// use apex_solver::core::loss_functions::{LossFunction, HuberLoss};
    /// use apex_solver::manifold::se2::SE2;
    /// use slotmap::SlotMap;
    /// # use apex_solver::core::CoreResult;
    /// # fn example() -> CoreResult<()> {
    ///
    /// let mut var_sm: SlotMap<VarKey, ()> = SlotMap::with_key();
    /// let k0 = var_sm.insert(());
    /// let k1 = var_sm.insert(());
    /// let mut fac_sm: SlotMap<FactorKey, ()> = SlotMap::with_key();
    /// let fk = fac_sm.insert(());
    ///
    /// let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
    /// let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send + Sync>);
    ///
    /// let block = ResidualBlock::new(
    ///     fk,                 // Block handle
    ///     0,                  // Starts at row 0
    ///     &[k0, k1],          // Connected variable handles
    ///     factor,
    ///     loss,
    /// );
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn new(
        residual_block_id: FactorKey,
        residual_row_start_idx: usize,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
    ) -> Self {
        Self::with_noise(
            residual_block_id,
            residual_row_start_idx,
            variable_keys,
            factor,
            loss_func,
            NoiseModel::Null,
        )
    }

    /// Like [`Self::new`], with a measurement noise model. The whitened
    /// residual and Jacobian drive both the linear system and the reported
    /// cost (`½·ρ(‖S·r‖²)`).
    pub fn with_noise(
        residual_block_id: FactorKey,
        residual_row_start_idx: usize,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
        noise: NoiseModel,
    ) -> Self {
        ResidualBlock {
            residual_block_id,
            residual_row_start_idx,
            variable_keys: variable_keys.to_vec(),
            factor,
            loss_func,
            noise,
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
    /// use apex_solver::core::{FactorKey, VarKey};
    /// use apex_solver::factors::Factor;
    /// use apex_solver::factors::pose::BetweenFactor;
    /// use apex_solver::core::variable::Variable;
    /// use apex_solver::manifold::se2::SE2;
    /// use slotmap::SlotMap;
    ///
    /// let mut var_sm: SlotMap<VarKey, ()> = SlotMap::with_key();
    /// let k0 = var_sm.insert(());
    /// let k1 = var_sm.insert(());
    /// let mut fac_sm: SlotMap<FactorKey, ()> = SlotMap::with_key();
    /// let fk = fac_sm.insert(());
    ///
    /// let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
    /// let block = ResidualBlock::new(fk, 0, &[k0, k1], factor, None);
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
        variables: &[&Variable<M>],
    ) -> CoreResult<(DVector<f64>, DMatrix<f64>)>
    where
        M: LieGroup + Clone,
        M::TangentVector: Tangent<M>,
    {
        let param_owned: Vec<M> = variables.iter().map(|v| v.value.clone()).collect();
        let param_slices: Vec<&[f64]> = param_owned.iter().map(|v| v.as_param_slice()).collect();

        let res_dim = self.factor.residual_dim();
        let (jac_rows, jac_cols) = self.factor.jacobian_shape();

        let mut residual_buf = vec![0.0f64; res_dim];
        let mut jacobian_buf = vec![0.0f64; jac_rows * jac_cols];
        let jac_mut =
            faer::mat::MatMut::from_column_major_slice_mut(&mut jacobian_buf, jac_rows, jac_cols);
        self.factor
            .linearize(&param_slices, &mut residual_buf, Some(jac_mut));

        // Whiten by the noise model before the robust corrector (same
        // ordering as `compute_block_into`).
        self.noise.whiten_residual(&mut residual_buf);
        self.noise
            .whiten_jacobian(&mut jacobian_buf, jac_rows, jac_cols);

        if let Some(loss_func) = self.loss_func.as_ref() {
            let squared_norm: f64 = residual_buf.iter().map(|x| x * x).sum();
            let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
            // Jacobian correction must read the original (un-corrected) residual.
            corrector.correct_jacobian_in_place(
                &residual_buf,
                &mut jacobian_buf,
                jac_rows,
                jac_cols,
            );
            corrector.correct_residual_in_place(&mut residual_buf);
        }

        let residual = DVector::from_vec(residual_buf);
        let jacobian = DMatrix::from_column_slice(jac_rows, jac_cols, &jacobian_buf);

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
    use crate::factors::pose::EuclideanPriorFactor;
    use crate::factors::pose::{BetweenFactor, PriorFactor};
    use apex_manifolds::{se2::SE2, se3::SE3};
    use nalgebra::{Quaternion, dvector, vector};
    use slotmap::SlotMap;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn make_keys(n_vars: usize) -> (FactorKey, Vec<VarKey>) {
        let mut fac_sm: SlotMap<FactorKey, ()> = SlotMap::with_key();
        let fk = fac_sm.insert(());
        let mut var_sm: SlotMap<VarKey, ()> = SlotMap::with_key();
        let keys = (0..n_vars).map(|_| var_sm.insert(())).collect();
        (fk, keys)
    }

    #[test]
    fn test_residual_block_creation() -> TestResult {
        let (fk, keys) = make_keys(2);
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
        let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send + Sync>);

        let block = ResidualBlock::new(fk, 0, &keys, factor, loss);

        assert_eq!(block.residual_block_id, fk);
        assert_eq!(block.residual_row_start_idx, 0);
        assert_eq!(block.variable_keys, keys);
        assert!(block.loss_func.is_some());

        Ok(())
    }

    #[test]
    fn test_residual_block_without_loss() -> TestResult {
        let (fk, keys) = make_keys(1);
        let factor = Box::new(EuclideanPriorFactor::new(dvector![0.0, 0.0, 0.0]));

        let block = ResidualBlock::new(fk, 3, &keys, factor, None);

        assert_eq!(block.residual_block_id, fk);
        assert_eq!(block.residual_row_start_idx, 3);
        assert_eq!(block.variable_keys, keys);
        assert!(block.loss_func.is_none());

        Ok(())
    }

    #[test]
    fn test_residual_and_jacobian_se2_between_factor() -> TestResult {
        let (fk, keys) = make_keys(2);
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.5, 0.1)));
        let block = ResidualBlock::new(fk, 0, &keys, factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(1.0, 0.5, 0.1));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        assert_eq!(residual.len(), 3);
        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 6);

        assert!(
            residual.norm() < 1e-10,
            "Residual norm: {}",
            residual.norm()
        );
        assert!(jacobian.norm() > 1e-10, "Jacobian should not be near zero");

        Ok(())
    }

    #[test]
    fn test_residual_and_jacobian_with_huber_loss() -> TestResult {
        let (fk, keys) = make_keys(2);
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0)));
        let loss = Some(Box::new(HuberLoss::new(1.0)?) as Box<dyn LossFunction + Send + Sync>);
        let block = ResidualBlock::new(fk, 0, &keys, factor, loss);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(5.0, 5.0, 2.0));
        let variables = vec![&var0, &var1];

        let (residual_with_loss, jacobian_with_loss) = block.residual_and_jacobian(&variables)?;

        let (fk2, keys2) = make_keys(2);
        let factor_no_loss = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0)));
        let block_no_loss = ResidualBlock::new(fk2, 0, &keys2, factor_no_loss, None);
        let (residual_no_loss, jacobian_no_loss) =
            block_no_loss.residual_and_jacobian(&variables)?;

        assert!((residual_with_loss - residual_no_loss).norm() > 1e-10);
        assert!((jacobian_with_loss - jacobian_no_loss).norm() > 1e-10);

        Ok(())
    }

    #[test]
    fn test_residual_block_se3_between_factor() -> TestResult {
        let (fk, keys) = make_keys(1);
        let se3_data = dvector![1.0, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0];
        let factor = Box::new(PriorFactor::<SE3>::new(SE3::from_param_slice(
            se3_data.as_slice(),
        )));
        let block = ResidualBlock::new(fk, 0, &keys, factor, None);

        let var0 = Variable::new(SE3::from_translation_quaternion(
            vector![1.0, 0.5, 0.2],
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
        ));
        let variables = vec![&var0];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        // Tangent-space prior: 6-dim residual and Jacobian on the SE(3)
        // variable (the old ambient prior produced 7 of each).
        assert_eq!(residual.len(), 6);
        assert_eq!(jacobian.nrows(), 6);
        assert_eq!(jacobian.ncols(), 6);

        // Variable is identity-rotation pose at (1, 0.5, 0.2); prior is the
        // same translation with identity rotation → zero residual.
        assert!(residual.norm() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_multiple_residual_blocks_different_ids() -> TestResult {
        let mut fac_sm: SlotMap<FactorKey, ()> = SlotMap::with_key();
        let mut var_sm: SlotMap<VarKey, ()> = SlotMap::with_key();
        let k0 = var_sm.insert(());
        let k1 = var_sm.insert(());

        let configs: Vec<(FactorKey, usize, Vec<VarKey>, bool)> = vec![
            (fac_sm.insert(()), 0, vec![k0, k1], false),
            (fac_sm.insert(()), 3, vec![k0, k1], true),
            (fac_sm.insert(()), 6, vec![k0], false),
        ];

        let factors: Vec<Box<dyn Factor + Send + Sync>> = vec![
            Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1))),
            Box::new(BetweenFactor::new(SE2::from_xy_angle(0.8, 0.2, -0.05))),
            Box::new(EuclideanPriorFactor::new(dvector![0.0, 0.0, 0.0])),
        ];

        let blocks: Vec<ResidualBlock> = configs
            .iter()
            .zip(factors)
            .map(|((fk, row, keys, has_loss), factor)| -> Result<ResidualBlock, Box<dyn std::error::Error>> {
                Ok(ResidualBlock::new(
                    *fk,
                    *row,
                    keys,
                    factor,
                    if *has_loss { Some(Box::new(HuberLoss::new(0.5)?)) } else { None },
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (i, (block, (fk, row, keys, has_loss))) in blocks.iter().zip(configs.iter()).enumerate()
        {
            assert_eq!(block.residual_block_id, *fk);
            assert_eq!(block.residual_row_start_idx, *row);
            assert_eq!(block.variable_keys.len(), keys.len(), "block {i}");
            assert_eq!(block.loss_func.is_some(), *has_loss, "block {i}");
        }

        Ok(())
    }

    #[test]
    fn test_residual_block_variable_ordering() -> TestResult {
        let mut fac_sm: SlotMap<FactorKey, ()> = SlotMap::with_key();
        let mut var_sm: SlotMap<VarKey, ()> = SlotMap::with_key();
        let fk = fac_sm.insert(());
        let k2 = var_sm.insert(());
        let k1 = var_sm.insert(());
        let k0 = var_sm.insert(());

        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
        let block = ResidualBlock::new(fk, 0, &[k2, k1, k0], factor, None);

        assert_eq!(block.variable_keys, vec![k2, k1, k0]);

        Ok(())
    }

    #[test]
    fn test_residual_block_numerical_stability() -> TestResult {
        let (fk, keys) = make_keys(2);
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1e-8, 1e-8, 1e-8)));
        let block = ResidualBlock::new(fk, 0, &keys, factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(1e-8, 1e-8, 1e-8));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        assert!(residual.iter().all(|&x| x.is_finite()));
        assert!(jacobian.iter().all(|&x| x.is_finite()));
        assert!(residual.norm() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_residual_block_large_values() -> TestResult {
        let (fk, keys) = make_keys(2);
        let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(100.0, -200.0, 1.5)));
        let block = ResidualBlock::new(fk, 0, &keys, factor, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(100.0, -200.0, 1.5));
        let variables = vec![&var0, &var1];

        let (residual, jacobian) = block.residual_and_jacobian(&variables)?;

        assert!(residual.iter().all(|&x| x.is_finite()));
        assert!(jacobian.iter().all(|&x| x.is_finite()));
        assert!(residual.norm() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_residual_block_loss_function_switching() -> TestResult {
        let (fk1, keys1) = make_keys(2);
        let (fk2, keys2) = make_keys(2);

        let factor1 = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
        let factor2 = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));

        let block_with_loss = ResidualBlock::new(
            fk1,
            0,
            &keys1,
            factor1,
            Some(Box::new(HuberLoss::new(0.1)?)),
        );
        let block_without_loss = ResidualBlock::new(fk2, 0, &keys2, factor2, None);

        let var0 = Variable::new(SE2::from_xy_angle(0.0, 0.0, 0.0));
        let var1 = Variable::new(SE2::from_xy_angle(2.0, 1.0, 0.2));
        let variables = vec![&var0, &var1];

        let (res_with, jac_with) = block_with_loss.residual_and_jacobian(&variables)?;
        let (res_without, jac_without) = block_without_loss.residual_and_jacobian(&variables)?;

        assert!((res_with.clone() - res_without.clone()).norm() > 1e-6);
        assert!((jac_with.clone() - jac_without.clone()).norm() > 1e-6);
        assert!(res_with.norm() < res_without.norm());

        Ok(())
    }
}
