//! SE(2) between factor for 2D pose graph optimization.

use super::Factor;
use crate::manifold::{LieGroup, se2::SE2};
use nalgebra::{DMatrix, DVector, Matrix3};

/// Between factor for SE(2) poses (2D pose graph constraint).
///
/// Represents a relative pose measurement between two SE(2) poses in 2D. This is the
/// fundamental building block for 2D SLAM, pose graph optimization, and trajectory estimation.
///
/// # Mathematical Formulation
///
/// Given two poses `T_i` and `T_j` in SE(2), and a measurement `T_ij`, the residual is:
///
/// ```text
/// r = log(T_ij⁻¹ ⊕ T_i⁻¹ ⊕ T_j)
/// ```
///
/// where:
/// - `⊕` is SE(2) composition
/// - `log` is the SE(2) logarithm map (converts to tangent space)
/// - The residual is a 3D vector `[dx, dy, dtheta]` in the tangent space
///
/// # Jacobian Computation
///
/// The Jacobian is computed analytically using the chain rule and Lie group derivatives:
///
/// ```text
/// J = ∂r/∂[T_i, T_j]
/// ```
///
/// This is a 3×6 matrix (3 residual dimensions, 6 DOF from two SE(2) poses).
///
/// # Use Cases
///
/// - 2D SLAM: Odometry measurements, loop closure constraints
/// - Pose graph optimization: Relative pose constraints
/// - 2D trajectory estimation
///
/// # Example
///
/// ```
/// use apex_solver::factors::{Factor, BetweenFactorSE2};
/// use nalgebra::DVector;
///
/// // Measurement: robot moved 1m forward and rotated 0.1 rad
/// let between = BetweenFactorSE2::new(1.0, 0.0, 0.1);
///
/// // Current pose estimates
/// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);     // Origin
/// let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.12]);  // Slightly off
///
/// // Compute residual (should be small if poses are consistent)
/// let (residual, jacobian) = between.linearize(&[pose_i, pose_j], true);
/// println!("Residual: {:?}", residual);  // Shows deviation from measurement
/// ```
#[derive(Clone)]
pub struct BetweenFactorSE2 {
    /// The measured relative pose transformation between the two connected poses
    pub relative_pose: SE2,
}

impl BetweenFactorSE2 {
    /// Create a new SE(2) between factor from translation and rotation components.
    ///
    /// # Arguments
    ///
    /// * `dx` - Relative translation in x direction (meters)
    /// * `dy` - Relative translation in y direction (meters)
    /// * `dtheta` - Relative rotation angle (radians)
    ///
    /// # Returns
    ///
    /// A new `BetweenFactorSE2` instance
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactorSE2;
    ///
    /// // Robot moved 2m forward, 0.5m left, rotated 0.1 rad counterclockwise
    /// let factor = BetweenFactorSE2::new(2.0, 0.5, 0.1);
    /// ```
    pub fn new(dx: f64, dy: f64, dtheta: f64) -> Self {
        let relative_pose = SE2::from_xy_angle(dx, dy, dtheta);
        Self { relative_pose }
    }

    /// Create a new SE(2) between factor from an existing SE2 transformation.
    ///
    /// # Arguments
    ///
    /// * `relative_pose` - The measured relative pose
    ///
    /// # Returns
    ///
    /// A new `BetweenFactorSE2` instance
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactorSE2;
    /// use apex_solver::manifold::se2::SE2;
    ///
    /// let relative = SE2::from_xy_angle(1.0, 0.0, 0.0);
    /// let factor = BetweenFactorSE2::from_se2(relative);
    /// ```
    pub fn from_se2(relative_pose: SE2) -> Self {
        Self { relative_pose }
    }
}

impl Factor for BetweenFactorSE2 {
    /// Compute residual and Jacobian for SE(2) between factor.
    ///
    /// # Arguments
    ///
    /// * `params` - Two SE(2) poses in G2O format: `[x, y, theta]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// - Residual: 3×1 vector `[dx_error, dy_error, dtheta_error]`
    /// - Jacobian: 3×6 matrix `[∂r/∂pose_i, ∂r/∂pose_j]`
    ///
    /// # Algorithm
    ///
    /// Uses analytical Jacobians computed via chain rule through:
    /// 1. Inverse: `T_i⁻¹`
    /// 2. Composition: `T_i⁻¹ ⊕ T_j`
    /// 3. Composition: `(T_i⁻¹ ⊕ T_j) ⊕ T_ij`
    /// 4. Logarithm: `log(...)`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Use analytical jacobians for SE2 between factor (same pattern as SE3)
        // Input: params = [x, y, theta] for each pose (G2O FORMAT)
        let se2_origin_k0 = SE2::from(params[0].clone());
        let se2_origin_k1 = SE2::from(params[1].clone());
        let se2_k0_k1_measured = &self.relative_pose;

        // Step 1: se2_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = Matrix3::zeros();
        let se2_k1_inv = se2_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se2_k1_inv * se2_origin_k0
        let mut j_compose1_wrt_k1_inv = Matrix3::zeros();
        let mut j_compose1_wrt_k0 = Matrix3::zeros();
        let se2_temp = se2_k1_inv.compose(
            &se2_origin_k0,
            Some(&mut j_compose1_wrt_k1_inv),
            Some(&mut j_compose1_wrt_k0),
        );

        // Step 3: se2_temp * se2_k0_k1_measured
        let mut j_compose2_wrt_temp = Matrix3::zeros();
        let se2_diff = se2_temp.compose(se2_k0_k1_measured, Some(&mut j_compose2_wrt_temp), None);

        // Step 4: se2_diff.log()
        let mut j_log_wrt_diff = Matrix3::zeros();
        let residual = se2_diff.log(Some(&mut j_log_wrt_diff));

        let jacobian = if compute_jacobian {
            // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
            let j_temp_wrt_k1 = j_compose1_wrt_k1_inv * j_k1_inv_wrt_k1;
            let j_diff_wrt_k0 = j_compose2_wrt_temp * j_compose1_wrt_k0;
            let j_diff_wrt_k1 = j_compose2_wrt_temp * j_temp_wrt_k1;

            let jacobian_wrt_k0 = j_log_wrt_diff * j_diff_wrt_k0;
            let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

            // Assemble full Jacobian: [∂r/∂pose_i | ∂r/∂pose_j]
            let mut jacobian = DMatrix::<f64>::zeros(3, 6);
            jacobian
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&jacobian_wrt_k0);
            jacobian
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&jacobian_wrt_k1);

            Some(jacobian)
        } else {
            None
        };

        (residual.into(), jacobian)
    }

    fn get_dimension(&self) -> usize {
        3 // SE(2) between factor has 3D residual: [dx, dy, dtheta]
    }
}
