//! SE(3) between factor for 3D pose graph optimization.

use super::Factor;
use crate::manifold::{LieGroup, se3::SE3};
use nalgebra::{DMatrix, DVector, Matrix6};

/// Between factor for SE(3) poses (3D pose graph constraint).
///
/// Represents a relative pose measurement between two SE(3) poses in 3D. This is the
/// fundamental building block for 3D SLAM, structure from motion, and bundle adjustment.
///
/// # Mathematical Formulation
///
/// Given two poses `T_i` and `T_j` in SE(3), and a measurement `T_ij`, the residual is:
///
/// ```text
/// r = log(T_ij⁻¹ ⊕ T_i⁻¹ ⊕ T_j)
/// ```
///
/// where:
/// - `⊕` is SE(3) composition
/// - `log` is the SE(3) logarithm map (converts to tangent space se(3))
/// - The residual is a 6D vector `[v_x, v_y, v_z, ω_x, ω_y, ω_z]` in the tangent space
///
/// # Tangent Space
///
/// The 6D tangent space se(3) consists of:
/// - **Translation**: `[v_x, v_y, v_z]` - Linear velocity
/// - **Rotation**: `[ω_x, ω_y, ω_z]` - Angular velocity (axis-angle)
///
/// # Jacobian Computation
///
/// The Jacobian is computed analytically using the chain rule and Lie group derivatives:
///
/// ```text
/// J = ∂r/∂[T_i, T_j]
/// ```
///
/// This is a 6×12 matrix (6 residual dimensions, 12 DOF from two SE(3) poses).
///
/// # Use Cases
///
/// - 3D SLAM: Visual odometry, loop closure constraints
/// - Pose graph optimization: Relative pose constraints
/// - Bundle adjustment: Camera pose relationships
/// - Multi-view geometry: Relative camera poses
///
/// # Example
///
/// ```
/// use apex_solver::factors::{Factor, BetweenFactorSE3};
/// use apex_solver::manifold::se3::SE3;
/// use nalgebra::{Vector3, Quaternion, DVector};
///
/// // Measurement: relative transformation between two poses
/// let relative_pose = SE3::from_translation_quaternion(
///     Vector3::new(1.0, 0.0, 0.0),        // 1m forward
///     Quaternion::new(1.0, 0.0, 0.0, 0.0) // No rotation
/// );
/// let between = BetweenFactorSE3::new(relative_pose);
///
/// // Current pose estimates (in [tx, ty, tz, qw, qx, qy, qz] format)
/// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
/// let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]);
///
/// // Compute residual
/// let (residual, jacobian) = between.linearize(&[pose_i, pose_j], true);
/// info!("Residual dimension: {}", residual.len());  // 6
/// if let Some(jac) = jacobian {
///     info!("Jacobian shape: {} x {}", jac.nrows(), jac.ncols());  // 6x12
/// }
/// ```
#[derive(Clone)]
pub struct BetweenFactorSE3 {
    /// The measured relative pose transformation between the two connected poses
    pub relative_pose: SE3,
}

impl BetweenFactorSE3 {
    /// Create a new SE(3) between factor from a relative pose measurement.
    ///
    /// # Arguments
    ///
    /// * `relative_pose` - The measured relative SE(3) transformation
    ///
    /// # Returns
    ///
    /// A new `BetweenFactorSE3` instance
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactorSE3;
    /// use apex_solver::manifold::se3::SE3;
    ///
    /// // Create relative pose: move 2m in x, rotate 90° around z-axis
    /// let relative = SE3::from_translation_euler(
    ///     2.0, 0.0, 0.0,                      // translation (x, y, z)
    ///     0.0, 0.0, std::f64::consts::FRAC_PI_2  // rotation (roll, pitch, yaw)
    /// );
    ///
    /// let factor = BetweenFactorSE3::new(relative);
    /// ```
    pub fn new(relative_pose: SE3) -> Self {
        Self { relative_pose }
    }
}

impl Factor for BetweenFactorSE3 {
    /// Compute residual and Jacobian for SE(3) between factor.
    ///
    /// # Arguments
    ///
    /// * `params` - Two SE(3) poses in format: `[tx, ty, tz, qw, qx, qy, qz]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// - Residual: 6×1 vector `[v_x, v_y, v_z, ω_x, ω_y, ω_z]` in tangent space
    /// - Jacobian: 6×12 matrix `[∂r/∂pose_i, ∂r/∂pose_j]`
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
        let se3_origin_k0 = SE3::from(params[0].clone());
        let se3_origin_k1 = SE3::from(params[1].clone());
        let se3_k0_k1_measured = &self.relative_pose;

        // Step 1: se3_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = Matrix6::zeros();
        let se3_k1_inv = se3_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se3_k1_inv * se3_origin_k0
        let mut j_compose1_wrt_k1_inv = Matrix6::zeros();
        let mut j_compose1_wrt_k0 = Matrix6::zeros();
        let se3_temp = se3_k1_inv.compose(
            &se3_origin_k0,
            Some(&mut j_compose1_wrt_k1_inv),
            Some(&mut j_compose1_wrt_k0),
        );

        // Step 3: se3_temp * se3_k0_k1_measured
        let mut j_compose2_wrt_temp = Matrix6::zeros();
        let se3_diff = se3_temp.compose(se3_k0_k1_measured, Some(&mut j_compose2_wrt_temp), None);

        // Step 4: se3_diff.log()
        let mut j_log_wrt_diff = Matrix6::zeros();
        let residual = se3_diff.log(Some(&mut j_log_wrt_diff));

        let jacobian = if compute_jacobian {
            // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
            let j_temp_wrt_k1 = j_compose1_wrt_k1_inv * j_k1_inv_wrt_k1;
            let j_diff_wrt_k0 = j_compose2_wrt_temp * j_compose1_wrt_k0;
            let j_diff_wrt_k1 = j_compose2_wrt_temp * j_temp_wrt_k1;

            let jacobian_wrt_k0 = j_log_wrt_diff * j_diff_wrt_k0;
            let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

            // Assemble full Jacobian: [∂r/∂pose_i | ∂r/∂pose_j]
            let mut jacobian = DMatrix::<f64>::zeros(6, 12);
            jacobian
                .fixed_view_mut::<6, 6>(0, 0)
                .copy_from(&jacobian_wrt_k0);
            jacobian
                .fixed_view_mut::<6, 6>(0, 6)
                .copy_from(&jacobian_wrt_k1);

            Some(jacobian)
        } else {
            None
        };
        (residual.into(), jacobian)
    }

    fn get_dimension(&self) -> usize {
        6 // SE(3) between factor has 6D residual: [translation (3D), rotation (3D)]
    }
}
