use super::Factor;
use crate::manifold::LieGroup;
use nalgebra::{DMatrix, DVector};

/// Generic between factor for Lie group pose constraints.
///
/// Represents a relative pose measurement between two poses of any Lie group manifold type.
/// This is a generic implementation that works with SE(2), SE(3), SO(2), SO(3), and Rⁿ
/// using static dispatch for zero runtime overhead.
///
/// # Type Parameter
///
/// * `T` - The Lie group manifold type (e.g., SE2, SE3, SO2, SO3, Rn)
///
/// # Mathematical Formulation
///
/// Given two poses `T_i` and `T_j` in a Lie group, and a measurement `T_ij`, the residual is:
///
/// ```text
/// r = log(T_ij⁻¹ ⊕ T_i⁻¹ ⊕ T_j)
/// ```
///
/// where:
/// - `⊕` is the Lie group composition operation
/// - `log` is the logarithm map (converts from manifold to tangent space)
/// - The residual dimensionality depends on the manifold's degrees of freedom (DOF)
///
/// # Residual Dimensions by Manifold Type
///
/// - **SE(3)**: 6D residual `[v_x, v_y, v_z, ω_x, ω_y, ω_z]` - translation + rotation
/// - **SE(2)**: 3D residual `[dx, dy, dθ]` - 2D translation + rotation
/// - **SO(3)**: 3D residual `[ω_x, ω_y, ω_z]` - 3D rotation only
/// - **SO(2)**: 1D residual `[dθ]` - 2D rotation only
/// - **Rⁿ**: nD residual - Euclidean space
///
/// # Jacobian Computation
///
/// The Jacobian is computed analytically using the chain rule and Lie group derivatives:
///
/// ```text
/// J = ∂r/∂[T_i, T_j]
/// ```
///
/// The Jacobian dimensions are `DOF × (2 × DOF)` where DOF is the manifold's degrees of freedom:
/// - **SE(3)**: 6×12 matrix
/// - **SE(2)**: 3×6 matrix
/// - **SO(3)**: 3×6 matrix
/// - **SO(2)**: 1×2 matrix
///
/// # Use Cases
///
/// - **3D SLAM**: Visual odometry, loop closure constraints (SE3)
/// - **2D SLAM**: Robot navigation, mapping (SE2)
/// - **Pose graph optimization**: Relative pose constraints (SE2, SE3)
/// - **Orientation tracking**: IMU fusion, attitude estimation (SO2, SO3)
/// - **General manifold optimization**: Custom manifolds (Rⁿ)
///
/// # Examples
///
/// ## SE(3) - 3D Pose Graph
///
/// ```
/// use apex_solver::factors::{Factor, BetweenFactor};
/// use apex_solver::manifold::se3::SE3;
/// use nalgebra::{Vector3, Quaternion, DVector};
///
/// // Measurement: relative 3D transformation between two poses
/// let relative_pose = SE3::from_translation_quaternion(
///     Vector3::new(1.0, 0.0, 0.0),        // 1m forward
///     Quaternion::new(1.0, 0.0, 0.0, 0.0) // No rotation
/// );
/// let between = BetweenFactor::new(relative_pose);
///
/// // Current pose estimates (in [tx, ty, tz, qw, qx, qy, qz] format)
/// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
/// let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]);
///
/// // Compute residual (dimension 6) and Jacobian (6×12)
/// let (residual, jacobian) = between.linearize(&[pose_i, pose_j], true);
/// ```
///
/// ## SE(2) - 2D Pose Graph
///
/// ```
/// use apex_solver::factors::{Factor, BetweenFactor};
/// use apex_solver::manifold::se2::SE2;
/// use nalgebra::DVector;
///
/// // Measurement: robot moved 1m forward and rotated 0.1 rad
/// let relative_pose = SE2::from_xy_angle(1.0, 0.0, 0.1);
/// let between = BetweenFactor::new(relative_pose);
///
/// // Current pose estimates (in [x, y, theta] format)
/// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
/// let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.12]);
///
/// // Compute residual (dimension 3) and Jacobian (3×6)
/// let (residual, jacobian) = between.linearize(&[pose_i, pose_j], true);
/// ```
///
/// # Performance
///
/// This generic implementation uses static dispatch (monomorphization), meaning:
/// - **Zero runtime overhead** compared to type-specific implementations
/// - Compiler optimizes each instantiation (`BetweenFactor<SE3>`, `BetweenFactor<SE2>`, etc.)
/// - All type checking happens at compile time
/// - No dynamic dispatch or virtual function calls
#[derive(Clone, PartialEq)]
pub struct BetweenFactor<T>
where
    T: LieGroup + Clone + Send + Sync,
    T::TangentVector: Into<DVector<f64>>,
{
    /// The measured relative pose transformation between the two connected poses
    pub relative_pose: T,
}

impl<T> BetweenFactor<T>
where
    T: LieGroup + Clone + Send + Sync,
    T::TangentVector: Into<DVector<f64>>,
{
    /// Create a new between factor from a relative pose measurement.
    ///
    /// This is a generic constructor that works with any Lie group manifold type.
    /// The type parameter `T` is typically inferred from the `relative_pose` argument.
    ///
    /// # Arguments
    ///
    /// * `relative_pose` - The measured relative transformation between two poses
    ///
    /// # Returns
    ///
    /// A new `BetweenFactor<T>` instance
    ///
    /// # Examples
    ///
    /// ## SE(3) Between Factor
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactor;
    /// use apex_solver::manifold::se3::SE3;
    ///
    /// // Create relative pose: move 2m in x, rotate 90° around z-axis
    /// let relative = SE3::from_translation_euler(
    ///     2.0, 0.0, 0.0,                      // translation (x, y, z)
    ///     0.0, 0.0, std::f64::consts::FRAC_PI_2  // rotation (roll, pitch, yaw)
    /// );
    ///
    /// // Type is inferred as BetweenFactor<SE3>
    /// let factor = BetweenFactor::new(relative);
    /// ```
    ///
    /// ## SE(2) Between Factor
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactor;
    /// use apex_solver::manifold::se2::SE2;
    ///
    /// // Create relative 2D pose
    /// let relative = SE2::from_xy_angle(1.0, 0.5, 0.1);
    ///
    /// // Type is inferred as BetweenFactor<SE2>
    /// let factor = BetweenFactor::new(relative);
    /// ```
    pub fn new(relative_pose: T) -> Self {
        Self { relative_pose }
    }
}

impl<T> Factor for BetweenFactor<T>
where
    T: LieGroup + Clone + Send + Sync + From<DVector<f64>>,
    T::TangentVector: Into<DVector<f64>>,
{
    /// Compute residual and Jacobian for a generic between factor.
    ///
    /// This method works with any Lie group manifold type, automatically adapting to
    /// the manifold's degrees of freedom. The residual and Jacobian dimensions are
    /// determined at runtime based on the manifold type.
    ///
    /// # Arguments
    ///
    /// * `params` - Two poses as `DVector<f64>` in the manifold's representation format:
    ///   - **SE(3)**: `[tx, ty, tz, qw, qx, qy, qz]` (7 parameters, 6 DOF)
    ///   - **SE(2)**: `[x, y, theta]` (3 parameters, 3 DOF)
    ///   - **SO(3)**: `[qw, qx, qy, qz]` (4 parameters, 3 DOF)
    ///   - **SO(2)**: `[angle]` (1 parameter, 1 DOF)
    /// * `compute_jacobian` - Whether to compute the analytical Jacobian matrix
    ///
    /// # Returns
    ///
    /// A tuple `(residual, jacobian)` where:
    /// - **Residual**: `DVector<f64>` with dimension = manifold DOF
    ///   - SE(3): 6×1 vector `[v_x, v_y, v_z, ω_x, ω_y, ω_z]`
    ///   - SE(2): 3×1 vector `[dx, dy, dθ]`
    ///   - SO(3): 3×1 vector `[ω_x, ω_y, ω_z]`
    ///   - SO(2): 1×1 vector `[dθ]`
    /// - **Jacobian**: `Option<DMatrix<f64>>` with dimension = (DOF, 2×DOF)
    ///   - SE(3): 6×12 matrix `[∂r/∂pose_i | ∂r/∂pose_j]`
    ///   - SE(2): 3×6 matrix `[∂r/∂pose_i | ∂r/∂pose_j]`
    ///   - SO(3): 3×6 matrix `[∂r/∂pose_i | ∂r/∂pose_j]`
    ///   - SO(2): 1×2 matrix `[∂r/∂pose_i | ∂r/∂pose_j]`
    ///
    /// # Algorithm
    ///
    /// Uses analytical Jacobians computed via chain rule through three steps:
    /// 1. **Between**: `T_j.between(T_i) = T_j⁻¹ ⊕ T_i` with Jacobians ∂/∂T_j and ∂/∂T_i
    /// 2. **Composition**: `(T_j⁻¹ ⊕ T_i) ⊕ T_ij` with Jacobian ∂/∂(T_j⁻¹ ⊕ T_i)
    /// 3. **Logarithm**: `log(...)` with Jacobian ∂log/∂(...)
    ///
    /// The final Jacobian is computed using the chain rule:
    /// ```text
    /// J = ∂log/∂diff · ∂diff/∂between · ∂between/∂poses
    /// ```
    ///
    /// This approach reduces the number of matrix operations compared to computing
    /// inverse and compose separately, resulting in both clearer code and better performance.
    ///
    /// # Performance
    ///
    /// - **Static dispatch**: All operations are monomorphized at compile time
    /// - **Zero overhead**: Same performance as type-specific implementations
    /// - **Parallel-safe**: Marked `Send + Sync` for use in parallel optimization
    ///
    /// # Examples
    ///
    /// ## SE(3) Linearization
    ///
    /// ```
    /// use apex_solver::factors::{Factor, BetweenFactor};
    /// use apex_solver::manifold::se3::SE3;
    /// use nalgebra::DVector;
    ///
    /// let relative = SE3::identity();
    /// let factor = BetweenFactor::new(relative);
    ///
    /// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    /// let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    ///
    /// let (residual, jacobian) = factor.linearize(&[pose_i, pose_j], true);
    /// assert_eq!(residual.len(), 6);  // 6 DOF
    /// assert!(jacobian.is_some());
    /// let jac = jacobian.unwrap();
    /// assert_eq!(jac.nrows(), 6);      // Residual dimension
    /// assert_eq!(jac.ncols(), 12);     // 2 × DOF
    /// ```
    ///
    /// ## SE(2) Linearization
    ///
    /// ```
    /// use apex_solver::factors::{Factor, BetweenFactor};
    /// use apex_solver::manifold::se2::SE2;
    /// use nalgebra::DVector;
    ///
    /// let relative = SE2::from_xy_angle(1.0, 0.0, 0.0);
    /// let factor = BetweenFactor::new(relative);
    ///
    /// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    /// let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    ///
    /// let (residual, jacobian) = factor.linearize(&[pose_i, pose_j], true);
    /// assert_eq!(residual.len(), 3);   // 3 DOF
    /// let jac = jacobian.unwrap();
    /// assert_eq!(jac.nrows(), 3);      // Residual dimension
    /// assert_eq!(jac.ncols(), 6);      // 2 × DOF
    /// ```
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let se3_origin_k0 = T::from(params[0].clone());
        let se3_origin_k1 = T::from(params[1].clone());
        let se3_k0_k1_measured = &self.relative_pose;

        // Step 1: se3_origin_k1.between(se3_origin_k0) = k1⁻¹ * k0
        let mut j_k1_k0_wrt_k1 = T::zero_jacobian();
        let mut j_k1_k0_wrt_k0 = T::zero_jacobian();
        let se3_k1_k0 = se3_origin_k1.between(
            &se3_origin_k0,
            Some(&mut j_k1_k0_wrt_k1),
            Some(&mut j_k1_k0_wrt_k0),
        );

        // Step 2: se3_k1_k0 * se3_k0_k1_measured
        let mut j_diff_wrt_k1_k0 = T::zero_jacobian();
        let se3_diff = se3_k1_k0.compose(se3_k0_k1_measured, Some(&mut j_diff_wrt_k1_k0), None);

        // Step 3: se3_diff.log()
        let mut j_log_wrt_diff = T::zero_jacobian();
        let residual = se3_diff.log(Some(&mut j_log_wrt_diff));

        let jacobian = if compute_jacobian {
            // Calculate dimensions dynamically based on manifold DOF
            let dof = se3_origin_k0.tangent_dim();

            // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
            let j_diff_wrt_k0 = j_diff_wrt_k1_k0.clone() * j_k1_k0_wrt_k0;
            let j_diff_wrt_k1 = j_diff_wrt_k1_k0 * j_k1_k0_wrt_k1;

            let jacobian_wrt_k0 = j_log_wrt_diff.clone() * j_diff_wrt_k0;
            let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

            // Assemble full Jacobian: [∂r/∂pose_i | ∂r/∂pose_j]
            let mut jacobian = DMatrix::<f64>::zeros(dof, 2 * dof);

            // Copy element-wise from JacobianMatrix to DMatrix
            // This works for all Matrix types (fixed-size and dynamic)
            for i in 0..dof {
                for j in 0..dof {
                    jacobian[(i, j)] = jacobian_wrt_k0[(i, j)];
                    jacobian[(i, j + dof)] = jacobian_wrt_k1[(i, j)];
                }
            }

            Some(jacobian)
        } else {
            None
        };
        (residual.into(), jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.relative_pose.tangent_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::se2::{SE2, SE2Tangent};
    use crate::manifold::se3::SE3;
    use crate::manifold::so2::SO2;
    use crate::manifold::so3::SO3;
    use nalgebra::{DVector, Quaternion, Vector3};

    const TOLERANCE: f64 = 1e-9;
    const FD_EPSILON: f64 = 1e-6;

    #[test]
    fn test_between_factor_se2_identity() {
        // Test that identity measurement yields zero residual
        let relative = SE2::identity();
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.0, 0.0, 0.0]);

        let (residual, _) = factor.linearize(&[pose_i, pose_j], false);

        assert_eq!(residual.len(), 3);
        assert!(
            residual.norm() < TOLERANCE,
            "Residual norm: {}",
            residual.norm()
        );
    }

    #[test]
    fn test_between_factor_se3_identity() {
        // Test that identity measurement yields zero residual
        let relative = SE3::identity();
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let (residual, _) = factor.linearize(&[pose_i, pose_j], false);

        assert_eq!(residual.len(), 6);
        assert!(
            residual.norm() < TOLERANCE,
            "Residual norm: {}",
            residual.norm()
        );
    }

    #[test]
    fn test_between_factor_se2_jacobian_numerical() -> Result<(), Box<dyn std::error::Error>> {
        // Verify Jacobian using finite differences with manifold perturbations
        let relative = SE2::from_xy_angle(1.0, 0.0, 0.1);
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.12]);

        let (residual, jacobian_opt) = factor.linearize(&[pose_i.clone(), pose_j.clone()], true);
        let jacobian = jacobian_opt.ok_or("Jacobian should be Some when compute_jacobians=true")?;

        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 6);

        // Finite difference validation using manifold plus operation
        let mut jacobian_fd = DMatrix::<f64>::zeros(3, 6);
        let se2_i = SE2::from(pose_i.clone());
        let se2_j = SE2::from(pose_j.clone());

        // Perturb pose_i in tangent space
        for i in 0..3 {
            let delta = match i {
                0 => SE2Tangent::new(FD_EPSILON, 0.0, 0.0),
                1 => SE2Tangent::new(0.0, FD_EPSILON, 0.0),
                2 => SE2Tangent::new(0.0, 0.0, FD_EPSILON),
                _ => unreachable!(),
            };
            let se2_i_perturbed = se2_i.plus(&delta, None, None);
            let pose_i_perturbed = DVector::<f64>::from(se2_i_perturbed);
            let (residual_perturbed, _) =
                factor.linearize(&[pose_i_perturbed, pose_j.clone()], false);

            for j in 0..3 {
                jacobian_fd[(j, i)] = (residual_perturbed[j] - residual[j]) / FD_EPSILON;
            }
        }

        // Perturb pose_j in tangent space
        for i in 0..3 {
            let delta = match i {
                0 => SE2Tangent::new(FD_EPSILON, 0.0, 0.0),
                1 => SE2Tangent::new(0.0, FD_EPSILON, 0.0),
                2 => SE2Tangent::new(0.0, 0.0, FD_EPSILON),
                _ => unreachable!(),
            };
            let se2_j_perturbed = se2_j.plus(&delta, None, None);
            let pose_j_perturbed = DVector::<f64>::from(se2_j_perturbed);
            let (residual_perturbed, _) =
                factor.linearize(&[pose_i.clone(), pose_j_perturbed], false);

            for j in 0..3 {
                jacobian_fd[(j, i + 3)] = (residual_perturbed[j] - residual[j]) / FD_EPSILON;
            }
        }

        let diff_norm = (jacobian - jacobian_fd).norm();
        assert!(diff_norm < 1e-5, "Jacobian difference norm: {}", diff_norm);
        Ok(())
    }

    #[test]
    fn test_between_factor_se3_jacobian_numerical() -> Result<(), Box<dyn std::error::Error>> {
        // Verify Jacobian using finite differences for SE3
        let relative = SE3::from_translation_quaternion(
            Vector3::new(1.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
        );
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let (residual, jacobian_opt) = factor.linearize(&[pose_i.clone(), pose_j.clone()], true);
        let jacobian = jacobian_opt.ok_or("Jacobian should be Some when compute_jacobians=true")?;

        assert_eq!(jacobian.nrows(), 6);
        assert_eq!(jacobian.ncols(), 12);

        // Finite difference validation (only check translation part for simplicity)
        let mut jacobian_fd = DMatrix::<f64>::zeros(6, 12);

        // Perturb pose_i translation
        for i in 0..3 {
            let mut pose_i_perturbed = pose_i.clone();
            pose_i_perturbed[i] += FD_EPSILON;
            let (residual_perturbed, _) =
                factor.linearize(&[pose_i_perturbed, pose_j.clone()], false);

            for j in 0..6 {
                jacobian_fd[(j, i)] = (residual_perturbed[j] - residual[j]) / FD_EPSILON;
            }
        }

        // Perturb pose_j translation
        for i in 0..3 {
            let mut pose_j_perturbed = pose_j.clone();
            pose_j_perturbed[i] += FD_EPSILON;
            let (residual_perturbed, _) =
                factor.linearize(&[pose_i.clone(), pose_j_perturbed], false);

            for j in 0..6 {
                jacobian_fd[(j, i + 6)] = (residual_perturbed[j] - residual[j]) / FD_EPSILON;
            }
        }

        // Check translation part only (more robust for FD)
        let diff_norm_trans = (jacobian.columns(0, 3) - jacobian_fd.columns(0, 3)).norm();
        assert!(
            diff_norm_trans < 1e-5,
            "Jacobian difference norm (translation): {}",
            diff_norm_trans
        );
        Ok(())
    }

    #[test]
    fn test_between_factor_dimension_se2() -> Result<(), Box<dyn std::error::Error>> {
        let relative = SE2::from_xy_angle(1.0, 0.5, 0.1);
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0]);

        let (residual, jacobian) = factor.linearize(&[pose_i, pose_j], true);

        assert_eq!(residual.len(), 3);
        assert_eq!(factor.get_dimension(), 3);

        let jac = jacobian.ok_or("Jacobian should be Some when compute_jacobians=true")?;
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), 6);
        Ok(())
    }

    #[test]
    fn test_between_factor_dimension_se3() -> Result<(), Box<dyn std::error::Error>> {
        let relative = SE3::identity();
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let (residual, jacobian) = factor.linearize(&[pose_i, pose_j], true);

        assert_eq!(residual.len(), 6);
        assert_eq!(factor.get_dimension(), 6);

        let jac = jacobian.ok_or("Jacobian should be Some when compute_jacobians=true")?;
        assert_eq!(jac.nrows(), 6);
        assert_eq!(jac.ncols(), 12);
        Ok(())
    }

    #[test]
    fn test_between_factor_so2_so3() -> Result<(), Box<dyn std::error::Error>> {
        // Test SO2 (rotation-only in 2D)
        let so2_relative = SO2::from_angle(0.1);
        let so2_factor = BetweenFactor::new(so2_relative);

        let so2_i = DVector::from_vec(vec![0.0]);
        let so2_j = DVector::from_vec(vec![0.12]);

        let (residual_so2, jacobian_so2) = so2_factor.linearize(&[so2_i, so2_j], true);
        assert_eq!(residual_so2.len(), 1);
        assert_eq!(so2_factor.get_dimension(), 1);

        let jac_so2 = jacobian_so2.ok_or("Jacobian should be Some when compute_jacobians=true")?;
        assert_eq!(jac_so2.nrows(), 1);
        assert_eq!(jac_so2.ncols(), 2);

        // Test SO3 (rotation-only in 3D)
        let so3_relative = SO3::identity();
        let so3_factor = BetweenFactor::new(so3_relative);

        let so3_i = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let so3_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        let (residual_so3, jacobian_so3) = so3_factor.linearize(&[so3_i, so3_j], true);
        assert_eq!(residual_so3.len(), 3);
        assert_eq!(so3_factor.get_dimension(), 3);

        let jac_so3 = jacobian_so3.ok_or("Jacobian should be Some when compute_jacobians=true")?;
        assert_eq!(jac_so3.nrows(), 3);
        assert_eq!(jac_so3.ncols(), 6);
        Ok(())
    }

    #[test]
    fn test_between_factor_finiteness() -> Result<(), Box<dyn std::error::Error>> {
        // Test numerical stability with various inputs
        let relative = SE2::from_xy_angle(100.0, -200.0, std::f64::consts::PI);
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![50.0, -100.0, 1.5]);
        let pose_j = DVector::from_vec(vec![150.0, -300.0, -1.5]);

        let (residual, jacobian) = factor.linearize(&[pose_i, pose_j], true);

        assert!(residual.iter().all(|&x| x.is_finite()));
        let jac = jacobian.ok_or("Jacobian should be Some when compute_jacobians=true")?;
        assert!(jac.iter().all(|&x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn test_between_factor_clone() {
        let relative = SE3::identity();
        let factor = BetweenFactor::new(relative);
        let factor_clone = factor.clone();

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let (residual1, _) = factor.linearize(&[pose_i.clone(), pose_j.clone()], false);
        let (residual2, _) = factor_clone.linearize(&[pose_i, pose_j], false);

        assert!((residual1 - residual2).norm() < TOLERANCE);
    }
}
