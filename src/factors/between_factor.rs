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
    /// Uses analytical Jacobians computed via chain rule through four steps:
    /// 1. **Inverse**: `T_j⁻¹` with Jacobian ∂(T_j⁻¹)/∂T_j
    /// 2. **Composition**: `T_j⁻¹ ⊕ T_i` with Jacobians ∂/∂T_j⁻¹ and ∂/∂T_i
    /// 3. **Composition**: `(T_j⁻¹ ⊕ T_i) ⊕ T_ij` with Jacobian ∂/∂(T_j⁻¹ ⊕ T_i)
    /// 4. **Logarithm**: `log(...)` with Jacobian ∂log/∂(...)
    ///
    /// The final Jacobian is computed using the chain rule:
    /// ```text
    /// J = ∂log/∂diff · ∂diff/∂poses
    /// ```
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

        // Step 1: se3_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = T::zero_jacobian();
        let se3_k1_inv = se3_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se3_k1_inv * se3_origin_k0
        let mut j_k1_k0_wrt_k1_inv = T::zero_jacobian();
        let mut j_k1_k0_wrt_k0 = T::zero_jacobian();
        let se3_k1_k0 = se3_k1_inv.compose(
            &se3_origin_k0,
            Some(&mut j_k1_k0_wrt_k1_inv),
            Some(&mut j_k1_k0_wrt_k0),
        );

        // Step 3: se3_k1_k0 * se3_k0_k1_measured
        let mut j_diff_wrt_k1_k0 = T::zero_jacobian();
        let se3_diff = se3_k1_k0.compose(se3_k0_k1_measured, Some(&mut j_diff_wrt_k1_k0), None);

        // Step 4: se3_diff.log()
        let mut j_log_wrt_diff = T::zero_jacobian();
        let residual = se3_diff.log(Some(&mut j_log_wrt_diff));

        let jacobian = if compute_jacobian {
            // Calculate dimensions dynamically based on manifold DOF
            let dof = se3_origin_k0.tangent_dim();

            // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
            let j_diff_wrt_k0 = j_diff_wrt_k1_k0.clone() * j_k1_k0_wrt_k0;
            let j_diff_wrt_k1 = j_diff_wrt_k1_k0 * j_k1_k0_wrt_k1_inv * j_k1_inv_wrt_k1;

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
