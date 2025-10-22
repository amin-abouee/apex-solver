//! Factor implementations for graph-based optimization problems.
//!
//! Factors (also called constraints or error functions) represent measurements or relationships
//! between variables in a factor graph. Each factor computes a residual (error) vector and its
//! Jacobian with respect to the connected variables.
//!
//! # Factor Graph Formulation
//!
//! In graph-based SLAM and bundle adjustment, the optimization problem is represented as:
//!
//! ```text
//! minimize Σ_i ||r_i(x)||²
//! ```
//!
//! where:
//! - `x` is the set of variables (poses, landmarks, etc.)
//! - `r_i(x)` is the residual function for factor i
//! - Each factor connects one or more variables
//!
//! # Factor Types
//!
//! - **Unary factors**: Connect to a single variable (e.g., [`PriorFactor`])
//! - **Binary factors**: Connect to two variables (e.g., [`BetweenFactorSE2`], [`BetweenFactorSE3`])
//! - **N-ary factors**: Connect to N variables (not yet implemented)
//!
//! # Linearization
//!
//! Each factor must provide a `linearize` method that computes:
//! 1. **Residual** `r(x)`: The error at the current variable values
//! 2. **Jacobian** `J = ∂r/∂x`: How the residual changes with each variable
//!
//! This information is used by the optimizer to compute parameter updates via Newton-type methods.
//!
//! # Example: Creating and Using a Factor
//!
//! ```
//! use apex_solver::core::factors::{Factor, BetweenFactorSE2};
//! use nalgebra::DVector;
//!
//! // Create a relative pose constraint between two SE2 poses
//! let between_factor = BetweenFactorSE2::new(
//!     1.0,  // dx: relative x translation
//!     0.0,  // dy: relative y translation
//!     0.1,  // dtheta: relative rotation
//! );
//!
//! // Current variable values (two SE2 poses in [x, y, theta] format)
//! let pose1 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
//! let pose2 = DVector::from_vec(vec![0.9, 0.1, 0.15]);
//!
//! // Linearize: compute residual and Jacobian
//! let params = vec![pose1, pose2];
//! let (residual, jacobian) = between_factor.linearize(&params);
//!
//! // residual: 3x1 vector showing how far pose2 deviates from expected
//! // jacobian: 3x6 matrix showing derivatives w.r.t. both poses
//! println!("Residual: {:?}", residual);
//! println!("Jacobian shape: {} x {}", jacobian.nrows(), jacobian.ncols());
//! ```

use nalgebra;

use crate::manifold::{LieGroup, se2::SE2, se3::SE3};

/// Trait for factor (constraint) implementations in factor graph optimization.
///
/// A factor represents a measurement or constraint connecting one or more variables.
/// It computes the residual (error) and Jacobian for the current variable values,
/// which are used by the optimizer to minimize the total cost.
///
/// # Implementing Custom Factors
///
/// To create a custom factor:
/// 1. Implement this trait
/// 2. Define the residual function `r(x)` (how to compute error from variable values)
/// 3. Compute the Jacobian `J = ∂r/∂x` (analytically or numerically)
/// 4. Return the residual dimension
///
/// # Thread Safety
///
/// Factors must be `Send + Sync` to enable parallel residual/Jacobian evaluation.
///
/// # Example
///
/// ```
/// use apex_solver::core::factors::Factor;
/// use nalgebra::{DMatrix, DVector};
///
/// // Simple 1D range measurement factor
/// struct RangeFactor {
///     measurement: f64,  // Measured distance
/// }
///
/// impl Factor for RangeFactor {
///     fn linearize(&self, params: &[DVector<f64>]) -> (DVector<f64>, DMatrix<f64>) {
///         // params[0] is a 2D point [x, y]
///         let x = params[0][0];
///         let y = params[0][1];
///
///         // Residual: measured distance - actual distance
///         let predicted_distance = (x * x + y * y).sqrt();
///         let residual = DVector::from_vec(vec![self.measurement - predicted_distance]);
///
///         // Jacobian: ∂(residual)/∂[x, y]
///         let jacobian = DMatrix::from_row_slice(1, 2, &[
///             -x / predicted_distance,
///             -y / predicted_distance,
///         ]);
///
///         (residual, jacobian)
///     }
///
///     fn get_dimension(&self) -> usize { 1 }
/// }
/// ```
pub trait Factor: Send + Sync {
    /// Compute the residual and Jacobian at the given parameter values.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice of variable values (one `DVector` per connected variable)
    ///
    /// # Returns
    ///
    /// Tuple `(residual, jacobian)` where:
    /// - `residual`: N-dimensional error vector
    /// - `jacobian`: N × M matrix where M is the total DOF of all variables
    ///
    /// # Example
    ///
    /// For a between factor connecting two SE2 poses (3 DOF each):
    /// - Input: `params = [pose1 (3×1), pose2 (3×1)]`
    /// - Output: `(residual (3×1), jacobian (3×6))`
    fn linearize(
        &self,
        params: &[nalgebra::DVector<f64>],
    ) -> (nalgebra::DVector<f64>, nalgebra::DMatrix<f64>);

    /// Get the dimension of the residual vector.
    ///
    /// # Returns
    ///
    /// Number of elements in the residual vector (number of constraints)
    ///
    /// # Example
    ///
    /// - SE2 between factor: 3 (dx, dy, dtheta)
    /// - SE3 between factor: 6 (translation + rotation)
    /// - Prior factor: dimension of the variable
    fn get_dimension(&self) -> usize;
}

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
/// use apex_solver::core::factors::{Factor, BetweenFactorSE2};
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
/// let (residual, jacobian) = between.linearize(&[pose_i, pose_j]);
/// println!("Residual: {:?}", residual);  // Shows deviation from measurement
/// ```
#[derive(Debug, Clone)]
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
    /// use apex_solver::core::factors::BetweenFactorSE2;
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
    /// use apex_solver::core::factors::BetweenFactorSE2;
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
        params: &[nalgebra::DVector<f64>],
    ) -> (nalgebra::DVector<f64>, nalgebra::DMatrix<f64>) {
        // Use analytical jacobians for SE2 between factor (same pattern as SE3)
        // Input: params = [x, y, theta] for each pose (G2O FORMAT)
        let se2_origin_k0 = SE2::from(params[0].clone());
        let se2_origin_k1 = SE2::from(params[1].clone());
        let se2_k0_k1_measured = &self.relative_pose;

        // Step 1: se2_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = nalgebra::Matrix3::zeros();
        let se2_k1_inv = se2_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se2_k1_inv * se2_origin_k0
        let mut j_compose1_wrt_k1_inv = nalgebra::Matrix3::zeros();
        let mut j_compose1_wrt_k0 = nalgebra::Matrix3::zeros();
        let se2_temp = se2_k1_inv.compose(
            &se2_origin_k0,
            Some(&mut j_compose1_wrt_k1_inv),
            Some(&mut j_compose1_wrt_k0),
        );

        // Step 3: se2_temp * se2_k0_k1_measured
        let mut j_compose2_wrt_temp = nalgebra::Matrix3::zeros();
        let se2_diff = se2_temp.compose(se2_k0_k1_measured, Some(&mut j_compose2_wrt_temp), None);

        // Step 4: se2_diff.log()
        let mut j_log_wrt_diff = nalgebra::Matrix3::zeros();
        let residual = se2_diff.log(Some(&mut j_log_wrt_diff));

        // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
        let j_temp_wrt_k1 = j_compose1_wrt_k1_inv * j_k1_inv_wrt_k1;
        let j_diff_wrt_k0 = j_compose2_wrt_temp * j_compose1_wrt_k0;
        let j_diff_wrt_k1 = j_compose2_wrt_temp * j_temp_wrt_k1;

        let jacobian_wrt_k0 = j_log_wrt_diff * j_diff_wrt_k0;
        let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

        // Assemble full Jacobian: [∂r/∂pose_i | ∂r/∂pose_j]
        let mut jacobian = nalgebra::DMatrix::<f64>::zeros(3, 6);
        jacobian
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&jacobian_wrt_k0);
        jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&jacobian_wrt_k1);

        (residual.into(), jacobian)
    }

    fn get_dimension(&self) -> usize {
        3 // SE(2) between factor has 3D residual: [dx, dy, dtheta]
    }
}

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
/// use apex_solver::core::factors::{Factor, BetweenFactorSE3};
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
/// let (residual, jacobian) = between.linearize(&[pose_i, pose_j]);
/// println!("Residual dimension: {}", residual.len());  // 6
/// println!("Jacobian shape: {} x {}", jacobian.nrows(), jacobian.ncols());  // 6x12
/// ```
#[derive(Debug, Clone)]
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
    /// use apex_solver::core::factors::BetweenFactorSE3;
    /// use apex_solver::manifold::se3::SE3;
    /// use nalgebra::{Vector3, Quaternion};
    ///
    /// // Create relative pose: move 2m in x, rotate 90° around z
    /// let relative = SE3::from_translation_quaternion(
    ///     Vector3::new(2.0, 0.0, 0.0),
    ///     Quaternion::from_axis_angle(
    ///         &nalgebra::Unit::new_normalize(Vector3::z()),
    ///         std::f64::consts::FRAC_PI_2
    ///     )
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
        params: &[nalgebra::DVector<f64>],
    ) -> (nalgebra::DVector<f64>, nalgebra::DMatrix<f64>) {
        let se3_origin_k0 = SE3::from(params[0].clone());
        let se3_origin_k1 = SE3::from(params[1].clone());
        let se3_k0_k1_measured = &self.relative_pose;

        // Step 1: se3_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = nalgebra::Matrix6::zeros();
        let se3_k1_inv = se3_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se3_k1_inv * se3_origin_k0
        let mut j_compose1_wrt_k1_inv = nalgebra::Matrix6::zeros();
        let mut j_compose1_wrt_k0 = nalgebra::Matrix6::zeros();
        let se3_temp = se3_k1_inv.compose(
            &se3_origin_k0,
            Some(&mut j_compose1_wrt_k1_inv),
            Some(&mut j_compose1_wrt_k0),
        );

        // Step 3: se3_temp * se3_k0_k1_measured
        let mut j_compose2_wrt_temp = nalgebra::Matrix6::zeros();
        let se3_diff = se3_temp.compose(se3_k0_k1_measured, Some(&mut j_compose2_wrt_temp), None);

        // Step 4: se3_diff.log()
        let mut j_log_wrt_diff = nalgebra::Matrix6::zeros();
        let residual = se3_diff.log(Some(&mut j_log_wrt_diff));

        // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
        let j_temp_wrt_k1 = j_compose1_wrt_k1_inv * j_k1_inv_wrt_k1;
        let j_diff_wrt_k0 = j_compose2_wrt_temp * j_compose1_wrt_k0;
        let j_diff_wrt_k1 = j_compose2_wrt_temp * j_temp_wrt_k1;

        let jacobian_wrt_k0 = j_log_wrt_diff * j_diff_wrt_k0;
        let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

        // Assemble full Jacobian: [∂r/∂pose_i | ∂r/∂pose_j]
        let mut jacobian = nalgebra::DMatrix::<f64>::zeros(6, 12);
        jacobian
            .fixed_view_mut::<6, 6>(0, 0)
            .copy_from(&jacobian_wrt_k0);
        jacobian
            .fixed_view_mut::<6, 6>(0, 6)
            .copy_from(&jacobian_wrt_k1);

        (residual.into(), jacobian)
    }

    fn get_dimension(&self) -> usize {
        6 // SE(3) between factor has 6D residual: [translation (3D), rotation (3D)]
    }
}

/// Prior factor (unary constraint) on a single variable.
///
/// Represents a direct measurement or prior belief about a variable's value. This is used
/// to anchor variables to known values or to incorporate prior knowledge into the optimization.
///
/// # Mathematical Formulation
///
/// The residual is simply the difference between the current value and the prior:
///
/// ```text
/// r = x - x_prior
/// ```
///
/// The Jacobian is the identity matrix: `J = I`.
///
/// # Use Cases
///
/// - **Anchoring**: Fix the first pose in SLAM to prevent drift
/// - **GPS measurements**: Constrain a pose to a known global position
/// - **Prior knowledge**: Incorporate measurements from other sensors
/// - **Regularization**: Prevent variables from drifting too far from initial values
///
/// # Example
///
/// ```
/// use apex_solver::core::factors::{Factor, PriorFactor};
/// use nalgebra::DVector;
///
/// // Prior: first pose should be at origin
/// let prior = PriorFactor {
///     data: DVector::from_vec(vec![0.0, 0.0, 0.0]),
/// };
///
/// // Current estimate (slightly off)
/// let current_pose = DVector::from_vec(vec![0.1, 0.05, 0.02]);
///
/// // Compute residual (shows deviation from prior)
/// let (residual, jacobian) = prior.linearize(&[current_pose]);
/// println!("Deviation from origin: {:?}", residual);
/// ```
///
/// # Implementation Note
///
/// This is a simple "Euclidean" prior that works for any vector space. For manifold
/// variables (SE2, SE3, etc.), consider using manifold-aware priors that respect the
/// geometry (not yet implemented).
#[derive(Debug, Clone)]
pub struct PriorFactor {
    /// The prior value (measurement or known value)
    pub data: nalgebra::DVector<f64>,
}

impl Factor for PriorFactor {
    /// Compute residual and Jacobian for prior factor.
    ///
    /// # Arguments
    ///
    /// * `params` - Single variable value
    ///
    /// # Returns
    ///
    /// - Residual: N×1 vector `(x_current - x_prior)`
    /// - Jacobian: N×N identity matrix
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::factors::{Factor, PriorFactor};
    /// use nalgebra::DVector;
    ///
    /// let prior = PriorFactor {
    ///     data: DVector::from_vec(vec![1.0, 2.0]),
    /// };
    ///
    /// let current = DVector::from_vec(vec![1.5, 2.3]);
    /// let (residual, jacobian) = prior.linearize(&[current]);
    ///
    /// // Residual shows difference
    /// assert!((residual[0] - 0.5).abs() < 1e-10);
    /// assert!((residual[1] - 0.3).abs() < 1e-10);
    ///
    /// // Jacobian is identity
    /// assert_eq!(jacobian[(0, 0)], 1.0);
    /// assert_eq!(jacobian[(1, 1)], 1.0);
    /// ```
    fn linearize(
        &self,
        params: &[nalgebra::DVector<f64>],
    ) -> (nalgebra::DVector<f64>, nalgebra::DMatrix<f64>) {
        let residual = &params[0] - &self.data;
        let jacobian = nalgebra::DMatrix::<f64>::identity(residual.nrows(), residual.nrows());
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.data.len()
    }
}
