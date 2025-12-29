//! Camera projection models and factors for bundle adjustment.
//!
//! This module provides a flexible framework for camera projection factors used in
//! bundle adjustment, self-calibration, and Structure-from-Motion (SfM).
//!
//! # Key Components
//!
//! - **`CameraModel` trait**: Defines the interface for different camera models
//! - **`ProjectionFactor`**: Generic factor for camera projection constraints
//! - **`OptimizeParams`**: Compile-time configuration for which parameters to optimize
//! - **Type aliases**: Common optimization patterns (BundleAdjustment, SelfCalibration, etc.)
//!
//! # Optimization Configuration
//!
//! The `OptimizeParams<POSE, LANDMARK, INTRINSIC>` type uses const generics to specify
//! which parameters to optimize at compile time:
//!
//! ```rust
//! use apex_solver::factors::camera::{OptimizeParams, ProjectionFactor, PinholeCamera};
//!
//! // Bundle Adjustment: optimize pose + landmarks (intrinsics fixed)
//! type BundleAdjustment = OptimizeParams<true, true, false>;
//!
//! // Self-Calibration: optimize everything
//! type SelfCalibration = OptimizeParams<true, true, true>;
//!
//! // Only intrinsics (pose and landmarks fixed)
//! type OnlyIntrinsics = OptimizeParams<false, false, true>;
//! ```
//!
//! # Examples
//!
//! ## Bundle Adjustment
//!
//! ```rust
//! use apex_solver::factors::camera::{ProjectionFactor, BundleAdjustment, PinholeCamera};
//! use nalgebra::{Matrix2xX, Vector2};
//!
//! let camera = PinholeCamera { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
//! let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
//!
//! let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
//!     ProjectionFactor::new(observations, camera);
//! ```
//!
//! ## Self-Calibration with Fixed Values
//!
//! ```rust
//! use apex_solver::factors::camera::{ProjectionFactor, OnlyIntrinsics, PinholeCamera};
//! use apex_solver::manifold::se3::SE3;
//! use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
//!
//! let camera = PinholeCamera { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
//! let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
//! let fixed_pose = SE3::identity();
//! let fixed_landmarks = Matrix3xX::from_columns(&[Vector3::new(0.0, 0.0, 1.0)]);
//!
//! let factor: ProjectionFactor<PinholeCamera, OnlyIntrinsics> =
//!     ProjectionFactor::new(observations, camera)
//!         .with_fixed_pose(fixed_pose)
//!         .with_fixed_landmarks(fixed_landmarks);
//! ```

use crate::manifold::se3::SE3;
use nalgebra::{DVector, Matrix2xX, Matrix3, Matrix3xX, SMatrix, Vector2, Vector3};

// ============================================================================
// OPTIMIZATION CONFIGURATION
// ============================================================================

/// Configuration for which parameters to optimize.
///
/// Uses const generic booleans for compile-time optimization selection.
/// This allows the compiler to eliminate unused code paths and provide
/// type-safe configuration.
///
/// # Type Parameters
///
/// - `POSE`: Whether to optimize camera pose (SE3 transformation)
/// - `LANDMARK`: Whether to optimize 3D landmark positions
/// - `INTRINSIC`: Whether to optimize camera intrinsic parameters
///
/// # Examples
///
/// ```rust
/// use apex_solver::factors::camera::OptimizeParams;
///
/// // Custom configuration: optimize pose + intrinsics only
/// type PoseAndIntrinsics = OptimizeParams<true, false, true>;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OptimizeParams<const POSE: bool, const LANDMARK: bool, const INTRINSIC: bool>;

impl<const P: bool, const L: bool, const I: bool> OptimizeParams<P, L, I> {
    /// Whether to optimize camera pose
    pub const POSE: bool = P;
    /// Whether to optimize 3D landmarks
    pub const LANDMARK: bool = L;
    /// Whether to optimize camera intrinsics
    pub const INTRINSIC: bool = I;
}

/// Bundle Adjustment: optimize pose + landmarks (intrinsics fixed).
///
/// Standard bundle adjustment where camera intrinsics are known and fixed.
/// Optimizes N observations of N landmarks from ONE camera pose.
pub type BundleAdjustment = OptimizeParams<true, true, false>;

/// Self-Calibration: optimize pose + landmarks + intrinsics.
///
/// Full Structure-from-Motion where camera intrinsics are also unknown.
/// Optimizes camera parameters along with structure and motion.
pub type SelfCalibration = OptimizeParams<true, true, true>;

/// Only Intrinsics: optimize intrinsics (pose and landmarks fixed).
///
/// Camera calibration scenario where 3D structure and camera pose are known.
/// Only the camera intrinsic parameters are optimized.
pub type OnlyIntrinsics = OptimizeParams<false, false, true>;

/// Only Pose: optimize pose (landmarks and intrinsics fixed).
///
/// Visual Odometry or camera localization where 3D landmarks and intrinsics are known.
/// Only the camera pose is optimized.
pub type OnlyPose = OptimizeParams<true, false, false>;

/// Only Landmarks: optimize landmarks (pose and intrinsics fixed).
///
/// Triangulation scenario where camera pose and intrinsics are known.
/// Only 3D landmark positions are optimized.
pub type OnlyLandmarks = OptimizeParams<false, true, false>;

/// Pose and Intrinsics: optimize pose + intrinsics (landmarks fixed).
///
/// Calibration from known 3D structure.
pub type PoseAndIntrinsics = OptimizeParams<true, false, true>;

/// Landmarks and Intrinsics: optimize landmarks + intrinsics (pose fixed).
///
/// Rarely used configuration for specific calibration scenarios.
pub type LandmarksAndIntrinsics = OptimizeParams<false, true, true>;

// ============================================================================
// CAMERA MODEL TRAIT
// ============================================================================

/// Trait for camera projection models.
///
/// This trait defines the interface for different camera models (pinhole, fisheye,
/// double sphere, etc.) used in bundle adjustment and Structure-from-Motion.
///
/// # Type Parameters
///
/// - `INTRINSIC_DIM`: Number of intrinsic parameters (e.g., 4 for pinhole: fx, fy, cx, cy)
/// - `IntrinsicJacobian`: Jacobian type for intrinsics (2 × INTRINSIC_DIM matrix)
/// - `PointJacobian`: Jacobian type for 3D point (2 × 3 matrix)
///
/// # Implementation Requirements
///
/// All camera models must:
/// - Be thread-safe (`Send + Sync`)
/// - Be cloneable and debuggable
/// - Provide projection and Jacobian methods
/// - Handle invalid projections (e.g., points behind camera)
///
/// # Examples
///
/// See `PinholeCamera` for a complete implementation example.
pub trait CameraModel: Send + Sync + Clone + std::fmt::Debug + 'static {
    /// Number of intrinsic parameters (compile-time constant).
    ///
    /// Examples:
    /// - Pinhole: 4 (fx, fy, cx, cy)
    /// - Pinhole with distortion: 8+ (fx, fy, cx, cy, k1, k2, p1, p2, ...)
    /// - Double Sphere: 6 (fx, fy, cx, cy, xi, alpha)
    const INTRINSIC_DIM: usize;

    /// Jacobian type for intrinsics: 2 × INTRINSIC_DIM.
    ///
    /// Typically `SMatrix<f64, 2, INTRINSIC_DIM>`.
    type IntrinsicJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Jacobian type for 3D point: 2 × 3.
    ///
    /// Typically `SMatrix<f64, 2, 3>`.
    type PointJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Mul<SMatrix<f64, 3, 6>, Output = SMatrix<f64, 2, 6>>
        + std::ops::Mul<Matrix3<f64>, Output = SMatrix<f64, 2, 3>>
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Project a single 3D point in camera frame to 2D image coordinates.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z)
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates (u, v) if projection is valid
    /// - `None` - If projection is invalid (e.g., point behind camera, z ≤ 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use apex_solver::factors::camera::{CameraModel, PinholeCamera};
    /// use nalgebra::Vector3;
    ///
    /// let camera = PinholeCamera { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
    /// let point = Vector3::new(1.0, 0.5, 2.0); // Point 2m in front of camera
    ///
    /// if let Some(uv) = camera.project(&point) {
    ///     println!("Projected to pixel: ({}, {})", uv.x, uv.y);
    /// }
    /// ```
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>>;

    /// Check if a 3D point is valid for projection.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// `true` if the point can be projected (e.g., z > 0 for pinhole cameras)
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool;

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Returns ∂(u,v)/∂(x,y,z) where (x,y,z) is the point in camera frame.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// 2×3 Jacobian matrix
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian;

    /// Jacobian of projection w.r.t. camera pose (2×6).
    ///
    /// Returns ∂(u,v)/∂(pose) for SE3 tangent space.
    ///
    /// This combines:
    /// - Jacobian w.r.t. point position: ∂(u,v)/∂(p_cam)
    /// - Jacobian of transformed point w.r.t. pose: ∂(p_cam)/∂(pose)
    ///
    /// For efficiency, returns both the projection Jacobian and the
    /// transformed point Jacobian separately so the factor can apply the chain rule.
    ///
    /// # Arguments
    ///
    /// * `p_world` - 3D point in world coordinate frame
    /// * `pose` - Camera pose (world-to-camera transformation)
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - Projection Jacobian (2×3): ∂(u,v)/∂(p_cam)
    /// - Pose Jacobian (3×6): ∂(p_cam)/∂(pose)
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>);

    /// Jacobian of projection w.r.t. intrinsic parameters (2×N).
    ///
    /// Returns ∂(u,v)/∂(intrinsics) where intrinsics are camera-specific parameters.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// 2×INTRINSIC_DIM Jacobian matrix
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian;

    /// Batch projection (default impl calls single-point version).
    ///
    /// Projects multiple 3D points to 2D image coordinates.
    ///
    /// # Arguments
    ///
    /// * `points_cam` - 3×N matrix of 3D points in camera frame
    ///
    /// # Returns
    ///
    /// 2×N matrix of projected 2D points. Invalid projections are set to (1e6, 1e6).
    fn project_batch(&self, points_cam: &Matrix3xX<f64>) -> Matrix2xX<f64> {
        let n = points_cam.ncols();
        let mut result = Matrix2xX::zeros(n);
        for i in 0..n {
            let p = Vector3::new(points_cam[(0, i)], points_cam[(1, i)], points_cam[(2, i)]);
            match self.project(&p) {
                Some(uv) => result.set_column(i, &uv),
                None => result.set_column(i, &Vector2::new(1e6, 1e6)),
            }
        }
        result
    }

    /// Get intrinsic parameters as dynamic vector.
    ///
    /// # Returns
    ///
    /// Vector of intrinsic parameters (length = INTRINSIC_DIM)
    fn intrinsics_vec(&self) -> DVector<f64>;

    /// Create camera from parameter slice.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice of intrinsic parameters (length ≥ INTRINSIC_DIM)
    ///
    /// # Returns
    ///
    /// New camera instance with the given intrinsics
    fn from_params(params: &[f64]) -> Self;
}

/// Compute skew-symmetric matrix from a 3D vector.
///
/// Returns the cross-product matrix [v]× such that [v]× w = v × w.
///
/// # Arguments
///
/// * `v` - 3D vector
///
/// # Returns
///
/// 3×3 skew-symmetric matrix:
/// ```text
/// [  0  -vz   vy ]
/// [ vz    0  -vx ]
/// [-vy   vx    0 ]
/// ```
///
/// # Examples
///
/// ```rust
/// use apex_solver::factors::camera::skew_symmetric;
/// use nalgebra::Vector3;
///
/// let v = Vector3::new(1.0, 2.0, 3.0);
/// let skew = skew_symmetric(&v);
/// assert_eq!(skew[(0, 1)], -3.0); // -vz
/// assert_eq!(skew[(1, 0)],  3.0); //  vz
/// ```
#[inline]
pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

// ============================================================================
// SUBMODULES
// ============================================================================

pub mod bal_pinhole;
pub mod double_sphere;
pub mod eucm;
pub mod fov;
pub mod kannala_brandt;
pub mod pinhole;
pub mod projection_factor;
pub mod rad_tan;
pub mod reprojection_factor;
pub mod ucm;

// Re-export main types
pub use bal_pinhole::BALPinholeCamera;
pub use double_sphere::DoubleSphereCamera;
pub use eucm::EucmCamera;
pub use fov::FovCamera;
pub use kannala_brandt::KannalaBrandtCamera;
pub use pinhole::PinholeCamera;
pub use projection_factor::ProjectionFactor;
pub use rad_tan::RadTanCamera;
pub use reprojection_factor::ReprojectionFactor;
pub use ucm::UcmCamera;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_params_constants() {
        // Use runtime variables to avoid clippy's constant assertion warnings
        let ba_pose: bool = BundleAdjustment::POSE;
        let ba_land: bool = BundleAdjustment::LANDMARK;
        let ba_intr: bool = BundleAdjustment::INTRINSIC;
        assert!(ba_pose);
        assert!(ba_land);
        assert!(!ba_intr);

        let sc_pose: bool = SelfCalibration::POSE;
        let sc_land: bool = SelfCalibration::LANDMARK;
        let sc_intr: bool = SelfCalibration::INTRINSIC;
        assert!(sc_pose);
        assert!(sc_land);
        assert!(sc_intr);

        let oi_pose: bool = OnlyIntrinsics::POSE;
        let oi_land: bool = OnlyIntrinsics::LANDMARK;
        let oi_intr: bool = OnlyIntrinsics::INTRINSIC;
        assert!(!oi_pose);
        assert!(!oi_land);
        assert!(oi_intr);

        let op_pose: bool = OnlyPose::POSE;
        let op_land: bool = OnlyPose::LANDMARK;
        let op_intr: bool = OnlyPose::INTRINSIC;
        assert!(op_pose);
        assert!(!op_land);
        assert!(!op_intr);

        let ol_pose: bool = OnlyLandmarks::POSE;
        let ol_land: bool = OnlyLandmarks::LANDMARK;
        let ol_intr: bool = OnlyLandmarks::INTRINSIC;
        assert!(!ol_pose);
        assert!(ol_land);
        assert!(!ol_intr);
    }

    #[test]
    fn test_skew_symmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let skew = skew_symmetric(&v);

        // Check structure
        assert_eq!(skew[(0, 0)], 0.0);
        assert_eq!(skew[(1, 1)], 0.0);
        assert_eq!(skew[(2, 2)], 0.0);

        // Check anti-symmetry
        assert_eq!(skew[(0, 1)], -skew[(1, 0)]);
        assert_eq!(skew[(0, 2)], -skew[(2, 0)]);
        assert_eq!(skew[(1, 2)], -skew[(2, 1)]);

        // Check specific values
        assert_eq!(skew[(0, 1)], -v.z);
        assert_eq!(skew[(0, 2)], v.y);
        assert_eq!(skew[(1, 0)], v.z);
        assert_eq!(skew[(1, 2)], -v.x);
        assert_eq!(skew[(2, 0)], -v.y);
        assert_eq!(skew[(2, 1)], v.x);

        // Verify cross product property: skew * w = v × w
        let w = Vector3::new(4.0, 5.0, 6.0);
        let cross_via_skew = skew * w;
        let cross_direct = v.cross(&w);
        assert!((cross_via_skew - cross_direct).norm() < 1e-10);
    }
}
