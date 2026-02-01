//! Camera projection models for bundle adjustment.
//!
//! This module provides camera models used in bundle adjustment, self-calibration,
//! and Structure-from-Motion (SfM).
//!
//! # Key Components
//!
//! - **`CameraModel` trait**: Interface for different camera models
//! - **`OptimizeParams`**: Compile-time configuration for which parameters to optimize
//! - **Type aliases**: Common optimization patterns (BundleAdjustment, SelfCalibration, etc.)
//!
//! # Available Camera Models
//!
//! - **Pinhole**: Standard perspective projection
//! - **Kannala-Brandt**: Fisheye model with polynomial distortion
//! - **Double Sphere**: Two-parameter fisheye model
//! - **Radial-Tangential**: OpenCV-compatible distortion model
//! - **EUCM**: Extended Unified Camera Model
//! - **FOV**: Field-of-view based fisheye model
//! - **UCM**: Unified Camera Model
//! - **BAL Pinhole**: Bundle Adjustment in the Large format

use apex_manifolds::se3::SE3;
use nalgebra::{Matrix2xX, Matrix3, Matrix3xX, SMatrix, Vector2, Vector3};

// ============================================================================
// Precision Constants
// ============================================================================

/// Precision constant for geometric validity checks (e.g., point in front of camera).
///
/// Used to determine if a 3D point is geometrically valid for projection.
/// Default: 1e-6 (micrometers at meter scale)
pub const GEOMETRIC_PRECISION: f64 = 1e-6;

/// Epsilon for numerical differentiation in Jacobian computation.
///
/// Used when computing numerical derivatives for validation and testing.
/// Default: 1e-7 (provides good balance between truncation and round-off error)
pub const NUMERICAL_DERIVATIVE_EPS: f64 = 1e-7;

/// Tolerance for numerical Jacobian validation in tests.
///
/// Maximum allowed difference between analytical and numerical Jacobians.
/// Default: 1e-5 (allows for small numerical errors in finite differences)
pub const JACOBIAN_TEST_TOLERANCE: f64 = 1e-5;

/// Tolerance for projection/unprojection test assertions.
///
/// Maximum allowed error in pixel coordinates for test assertions.
/// Default: 1e-10 (essentially exact for double precision)
pub const PROJECTION_TEST_TOLERANCE: f64 = 1e-10;

/// Minimum depth for valid 3D points (meters).
///
/// Points closer than this to the camera are considered invalid.
/// Default: 1e-6 meters (1 micrometer)
pub const MIN_DEPTH: f64 = 1e-6;

/// Convergence threshold for iterative unprojection algorithms.
///
/// Used in camera models that require iterative solving (e.g., Kannala-Brandt).
/// Default: 1e-6
pub const CONVERGENCE_THRESHOLD: f64 = 1e-6;

/// Camera model errors.
#[derive(thiserror::Error, Debug)]
pub enum CameraModelError {
    #[error("Projection is outside the image")]
    ProjectionOutSideImage,
    #[error("Input point is outside the image")]
    PointIsOutSideImage,
    #[error("z is close to zero, point is at camera center")]
    PointAtCameraCenter,
    #[error("Focal length must be positive")]
    FocalLengthMustBePositive,
    #[error("Principal point must be finite")]
    PrincipalPointMustBeFinite,
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
    #[error("NumericalError: {0}")]
    NumericalError(String),
}

/// The "Common 4" - Linear intrinsic parameters.
///
/// These define the projection matrix K for the pinhole camera model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeParams {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x-coordinate (pixels)
    pub cx: f64,
    /// Principal point y-coordinate (pixels)
    pub cy: f64,
}

impl PinholeParams {
    /// Create new pinhole parameters with validation.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Result<Self, CameraModelError> {
        if fx <= 0.0 || fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }
        if !cx.is_finite() || !cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }
        Ok(Self { fx, fy, cx, cy })
    }
}

/// Lens distortion models.
///
/// This enum captures all supported distortion models with their parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistortionModel {
    /// Perfect pinhole (no distortion)
    None,

    /// BAL-style radial distortion (2 parameters: k1, k2)
    ///
    /// Used in Bundle Adjustment in the Large datasets.
    Radial { k1: f64, k2: f64 },

    /// Brown-Conrady / OpenCV model (5 parameters)
    ///
    /// Standard model used in OpenCV: k1, k2, k3 (radial), p1, p2 (tangential)
    BrownConrady {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    },

    /// Kannala-Brandt fisheye model (4 parameters)
    ///
    /// Polynomial distortion model for fisheye lenses.
    KannalaBrandt { k1: f64, k2: f64, k3: f64, k4: f64 },

    /// Field-of-View model (1 parameter)
    ///
    /// Single-parameter fisheye model based on field of view.
    FOV { w: f64 },

    /// Unified Camera Model (2 parameters)
    ///
    /// Single-viewpoint catadioptric camera model.
    UCM { alpha: f64, beta: f64 },

    /// Extended Unified Camera Model (2 parameters)
    ///
    /// Extension of UCM with improved accuracy.
    EUCM { alpha: f64, beta: f64 },

    /// Double Sphere model (2 parameters)
    ///
    /// Two-parameter fisheye model with improved wide-angle accuracy.
    DoubleSphere { xi: f64, alpha: f64 },
}

/// Represents the resolution of a camera image.
///
/// This struct holds the width and height of the image sensor in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolution {
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
}

/// Validates that a 3D point's z-coordinate is positive (in front of camera).
pub fn validate_point_in_front(z: f64) -> Result<(), CameraModelError> {
    if z < f64::EPSILON.sqrt() {
        return Err(CameraModelError::PointAtCameraCenter);
    }
    Ok(())
}

// Camera model modules

pub mod bal_pinhole;
pub mod double_sphere;
pub mod eucm;
pub mod fov;
pub mod kannala_brandt;
pub mod pinhole;
pub mod rad_tan;
pub mod ucm;

// Re-export camera types
pub use bal_pinhole::BALPinholeCameraStrict;
pub use double_sphere::DoubleSphereCamera;
pub use eucm::EucmCamera;
pub use fov::FovCamera;
pub use kannala_brandt::KannalaBrandtCamera;
pub use pinhole::PinholeCamera;
pub use rad_tan::RadTanCamera;
pub use ucm::UcmCamera;

// Optimization configuration

/// Configuration for which parameters to optimize.
///
/// Uses const generic booleans for compile-time optimization selection.
///
/// # Type Parameters
///
/// - `POSE`: Whether to optimize camera pose (SE3 transformation)
/// - `LANDMARK`: Whether to optimize 3D landmark positions
/// - `INTRINSIC`: Whether to optimize camera intrinsic parameters
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
pub type BundleAdjustment = OptimizeParams<true, true, false>;

/// Self-Calibration: optimize pose + landmarks + intrinsics.
pub type SelfCalibration = OptimizeParams<true, true, true>;

/// Only Intrinsics: optimize intrinsics (pose and landmarks fixed).
pub type OnlyIntrinsics = OptimizeParams<false, false, true>;

/// Only Pose: optimize pose (landmarks and intrinsics fixed).
pub type OnlyPose = OptimizeParams<true, false, false>;

/// Only Landmarks: optimize landmarks (pose and intrinsics fixed).
pub type OnlyLandmarks = OptimizeParams<false, true, false>;

/// Pose and Intrinsics: optimize pose + intrinsics (landmarks fixed).
pub type PoseAndIntrinsics = OptimizeParams<true, false, true>;

/// Landmarks and Intrinsics: optimize landmarks + intrinsics (pose fixed).
pub type LandmarksAndIntrinsics = OptimizeParams<false, true, true>;

// Camera Model Trait

/// Trait for camera projection models.
///
/// Defines the interface for camera models used in bundle adjustment and SfM.
///
/// # Type Parameters
///
/// - `INTRINSIC_DIM`: Number of intrinsic parameters
/// - `IntrinsicJacobian`: Jacobian type for intrinsics (2 × INTRINSIC_DIM)
/// - `PointJacobian`: Jacobian type for 3D point (2 × 3)
pub trait CameraModel: Send + Sync + Clone + std::fmt::Debug + 'static {
    /// Number of intrinsic parameters (compile-time constant).
    const INTRINSIC_DIM: usize;

    /// Jacobian type for intrinsics: 2 × INTRINSIC_DIM.
    type IntrinsicJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Jacobian type for 3D point: 2 × 3.
    type PointJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Mul<SMatrix<f64, 3, 6>, Output = SMatrix<f64, 2, 6>>
        + std::ops::Mul<Matrix3<f64>, Output = SMatrix<f64, 2, 3>>
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z)
    ///
    /// # Returns
    ///
    /// - `Ok(uv)` - 2D image coordinates if projection is valid
    /// - `Err(CameraModelError)` - If point cannot be projected with specific error reason
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError>;

    /// Unprojects a 2D image point to a 3D ray in camera frame.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates (u, v)
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err(CameraModelError)` - If unprojection fails
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError>;

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// `true` if the point satisfies projection constraints
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool;

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Returns ∂(u,v)/∂(x,y,z) where (x,y,z) is the point in camera frame.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian;

    /// Jacobian of projection w.r.t. camera pose (2×6).
    ///
    /// Returns both the projection Jacobian and the transformed point Jacobian
    /// for efficient chain rule application.
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
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian;

    /// Batch projection of multiple 3D points.
    ///
    /// Default implementation calls single-point projection for each point.
    /// Invalid projections are set to (1e6, 1e6).
    fn project_batch(&self, points_cam: &Matrix3xX<f64>) -> Matrix2xX<f64> {
        let n = points_cam.ncols();
        let mut result = Matrix2xX::zeros(n);
        for i in 0..n {
            let p = Vector3::new(points_cam[(0, i)], points_cam[(1, i)], points_cam[(2, i)]);
            match self.project(&p) {
                Ok(uv) => result.set_column(i, &uv),
                Err(_) => result.set_column(i, &Vector2::new(1e6, 1e6)),
            }
        }
        result
    }

    /// Validates camera parameters.
    ///
    /// # Returns
    ///
    /// - `Ok(())` - All parameters are valid
    /// - `Err(CameraModelError)` - Invalid parameter detected
    fn validate_params(&self) -> Result<(), CameraModelError>;

    /// Get pinhole parameters.
    fn get_pinhole_params(&self) -> PinholeParams;

    /// Get distortion parameters (model-specific).
    fn get_distortion(&self) -> DistortionModel;

    /// Get model name identifier.
    fn get_model_name(&self) -> &'static str;
}

/// Compute skew-symmetric matrix from a 3D vector.
///
/// Returns the cross-product matrix [v]× such that [v]× w = v × w.
///
/// # Mathematical Form
///
/// ```text
/// [  0  -vz   vy ]
/// [ vz    0  -vx ]
/// [-vy   vx    0 ]
/// ```
#[inline]
pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_params_constants() {
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

        assert_eq!(skew[(0, 0)], 0.0);
        assert_eq!(skew[(1, 1)], 0.0);
        assert_eq!(skew[(2, 2)], 0.0);

        assert_eq!(skew[(0, 1)], -skew[(1, 0)]);
        assert_eq!(skew[(0, 2)], -skew[(2, 0)]);
        assert_eq!(skew[(1, 2)], -skew[(2, 1)]);

        assert_eq!(skew[(0, 1)], -v.z);
        assert_eq!(skew[(0, 2)], v.y);
        assert_eq!(skew[(1, 0)], v.z);
        assert_eq!(skew[(1, 2)], -v.x);
        assert_eq!(skew[(2, 0)], -v.y);
        assert_eq!(skew[(2, 1)], v.x);

        let w = Vector3::new(4.0, 5.0, 6.0);
        let cross_via_skew = skew * w;
        let cross_direct = v.cross(&w);
        assert!((cross_via_skew - cross_direct).norm() < 1e-10);
    }
}
