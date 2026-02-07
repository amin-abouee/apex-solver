//! Camera projection models for computer vision applications.
//!
//! This crate provides a comprehensive collection of camera projection models commonly used in
//! bundle adjustment, SLAM, visual odometry, and Structure-from-Motion (SfM). All models implement
//! the [`CameraModel`] trait providing a unified interface for projection, unprojection, and
//! analytic Jacobian computation.
//!
//! # Core Architecture
//!
//! ## CameraModel Trait
//!
//! The [`CameraModel`] trait defines the interface that all camera models must implement:
//!
//! - **Projection**: 3D point (x,y,z) → 2D pixel (u,v)
//! - **Unprojection**: 2D pixel (u,v) → 3D unit ray
//! - **Jacobians**: Analytic derivatives for optimization
//!   - Point Jacobian: ∂(u,v)/∂(x,y,z) — 2×3 matrix
//!   - Pose Jacobian: ∂(u,v)/∂(pose) — 2×6 matrix (SE3 tangent space)
//!   - Intrinsic Jacobian: ∂(u,v)/∂(params) — 2×N matrix (N = parameter count)
//!
//! ## Error Handling
//!
//! All operations return [`Result`] with [`CameraModelError`] providing structured error variants
//! that include actual parameter values for debugging:
//!
//! - Parameter validation: `FocalLengthNotPositive`, `FocalLengthNotFinite`, `ParameterOutOfRange`
//! - Projection errors: `PointBehindCamera`, `ProjectionOutOfBounds`, `DenominatorTooSmall`
//! - Numerical errors: `NumericalError` with operation context
//!
//! # Available Camera Models
//!
//! ## Standard Models (FOV < 90°)
//! - **Pinhole**: Standard perspective projection (4 params: fx, fy, cx, cy)
//! - **Radial-Tangential**: OpenCV Brown-Conrady model (9 params with k1,k2,k3,p1,p2)
//!
//! ## Wide-Angle Models (FOV 90°-180°)
//! - **Kannala-Brandt**: Polynomial fisheye model (8 params, θ-based distortion)
//! - **FOV**: Field-of-view model (5 params, atan-based)
//!
//! ## Omnidirectional Models (FOV > 180°)
//! - **UCM**: Unified Camera Model (5 params, α parameter)
//! - **EUCM**: Extended Unified Camera Model (6 params, α, β parameters)
//! - **Double Sphere**: Two-sphere projection (6 params, ξ, α parameters)
//!
//! ## Specialized Models
//! - **BAL Pinhole**: Bundle Adjustment in the Large format (6 params, -Z convention)

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{Matrix2xX, Matrix3, Matrix3xX, SMatrix, Vector2, Vector3};

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
    /// Focal length must be positive: fx={fx}, fy={fy}
    #[error("Focal length must be positive: fx={fx}, fy={fy}")]
    FocalLengthNotPositive { fx: f64, fy: f64 },

    /// Focal length must be finite: fx={fx}, fy={fy}
    #[error("Focal length must be finite: fx={fx}, fy={fy}")]
    FocalLengthNotFinite { fx: f64, fy: f64 },

    /// Principal point must be finite: cx={cx}, cy={cy}
    #[error("Principal point must be finite: cx={cx}, cy={cy}")]
    PrincipalPointNotFinite { cx: f64, cy: f64 },

    /// Distortion coefficient must be finite
    #[error("Distortion coefficient '{name}' must be finite, got {value}")]
    DistortionNotFinite { name: String, value: f64 },

    /// Parameter out of range
    #[error("Parameter '{param}' must be in range [{min}, {max}], got {value}")]
    ParameterOutOfRange {
        param: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// Point behind camera
    #[error("Point behind camera: z={z} (must be > {min_z})")]
    PointBehindCamera { z: f64, min_z: f64 },

    /// Point at camera center
    #[error("Point at camera center: 3D point too close to optical axis")]
    PointAtCameraCenter,

    /// Projection denominator too small
    #[error("Projection denominator too small: denom={denom} (threshold={threshold})")]
    DenominatorTooSmall { denom: f64, threshold: f64 },

    /// Projection outside valid image region
    #[error("Projection outside valid image region")]
    ProjectionOutOfBounds,

    /// Point outside image bounds
    #[error("Point outside image bounds: ({x}, {y}) not in valid region")]
    PointOutsideImage { x: f64, y: f64 },

    /// Numerical error
    #[error("Numerical error in {operation}: {details}")]
    NumericalError { operation: String, details: String },

    /// Generic invalid parameters
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
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
            return Err(CameraModelError::FocalLengthNotPositive { fx, fy });
        }
        if !fx.is_finite() || !fy.is_finite() {
            return Err(CameraModelError::FocalLengthNotFinite { fx, fy });
        }
        if !cx.is_finite() || !cy.is_finite() {
            return Err(CameraModelError::PrincipalPointNotFinite { cx, cy });
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

    /// Unified Camera Model (1 parameter)
    ///
    /// Single-viewpoint catadioptric camera model.
    UCM { alpha: f64 },

    /// Extended Unified Camera Model (2 parameters)
    ///
    /// Extension of UCM with improved accuracy.
    EUCM { alpha: f64, beta: f64 },

    /// Double Sphere model (2 parameters)
    ///
    /// Two-parameter fisheye model with improved wide-angle accuracy.
    DoubleSphere { xi: f64, alpha: f64 },
}

/// Validates that a 3D point is in front of the camera.
///
/// A point must have positive z-coordinate (in camera frame) to be valid for projection.
/// Points too close to the camera center (z ≈ 0) are rejected to avoid numerical instability.
///
/// # Arguments
///
/// * `z` - Z-coordinate of the point in camera frame (meters)
///
/// # Returns
///
/// - `Ok(())` if z > √ε (approximately 1.5e-8)
/// - `Err(CameraModelError::PointAtCameraCenter)` if z is too small
///
/// # Mathematical Condition
///
/// The validation ensures the point is geometrically in front of the camera:
/// ```text
/// z > √ε ≈ 1.49 × 10^-8
/// ```
/// where ε is machine epsilon for f64 (≈ 2.22 × 10^-16).
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

    /// Projects a 3D point in camera coordinates to 2D image coordinates.
    ///
    /// The projection pipeline is: 3D point → normalized coordinates → distortion → pixel coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// For a 3D point p = (x, y, z) in camera frame:
    /// ```text
    /// (u, v) = K · distort(x/z, y/z)
    /// ```
    /// where K is the intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1].
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z) in meters
    ///
    /// # Returns
    ///
    /// - `Ok(Vector2)` - 2D image coordinates (u, v) in pixels if projection is valid
    /// - `Err(CameraModelError)` - If point is behind camera, at center, or causes numerical issues
    ///
    /// # Errors
    ///
    /// - `PointBehindCamera` - If z ≤ GEOMETRIC_PRECISION (point behind or too close)
    /// - `PointAtCameraCenter` - If point is too close to optical axis
    /// - `DenominatorTooSmall` - If projection causes numerical instability
    /// - `ProjectionOutOfBounds` - If projection falls outside valid image region
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError>;

    /// Unprojects a 2D image point to a normalized 3D ray in camera frame.
    ///
    /// The inverse of projection: 2D pixel → undistortion → normalized 3D ray.
    /// Some models use iterative methods (e.g., Newton-Raphson) for undistortion.
    ///
    /// # Mathematical Formula
    ///
    /// For a 2D point (u, v) in image coordinates:
    /// ```text
    /// (mx, my) = ((u - cx)/fx, (v - cy)/fy)
    /// ray = normalize(undistort(mx, my))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates (u, v) in pixels
    ///
    /// # Returns
    ///
    /// - `Ok(Vector3)` - Normalized 3D ray direction (unit vector)
    /// - `Err(CameraModelError)` - If point is outside valid image region or numerical issues occur
    ///
    /// # Errors
    ///
    /// - `PointOutsideImage` - If 2D point is outside valid unprojection region
    /// - `NumericalError` - If iterative solver fails to converge
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError>;

    /// Jacobian of projection with respect to 3D point coordinates.
    ///
    /// Returns the 2×3 matrix J where J[i,j] = ∂(u,v)[i] / ∂(x,y,z)[j].
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// J = ∂(u,v)/∂(x,y,z) = [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]
    ///                       [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]
    /// ```
    ///
    /// This Jacobian is used for:
    /// - Structure optimization (adjusting 3D landmark positions)
    /// - Triangulation refinement
    /// - Bundle adjustment with landmark optimization
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z) in meters
    ///
    /// # Returns
    ///
    /// 2×3 Jacobian matrix of projection w.r.t. point coordinates
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian;

    /// Jacobian of projection w.r.t. camera pose (SE3).
    ///
    /// # Mathematical Derivation
    ///
    /// The camera pose transformation converts a world point to camera coordinates:
    ///
    /// ```text
    /// p_cam = T⁻¹ · p_world = R^T · (p_world - t)
    /// ```
    ///
    /// where T = (R, t) is the camera pose (world-to-camera transform).
    ///
    /// ## Perturbation Model (Right Jacobian)
    ///
    /// We perturb the pose in the tangent space of SE(3):
    ///
    /// ```text
    /// T(δξ) = T · exp(δξ^)
    /// ```
    ///
    /// where δξ = (δω, δv) ∈ ℝ⁶ with:
    /// - δω ∈ ℝ³: rotation perturbation (so(3) algebra)
    /// - δv ∈ ℝ³: translation perturbation
    ///
    /// The perturbed camera-frame point becomes:
    ///
    /// ```text
    /// p_cam(δξ) = [T · exp(δξ^)]⁻¹ · p_world
    ///           = exp(-δξ^) · T⁻¹ · p_world
    ///           ≈ (I - δξ^) · p_cam     (first-order approximation)
    /// ```
    ///
    /// ## Jacobian w.r.t. Pose Perturbation
    ///
    /// For small perturbations δξ:
    ///
    /// ```text
    /// p_cam(δξ) ≈ p_cam - [p_cam]× · δω - R^T · δv
    /// ```
    ///
    /// where [p_cam]× is the skew-symmetric matrix of p_cam.
    ///
    /// Taking derivatives:
    ///
    /// ```text
    /// ∂p_cam/∂δω = -[p_cam]×
    /// ∂p_cam/∂δv = -R^T
    /// ```
    ///
    /// Therefore, the Jacobian of p_cam w.r.t. pose perturbation δξ is:
    ///
    /// ```text
    /// J_pose = ∂p_cam/∂δξ = [ -R^T | [p_cam]× ]  (3×6 matrix)
    /// ```
    ///
    /// where:
    /// - First 3 columns correspond to translation perturbation δv
    /// - Last 3 columns correspond to rotation perturbation δω
    ///
    /// ## Chain Rule to Pixel Coordinates
    ///
    /// The full Jacobian chain is:
    ///
    /// ```text
    /// J_pixel_pose = J_pixel_point · J_point_pose
    ///              = (∂u/∂p_cam) · (∂p_cam/∂δξ)
    /// ```
    ///
    /// where J_pixel_point is computed by `jacobian_point()`.
    ///
    /// ## Return Value
    ///
    /// Returns a tuple `(J_pixel_point, J_point_pose)`:
    /// - `J_pixel_point`: 2×3 Jacobian ∂uv/∂p_cam (from jacobian_point)
    /// - `J_point_pose`: 3×6 Jacobian ∂p_cam/∂δξ
    ///
    /// The caller multiplies these to get the full 2×6 Jacobian ∂uv/∂δξ.
    ///
    /// ## SE(3) Conventions
    ///
    /// - **Parameterization**: δξ = [δv_x, δv_y, δv_z, δω_x, δω_y, δω_z]
    /// - **Perturbation**: Right perturbation T(δξ) = T · exp(δξ^)
    /// - **Coordinate frame**: Perturbations are in the camera frame
    ///
    /// ## References
    ///
    /// - Barfoot, "State Estimation for Robotics", Chapter 7 (Lie group optimization)
    /// - Sola et al., "A micro Lie theory for state estimation in robotics", arXiv:1812.01537
    /// - Blanco, "A tutorial on SE(3) transformation parameterizations and on-manifold optimization"
    ///
    /// ## Implementation Notes
    ///
    /// The skew-symmetric matrix [p_cam]× is computed as:
    ///
    /// ```text
    /// [p_cam]× = [  0      -p_z    p_y  ]
    ///            [  p_z     0     -p_x  ]
    ///            [ -p_y    p_x     0   ]
    /// ```
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        // Transform world point to camera frame via inverse pose
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        // 2×3 projection Jacobian ∂(u,v)/∂(p_cam)
        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        // 3×6 transformation Jacobian ∂(p_cam)/∂(pose)
        // p_cam = R^T · (p_world - t)
        // Translation part: -R^T  (columns 0-2)
        // Rotation part: [p_cam]× (columns 3-5)
        let r_transpose = pose_inv.rotation_so3().rotation_matrix();
        let p_cam_skew = skew_symmetric(&p_cam);

        let d_pcam_d_pose = SMatrix::<f64, 3, 6>::from_fn(|r, c| {
            if c < 3 {
                -r_transpose[(r, c)]
            } else {
                p_cam_skew[(r, c - 3)]
            }
        });

        (d_uv_d_pcam, d_pcam_d_pose)
    }

    /// Jacobian of projection with respect to intrinsic parameters.
    ///
    /// Returns the 2×N matrix where N = INTRINSIC_DIM (model-dependent).
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// J = ∂(u,v)/∂(params) = [ ∂u/∂fx  ∂u/∂fy  ∂u/∂cx  ∂u/∂cy  ∂u/∂k1  ... ]
    ///                       [ ∂v/∂fx  ∂v/∂fy  ∂v/∂cx  ∂v/∂cy  ∂v/∂k1  ... ]
    /// ```
    ///
    /// Intrinsic parameters vary by model:
    /// - Pinhole: [fx, fy, cx, cy] (4 params)
    /// - RadTan: [fx, fy, cx, cy, k1, k2, p1, p2, k3] (9 params)
    /// - Kannala-Brandt: [fx, fy, cx, cy, k1, k2, k3, k4] (8 params)
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z) in meters
    ///
    /// # Returns
    ///
    /// 2×N Jacobian matrix of projection w.r.t. intrinsic parameters
    ///
    /// # Usage
    ///
    /// Used in camera calibration and self-calibration bundle adjustment.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian;

    /// Batch projection of multiple 3D points to 2D image coordinates.
    ///
    /// Projects N 3D points efficiently. Invalid projections are marked with a sentinel
    /// value (1e6, 1e6) rather than returning an error.
    ///
    /// # Arguments
    ///
    /// * `points_cam` - 3×N matrix where each column is a 3D point (x, y, z) in camera frame
    ///
    /// # Returns
    ///
    /// 2×N matrix where each column is the projected 2D point (u, v) in pixels.
    /// Invalid projections are set to (1e6, 1e6).
    ///
    /// # Performance
    ///
    /// Default implementation iterates over points. Camera models may override
    /// with vectorized implementations for better performance.
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

    /// Validates camera intrinsic and distortion parameters.
    ///
    /// Performs comprehensive validation including:
    /// - Focal lengths: must be positive and finite
    /// - Principal point: must be finite
    /// - Distortion coefficients: must be finite
    /// - Model-specific constraints (e.g., UCM α ∈ [0,1], Double Sphere ξ ∈ [-1,1])
    ///
    /// # Validation Rules
    ///
    /// Common validations across all models:
    /// - `fx > 0`, `fy > 0` (focal lengths must be positive)
    /// - `fx`, `fy` finite (no NaN or Inf)
    /// - `cx`, `cy` finite (principal point must be valid)
    ///
    /// Model-specific validations:
    /// - **UCM**: α ∈ [0, 1]
    /// - **EUCM**: α ∈ [0, 1], β > 0
    /// - **Double Sphere**: ξ ∈ [-1, 1], α ∈ (0, 1]
    /// - **FOV**: w ∈ (0, π]
    ///
    /// # Returns
    ///
    /// - `Ok(())` - All parameters satisfy validation rules
    /// - `Err(CameraModelError)` - Specific error indicating which parameter is invalid
    fn validate_params(&self) -> Result<(), CameraModelError>;

    /// Returns the pinhole parameters (fx, fy, cx, cy).
    ///
    /// # Returns
    ///
    /// [`PinholeParams`] struct containing focal lengths and principal point.
    fn get_pinhole_params(&self) -> PinholeParams;

    /// Returns the distortion model and parameters.
    ///
    /// # Returns
    ///
    /// [`DistortionModel`] enum variant with model-specific parameters.
    /// Returns `DistortionModel::None` for pinhole cameras without distortion.
    fn get_distortion(&self) -> DistortionModel;

    /// Returns the camera model name identifier.
    ///
    /// # Returns
    ///
    /// Static string identifier for the camera model type:
    /// - `"pinhole"`, `"rad_tan"`, `"kannala_brandt"`, etc.
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
