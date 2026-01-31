//! Pinhole Camera Model
//!
//! The simplest perspective camera model with no lens distortion.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! u = fx · (x/z) + cx
//! v = fy · (y/z) + cy
//! ```
//!
//! where:
//! - (fx, fy) are focal lengths in pixels
//! - (cx, cy) is the principal point in pixels
//! - z > 0 (point in front of camera)
//!
//! ## Unprojection (2D → 3D)
//!
//! For a 2D point (u, v) in image coordinates, the unprojected ray is:
//!
//! ```text
//! mx = (u - cx) / fx
//! my = (v - cy) / fy
//! ray = normalize([mx, my, 1])
//! ```
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy (4 parameters)
//! - **Distortion**: None
//!
//! # Use Cases
//!
//! - Standard narrow field-of-view cameras
//! - Initial calibration estimates
//! - Testing and validation
//!
//! # References
//!
//! - Hartley & Zisserman, "Multiple View Geometry in Computer Vision"

use crate::{
    Camera, CameraModel, CameraModelError, DistortionModel, PinholeParams, skew_symmetric,
};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Pinhole camera model with 4 intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub camera: Camera,
}

impl PinholeCamera {
    /// Creates a new Pinhole camera model.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole camera parameters (fx, fy, cx, cy)
    /// * `distortion` - Distortion model (must be DistortionModel::None)
    /// * `resolution` - Image resolution (width, height)
    ///
    /// # Returns
    ///
    /// - `Ok(PinholeCamera)` - Successfully created camera
    /// - `Err(CameraModelError)` - Invalid parameters or wrong distortion model
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{PinholeCamera, PinholeParams, DistortionModel, Resolution};
    ///
    /// let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::None;
    /// let resolution = Resolution { width: 640, height: 480 };
    /// let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
        resolution: crate::Resolution,
    ) -> Result<Self, CameraModelError> {
        let camera = Self {
            camera: Camera {
                pinhole,
                distortion,
                resolution,
            },
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Checks the geometric condition for a valid projection.
    pub fn check_projection_condition(&self, z: f64) -> bool {
        z >= 1e-6
    }

    /// Helper method to validate distortion model.
    ///
    /// # Returns
    ///
    /// - `Ok(())` - Distortion model is None (valid for Pinhole)
    /// - `Err(CameraModelError::InvalidParams)` - If distortion model is not None
    fn check_distortion_model(&self) -> Result<(), CameraModelError> {
        match self.camera.distortion {
            DistortionModel::None => Ok(()),
            _ => Err(CameraModelError::InvalidParams(
                "Invalid distortion model for Pinhole camera - expected DistortionModel::None"
                    .to_string(),
            )),
        }
    }
}

impl CameraModel for PinholeCamera {
    const INTRINSIC_DIM: usize = 4;
    type IntrinsicJacobian = SMatrix<f64, 2, 4>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// u = fx · (x/z) + cx
    /// v = fy · (y/z) + cy
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z)
    ///
    /// # Returns
    ///
    /// - `Ok(uv)` - 2D image coordinates if z > MIN_DEPTH
    /// - `Err(CameraModelError::PointAtCameraCenter)` - If point is behind or at camera (z ≤ MIN_DEPTH)
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if !self.check_projection_condition(p_cam.z) {
            return Err(CameraModelError::PointAtCameraCenter);
        }
        let inv_z = 1.0 / p_cam.z;
        Ok(Vector2::new(
            self.camera.pinhole.fx * p_cam.x * inv_z + self.camera.pinhole.cx,
            self.camera.pinhole.fy * p_cam.y * inv_z + self.camera.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray in camera frame.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// mx = (u - cx) / fx
    /// my = (v - cy) / fy
    /// ray = normalize([mx, my, 1])
    /// ```
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates (u, v)
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - Never fails for pinhole model
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let mx = (point_2d.x - self.camera.pinhole.cx) / self.camera.pinhole.fx;
        let my = (point_2d.y - self.camera.pinhole.cy) / self.camera.pinhole.fy;

        let r2 = mx * mx + my * my;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(mx * norm_inv, my * norm_inv, norm_inv))
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// `true` if the point satisfies projection constraints.
    ///
    /// # Validity Conditions
    ///
    /// - z ≥ MIN_DEPTH (point in front of camera)
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        self.check_projection_condition(p_cam.z)
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Computes ∂π/∂p where π is the projection function and p = (x, y, z) is the 3D point.
    ///
    /// # Mathematical Derivation
    ///
    /// Given the pinhole projection model:
    /// ```text
    /// u = fx · (x/z) + cx
    /// v = fy · (y/z) + cy
    /// ```
    ///
    /// ## Derivatives of u-coordinate
    ///
    /// Using the quotient rule for u = fx · (x/z) + cx:
    /// ```text
    /// ∂u/∂x = fx · ∂(x/z)/∂x = fx · (1/z) = fx/z
    ///
    /// ∂u/∂y = fx · ∂(x/z)/∂y = 0          (x/z doesn't depend on y)
    ///
    /// ∂u/∂z = fx · ∂(x/z)/∂z
    ///       = fx · (-x/z²)                 (quotient rule: d(x/z)/dz = -x/z²)
    ///       = -fx·x/z²
    /// ```
    ///
    /// ## Derivatives of v-coordinate
    ///
    /// Similarly for v = fy · (y/z) + cy:
    /// ```text
    /// ∂v/∂x = 0                            (y/z doesn't depend on x)
    ///
    /// ∂v/∂y = fy · ∂(y/z)/∂y = fy · (1/z) = fy/z
    ///
    /// ∂v/∂z = fy · ∂(y/z)/∂z = fy · (-y/z²) = -fy·y/z²
    /// ```
    ///
    /// ## Final Jacobian Matrix (2×3)
    ///
    /// ```text
    /// J = [ ∂u/∂x   ∂u/∂y   ∂u/∂z  ]   [ fx/z    0      -fx·x/z² ]
    ///     [ ∂v/∂x   ∂v/∂y   ∂v/∂z  ] = [  0     fy/z    -fy·y/z² ]
    /// ```
    ///
    /// # Implementation Note
    ///
    /// The implementation uses `inv_z = 1/z` and `x_norm = x/z`, `y_norm = y/z`
    /// to avoid redundant divisions and improve numerical stability.
    ///
    /// # References
    ///
    /// - Hartley & Zisserman, "Multiple View Geometry", Chapter 6
    /// - Standard perspective projection derivatives
    /// - Verified against numerical differentiation in tests
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_norm = p_cam.x * inv_z;
        let y_norm = p_cam.y * inv_z;

        // Jacobian ∂(u,v)/∂(x,y,z) where (x,y,z) is point in camera frame
        SMatrix::<f64, 2, 3>::new(
            self.camera.pinhole.fx * inv_z,
            0.0,
            -self.camera.pinhole.fx * x_norm * inv_z,
            0.0,
            self.camera.pinhole.fy * inv_z,
            -self.camera.pinhole.fy * y_norm * inv_z,
        )
    }

    /// Jacobian of projection w.r.t. camera pose (SE3).
    ///
    /// Computes the full chain: ∂π/∂ξ = (∂π/∂p_cam) · (∂p_cam/∂ξ)
    ///
    /// # Mathematical Derivation
    ///
    /// ## Chain Rule Decomposition
    ///
    /// ```text
    /// ∂π/∂ξ = ∂π/∂p_cam · ∂p_cam/∂ξ
    /// ```
    ///
    /// where:
    /// - `π` is the projection function (3D → 2D)
    /// - `p_cam` is the point in camera coordinates
    /// - `ξ ∈ se(3)` is the camera pose (Lie algebra representation)
    ///
    /// ## Part 1: Point Jacobian (∂π/∂p_cam)
    ///
    /// This is the standard point Jacobian (2×3) computed by `jacobian_point()`.
    /// See that method's documentation for details.
    ///
    /// ## Part 2: Pose Transformation Jacobian (∂p_cam/∂ξ)
    ///
    /// The camera frame point is related to the world frame point by:
    /// ```text
    /// p_cam = T⁻¹ · p_world = (R, t)⁻¹ · p_world
    /// p_cam = R^T · (p_world - t)
    /// ```
    ///
    /// The SE(3) pose is parameterized as ξ = (ω, v) where:
    /// - ω ∈ ℝ³ is the rotation (so(3) Lie algebra, axis-angle representation)
    /// - v ∈ ℝ³ is the translation
    ///
    /// ### Translation Part (∂p_cam/∂v):
    ///
    /// ```text
    /// ∂p_cam/∂v = ∂(R^T·(p_world - t))/∂v
    ///           = -R^T · ∂t/∂v
    ///           = -R^T · I
    ///           = -R^T               (3×3 matrix)
    /// ```
    ///
    /// ### Rotation Part (∂p_cam/∂ω):
    ///
    /// Using the Lie group adjoint relationship:
    /// ```text
    /// ∂p_cam/∂ω = [p_cam]×          (3×3 skew-symmetric matrix)
    /// ```
    ///
    /// where `[p_cam]×` is the skew-symmetric cross-product matrix:
    /// ```text
    /// [p_cam]× = [  0    -pz    py ]
    ///            [  pz     0   -px ]
    ///            [ -py    px     0 ]
    /// ```
    ///
    /// This comes from the derivative of the rotation action on a point.
    ///
    /// ### Combined Jacobian (3×6):
    ///
    /// ```text
    /// ∂p_cam/∂ξ = [ -R^T | [p_cam]× ]     (3×6)
    ///              ︸───︸   ︸──────︸
    ///               ∂/∂v     ∂/∂ω
    /// ```
    ///
    /// ## Final Result (2×6)
    ///
    /// ```text
    /// ∂π/∂ξ = (∂π/∂p_cam) · (∂p_cam/∂ξ)
    ///       = (2×3) · (3×6)
    ///       = (2×6)
    /// ```
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// 1. Point Jacobian (∂π/∂p_cam): 2×3 matrix
    /// 2. Pose Transformation Jacobian (∂p_cam/∂ξ): 3×6 matrix
    ///
    /// The caller can multiply these to get the full pose Jacobian.
    ///
    /// # References
    ///
    /// - Barfoot, "State Estimation for Robotics", Chapter 7 (Lie Groups)
    /// - Sola et al., "A micro Lie theory for state estimation in robotics", 2021
    ///
    /// # Implementation Note
    ///
    /// The rotation Jacobian uses the skew-symmetric matrix `[p_cam]×` which
    /// is provided by the `skew_symmetric()` helper function.
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        // Jacobian of transformed point w.r.t. pose
        // p_cam = R^T * (p_world - t)
        // ∂p_cam/∂[δt; δω] = [-R^T | [p_cam]×]

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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×4).
    ///
    /// Computes ∂π/∂K where K = [fx, fy, cx, cy] are the intrinsic parameters.
    ///
    /// # Mathematical Derivation
    ///
    /// The intrinsic parameters for the pinhole model are:
    /// - **Focal lengths**: fx, fy (scaling factors)
    /// - **Principal point**: cx, cy (image center offset)
    ///
    /// ## Projection Model Recap
    ///
    /// ```text
    /// u = fx · (x/z) + cx
    /// v = fy · (y/z) + cy
    /// ```
    ///
    /// ## Derivatives w.r.t. Focal Lengths
    ///
    /// ### For fx:
    /// ```text
    /// ∂u/∂fx = ∂(fx · x/z + cx)/∂fx
    ///        = x/z                    (coefficient of fx in u)
    ///
    /// ∂v/∂fx = 0                      (fx doesn't appear in v)
    /// ```
    ///
    /// ### For fy:
    /// ```text
    /// ∂u/∂fy = 0                      (fy doesn't appear in u)
    ///
    /// ∂v/∂fy = ∂(fy · y/z + cy)/∂fy
    ///        = y/z                    (coefficient of fy in v)
    /// ```
    ///
    /// ## Derivatives w.r.t. Principal Point
    ///
    /// ### For cx:
    /// ```text
    /// ∂u/∂cx = ∂(fx · x/z + cx)/∂cx
    ///        = 1                      (additive constant)
    ///
    /// ∂v/∂cx = 0                      (cx doesn't appear in v)
    /// ```
    ///
    /// ### For cy:
    /// ```text
    /// ∂u/∂cy = 0                      (cy doesn't appear in u)
    ///
    /// ∂v/∂cy = ∂(fy · y/z + cy)/∂cy
    ///        = 1                      (additive constant)
    /// ```
    ///
    /// ## Final Jacobian Matrix (2×4)
    ///
    /// ```text
    /// J = [ ∂u/∂fx  ∂u/∂fy  ∂u/∂cx  ∂u/∂cy ]
    ///     [ ∂v/∂fx  ∂v/∂fy  ∂v/∂cx  ∂v/∂cy ]
    ///
    ///   = [ x/z      0       1       0    ]
    ///     [  0      y/z      0       1    ]
    /// ```
    ///
    /// # Implementation Note
    ///
    /// The implementation uses precomputed normalized coordinates:
    /// - `x_norm = x/z`
    /// - `y_norm = y/z`
    ///
    /// This avoids redundant divisions and improves efficiency.
    ///
    /// # References
    ///
    /// - Standard camera calibration literature
    /// - Hartley & Zisserman, "Multiple View Geometry", Chapter 6
    /// - Verified against numerical differentiation in tests
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_norm = p_cam.x * inv_z;
        let y_norm = p_cam.y * inv_z;

        // Jacobian ∂(u,v)/∂(fx,fy,cx,cy)
        SMatrix::<f64, 2, 4>::new(x_norm, 0.0, 1.0, 0.0, 0.0, y_norm, 0.0, 1.0)
    }

    /// Validates camera parameters.
    ///
    /// # Returns
    ///
    /// - `Ok(())` - All parameters are valid
    /// - `Err(CameraModelError)` - Invalid parameter detected
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite (not NaN or infinity)
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.camera.pinhole.fx <= 0.0 || self.camera.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }
        if !self.camera.pinhole.cx.is_finite() || !self.camera.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }
        self.check_distortion_model()?;
        Ok(())
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.camera.pinhole.fx,
            fy: self.camera.pinhole.fy,
            cx: self.camera.pinhole.cx,
            cy: self.camera.pinhole.cy,
        }
    }

    fn get_distortion(&self) -> DistortionModel {
        self.camera.distortion.clone()
    }

    fn get_model_name(&self) -> &'static str {
        "pinhole"
    }
}

// ============================================================================
// From/Into Trait Implementations
// ============================================================================

/// Convert PinholeCamera to parameter vector.
///
/// Returns intrinsic parameters in the order: [fx, fy, cx, cy]
impl From<&PinholeCamera> for DVector<f64> {
    fn from(camera: &PinholeCamera) -> Self {
        DVector::from_vec(vec![
            camera.camera.pinhole.fx,
            camera.camera.pinhole.fy,
            camera.camera.pinhole.cx,
            camera.camera.pinhole.cy,
        ])
    }
}

/// Convert PinholeCamera to fixed-size parameter array.
///
/// Returns intrinsic parameters as [fx, fy, cx, cy]
impl From<&PinholeCamera> for [f64; 4] {
    fn from(camera: &PinholeCamera) -> Self {
        [
            camera.camera.pinhole.fx,
            camera.camera.pinhole.fy,
            camera.camera.pinhole.cx,
            camera.camera.pinhole.cy,
        ]
    }
}

/// Create PinholeCamera from parameter slice.
///
/// # Panics
///
/// Panics if the slice has fewer than 4 elements.
///
/// # Parameter Order
///
/// params = [fx, fy, cx, cy]
impl From<&[f64]> for PinholeCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 4,
            "PinholeCamera requires at least 4 parameters, got {}",
            params.len()
        );
        Self {
            camera: Camera {
                pinhole: PinholeParams {
                    fx: params[0],
                    fy: params[1],
                    cx: params[2],
                    cy: params[3],
                },
                distortion: DistortionModel::None,
                resolution: crate::Resolution {
                    width: 0,
                    height: 0,
                },
            },
        }
    }
}

/// Create PinholeCamera from fixed-size parameter array.
///
/// # Parameter Order
///
/// params = [fx, fy, cx, cy]
impl From<[f64; 4]> for PinholeCamera {
    fn from(params: [f64; 4]) -> Self {
        Self {
            camera: Camera {
                pinhole: PinholeParams {
                    fx: params[0],
                    fy: params[1],
                    cx: params[2],
                    cy: params[3],
                },
                distortion: DistortionModel::None,
                resolution: crate::Resolution {
                    width: 0,
                    height: 0,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn assert_approx_eq(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "Values {} and {} differ by more than {}",
            a,
            b,
            eps
        );
    }

    #[test]
    fn test_pinhole_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        assert_eq!(camera.camera.pinhole.fx, 500.0);
        assert_eq!(camera.camera.pinhole.fy, 500.0);
        assert_eq!(camera.camera.pinhole.cx, 320.0);
        assert_eq!(camera.camera.pinhole.cy, 240.0);
        Ok(())
    }

    #[test]
    fn test_pinhole_from_params() {
        let params = vec![600.0, 600.0, 320.0, 240.0];
        let camera = PinholeCamera::from(params.as_slice());
        assert_eq!(camera.camera.pinhole.fx, 600.0);
        let params_vec: DVector<f64> = (&camera).into();
        assert_eq!(params_vec, DVector::from_vec(params));
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);

        let uv = camera.project(&p_cam)?;

        assert_approx_eq(uv.x, 320.0, 1e-10);
        assert_approx_eq(uv.y, 240.0, 1e-10);

        Ok(())
    }

    #[test]
    fn test_projection_off_axis() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let uv = camera.project(&p_cam)?;

        assert_approx_eq(uv.x, 370.0, 1e-10);
        assert_approx_eq(uv.y, 340.0, 1e-10);

        Ok(())
    }

    #[test]
    fn test_projection_behind_camera() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.0, 0.0, -1.0);

        let result = camera.project(&p_cam);
        assert!(result.is_err());
        assert!(!camera.is_valid_point(&p_cam));
        Ok(())
    }

    #[test]
    fn test_jacobian_point_dimensions() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac = camera.jacobian_point(&p_cam);

        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 3);

        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_dimensions() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac = camera.jacobian_intrinsics(&p_cam);

        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 4);
        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);

        let eps = crate::NUMERICAL_DERIVATIVE_EPS;
        for i in 0..3 {
            let mut p_plus = p_cam;
            let mut p_minus = p_cam;
            p_plus[i] += eps;
            p_minus[i] -= eps;

            let uv_plus = camera.project(&p_plus)?;
            let uv_minus = camera.project(&p_minus)?;

            let numerical_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let analytical = jac_analytical[(r, i)];
                let numerical = numerical_jac[r];
                let rel_error = (analytical - numerical).abs() / (1.0 + numerical.abs());
                assert!(
                    rel_error < crate::JACOBIAN_TEST_TOLERANCE,
                    "Jacobian mismatch at ({}, {}): analytical={}, numerical={}, rel_error={}",
                    r,
                    i,
                    analytical,
                    numerical,
                    rel_error
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);

        let eps = crate::NUMERICAL_DERIVATIVE_EPS;
        let params: DVector<f64> = (&camera).into();

        for i in 0..4 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = PinholeCamera::from(params_plus.as_slice());
            let cam_minus = PinholeCamera::from(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam)?;
            let uv_minus = cam_minus.project(&p_cam)?;

            let numerical_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let analytical = jac_analytical[(r, i)];
                let numerical = numerical_jac[r];
                let rel_error = (analytical - numerical).abs() / (1.0 + numerical.abs());
                assert!(
                    rel_error < crate::JACOBIAN_TEST_TOLERANCE,
                    "Intrinsics Jacobian mismatch at ({}, {}): analytical={}, numerical={}, rel_error={}",
                    r,
                    i,
                    analytical,
                    numerical,
                    rel_error
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_jacobian_pose() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = PinholeCamera::new(pinhole, distortion, resolution)?;
        let pose = SE3::identity();
        let p_world = Vector3::new(0.1, 0.2, 1.0);

        let (d_uv_d_pcam, d_pcam_d_pose) = camera.jacobian_pose(&p_world, &pose);

        assert_eq!(d_uv_d_pcam.nrows(), 2);
        assert_eq!(d_uv_d_pcam.ncols(), 3);
        assert_eq!(d_pcam_d_pose.nrows(), 3);
        assert_eq!(d_pcam_d_pose.ncols(), 6);

        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(&p_world, None, None);
        assert_approx_eq((p_cam - p_world).norm(), 0.0, 1e-10);

        Ok(())
    }
}
