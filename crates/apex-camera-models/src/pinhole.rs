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

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Pinhole camera model with 4 intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl PinholeCamera {
    /// Creates a new Pinhole camera model.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole camera parameters (fx, fy, cx, cy).
    /// * `distortion` - Distortion model (must be [`DistortionModel::None`]).
    ///
    /// # Returns
    ///
    /// Returns a new `PinholeCamera` instance if the parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if:
    /// - The distortion model is not `None`.
    /// - Parameters are invalid (e.g., negative focal length, infinite principal point).
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{PinholeCamera, PinholeParams, DistortionModel};
    ///
    /// let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::None;
    /// let camera = PinholeCamera::new(pinhole, distortion)?;
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
    ) -> Result<Self, CameraModelError> {
        let camera = Self {
            pinhole,
            distortion,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Checks the geometric condition for a valid projection.
    ///
    /// # Arguments
    ///
    /// * `z` - The z-coordinate of the point in the camera frame.
    ///
    /// # Returns
    ///
    /// Returns `true` if `z >= 1e-6`, `false` otherwise.
    fn check_projection_condition(&self, z: f64) -> bool {
        z >= 1e-6
    }
}

/// Convert PinholeCamera to parameter vector.
///
/// Returns intrinsic parameters in the order: [fx, fy, cx, cy]
impl From<&PinholeCamera> for DVector<f64> {
    fn from(camera: &PinholeCamera) -> Self {
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
        ])
    }
}

/// Convert PinholeCamera to fixed-size parameter array.
///
/// Returns intrinsic parameters as [fx, fy, cx, cy]
impl From<&PinholeCamera> for [f64; 4] {
    fn from(camera: &PinholeCamera) -> Self {
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
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
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::None,
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
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::None,
        }
    }
}

/// Creates a `PinholeCamera` from a parameter slice with validation.
///
/// Unlike `From<&[f64]>`, this constructor validates all parameters
/// and returns a `Result` instead of panicking on invalid input.
///
/// # Errors
///
/// Returns `CameraModelError::InvalidParams` if fewer than 4 parameters are provided.
/// Returns validation errors if focal lengths are non-positive or parameters are non-finite.
pub fn try_from_params(params: &[f64]) -> Result<PinholeCamera, CameraModelError> {
    if params.len() < 4 {
        return Err(CameraModelError::InvalidParams(format!(
            "PinholeCamera requires at least 4 parameters, got {}",
            params.len()
        )));
    }
    let camera = PinholeCamera::from(params);
    camera.validate_params()?;
    Ok(camera)
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
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z).
    ///
    /// # Returns
    ///
    /// Returns the 2D image coordinates if valid.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::PointAtCameraCenter`] if the point is behind or at the camera center (`z <= MIN_DEPTH`).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if !self.check_projection_condition(p_cam.z) {
            return Err(CameraModelError::PointBehindCamera {
                z: p_cam.z,
                min_z: crate::GEOMETRIC_PRECISION,
            });
        }
        let inv_z = 1.0 / p_cam.z;
        Ok(Vector2::new(
            self.pinhole.fx * p_cam.x * inv_z + self.pinhole.cx,
            self.pinhole.fy * p_cam.y * inv_z + self.pinhole.cy,
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
    /// * `point_2d` - 2D point in image coordinates (u, v).
    ///
    /// # Returns
    ///
    /// Returns the normalized 3D ray direction.
    ///
    /// # Errors
    ///
    /// This function never fails for the simple pinhole model, but returns a `Result` for trait consistency.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let mx = (point_2d.x - self.pinhole.cx) / self.pinhole.fx;
        let my = (point_2d.y - self.pinhole.cy) / self.pinhole.fy;

        let r2 = mx * mx + my * my;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(mx * norm_inv, my * norm_inv, norm_inv))
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
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns the 2x3 Jacobian matrix.
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
            self.pinhole.fx * inv_z,
            0.0,
            -self.pinhole.fx * x_norm * inv_z,
            0.0,
            self.pinhole.fy * inv_z,
            -self.pinhole.fy * y_norm * inv_z,
        )
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
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns the 2x4 Intrinsic Jacobian matrix.
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
    /// # Validation Rules
    ///
    /// - `fx` and `fy` must be positive.
    /// - `fx` and `fy` must be finite.
    /// - `cx` and `cy` must be finite.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if any parameter violates validation rules.
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.pinhole.fx <= 0.0 || self.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthNotPositive {
                fx: self.pinhole.fx,
                fy: self.pinhole.fy,
            });
        }
        if !self.pinhole.fx.is_finite() || !self.pinhole.fy.is_finite() {
            return Err(CameraModelError::FocalLengthNotFinite {
                fx: self.pinhole.fx,
                fy: self.pinhole.fy,
            });
        }
        if !self.pinhole.cx.is_finite() || !self.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointNotFinite {
                cx: self.pinhole.cx,
                cy: self.pinhole.cy,
            });
        }
        Ok(())
    }

    /// Returns the pinhole parameters of the camera.
    ///
    /// # Returns
    ///
    /// A [`PinholeParams`] struct containing the focal lengths (fx, fy) and principal point (cx, cy).
    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
        }
    }

    /// Returns the distortion model and parameters of the camera.
    ///
    /// # Returns
    ///
    /// The [`DistortionModel`] associated with this camera (typically [`DistortionModel::None`] for pinhole).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the string identifier for the camera model.
    ///
    /// # Returns
    ///
    /// The string `"pinhole"`.
    fn get_model_name(&self) -> &'static str {
        "pinhole"
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
        let camera = PinholeCamera::new(pinhole, distortion)?;
        assert_eq!(camera.pinhole.fx, 500.0);
        assert_eq!(camera.pinhole.fy, 500.0);
        assert_eq!(camera.pinhole.cx, 320.0);
        assert_eq!(camera.pinhole.cy, 240.0);
        Ok(())
    }

    #[test]
    fn test_pinhole_from_params() {
        let params = vec![600.0, 600.0, 320.0, 240.0];
        let camera = PinholeCamera::from(params.as_slice());
        assert_eq!(camera.pinhole.fx, 600.0);
        let params_vec: DVector<f64> = (&camera).into();
        assert_eq!(params_vec, DVector::from_vec(params));
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let camera = PinholeCamera::new(pinhole, distortion)?;
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
        let camera = PinholeCamera::new(pinhole, distortion)?;
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
        let camera = PinholeCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.0, 0.0, -1.0);

        let result = camera.project(&p_cam);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_jacobian_point_dimensions() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let distortion = DistortionModel::None;
        let camera = PinholeCamera::new(pinhole, distortion)?;
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
        let camera = PinholeCamera::new(pinhole, distortion)?;
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
        let camera = PinholeCamera::new(pinhole, distortion)?;
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
                assert!(
                    analytical.is_finite(),
                    "Jacobian point [{r},{i}] is not finite"
                );
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
        let camera = PinholeCamera::new(pinhole, distortion)?;
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
                assert!(
                    analytical.is_finite(),
                    "Jacobian intrinsics [{r},{i}] is not finite"
                );
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
    fn test_project_unproject_round_trip() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;

        let test_points = [
            Vector3::new(0.1, 0.2, 1.0),
            Vector3::new(-0.3, 0.1, 2.0),
            Vector3::new(0.05, -0.1, 0.5),
        ];

        for p_cam in &test_points {
            let uv = camera.project(p_cam)?;
            let ray = camera.unproject(&uv)?;
            let dot = ray.dot(&p_cam.normalize());
            assert!(
                (dot - 1.0).abs() < 1e-6,
                "Round-trip failed: dot={dot}, expected ~1.0"
            );
        }

        Ok(())
    }

    #[test]
    fn test_project_returns_error_behind_camera() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;
        assert!(camera.project(&Vector3::new(0.0, 0.0, -1.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_project_at_min_depth_boundary() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;
        let p_min = Vector3::new(0.0, 0.0, crate::MIN_DEPTH);
        if let Ok(uv) = camera.project(&p_min) {
            assert!(uv.x.is_finite() && uv.y.is_finite());
        }
        Ok(())
    }
}
