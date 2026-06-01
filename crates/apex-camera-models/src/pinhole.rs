//! Pinhole camera model.
//!
//! The simplest perspective camera: a 3D point is divided by its depth, then
//! scaled and shifted to pixel coordinates. There is no lens distortion.
//! Suitable for narrow FOV lenses and as a baseline for calibration.
//! See the [cookbook](../doc/cookbook/src/pinhole.html) for the formulation.

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

    /// True if `z` is far enough from the optical centre for a numerically
    /// stable projection.
    fn check_projection_condition(&self, z: f64) -> bool {
        z >= 1e-6
    }
}

/// Parameter order: `[fx, fy, cx, cy]`.
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

/// Parameter order: `[fx, fy, cx, cy]`.
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

/// Parameter order: `[fx, fy, cx, cy]`. Returns an error if the slice has
/// fewer than 4 elements.
impl TryFrom<&[f64]> for PinholeCamera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < 4 {
            return Err(CameraModelError::InvalidParams(format!(
                "PinholeCamera requires at least 4 parameters, got {}",
                params.len()
            )));
        }
        Ok(Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::None,
        })
    }
}

/// Parameter order: `[fx, fy, cx, cy]`.
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

/// Like [`<PinholeCamera as TryFrom<&[f64]>>::try_from`] but also validates
/// the resulting parameters. Returns
/// [`CameraModelError::InvalidParams`] on a short slice and a validation
/// error otherwise.
pub fn try_from_params(params: &[f64]) -> Result<PinholeCamera, CameraModelError> {
    let camera = PinholeCamera::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for PinholeCamera {
    const INTRINSIC_DIM: usize = 4;
    type IntrinsicJacobian = SMatrix<f64, 2, 4>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D pixel coordinates. See the
    /// [cookbook](../doc/cookbook/src/pinhole.html#projection) for the
    /// formula.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::PointBehindCamera`] when `z ≤ MIN_DEPTH`.
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

    /// Unprojects a pixel to a unit ray. Algebraic. See the
    /// [cookbook](../doc/cookbook/src/pinhole.html#unprojection).
    ///
    /// # Errors
    ///
    /// Never fails; the `Result` is for trait uniformity.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let mx = (point_2d.x - self.pinhole.cx) / self.pinhole.fx;
        let my = (point_2d.y - self.pinhole.cy) / self.pinhole.fy;

        let r2 = mx * mx + my * my;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(mx * norm_inv, my * norm_inv, norm_inv))
    }

    /// ∂(u,v)/∂(x,y,z) — 2×3 projection Jacobian.
    /// See the [cookbook](../doc/cookbook/src/pinhole.html#point-jacobian).
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_norm = p_cam.x * inv_z;
        let y_norm = p_cam.y * inv_z;

        SMatrix::<f64, 2, 3>::new(
            self.pinhole.fx * inv_z,
            0.0,
            -self.pinhole.fx * x_norm * inv_z,
            0.0,
            self.pinhole.fy * inv_z,
            -self.pinhole.fy * y_norm * inv_z,
        )
    }

    /// ∂(u,v)/∂(fx, fy, cx, cy) — 2×4 intrinsic Jacobian. Parameter
    /// order: `[fx, fy, cx, cy]`. See the
    /// [cookbook](../doc/cookbook/src/pinhole.html#intrinsic-jacobian).
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_norm = p_cam.x * inv_z;
        let y_norm = p_cam.y * inv_z;

        SMatrix::<f64, 2, 4>::new(x_norm, 0.0, 1.0, 0.0, 0.0, y_norm, 0.0, 1.0)
    }

    /// Validates pinhole intrinsics. Rules (mirrored in the
    /// [cookbook](../doc/cookbook/src/pinhole.html#validation-rules)):
    ///
    /// - `fx > 0`, `fy > 0` and finite.
    /// - `cx`, `cy` finite.
    fn validate_params(&self) -> Result<(), CameraModelError> {
        self.pinhole.validate()?;
        self.get_distortion().validate()
    }

    /// Returns the linear intrinsics `(fx, fy, cx, cy)`.
    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
        }
    }

    /// Returns the distortion model (always [`DistortionModel::None`] for this camera).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the model identifier `"pinhole"`.
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
    fn test_pinhole_from_params() -> TestResult {
        let params = vec![600.0, 600.0, 320.0, 240.0];
        let camera = PinholeCamera::try_from(params.as_slice())?;
        assert_eq!(camera.pinhole.fx, 600.0);
        let params_vec: DVector<f64> = (&camera).into();
        assert_eq!(params_vec, DVector::from_vec(params));
        Ok(())
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

            let cam_plus = PinholeCamera::try_from(params_plus.as_slice())?;
            let cam_minus = PinholeCamera::try_from(params_minus.as_slice())?;

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
