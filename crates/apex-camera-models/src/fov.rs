//! Field-of-View (FOV) camera model.
//!
//! Fisheye model parameterised by a single FOV coefficient `w` rather than a polynomial
//! series, suitable for wide-FOV SLAM lenses. Has 5 intrinsic parameters. See the
//! [fov cookbook chapter](../doc/cookbook/src/fov.html) for the full projection,
//! unprojection, and Jacobian derivations.

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// FOV camera model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl FovCamera {
    /// Creates a new FOV camera.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not
    /// [`DistortionModel::FOV`].
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, DistortionModel, FovCamera, PinholeParams};
    ///
    /// let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::FOV { w: 1.5 };
    /// let camera = FovCamera::new(pinhole, distortion)?;
    /// assert_eq!(camera.get_model_name(), "fov");
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

    /// Returns the FOV parameter `w`. Returns `0.0` if the model is not FOV.
    fn distortion_params(&self) -> f64 {
        match self.distortion {
            DistortionModel::FOV { w } => w,
            _ => 0.0,
        }
    }

    /// Estimates the `w` parameter by 1-D grid search over candidate values.
    /// Requires the intrinsics `[fx, fy, cx, cy]` to already be set; needs at least
    /// 2 correspondences. Note: this is a search, not a closed-form LS solve.
    pub fn linear_estimation(
        &mut self,
        points_3d: &nalgebra::Matrix3xX<f64>,
        points_2d: &nalgebra::Matrix2xX<f64>,
    ) -> Result<(), CameraModelError> {
        if points_2d.ncols() != points_3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let num_points = points_2d.ncols();

        if num_points < 2 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 2 point correspondences for linear estimation".to_string(),
            ));
        }

        let mut best_w = 1.0;
        let mut best_error = f64::INFINITY;

        for w_test in (10..300).map(|i| i as f64 / 100.0) {
            let mut error_sum = 0.0;
            let mut valid_count = 0;

            for i in 0..num_points {
                let x = points_3d[(0, i)];
                let y = points_3d[(1, i)];
                let z = points_3d[(2, i)];
                let u_observed = points_2d[(0, i)];
                let v_observed = points_2d[(1, i)];

                let r2 = x * x + y * y;
                let r = r2.sqrt();

                let tan_w_half = (w_test / 2.0).tan();
                let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

                let eps_sqrt = f64::EPSILON.sqrt();
                let rd = if r2 < eps_sqrt {
                    2.0 * tan_w_half / w_test
                } else {
                    atan_wrd / (r * w_test)
                };

                let mx = x * rd;
                let my = y * rd;

                let u_predicted = self.pinhole.fx * mx + self.pinhole.cx;
                let v_predicted = self.pinhole.fy * my + self.pinhole.cy;

                let error = ((u_predicted - u_observed).powi(2)
                    + (v_predicted - v_observed).powi(2))
                .sqrt();

                if error.is_finite() {
                    error_sum += error;
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                let avg_error = error_sum / valid_count as f64;
                if avg_error < best_error {
                    best_error = avg_error;
                    best_w = w_test;
                }
            }
        }

        self.distortion = DistortionModel::FOV { w: best_w };

        self.validate_params()?;

        Ok(())
    }
}

/// Converts the camera to a dynamic vector with layout `[fx, fy, cx, cy, w]`.
impl From<&FovCamera> for DVector<f64> {
    fn from(camera: &FovCamera) -> Self {
        let w = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            w,
        ])
    }
}

/// Converts the camera to a fixed-size array with layout `[fx, fy, cx, cy, w]`.
impl From<&FovCamera> for [f64; 5] {
    fn from(camera: &FovCamera) -> Self {
        let w = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            w,
        ]
    }
}

/// Creates a camera from a slice with layout `[fx, fy, cx, cy, w]`.
/// Returns an error if the slice has fewer than 5 elements.
impl TryFrom<&[f64]> for FovCamera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < 5 {
            return Err(CameraModelError::InvalidParams(format!(
                "FovCamera requires at least 5 parameters, got {}",
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
            distortion: DistortionModel::FOV { w: params[4] },
        })
    }
}

/// Creates a camera from a fixed-size array with layout `[fx, fy, cx, cy, w]`.
impl From<[f64; 5]> for FovCamera {
    fn from(params: [f64; 5]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::FOV { w: params[4] },
        }
    }
}

/// Creates an `FovCamera` from a parameter slice with full validation.
/// Unlike [`<FovCamera as TryFrom<&[f64]>>::try_from`], this also calls
/// [`CameraModel::validate_params`] and returns any validation errors.
pub fn try_from_params(params: &[f64]) -> Result<FovCamera, CameraModelError> {
    let camera = FovCamera::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for FovCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point in the camera frame to 2D image coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::ProjectionOutOfBounds`] if `z` is too small.
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        if z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::ProjectionOutOfBounds);
        }

        let r = (x * x + y * y).sqrt();
        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let rd = if r > crate::GEOMETRIC_PRECISION {
            let atan_wrd = (mul2tanwby2 * r / z).atan();
            atan_wrd / (r * w)
        } else {
            mul2tanwby2 / w
        };

        let mx = x * rd;
        let my = y * rd;

        Ok(Vector2::new(
            self.pinhole.fx * mx + self.pinhole.cx,
            self.pinhole.fy * my + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a unit 3D ray. Inverts the projection via the
    /// trigonometric relationship of the FOV model.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let mx = (u - self.pinhole.cx) / self.pinhole.fx;
        let my = (v - self.pinhole.cy) / self.pinhole.fy;

        let r2 = mx * mx + my * my;
        let rd = r2.sqrt();

        if rd < crate::GEOMETRIC_PRECISION {
            return Ok(Vector3::new(0.0, 0.0, 1.0));
        }

        let ru = (rd * w).tan() / mul2tanwby2;

        let norm_factor = (1.0 + ru * ru).sqrt();
        let x = mx * ru / (rd * norm_factor);
        let y = my * ru / (rd * norm_factor);
        let z = 1.0 / norm_factor;

        Ok(Vector3::new(x, y, z))
    }

    /// 2×3 Jacobian ∂(u,v)/∂(x,y,z). See the
    /// [cookbook](../doc/cookbook/src/fov.html#jacobians) for the full derivation.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r = (x * x + y * y).sqrt();
        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        if r < crate::GEOMETRIC_PRECISION {
            let rd = mul2tanwby2 / w;
            return SMatrix::<f64, 2, 3>::new(
                self.pinhole.fx * rd,
                0.0,
                0.0,
                0.0,
                self.pinhole.fy * rd,
                0.0,
            );
        }

        let atan_wrd = (mul2tanwby2 * r / z).atan();
        let rd = atan_wrd / (r * w);

        // Derivatives
        let datan_dr = mul2tanwby2 * z / (z * z + mul2tanwby2 * mul2tanwby2 * r * r);
        let datan_dz = -mul2tanwby2 * r / (z * z + mul2tanwby2 * mul2tanwby2 * r * r);

        let drd_dr = (datan_dr * r - atan_wrd) / (r * r * w);
        let drd_dz = datan_dz / (r * w);

        let dr_dx = x / r;
        let dr_dy = y / r;

        let dmx_dx = rd + x * drd_dr * dr_dx;
        let dmx_dy = x * drd_dr * dr_dy;
        let dmx_dz = x * drd_dz;

        let dmy_dx = y * drd_dr * dr_dx;
        let dmy_dy = rd + y * drd_dr * dr_dy;
        let dmy_dz = y * drd_dz;

        SMatrix::<f64, 2, 3>::new(
            self.pinhole.fx * dmx_dx,
            self.pinhole.fx * dmx_dy,
            self.pinhole.fx * dmx_dz,
            self.pinhole.fy * dmy_dx,
            self.pinhole.fy * dmy_dy,
            self.pinhole.fy * dmy_dz,
        )
    }

    /// 2×5 Jacobian ∂(u,v)/∂[fx, fy, cx, cy, w]. See the
    /// [cookbook](../doc/cookbook/src/fov.html#jacobians) for the full derivation.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r = (x * x + y * y).sqrt();
        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let rd = if r > crate::GEOMETRIC_PRECISION {
            let atan_wrd = (mul2tanwby2 * r / z).atan();
            atan_wrd / (r * w)
        } else {
            mul2tanwby2 / w
        };

        let mx = x * rd;
        let my = y * rd;

        // ∂u/∂fx = mx, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = my, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For w derivative: ∂rd/∂w
        let drd_dw = if r > crate::GEOMETRIC_PRECISION {
            let tan_w_2 = (w / 2.0).tan();
            let alpha = 2.0 * tan_w_2 * r / z;
            let atan_alpha = alpha.atan();

            // sec²(w/2) = 1 + tan²(w/2)
            let sec2_w_2 = 1.0 + tan_w_2 * tan_w_2;
            let dalpha_dw = sec2_w_2 * r / z;

            // ∂rd/∂w = [1/(1+α²) · ∂α/∂w · r·w - atan(α) · r] / (r·w)²
            let datan_dw = dalpha_dw / (1.0 + alpha * alpha);
            (datan_dw * r * w - atan_alpha * r) / (r * r * w * w)
        } else {
            let tan_w_2 = (w / 2.0).tan();
            let sec2_w_2 = 1.0 + tan_w_2 * tan_w_2;
            // rd = 2·tan(w/2) / w
            // ∂rd/∂w = [2·sec²(w/2)/2 · w - 2·tan(w/2)] / w²
            //        = [sec²(w/2) · w - 2·tan(w/2)] / w²
            (sec2_w_2 * w - 2.0 * tan_w_2) / (w * w)
        };

        let du_dw = self.pinhole.fx * x * drd_dw;
        let dv_dw = self.pinhole.fy * y * drd_dw;

        SMatrix::<f64, 2, 5>::new(mx, 0.0, 1.0, 0.0, du_dw, 0.0, my, 0.0, 1.0, dv_dw)
    }

    /// Validates the camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - `fx`, `fy` must be positive (> 0) and finite
    /// - `cx`, `cy` must be finite
    /// - `w` must be in `(0, π]`
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if any rule is violated.
    fn validate_params(&self) -> Result<(), CameraModelError> {
        self.pinhole.validate()?;
        self.get_distortion().validate()
    }

    /// Returns the pinhole parameters.
    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
        }
    }

    /// Returns the distortion model (must be [`DistortionModel::FOV`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the model name: `"fov"`.
    fn get_model_name(&self) -> &'static str {
        "fov"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_fov_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

        assert_eq!(camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params(), 1.5);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < 1e-4);
        assert!((uv.y - 240.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

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
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                assert!(
                    jac_analytical[(r, i)].is_finite(),
                    "Jacobian [{r},{i}] is not finite"
                );
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < crate::JACOBIAN_TEST_TOLERANCE,
                    "Mismatch at ({}, {})",
                    r,
                    i
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = FovCamera::try_from(params_plus.as_slice())?;
            let cam_minus = FovCamera::try_from(params_minus.as_slice())?;

            let uv_plus = cam_plus.project(&p_cam)?;
            let uv_minus = cam_minus.project(&p_cam)?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                assert!(
                    jac_analytical[(r, i)].is_finite(),
                    "Jacobian [{r},{i}] is not finite"
                );
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(diff < 1e-4, "Mismatch at ({}, {})", r, i);
            }
        }
        Ok(())
    }

    #[test]
    fn test_fov_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(400.0, 410.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.8 };
        let camera = FovCamera::new(pinhole, distortion)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 5);
        assert_eq!(params[0], 400.0);
        assert_eq!(params[1], 410.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 1.8);

        // Test conversion to array
        let arr: [f64; 5] = (&camera).into();
        assert_eq!(arr, [400.0, 410.0, 320.0, 240.0, 1.8]);

        // Test conversion from slice
        let params_slice = [450.0, 460.0, 330.0, 250.0, 2.0];
        let camera2 = FovCamera::try_from(&params_slice[..])?;
        assert_eq!(camera2.pinhole.fx, 450.0);
        assert_eq!(camera2.pinhole.fy, 460.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        assert_eq!(camera2.distortion_params(), 2.0);

        // Test conversion from array
        let camera3 = FovCamera::from([500.0, 510.0, 340.0, 260.0, 2.5]);
        assert_eq!(camera3.pinhole.fx, 500.0);
        assert_eq!(camera3.pinhole.fy, 510.0);
        assert_eq!(camera3.distortion_params(), 2.5);

        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth FOV camera
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::FOV { w: 1.0 };
        let gt_camera = FovCamera::new(gt_pinhole, gt_distortion)?;

        // Generate synthetic 3D points in camera frame
        let n_points = 50;
        let mut pts_3d = Matrix3xX::zeros(n_points);
        let mut pts_2d = Matrix2xX::zeros(n_points);
        let mut valid = 0;

        for i in 0..n_points {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n_points as f64;
            let r = 0.1 + 0.3 * (i as f64 / n_points as f64);
            let p3d = Vector3::new(r * angle.cos(), r * angle.sin(), 1.0);

            if let Ok(p2d) = gt_camera.project(&p3d) {
                pts_3d.set_column(valid, &p3d);
                pts_2d.set_column(valid, &p2d);
                valid += 1;
            }
        }
        let pts_3d = pts_3d.columns(0, valid).into_owned();
        let pts_2d = pts_2d.columns(0, valid).into_owned();

        // Initial camera with default w (grid search will find best)
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::FOV { w: 0.5 };
        let mut camera = FovCamera::new(init_pinhole, init_distortion)?;

        camera.linear_estimation(&pts_3d, &pts_2d)?;

        // FOV uses grid search so tolerance is looser
        for i in 0..valid {
            let col = pts_3d.column(i);
            let projected = camera.project(&Vector3::new(col[0], col[1], col[2]))?;
            let err = ((projected.x - pts_2d[(0, i)]).powi(2)
                + (projected.y - pts_2d[(1, i)]).powi(2))
            .sqrt();
            assert!(err < 5.0, "Reprojection error too large: {err}");
        }

        Ok(())
    }

    #[test]
    fn test_project_unproject_round_trip() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

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
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;
        assert!(camera.project(&Vector3::new(0.0, 0.0, -1.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_project_at_min_depth_boundary() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;
        let p_min = Vector3::new(0.0, 0.0, crate::MIN_DEPTH);
        if let Ok(uv) = camera.project(&p_min) {
            assert!(uv.x.is_finite() && uv.y.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_projection_off_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.3, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;
        assert!(
            uv.x > 320.0,
            "off-axis point should project right of principal point"
        );
        assert!(
            (uv.y - 240.0).abs() < 1.0,
            "y should be close to cy for horizontal offset"
        );
        Ok(())
    }

    #[test]
    fn test_unproject_center_pixel() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;
        let uv = Vector2::new(320.0, 240.0);
        let ray = camera.unproject(&uv)?;
        assert!(ray.x.abs() < 1e-6, "x should be ~0, got {}", ray.x);
        assert!(ray.y.abs() < 1e-6, "y should be ~0, got {}", ray.y);
        assert!((ray.z - 1.0).abs() < 1e-6, "z should be ~1, got {}", ray.z);
        Ok(())
    }

    #[test]
    fn test_batch_projection_matches_individual() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;
        let pts = Matrix3xX::from_columns(&[
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.3, 0.2, 1.5),
            Vector3::new(-0.4, 0.1, 2.0),
        ]);
        let batch = camera.project_batch(&pts);
        for i in 0..3 {
            let col = pts.column(i);
            let p = camera.project(&Vector3::new(col[0], col[1], col[2]))?;
            assert!(
                (batch[(0, i)] - p.x).abs() < 1e-10,
                "batch u mismatch at col {i}"
            );
            assert!(
                (batch[(1, i)] - p.y).abs() < 1e-10,
                "batch v mismatch at col {i}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_jacobian_dimensions() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);
        let jac_point = camera.jacobian_point(&p_cam);
        assert_eq!(jac_point.nrows(), 2);
        assert_eq!(jac_point.ncols(), 3);
        let jac_intr = camera.jacobian_intrinsics(&p_cam);
        assert_eq!(jac_intr.nrows(), 2);
        assert_eq!(jac_intr.ncols(), 5); // FovCamera::INTRINSIC_DIM = 5
        Ok(())
    }
}
