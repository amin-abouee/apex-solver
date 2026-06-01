//! Unified Camera Model (UCM).
//!
//! Projects via a single shape parameter α onto a virtual unit sphere, generalising
//! catadioptric and fisheye cameras. Has 5 intrinsic parameters. See the
//! [ucm cookbook chapter](../doc/cookbook/src/ucm.html) for the full projection,
//! unprojection, and Jacobian derivations.

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Unified Camera Model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UcmCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl UcmCamera {
    /// Creates a new UCM camera.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not
    /// [`DistortionModel::UCM`].
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, DistortionModel, PinholeParams, UcmCamera};
    ///
    /// let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::UCM { alpha: 0.5 };
    /// let camera = UcmCamera::new(pinhole, distortion)?;
    /// assert_eq!(camera.get_model_name(), "ucm");
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

    /// Returns the UCM `alpha` parameter. Returns `0.0` if the model is not UCM.
    fn distortion_params(&self) -> f64 {
        match self.distortion {
            DistortionModel::UCM { alpha } => alpha,
            _ => 0.0,
        }
    }

    /// Returns `true` if the projection is valid for the given `z` and Euclidean
    /// distance `d = √(x² + y² + z²)`. The condition is `z > -w·d` where
    /// `w` depends on `alpha`.
    fn check_projection_condition(&self, z: f64, d: f64) -> bool {
        let alpha = self.distortion_params();
        let w = if alpha <= 0.5 {
            alpha / (1.0 - alpha)
        } else {
            (1.0 - alpha) / alpha
        };
        z > -w * d
    }

    /// Returns `true` if the squared normalised radius is within the model's
    /// unprojection domain (a constraint that only binds for `alpha > 0.5`).
    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let alpha = self.distortion_params();
        if alpha > 0.5 {
            let gamma = 1.0 - alpha;
            r_squared <= gamma * gamma / (2.0 * alpha - 1.0)
        } else {
            true
        }
    }

    /// Estimates the `alpha` parameter via linear least-squares given 3D–2D
    /// correspondences. Requires the intrinsics `[fx, fy, cx, cy]` to already
    /// be set; needs at least 1 correspondence.
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
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let x = points_3d[(0, i)];
            let y = points_3d[(1, i)];
            let z = points_3d[(2, i)];
            let u = points_2d[(0, i)];
            let v = points_2d[(1, i)];

            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - self.pinhole.cx;
            let v_cy = v - self.pinhole.cy;

            a[(i * 2, 0)] = u_cx * (d - z);
            a[(i * 2 + 1, 0)] = v_cy * (d - z);

            b[i * 2] = (self.pinhole.fx * x) - (u_cx * z);
            b[i * 2 + 1] = (self.pinhole.fy * y) - (v_cy * z);
        }

        let svd = a.svd(true, true);
        let alpha = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol[0],
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError {
                    operation: "svd_solve".to_string(),
                    details: err_msg.to_string(),
                });
            }
        };

        self.distortion = DistortionModel::UCM { alpha };

        self.validate_params()?;

        Ok(())
    }
}

/// Converts the camera to a dynamic vector with layout `[fx, fy, cx, cy, alpha]`.
impl From<&UcmCamera> for DVector<f64> {
    fn from(camera: &UcmCamera) -> Self {
        let alpha = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            alpha,
        ])
    }
}

/// Converts the camera to a fixed-size array with layout `[fx, fy, cx, cy, alpha]`.
impl From<&UcmCamera> for [f64; 5] {
    fn from(camera: &UcmCamera) -> Self {
        let alpha = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            alpha,
        ]
    }
}

/// Creates a camera from a slice with layout `[fx, fy, cx, cy, alpha]`.
/// Returns an error if the slice has fewer than 5 elements.
impl TryFrom<&[f64]> for UcmCamera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < 5 {
            return Err(CameraModelError::InvalidParams(format!(
                "UcmCamera requires at least 5 parameters, got {}",
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
            distortion: DistortionModel::UCM { alpha: params[4] },
        })
    }
}

/// Creates a camera from a fixed-size array with layout `[fx, fy, cx, cy, alpha]`.
impl From<[f64; 5]> for UcmCamera {
    fn from(params: [f64; 5]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::UCM { alpha: params[4] },
        }
    }
}

/// Creates a `UcmCamera` from a parameter slice with full validation.
/// Unlike [`<UcmCamera as TryFrom<&[f64]>>::try_from`], this also calls
/// [`CameraModel::validate_params`] and returns any validation errors.
pub fn try_from_params(params: &[f64]) -> Result<UcmCamera, CameraModelError> {
    let camera = UcmCamera::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for UcmCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point in the camera frame to 2D image coordinates.
    /// Returns [`CameraModelError::PointBehindCamera`] / `PointOutsideImage` if the
    /// point violates the model's domain (`check_projection_condition`).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let alpha = self.distortion_params();
        let denom = alpha * d + (1.0 - alpha) * z;

        // Check projection validity
        if !self.check_projection_condition(z, d) {
            return Err(CameraModelError::PointBehindCamera {
                z,
                min_z: crate::GEOMETRIC_PRECISION,
            });
        }

        if denom < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::DenominatorTooSmall {
                denom,
                threshold: crate::GEOMETRIC_PRECISION,
            });
        }

        Ok(Vector2::new(
            self.pinhole.fx * x / denom + self.pinhole.cx,
            self.pinhole.fy * y / denom + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a unit 3D ray via the UCM algebraic inverse.
    /// Returns [`CameraModelError::PointOutsideImage`] if the unprojection domain
    /// (`check_unprojection_condition`) is violated.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;
        let alpha = self.distortion_params();
        let gamma = 1.0 - alpha;
        let xi = alpha / gamma;
        let mx = (u - self.pinhole.cx) / self.pinhole.fx * gamma;
        let my = (v - self.pinhole.cy) / self.pinhole.fy * gamma;

        let r_squared = mx * mx + my * my;
        if !self.check_unprojection_condition(r_squared) {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        // Mei xi-sphere inverse: bx = mx·f, by = my·f, bz = f − ξ
        // with f = (ξ + √(1 + (1−ξ²)·R²)) / (1 + R²).
        let num = xi + (1.0 + (1.0 - xi * xi) * r_squared).sqrt();
        let denom = 1.0 + r_squared;

        if denom < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let coeff = num / denom;

        let point3d = Vector3::new(coeff * mx, coeff * my, coeff) - Vector3::new(0.0, 0.0, xi);

        Ok(point3d.normalize())
    }

    /// 2×3 Jacobian ∂(u,v)/∂(x,y,z). See the
    /// [cookbook](../doc/cookbook/src/ucm.html#jacobians) for the full derivation.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let rho = (x * x + y * y + z * z).sqrt();
        let alpha = self.distortion_params();

        // Denominator D = alpha * rho + (1 - alpha) * z
        // Partial derivatives of D:
        // ∂D/∂x = alpha * x / rho
        // ∂D/∂y = alpha * y / rho
        // ∂D/∂z = alpha * z / rho + (1 - alpha)

        let d_denom_dx = alpha * x / rho;
        let d_denom_dy = alpha * y / rho;
        let d_denom_dz = alpha * z / rho + (1.0 - alpha);

        let denom = alpha * rho + (1.0 - alpha) * z;

        // u = fx * x / denom + cx
        // v = fy * y / denom + cy

        // ∂u/∂x = fx * (denom - x * ∂D/∂x) / denom²
        // ∂u/∂y = fx * (-x * ∂D/∂y) / denom²
        // ∂u/∂z = fx * (-x * ∂D/∂z) / denom²

        // ∂v/∂x = fy * (-y * ∂D/∂x) / denom²
        // ∂v/∂y = fy * (denom - y * ∂D/∂y) / denom²
        // ∂v/∂z = fy * (-y * ∂D/∂z) / denom²

        let denom2 = denom * denom;

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        jac[(0, 0)] = self.pinhole.fx * (denom - x * d_denom_dx) / denom2;
        jac[(0, 1)] = self.pinhole.fx * (-x * d_denom_dy) / denom2;
        jac[(0, 2)] = self.pinhole.fx * (-x * d_denom_dz) / denom2;

        jac[(1, 0)] = self.pinhole.fy * (-y * d_denom_dx) / denom2;
        jac[(1, 1)] = self.pinhole.fy * (denom - y * d_denom_dy) / denom2;
        jac[(1, 2)] = self.pinhole.fy * (-y * d_denom_dz) / denom2;

        jac
    }

    /// 2×5 Jacobian ∂(u,v)/∂[fx, fy, cx, cy, alpha]. See the
    /// [cookbook](../doc/cookbook/src/ucm.html#jacobians) for the full derivation.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let rho = (x * x + y * y + z * z).sqrt();
        let alpha = self.distortion_params();
        let denom = alpha * rho + (1.0 - alpha) * z;

        let x_norm = x / denom;
        let y_norm = y / denom;

        let u_cx = self.pinhole.fx * x_norm;
        let v_cy = self.pinhole.fy * y_norm;

        let mut jac = SMatrix::<f64, 2, 5>::zeros();

        jac[(0, 0)] = x_norm;
        jac[(1, 1)] = y_norm;
        jac[(0, 2)] = 1.0;
        jac[(1, 3)] = 1.0;

        let d_denom_d_alpha = rho - z;
        jac[(0, 4)] = -u_cx * d_denom_d_alpha / denom;
        jac[(1, 4)] = -v_cy * d_denom_d_alpha / denom;

        jac
    }

    /// Validates the camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - `fx`, `fy` must be positive (> 0) and finite
    /// - `cx`, `cy` must be finite
    /// - `α` must be in `[0, 1]`
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

    /// Returns the distortion model (must be [`DistortionModel::UCM`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the model name: `"ucm"`.
    fn get_model_name(&self) -> &'static str {
        "ucm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_ucm_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;

        assert_eq!(camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params(), 0.5);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;
        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.6 };
        let camera = UcmCamera::new(pinhole, distortion)?;

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
                    "Mismatch at ({}, {}): {} vs {}",
                    r,
                    i,
                    jac_analytical[(r, i)],
                    num_jac[r]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.6 };
        let camera = UcmCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = UcmCamera::try_from(params_plus.as_slice())?;
            let cam_minus = UcmCamera::try_from(params_minus.as_slice())?;

            let uv_plus = cam_plus.project(&p_cam)?;
            let uv_minus = cam_minus.project(&p_cam)?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                assert!(
                    jac_analytical[(r, i)].is_finite(),
                    "Jacobian [{r},{i}] is not finite"
                );
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < crate::JACOBIAN_TEST_TOLERANCE,
                    "Mismatch at ({}, {}): {} vs {}",
                    r,
                    i,
                    jac_analytical[(r, i)],
                    num_jac[r]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_ucm_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(400.0, 410.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.7 };
        let camera = UcmCamera::new(pinhole, distortion)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 5);
        assert_eq!(params[0], 400.0);
        assert_eq!(params[1], 410.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 0.7);

        // Test conversion to array
        let arr: [f64; 5] = (&camera).into();
        assert_eq!(arr, [400.0, 410.0, 320.0, 240.0, 0.7]);

        // Test conversion from slice
        let params_slice = [450.0, 460.0, 330.0, 250.0, 0.8];
        let camera2 = UcmCamera::try_from(&params_slice[..])?;
        assert_eq!(camera2.pinhole.fx, 450.0);
        assert_eq!(camera2.pinhole.fy, 460.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        assert_eq!(camera2.distortion_params(), 0.8);

        // Test conversion from array
        let camera3 = UcmCamera::from([500.0, 510.0, 340.0, 260.0, 0.9]);
        assert_eq!(camera3.pinhole.fx, 500.0);
        assert_eq!(camera3.pinhole.fy, 510.0);
        assert_eq!(camera3.distortion_params(), 0.9);

        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth UCM camera
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::UCM { alpha: 0.5 };
        let gt_camera = UcmCamera::new(gt_pinhole, gt_distortion)?;

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

        // Initial camera with zero alpha
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::UCM { alpha: 0.0 };
        let mut camera = UcmCamera::new(init_pinhole, init_distortion)?;

        camera.linear_estimation(&pts_3d, &pts_2d)?;

        // Verify reprojection error
        for i in 0..valid {
            let col = pts_3d.column(i);
            let projected = camera.project(&Vector3::new(col[0], col[1], col[2]))?;
            let err = ((projected.x - pts_2d[(0, i)]).powi(2)
                + (projected.y - pts_2d[(1, i)]).powi(2))
            .sqrt();
            assert!(err < 1.0, "Reprojection error too large: {err}");
        }

        Ok(())
    }

    #[test]
    fn test_project_unproject_round_trip() -> TestResult {
        // Use α = 0.6 so the (1 − ξ²)·R² term in the unprojection is
        // non-zero — this catches sign-of-denominator regressions that
        // pass at the degenerate α = 0.5.
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.6 };
        let camera = UcmCamera::new(pinhole, distortion)?;

        // Include off-axis bearings that reach the fisheye periphery
        // (around 60° from optical axis), where the round-trip error
        // dominates if the inverse formula is wrong.
        let test_points = [
            Vector3::new(0.1, 0.2, 1.0),
            Vector3::new(-0.3, 0.1, 2.0),
            Vector3::new(0.05, -0.1, 0.5),
            Vector3::new(0.6, 0.0, 0.8),
            Vector3::new(0.4, -0.5, 0.7),
        ];

        for p_cam in &test_points {
            let uv = camera.project(p_cam)?;
            let ray = camera.unproject(&uv)?;
            let dot = ray.dot(&p_cam.normalize());
            assert!(
                (dot - 1.0).abs() < 1e-8,
                "Round-trip failed: dot={dot}, expected ~1.0 (p_cam = {p_cam:?})"
            );
        }

        Ok(())
    }

    #[test]
    fn test_project_returns_error_behind_camera() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;
        assert!(camera.project(&Vector3::new(0.0, 0.0, -1.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_project_at_min_depth_boundary() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;
        let p_min = Vector3::new(0.0, 0.0, crate::MIN_DEPTH);
        if let Ok(uv) = camera.project(&p_min) {
            assert!(uv.x.is_finite() && uv.y.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_projection_off_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);
        let jac_point = camera.jacobian_point(&p_cam);
        assert_eq!(jac_point.nrows(), 2);
        assert_eq!(jac_point.ncols(), 3);
        let jac_intr = camera.jacobian_intrinsics(&p_cam);
        assert_eq!(jac_intr.nrows(), 2);
        assert_eq!(jac_intr.ncols(), 5); // UcmCamera::INTRINSIC_DIM = 5
        Ok(())
    }
}
