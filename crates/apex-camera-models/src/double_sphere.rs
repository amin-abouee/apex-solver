//! Double Sphere camera model.
//!
//! Two-parameter fisheye model that combines two consecutive sphere projections, providing
//! better accuracy than UCM at extreme wide angles. Has 6 intrinsic parameters. See the
//! [double-sphere cookbook chapter](../doc/cookbook/src/double-sphere.html) for the full
//! projection, unprojection, and Jacobian derivations.

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};
use std::fmt;

/// Double Sphere camera model with 6 parameters.
#[derive(Clone, Copy, PartialEq)]
pub struct DoubleSphereCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl DoubleSphereCamera {
    /// Creates a new Double Sphere camera.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not
    /// [`DistortionModel::DoubleSphere`], or if validation fails (e.g. non-positive
    /// focal length, alpha out of range).
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, DistortionModel, DoubleSphereCamera, PinholeParams};
    ///
    /// let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::DoubleSphere { xi: -0.2, alpha: 0.6 };
    /// let camera = DoubleSphereCamera::new(pinhole, distortion)?;
    /// assert_eq!(camera.get_model_name(), "double_sphere");
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
    ) -> Result<Self, CameraModelError> {
        let model = Self {
            pinhole,
            distortion,
        };
        model.validate_params()?;
        Ok(model)
    }

    /// Returns the Double Sphere parameters as `(xi, alpha)`. Returns `(0.0, 0.0)` if
    /// the model is not Double Sphere.
    fn distortion_params(&self) -> (f64, f64) {
        match self.distortion {
            DistortionModel::DoubleSphere { xi, alpha } => (xi, alpha),
            _ => (0.0, 0.0),
        }
    }

    /// Returns `Ok(true)` if the projection is valid for the given `z` and Euclidean
    /// distance `d1`; `Ok(false)` if not. Returns an error only for unrecoverable
    /// numerical failures.
    fn check_projection_condition(&self, z: f64, d1: f64) -> Result<bool, CameraModelError> {
        let (xi, alpha) = self.distortion_params();
        let w1 = if alpha > 0.5 {
            (1.0 - alpha) / alpha
        } else {
            alpha / (1.0 - alpha)
        };
        let w2 = (w1 + xi) / (2.0 * w1 * xi + xi * xi + 1.0).sqrt();
        Ok(z > -w2 * d1)
    }

    /// Returns `Ok(true)` if the squared normalised radius is within the unprojection
    /// domain (only constrains for `alpha > 0.5`).
    fn check_unprojection_condition(&self, r_squared: f64) -> Result<bool, CameraModelError> {
        let (_, alpha) = self.distortion_params();
        if alpha > 0.5 && r_squared > 1.0 / (2.0 * alpha - 1.0) {
            return Ok(false);
        }
        Ok(true)
    }

    /// Estimates the `alpha` parameter via linear least-squares given 3D–2D
    /// correspondences. `xi` is reset to `0.0`. Requires the intrinsics
    /// `[fx, fy, cx, cy]` to already be set; needs at least 1 correspondence.
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

        self.distortion = DistortionModel::DoubleSphere { xi: 0.0, alpha };

        self.validate_params()?;

        Ok(())
    }
}

/// Debug formatter for [`DoubleSphereCamera`].
impl fmt::Debug for DoubleSphereCamera {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (xi, alpha) = self.distortion_params();
        write!(
            f,
            "DoubleSphere [fx: {} fy: {} cx: {} cy: {} alpha: {} xi: {}]",
            self.pinhole.fx, self.pinhole.fy, self.pinhole.cx, self.pinhole.cy, alpha, xi
        )
    }
}

/// Converts the camera to a dynamic vector with layout `[fx, fy, cx, cy, xi, alpha]`.
impl From<&DoubleSphereCamera> for DVector<f64> {
    fn from(camera: &DoubleSphereCamera) -> Self {
        let (xi, alpha) = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            xi,
            alpha,
        ])
    }
}

/// Converts the camera to a fixed-size array with layout `[fx, fy, cx, cy, xi, alpha]`.
impl From<&DoubleSphereCamera> for [f64; 6] {
    fn from(camera: &DoubleSphereCamera) -> Self {
        let (xi, alpha) = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            xi,
            alpha,
        ]
    }
}

/// Creates a camera from a slice with layout `[fx, fy, cx, cy, xi, alpha]`.
/// Returns an error if the slice has fewer than 6 elements.
impl TryFrom<&[f64]> for DoubleSphereCamera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < 6 {
            return Err(CameraModelError::InvalidParams(format!(
                "DoubleSphereCamera requires at least 6 parameters, got {}",
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
            distortion: DistortionModel::DoubleSphere {
                xi: params[4],
                alpha: params[5],
            },
        })
    }
}

/// Creates a camera from a fixed-size array with layout `[fx, fy, cx, cy, xi, alpha]`.
impl From<[f64; 6]> for DoubleSphereCamera {
    fn from(params: [f64; 6]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::DoubleSphere {
                xi: params[4],
                alpha: params[5],
            },
        }
    }
}

/// Creates a `DoubleSphereCamera` from a parameter slice with full validation.
/// Unlike [`<DoubleSphereCamera as TryFrom<&[f64]>>::try_from`], this also calls
/// [`CameraModel::validate_params`] and returns any validation errors.
pub fn try_from_params(params: &[f64]) -> Result<DoubleSphereCamera, CameraModelError> {
    let camera = DoubleSphereCamera::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for DoubleSphereCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point in the camera frame to 2D image coordinates.
    /// Returns [`CameraModelError::PointBehindCamera`] / `PointOutsideImage` if the
    /// point violates the model's domain (`check_projection_condition`).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (xi, alpha) = self.distortion_params();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();

        if !self.check_projection_condition(z, d1)? {
            return Err(CameraModelError::ProjectionOutOfBounds);
        }

        let xi_d1_z = xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = alpha * d2 + (1.0 - alpha) * xi_d1_z;

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

    /// Unprojects a 2D image point to a unit 3D ray via the double-sphere algebraic
    /// inverse. Returns [`CameraModelError::PointOutsideImage`] if the unprojection
    /// domain (`check_unprojection_condition`) is violated.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let (xi, alpha) = self.distortion_params();
        let mx = (u - self.pinhole.cx) / self.pinhole.fx;
        let my = (v - self.pinhole.cy) / self.pinhole.fy;
        let r2 = mx * mx + my * my;

        if !self.check_unprojection_condition(r2)? {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let mz_num = 1.0 - alpha * alpha * r2;
        let mz_denom = alpha * (1.0 - (2.0 * alpha - 1.0) * r2).sqrt() + (1.0 - alpha);
        let mz = mz_num / mz_denom;

        let mz2 = mz * mz;

        let num_term = mz * xi + (mz2 + (1.0 - xi * xi) * r2).sqrt();
        let denom_term = mz2 + r2;

        if denom_term < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let k = num_term / denom_term;

        let x = k * mx;
        let y = k * my;
        let z = k * mz - xi;

        // Manual normalization to reuse computed norm
        let norm = (x * x + y * y + z * z).sqrt();
        Ok(Vector3::new(x / norm, y / norm, z / norm))
    }

    /// 2×3 Jacobian ∂(u,v)/∂(x,y,z). See the
    /// [cookbook](../doc/cookbook/src/double-sphere.html#jacobians) for the full derivation.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (xi, alpha) = self.distortion_params();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let xi_d1_z = xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = alpha * d2 + (1.0 - alpha) * xi_d1_z;

        // Cache reciprocals to avoid repeated divisions
        let inv_d1 = 1.0 / d1;
        let inv_d2 = 1.0 / d2;

        // ∂d₁/∂x = x/d₁, ∂d₁/∂y = y/d₁, ∂d₁/∂z = z/d₁
        let dd1_dx = x * inv_d1;
        let dd1_dy = y * inv_d1;
        let dd1_dz = z * inv_d1;

        // ∂(ξ·d₁+z)/∂x = ξ·∂d₁/∂x
        let d_xi_d1_z_dx = xi * dd1_dx;
        let d_xi_d1_z_dy = xi * dd1_dy;
        let d_xi_d1_z_dz = xi * dd1_dz + 1.0;

        // ∂d₂/∂x = (x + (ξ·d₁+z)·∂(ξ·d₁+z)/∂x) / d₂
        let dd2_dx = (x + xi_d1_z * d_xi_d1_z_dx) * inv_d2;
        let dd2_dy = (y + xi_d1_z * d_xi_d1_z_dy) * inv_d2;
        let dd2_dz = (xi_d1_z * d_xi_d1_z_dz) * inv_d2;

        // ∂denom/∂x = α·∂d₂/∂x + (1-α)·∂(ξ·d₁+z)/∂x
        let ddenom_dx = alpha * dd2_dx + (1.0 - alpha) * d_xi_d1_z_dx;
        let ddenom_dy = alpha * dd2_dy + (1.0 - alpha) * d_xi_d1_z_dy;
        let ddenom_dz = alpha * dd2_dz + (1.0 - alpha) * d_xi_d1_z_dz;

        let denom2 = denom * denom;

        // ∂(x/denom)/∂x = (denom - x·∂denom/∂x) / denom²
        let du_dx = self.pinhole.fx * (denom - x * ddenom_dx) / denom2;
        let du_dy = self.pinhole.fx * (-x * ddenom_dy) / denom2;
        let du_dz = self.pinhole.fx * (-x * ddenom_dz) / denom2;

        let dv_dx = self.pinhole.fy * (-y * ddenom_dx) / denom2;
        let dv_dy = self.pinhole.fy * (denom - y * ddenom_dy) / denom2;
        let dv_dz = self.pinhole.fy * (-y * ddenom_dz) / denom2;

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// 2×6 Jacobian ∂(u,v)/∂[fx, fy, cx, cy, xi, alpha]. See the
    /// [cookbook](../doc/cookbook/src/double-sphere.html#jacobians) for the full derivation.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (xi, alpha) = self.distortion_params();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let xi_d1_z = xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = alpha * d2 + (1.0 - alpha) * xi_d1_z;

        // Cache reciprocals to avoid repeated divisions
        let inv_denom = 1.0 / denom;
        let inv_d2 = 1.0 / d2;

        let x_norm = x * inv_denom;
        let y_norm = y * inv_denom;

        // ∂u/∂fx = x/denom, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y/denom, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For ξ and α derivatives
        let d_xi_d1_z_dxi = d1;
        let dd2_dxi = (xi_d1_z * d_xi_d1_z_dxi) * inv_d2;
        let ddenom_dxi = alpha * dd2_dxi + (1.0 - alpha) * d_xi_d1_z_dxi;

        let ddenom_dalpha = d2 - xi_d1_z;

        let inv_denom2 = inv_denom * inv_denom;

        let du_dxi = -self.pinhole.fx * x * ddenom_dxi * inv_denom2;
        let dv_dxi = -self.pinhole.fy * y * ddenom_dxi * inv_denom2;

        let du_dalpha = -self.pinhole.fx * x * ddenom_dalpha * inv_denom2;
        let dv_dalpha = -self.pinhole.fy * y * ddenom_dalpha * inv_denom2;

        SMatrix::<f64, 2, 6>::new(
            x_norm, 0.0, 1.0, 0.0, du_dxi, du_dalpha, 0.0, y_norm, 0.0, 1.0, dv_dxi, dv_dalpha,
        )
    }

    /// Validates the camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - `fx`, `fy` must be positive (> 0) and finite
    /// - `cx`, `cy` must be finite
    /// - `ξ` must be finite
    /// - `α` must be in `(0, 1]`
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

    /// Returns the distortion model (must be [`DistortionModel::DoubleSphere`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the model name: `"double_sphere"`.
    fn get_model_name(&self) -> &'static str {
        "double_sphere"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_double_sphere_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        assert_eq!(camera.pinhole.fx, 300.0);
        let (xi, alpha) = camera.distortion_params();
        assert_eq!(alpha, 0.6);
        assert_eq!(xi, -0.2);

        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..6 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = DoubleSphereCamera::try_from(params_plus.as_slice())?;
            let cam_minus = DoubleSphereCamera::try_from(params_minus.as_slice())?;

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
                    "Mismatch at ({}, {})",
                    r,
                    i
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth DoubleSphere camera with xi=0.0 (linear_estimation fixes xi=0.0)
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.6,
        };
        let gt_camera = DoubleSphereCamera::new(gt_pinhole, gt_distortion)?;

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

        // Initial camera with small alpha (alpha must be > 0 for validation)
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.1,
        };
        let mut camera = DoubleSphereCamera::new(init_pinhole, init_distortion)?;

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
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;

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
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        assert!(camera.project(&Vector3::new(0.0, 0.0, -1.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_project_at_min_depth_boundary() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let p_min = Vector3::new(0.0, 0.0, crate::MIN_DEPTH);
        if let Ok(uv) = camera.project(&p_min) {
            assert!(uv.x.is_finite() && uv.y.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_debug_format() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let s = format!("{:?}", camera);
        assert!(
            s.contains("DoubleSphere"),
            "Debug output should contain 'DoubleSphere', got: {s}"
        );
        assert!(
            s.contains("300"),
            "Debug output should contain focal length, got: {s}"
        );
        Ok(())
    }

    #[test]
    fn test_from_camera_to_fixed_array() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let arr: [f64; 6] = (&camera).into();
        assert_eq!(arr[0], 300.0); // fx
        assert_eq!(arr[1], 300.0); // fy
        assert_eq!(arr[2], 320.0); // cx
        assert_eq!(arr[3], 240.0); // cy
        assert!((arr[4] - (-0.2)).abs() < 1e-15); // xi
        assert!((arr[5] - 0.6).abs() < 1e-15); // alpha
        Ok(())
    }

    #[test]
    fn test_from_fixed_array_to_camera() {
        let arr = [300.0f64, 300.0, 320.0, 240.0, -0.2, 0.6];
        let camera = DoubleSphereCamera::from(arr);
        assert_eq!(camera.pinhole.fx, 300.0);
        let (xi, alpha) = camera.distortion_params();
        assert!((xi - (-0.2)).abs() < 1e-15);
        assert!((alpha - 0.6).abs() < 1e-15);
    }

    #[test]
    fn test_try_from_params_valid() -> TestResult {
        let params = [300.0f64, 300.0, 320.0, 240.0, -0.2, 0.6];
        let camera = try_from_params(&params)?;
        assert_eq!(camera.pinhole.fx, 300.0);
        Ok(())
    }

    #[test]
    fn test_try_from_params_too_few() {
        let params = [300.0f64, 300.0, 320.0];
        let result = try_from_params(&params);
        assert!(result.is_err(), "Should fail with fewer than 6 params");
    }

    #[test]
    fn test_try_from_params_invalid_alpha() {
        let params = [300.0f64, 300.0, 320.0, 240.0, 0.0, 0.0]; // alpha = 0 is invalid
        let result = try_from_params(&params);
        assert!(result.is_err(), "Should fail with alpha = 0 (must be > 0)");
    }

    #[test]
    fn test_get_pinhole_params() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 301.0, 320.0, 241.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let p = camera.get_pinhole_params();
        assert_eq!(p.fx, 300.0);
        assert_eq!(p.fy, 301.0);
        assert_eq!(p.cx, 320.0);
        assert_eq!(p.cy, 241.0);
        Ok(())
    }

    #[test]
    fn test_get_distortion() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let d = camera.get_distortion();
        assert_eq!(
            d,
            DistortionModel::DoubleSphere {
                xi: -0.2,
                alpha: 0.6
            }
        );
        Ok(())
    }

    #[test]
    fn test_get_model_name() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        assert_eq!(camera.get_model_name(), "double_sphere");
        Ok(())
    }

    #[test]
    fn test_validate_params_invalid_alpha_zero() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.0,
        };
        let result = DoubleSphereCamera::new(pinhole, distortion);
        assert!(result.is_err(), "alpha = 0 should be invalid");
        Ok(())
    }

    #[test]
    fn test_validate_params_invalid_xi_out_of_range() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: 2.0,
            alpha: 0.6,
        };
        let result = DoubleSphereCamera::new(pinhole, distortion);
        assert!(result.is_err(), "xi = 2.0 should be invalid");
        Ok(())
    }

    #[test]
    fn test_validate_params_invalid_focal_length() {
        let pinhole = PinholeParams {
            fx: -1.0,
            fy: 300.0,
            cx: 320.0,
            cy: 240.0,
        };
        let distortion = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.6,
        };
        let result = DoubleSphereCamera::new(pinhole, distortion);
        assert!(result.is_err(), "negative focal length should be invalid");
    }

    #[test]
    fn test_unproject_outside_image_returns_error() -> TestResult {
        // check_unprojection_condition reads distortion_params() as (alpha, _),
        // where distortion_params() returns (xi, alpha). So the condition triggers
        // when xi > 0.5 and r² > 1/(2*xi - 1).
        // With xi=0.6: threshold = 1/(2*0.6-1) = 5.0.
        // Use fx=fy=1, cx=cy=0 so mx=u, my=v.  u=3 → r²=9 > 5.0 ✓
        let pinhole = PinholeParams::new(1.0, 1.0, 0.0, 0.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: 0.6,
            alpha: 0.9,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let result = camera.unproject(&Vector2::new(3.0, 0.0));
        assert!(
            result.is_err(),
            "Point with r² > threshold should return PointOutsideImage"
        );
        Ok(())
    }
}
