//! Radial-Tangential (Brown-Conrady) camera model.
//!
//! Standard OpenCV distortion model combining radial and tangential terms, suitable for narrow
//! to moderate field-of-view lenses. Has 9 intrinsic parameters. See the
//! [rad-tan cookbook chapter](../doc/cookbook/src/rad-tan.html) for the full projection,
//! unprojection, and Jacobian derivations.

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams};
use nalgebra::{DVector, Matrix2, SMatrix, Vector2, Vector3};

/// A Radial-Tangential camera model with 9 intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadTanCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl RadTanCamera {
    /// Creates a new Radial-Tangential (Brown-Conrady) camera.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not
    /// [`DistortionModel::BrownConrady`].
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, DistortionModel, PinholeParams, RadTanCamera};
    ///
    /// let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::BrownConrady { k1: 0.1, k2: 0.01, p1: 0.001, p2: 0.002, k3: 0.001 };
    /// let camera = RadTanCamera::new(pinhole, distortion)?;
    /// assert_eq!(camera.get_model_name(), "rad_tan");
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

    /// Returns `true` if the point's z-coordinate is at or above [`crate::GEOMETRIC_PRECISION`].
    fn check_projection_condition(&self, z: f64) -> bool {
        z >= crate::GEOMETRIC_PRECISION
    }

    /// Returns the Brown-Conrady distortion parameters as `(k1, k2, p1, p2, k3)`.
    /// Returns zeros if the model is not Brown-Conrady.
    fn distortion_params(&self) -> (f64, f64, f64, f64, f64) {
        match self.distortion {
            DistortionModel::BrownConrady { k1, k2, p1, p2, k3 } => (k1, k2, p1, p2, k3),
            _ => (0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Initializes `[k1, k2, k3]` via linear least-squares given 3D–2D correspondences.
    /// Tangential parameters `[p1, p2]` are reset to zero. Requires the intrinsics
    /// `[fx, fy, cx, cy]` to already be set; needs at least 3 correspondences.
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
        if num_points < 3 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 3 points for RadTan linear estimation".to_string(),
            ));
        }

        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 3);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        let fx = self.pinhole.fx;
        let fy = self.pinhole.fy;
        let cx = self.pinhole.cx;
        let cy = self.pinhole.cy;

        for i in 0..num_points {
            let x = points_3d[(0, i)];
            let y = points_3d[(1, i)];
            let z = points_3d[(2, i)];
            let u = points_2d[(0, i)];
            let v = points_2d[(1, i)];

            let x_norm = x / z;
            let y_norm = y / z;
            let r2 = x_norm * x_norm + y_norm * y_norm;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            let u_undist = fx * x_norm + cx;
            let v_undist = fy * y_norm + cy;

            a[(i * 2, 0)] = fx * x_norm * r2;
            a[(i * 2, 1)] = fx * x_norm * r4;
            a[(i * 2, 2)] = fx * x_norm * r6;

            a[(i * 2 + 1, 0)] = fy * y_norm * r2;
            a[(i * 2 + 1, 1)] = fy * y_norm * r4;
            a[(i * 2 + 1, 2)] = fy * y_norm * r6;

            b[i * 2] = u - u_undist;
            b[i * 2 + 1] = v - v_undist;
        }

        let svd = a.svd(true, true);
        let distortion_coeffs = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol,
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError {
                    operation: "svd_solve".to_string(),
                    details: err_msg.to_string(),
                });
            }
        };

        self.distortion = DistortionModel::BrownConrady {
            k1: distortion_coeffs[0],
            k2: distortion_coeffs[1],
            p1: 0.0,
            p2: 0.0,
            k3: distortion_coeffs[2],
        };

        self.validate_params()?;
        Ok(())
    }
}

/// Converts the camera parameters to a dynamic vector with layout
/// `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`.
impl From<&RadTanCamera> for DVector<f64> {
    fn from(camera: &RadTanCamera) -> Self {
        let (k1, k2, p1, p2, k3) = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        ])
    }
}

/// Converts the camera parameters to a fixed-size array with layout
/// `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`.
impl From<&RadTanCamera> for [f64; 9] {
    fn from(camera: &RadTanCamera) -> Self {
        let (k1, k2, p1, p2, k3) = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        ]
    }
}

/// Creates a camera from a slice of intrinsic parameters with layout
/// `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`. Returns an error if the slice
/// has fewer than 9 elements.
impl TryFrom<&[f64]> for RadTanCamera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < 9 {
            return Err(CameraModelError::InvalidParams(format!(
                "RadTanCamera requires at least 9 parameters, got {}",
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
            distortion: DistortionModel::BrownConrady {
                k1: params[4],
                k2: params[5],
                p1: params[6],
                p2: params[7],
                k3: params[8],
            },
        })
    }
}

/// Creates a camera from a fixed-size array with layout
/// `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`.
impl From<[f64; 9]> for RadTanCamera {
    fn from(params: [f64; 9]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::BrownConrady {
                k1: params[4],
                k2: params[5],
                p1: params[6],
                p2: params[7],
                k3: params[8],
            },
        }
    }
}

/// Creates a `RadTanCamera` from a parameter slice with full validation.
/// Unlike [`<RadTanCamera as TryFrom<&[f64]>>::try_from`], this also calls
/// [`CameraModel::validate_params`] and returns any validation errors.
pub fn try_from_params(params: &[f64]) -> Result<RadTanCamera, CameraModelError> {
    let camera = RadTanCamera::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for RadTanCamera {
    const INTRINSIC_DIM: usize = 9;
    type IntrinsicJacobian = SMatrix<f64, 2, 9>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point in the camera frame to 2D image coordinates.
    /// Returns [`CameraModelError::PointBehindCamera`] if the point lies at or behind
    /// the camera, and [`CameraModelError::PointOutsideImage`] if the radial polynomial
    /// becomes non-positive (the model cannot represent such rays).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if !self.check_projection_condition(p_cam.z) {
            return Err(CameraModelError::PointBehindCamera {
                z: p_cam.z,
                min_z: crate::GEOMETRIC_PRECISION,
            });
        }

        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (k1, k2, p1, p2, k3) = self.distortion_params();

        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        if radial <= crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointOutsideImage {
                x: x_prime,
                y: y_prime,
            });
        }

        let xy = x_prime * y_prime;
        let dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x_prime * x_prime);
        let dy = p1 * (r2 + 2.0 * y_prime * y_prime) + 2.0 * p2 * xy;

        let x_distorted = radial * x_prime + dx;
        let y_distorted = radial * y_prime + dy;

        Ok(Vector2::new(
            self.pinhole.fx * x_distorted + self.pinhole.cx,
            self.pinhole.fy * y_distorted + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a unit 3D ray. Uses Newton-Raphson on the
    /// distortion residual; returns [`CameraModelError::NumericalError`] on singular
    /// Jacobian or non-convergence.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let x_distorted = (u - self.pinhole.cx) / self.pinhole.fx;
        let y_distorted = (v - self.pinhole.cy) / self.pinhole.fy;
        let target_distorted_point = Vector2::new(x_distorted, y_distorted);

        let mut point = target_distorted_point;

        const CONVERGENCE_EPS: f64 = crate::CONVERGENCE_THRESHOLD;
        const MAX_ITERATIONS: u32 = 100;

        let (k1, k2, p1, p2, k3) = self.distortion_params();

        for iteration in 0..MAX_ITERATIONS {
            let x = point.x;
            let y = point.y;

            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

            let xy = x * y;
            let dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy;

            let x_dist = radial * x + dx;
            let y_dist = radial * y + dy;

            let fx = x_dist - target_distorted_point.x;
            let fy = y_dist - target_distorted_point.y;

            if fx.abs() < CONVERGENCE_EPS && fy.abs() < CONVERGENCE_EPS {
                break;
            }

            let dradial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

            let dfx_dx = radial + 2.0 * x * dradial_dr2 * x + 2.0 * p1 * y + 2.0 * p2 * (3.0 * x);
            let dfx_dy = 2.0 * x * dradial_dr2 * y + 2.0 * p1 * x + 2.0 * p2 * y;
            let dfy_dx = 2.0 * y * dradial_dr2 * x + 2.0 * p1 * x + 2.0 * p2 * y;
            let dfy_dy = radial + 2.0 * y * dradial_dr2 * y + 2.0 * p1 * (3.0 * y) + 2.0 * p2 * x;

            let jacobian = Matrix2::new(dfx_dx, dfx_dy, dfy_dx, dfy_dy);

            let det = jacobian[(0, 0)] * jacobian[(1, 1)] - jacobian[(0, 1)] * jacobian[(1, 0)];

            if det.abs() < crate::GEOMETRIC_PRECISION {
                return Err(CameraModelError::NumericalError {
                    operation: "unprojection".to_string(),
                    details: "Singular Jacobian in RadTan unprojection".to_string(),
                });
            }

            let inv_det = 1.0 / det;
            let delta_x = inv_det * (jacobian[(1, 1)] * (-fx) - jacobian[(0, 1)] * (-fy));
            let delta_y = inv_det * (-jacobian[(1, 0)] * (-fx) + jacobian[(0, 0)] * (-fy));

            point.x += delta_x;
            point.y += delta_y;

            if iteration == MAX_ITERATIONS - 1 {
                return Err(CameraModelError::NumericalError {
                    operation: "unprojection".to_string(),
                    details: "RadTan unprojection did not converge".to_string(),
                });
            }
        }

        let r2 = point.x * point.x + point.y * point.y;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(
            point.x * norm_inv,
            point.y * norm_inv,
            norm_inv,
        ))
    }

    /// 2×3 Jacobian ∂(u,v)/∂(x,y,z). See the
    /// [cookbook](../doc/cookbook/src/rad-tan.html#jacobians) for the full derivation.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (k1, k2, p1, p2, k3) = self.distortion_params();

        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let dradial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        let dx_dist_dx_prime = radial
            + 2.0 * x_prime * x_prime * dradial_dr2
            + 2.0 * p1 * y_prime
            + 6.0 * p2 * x_prime;

        let dx_dist_dy_prime =
            2.0 * x_prime * y_prime * dradial_dr2 + 2.0 * p1 * x_prime + 2.0 * p2 * y_prime;

        let dy_dist_dx_prime =
            2.0 * y_prime * x_prime * dradial_dr2 + 2.0 * p1 * x_prime + 2.0 * p2 * y_prime;

        let dy_dist_dy_prime = radial
            + 2.0 * y_prime * y_prime * dradial_dr2
            + 6.0 * p1 * y_prime
            + 2.0 * p2 * x_prime;

        let du_dx = self.pinhole.fx * (dx_dist_dx_prime * inv_z);
        let du_dy = self.pinhole.fx * (dx_dist_dy_prime * inv_z);
        let du_dz = self.pinhole.fx
            * (dx_dist_dx_prime * (-x_prime * inv_z) + dx_dist_dy_prime * (-y_prime * inv_z));

        let dv_dx = self.pinhole.fy * (dy_dist_dx_prime * inv_z);
        let dv_dy = self.pinhole.fy * (dy_dist_dy_prime * inv_z);
        let dv_dz = self.pinhole.fy
            * (dy_dist_dx_prime * (-x_prime * inv_z) + dy_dist_dy_prime * (-y_prime * inv_z));

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// 2×9 Jacobian ∂(u,v)/∂[fx, fy, cx, cy, k1, k2, p1, p2, k3]. See the
    /// [cookbook](../doc/cookbook/src/rad-tan.html#jacobians) for the full derivation.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (k1, k2, p1, p2, k3) = self.distortion_params();

        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        let xy = x_prime * y_prime;
        let dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x_prime * x_prime);
        let dy = p1 * (r2 + 2.0 * y_prime * y_prime) + 2.0 * p2 * xy;

        let x_distorted = radial * x_prime + dx;
        let y_distorted = radial * y_prime + dy;

        let du_dk1 = self.pinhole.fx * x_prime * r2;
        let du_dk2 = self.pinhole.fx * x_prime * r4;
        let du_dp1 = self.pinhole.fx * 2.0 * xy;
        let du_dp2 = self.pinhole.fx * (r2 + 2.0 * x_prime * x_prime);
        let du_dk3 = self.pinhole.fx * x_prime * r6;

        let dv_dk1 = self.pinhole.fy * y_prime * r2;
        let dv_dk2 = self.pinhole.fy * y_prime * r4;
        let dv_dp1 = self.pinhole.fy * (r2 + 2.0 * y_prime * y_prime);
        let dv_dp2 = self.pinhole.fy * 2.0 * xy;
        let dv_dk3 = self.pinhole.fy * y_prime * r6;

        SMatrix::<f64, 2, 9>::from_row_slice(&[
            x_distorted,
            0.0,
            1.0,
            0.0,
            du_dk1,
            du_dk2,
            du_dp1,
            du_dp2,
            du_dk3,
            0.0,
            y_distorted,
            0.0,
            1.0,
            dv_dk1,
            dv_dk2,
            dv_dp1,
            dv_dp2,
            dv_dk3,
        ])
    }

    /// Validates the camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - `fx`, `fy` must be positive (> 0) and finite
    /// - `cx`, `cy` must be finite
    /// - `k₁`, `k₂`, `p₁`, `p₂`, `k₃` must be finite
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
        self.pinhole
    }

    /// Returns the distortion model (must be [`DistortionModel::BrownConrady`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the model name: `"rad_tan"`.
    fn get_model_name(&self) -> &'static str {
        "rad_tan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_radtan_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        assert_eq!(camera.pinhole.fx, 300.0);
        let (k1, _, p1, _, _) = camera.distortion_params();
        assert_eq!(k1, 0.1);
        assert_eq!(p1, 0.001);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..9 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = RadTanCamera::try_from(params_plus.as_slice())?;
            let cam_minus = RadTanCamera::try_from(params_minus.as_slice())?;

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
    fn test_rad_tan_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 9);
        assert_eq!(params[0], 300.0);
        assert_eq!(params[1], 300.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 0.1);
        assert_eq!(params[5], 0.01);
        assert_eq!(params[6], 0.001);
        assert_eq!(params[7], 0.002);
        assert_eq!(params[8], 0.001);

        // Test conversion to array
        let arr: [f64; 9] = (&camera).into();
        assert_eq!(
            arr,
            [300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002, 0.001]
        );

        // Test conversion from slice
        let params_slice = [350.0, 350.0, 330.0, 250.0, 0.2, 0.02, 0.002, 0.003, 0.002];
        let camera2 = RadTanCamera::try_from(&params_slice[..])?;
        assert_eq!(camera2.pinhole.fx, 350.0);
        assert_eq!(camera2.pinhole.fy, 350.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        let (k1, k2, p1, p2, k3) = camera2.distortion_params();
        assert_eq!(k1, 0.2);
        assert_eq!(k2, 0.02);
        assert_eq!(p1, 0.002);
        assert_eq!(p2, 0.003);
        assert_eq!(k3, 0.002);

        // Test conversion from array
        let camera3 =
            RadTanCamera::from([400.0, 400.0, 340.0, 260.0, 0.3, 0.03, 0.003, 0.004, 0.003]);
        assert_eq!(camera3.pinhole.fx, 400.0);
        assert_eq!(camera3.pinhole.fy, 400.0);
        let (k1, k2, p1, p2, k3) = camera3.distortion_params();
        assert_eq!(k1, 0.3);
        assert_eq!(k2, 0.03);
        assert_eq!(p1, 0.003);
        assert_eq!(p2, 0.004);
        assert_eq!(k3, 0.003);

        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth camera with radial distortion only (p1=p2=0)
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::BrownConrady {
            k1: 0.05,
            k2: 0.01,
            p1: 0.0,
            p2: 0.0,
            k3: 0.001,
        };
        let gt_camera = RadTanCamera::new(gt_pinhole, gt_distortion)?;

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

        // Initial camera with zero distortion
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        let mut camera = RadTanCamera::new(init_pinhole, init_distortion)?;

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
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;

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
                (dot - 1.0).abs() < 1e-5,
                "Round-trip failed: dot={dot}, expected ~1.0"
            );
        }

        Ok(())
    }

    #[test]
    fn test_project_returns_error_behind_camera() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        assert!(camera.project(&Vector3::new(0.0, 0.0, -1.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_project_at_min_depth_boundary() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        let p_min = Vector3::new(0.0, 0.0, crate::MIN_DEPTH);
        if let Ok(uv) = camera.project(&p_min) {
            assert!(uv.x.is_finite() && uv.y.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_projection_off_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.3, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;
        assert!(
            uv.x > 320.0,
            "off-axis point should project right of principal point"
        );
        assert!(
            (uv.y - 240.0).abs() < 5.0,
            "y should be close to cy for horizontal offset"
        );
        Ok(())
    }

    #[test]
    fn test_unproject_center_pixel() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.0,
            p2: 0.0,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        let uv = Vector2::new(320.0, 240.0);
        let ray = camera.unproject(&uv)?;
        assert!(ray.x.abs() < 1e-5, "x should be ~0, got {}", ray.x);
        assert!(ray.y.abs() < 1e-5, "y should be ~0, got {}", ray.y);
        assert!((ray.z - 1.0).abs() < 1e-5, "z should be ~1, got {}", ray.z);
        Ok(())
    }

    #[test]
    fn test_batch_projection_matches_individual() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let camera = RadTanCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);
        let jac_point = camera.jacobian_point(&p_cam);
        assert_eq!(jac_point.nrows(), 2);
        assert_eq!(jac_point.ncols(), 3);
        let jac_intr = camera.jacobian_intrinsics(&p_cam);
        assert_eq!(jac_intr.nrows(), 2);
        assert_eq!(jac_intr.ncols(), 9); // RadTanCamera::INTRINSIC_DIM = 9
        Ok(())
    }
}
