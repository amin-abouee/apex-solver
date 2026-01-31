//! Field-of-View (FOV) Camera Model
//!
//! A fisheye camera model using a field-of-view parameter for radial distortion.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! r = √(x² + y²)
//! atan_wrd = atan2(2·tan(w/2)·r, z)
//! rd = atan_wrd / (r·w)    (if r > 0)
//! rd = 2·tan(w/2) / w       (if r ≈ 0)
//!
//! mx = x · rd
//! my = y · rd
//! u = fx · mx + cx
//! v = fy · my + cy
//! ```
//!
//! where w is the field-of-view parameter (0 < w ≤ π).
//!
//! ## Unprojection (2D → 3D)
//!
//! Uses trigonometric inverse with special handling near optical axis.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: w (field-of-view parameter) (5 parameters total)
//!
//! # Use Cases
//!
//! - Fisheye cameras in SLAM applications
//! - Wide field-of-view lenses
//!
//! # References
//!
//! - Zhang et al., "Simultaneous Localization and Mapping with Fisheye Cameras"
//!   https://arxiv.org/pdf/1807.08957

use crate::{
    Camera, CameraModel, CameraModelError, DistortionModel, PinholeParams, skew_symmetric,
};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// FOV camera model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovCamera {
    pub camera: Camera,
}

impl FovCamera {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, w: f64) -> Result<Self, CameraModelError> {
        let camera = Self {
            camera: Camera {
                pinhole: PinholeParams { fx, fy, cx, cy },
                distortion: DistortionModel::FOV { w },
            },
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Helper method to extract distortion parameter.
    fn distortion_params(&self) -> f64 {
        match self.camera.distortion {
            DistortionModel::FOV { w } => w,
            _ => panic!("Invalid distortion model for FovCamera"),
        }
    }

    /// Validates camera parameters.
    pub fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.camera.pinhole.fx <= 0.0 || self.camera.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.camera.pinhole.cx.is_finite() || !self.camera.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        let w = self.distortion_params();
        if !w.is_finite() || w <= 0.0 || w > std::f64::consts::PI {
            return Err(CameraModelError::InvalidParams(
                "w must be in (0, π]".to_string(),
            ));
        }

        Ok(())
    }
}

impl CameraModel for FovCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// Uses atan-based radial distortion with FOV parameter w.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates if valid
    /// - `None` - If projection fails
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        // Check if z is valid (too close to camera center)
        if z < f64::EPSILON.sqrt() {
            return None;
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

        Some(Vector2::new(
            self.camera.pinhole.fx * mx + self.camera.pinhole.cx,
            self.camera.pinhole.fy * my + self.camera.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Trigonometric inverse using sin/cos relationships.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - If unprojection fails
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let mx = (u - self.camera.pinhole.cx) / self.camera.pinhole.fx;
        let my = (v - self.camera.pinhole.cy) / self.camera.pinhole.fy;

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

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - Always returns true for FOV model (wide acceptance range)
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let z = p_cam[2];
        z >= f64::EPSILON.sqrt()
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
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
                self.camera.pinhole.fx * rd,
                0.0,
                0.0,
                0.0,
                self.camera.pinhole.fy * rd,
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
            self.camera.pinhole.fx * dmx_dx,
            self.camera.pinhole.fx * dmx_dy,
            self.camera.pinhole.fx * dmx_dz,
            self.camera.pinhole.fy * dmy_dx,
            self.camera.pinhole.fy * dmy_dy,
            self.camera.pinhole.fy * dmy_dz,
        )
    }

    /// Jacobian of projection w.r.t. camera pose (SE3).
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        let d_uv_d_pcam = self.jacobian_point(&p_cam);

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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×5).
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

        let du_dw = self.camera.pinhole.fx * x * drd_dw;
        let dv_dw = self.camera.pinhole.fy * y * drd_dw;

        SMatrix::<f64, 2, 5>::new(mx, 0.0, 1.0, 0.0, du_dw, 0.0, my, 0.0, 1.0, dv_dw)
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        let w = self.distortion_params();
        DVector::from_vec(vec![
            self.camera.pinhole.fx,
            self.camera.pinhole.fy,
            self.camera.pinhole.cx,
            self.camera.pinhole.cy,
            w,
        ])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 5,
            "FovCamera requires at least 5 parameters"
        );
        Self {
            camera: Camera {
                pinhole: PinholeParams {
                    fx: params[0],
                    fy: params[1],
                    cx: params[2],
                    cy: params[3],
                },
                distortion: DistortionModel::FOV { w: params[4] },
            },
        }
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - w must be in (0, π]
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.camera.pinhole.fx <= 0.0 || self.camera.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.camera.pinhole.cx.is_finite() || !self.camera.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        let w = self.distortion_params();
        if !w.is_finite() || w <= 0.0 || w > std::f64::consts::PI {
            return Err(CameraModelError::InvalidParams(
                "w must be in (0, π]".to_string(),
            ));
        }

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

    fn get_distortion(&self) -> Vec<f64> {
        vec![self.distortion_params()]
    }

    fn get_model_name(&self) -> &'static str {
        "fov"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_fov_camera_creation() -> TestResult {
        let camera = FovCamera::new(300.0, 300.0, 320.0, 240.0, 1.5)?;
        assert_eq!(camera.camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params(), 1.5);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = FovCamera::new(300.0, 300.0, 320.0, 240.0, 1.5)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert!((uv.x - 320.0).abs() < 1e-4);
        assert!((uv.y - 240.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = FovCamera::new(300.0, 300.0, 320.0, 240.0, 1.5)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..3 {
            let mut p_plus = p_cam;
            let mut p_minus = p_cam;
            p_plus[i] += eps;
            p_minus[i] -= eps;

            let uv_plus = camera.project(&p_plus).ok_or("Projection failed")?;
            let uv_minus = camera.project(&p_minus).ok_or("Projection failed")?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
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
        let camera = FovCamera::new(300.0, 300.0, 320.0, 240.0, 1.5)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = FovCamera::from_params(params_plus.as_slice());
            let cam_minus = FovCamera::from_params(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam).ok_or("Projection failed")?;
            let uv_minus = cam_minus.project(&p_cam).ok_or("Projection failed")?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(diff < 1e-4, "Mismatch at ({}, {})", r, i);
            }
        }
        Ok(())
    }
}
