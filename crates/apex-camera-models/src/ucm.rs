//! Unified Camera Model (UCM)
//!
//! A generic camera model for catadioptric and fisheye cameras using
//! a single parameter α for projection onto a unit sphere.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d = √(x² + y² + z²)
//! denom = α·d + (1-α)·z
//! u = fx · (x/denom) + cx
//! v = fy · (y/denom) + cy
//! ```
//!
//! where α is the projection parameter (typically α ∈ [0, 1]).
//!
//! ## Unprojection (2D → 3D)
//!
//! Algebraic solution using the UCM inverse equations.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: α (projection parameter) (5 parameters total)
//!
//! # Use Cases
//!
//! - Catadioptric systems (mirror-based omnidirectional cameras)
//! - Fisheye lenses
//! - Wide-angle cameras
//!
//! # References
//!
//! - Geyer & Daniilidis, "A Unifying Theory for Central Panoramic Systems"

use super::{CameraModel, CameraModelError, Intrinsics, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-3;

/// Unified Camera Model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UcmCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub alpha: f64,
}

impl UcmCamera {
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64, alpha: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            alpha,
        }
    }
}

impl CameraModel for UcmCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d = √(x² + y² + z²)
    /// denom = α·d + (1-α)·z
    /// u = fx · (x/denom) + cx
    /// v = fy · (y/denom) + cy
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates if valid
    /// - `None` - If denom < PRECISION or projection condition fails
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        // Check projection validity
        let w = if self.alpha <= 0.5 {
            self.alpha / (1.0 - self.alpha)
        } else {
            (1.0 - self.alpha) / self.alpha
        };
        let check_projection = z > -w * d;

        if denom < PRECISION || !check_projection {
            return None;
        }

        Some(Vector2::new(
            self.fx * x / denom + self.cx,
            self.fy * y / denom + self.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Algebraic solution for UCM inverse projection.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - If unprojection condition fails
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;
        let gamma = 1.0 - self.alpha;
        let xi = self.alpha / gamma;
        let mx = (u - self.cx) / self.fx * gamma;
        let my = (v - self.cy) / self.fy * gamma;

        let r_squared = mx * mx + my * my;

        // Check unprojection condition
        if self.alpha > 0.5 {
            let gamma_sq = gamma * gamma;
            if r_squared > gamma_sq / (2.0 * self.alpha - 1.0) {
                return Err(CameraModelError::PointIsOutSideImage);
            }
        }

        let num = xi + (1.0 + (1.0 - xi * xi) * r_squared).sqrt();
        let denom = 1.0 - r_squared;

        if denom < PRECISION {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let coeff = num / denom;

        let point3d = Vector3::new(coeff * mx, coeff * my, coeff) - Vector3::new(0.0, 0.0, xi);

        Ok(point3d.normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - denom = α·d + (1-α)·z must be ≥ PRECISION
    /// - Projection condition: z > -w·d where w depends on α
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        let w = if self.alpha <= 0.5 {
            self.alpha / (1.0 - self.alpha)
        } else {
            (1.0 - self.alpha) / self.alpha
        };
        let check_projection = z > -w * d;

        denom >= PRECISION && check_projection
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let rho = (x * x + y * y + z * z).sqrt();

        // Denominator D = alpha * rho + (1 - alpha) * z
        // Partial derivatives of D:
        // ∂D/∂x = alpha * x / rho
        // ∂D/∂y = alpha * y / rho
        // ∂D/∂z = alpha * z / rho + (1 - alpha)

        let d_denom_dx = self.alpha * x / rho;
        let d_denom_dy = self.alpha * y / rho;
        let d_denom_dz = self.alpha * z / rho + (1.0 - self.alpha);

        let denom = self.alpha * rho + (1.0 - self.alpha) * z;

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

        jac[(0, 0)] = self.fx * (denom - x * d_denom_dx) / denom2;
        jac[(0, 1)] = self.fx * (-x * d_denom_dy) / denom2;
        jac[(0, 2)] = self.fx * (-x * d_denom_dz) / denom2;

        jac[(1, 0)] = self.fy * (-y * d_denom_dx) / denom2;
        jac[(1, 1)] = self.fy * (denom - y * d_denom_dy) / denom2;
        jac[(1, 2)] = self.fy * (-y * d_denom_dz) / denom2;

        jac
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

        let rho = (x * x + y * y + z * z).sqrt();
        let denom = self.alpha * rho + (1.0 - self.alpha) * z;

        let x_norm = x / denom;
        let y_norm = y / denom;

        let u_cx = self.fx * x_norm;
        let v_cy = self.fy * y_norm;

        let mut jac = SMatrix::<f64, 2, 5>::zeros();

        // ∂u/∂fx = x / denom
        jac[(0, 0)] = x_norm;

        // ∂v/∂fy = y / denom
        jac[(1, 1)] = y_norm;

        // ∂u/∂cx = 1
        jac[(0, 2)] = 1.0;

        // ∂v/∂cy = 1
        jac[(1, 3)] = 1.0;

        // ∂denom/∂alpha = rho - z
        let d_denom_d_alpha = rho - z;

        // ∂u/∂alpha = -fx * x / denom² * (rho - z) = -u_cx * (rho - z) / denom
        jac[(0, 4)] = -u_cx * d_denom_d_alpha / denom;
        jac[(1, 4)] = -v_cy * d_denom_d_alpha / denom;

        jac
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.fx, self.fy, self.cx, self.cy, self.alpha])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 5,
            "UcmCamera requires at least 5 parameters"
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            alpha: params[4],
        }
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - α must be finite
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        if !self.alpha.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "alpha must be finite".to_string(),
            ));
        }

        Ok(())
    }

    fn get_intrinsics(&self) -> Intrinsics {
        Intrinsics {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
        }
    }

    fn get_distortion(&self) -> Vec<f64> {
        vec![self.alpha]
    }

    fn get_model_name(&self) -> &'static str {
        "ucm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_ucm_camera_creation() {
        let camera = UcmCamera::new(300.0, 300.0, 320.0, 240.0, 0.5);
        assert_eq!(camera.fx, 300.0);
        assert_eq!(camera.alpha, 0.5);
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = UcmCamera::new(300.0, 300.0, 320.0, 240.0, 0.5);
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;
        assert!((uv.x - 320.0).abs() < 1e-10);
        assert!((uv.y - 240.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = UcmCamera::new(300.0, 300.0, 320.0, 240.0, 0.6);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);
        let eps = 1e-7;

        for i in 0..3 {
            let mut p_plus = p_cam;
            let mut p_minus = p_cam;
            p_plus[i] += eps;
            p_minus[i] -= eps;

            let uv_plus = camera
                .project(&p_plus)
                .ok_or("Projection failed for p_plus")?;
            let uv_minus = camera
                .project(&p_minus)
                .ok_or("Projection failed for p_minus")?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < 1e-5,
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
        let camera = UcmCamera::new(300.0, 300.0, 320.0, 240.0, 0.6);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = 1e-7;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = UcmCamera::from_params(params_plus.as_slice());
            let cam_minus = UcmCamera::from_params(params_minus.as_slice());

            let uv_plus = cam_plus
                .project(&p_cam)
                .ok_or("Projection failed for cam_plus")?;
            let uv_minus = cam_minus
                .project(&p_cam)
                .ok_or("Projection failed for cam_minus")?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < 1e-5,
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
}
