//! Extended Unified Camera Model (EUCM)
//!
//! An extension of the Unified Camera Model with an additional parameter for
//! improved modeling of wide-angle and fisheye lenses.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d = √(β(x² + y²) + z²)
//! denom = α·d + (1-α)·z
//! u = fx · (x/denom) + cx
//! v = fy · (y/denom) + cy
//! ```
//!
//! where:
//! - α ∈ [0, 1] is the projection parameter
//! - β > 0 is the distortion parameter
//! - (fx, fy, cx, cy) are standard intrinsics
//!
//! ## Unprojection (2D → 3D)
//!
//! Uses algebraic solution to recover the 3D ray direction.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: α (projection), β (distortion) (6 parameters total)
//!
//! # Use Cases
//!
//! - Wide-angle cameras
//! - Fisheye lenses
//! - More flexible than UCM due to β parameter
//!
//! # References
//!
//! - Khomutenko et al., "An Enhanced Unified Camera Model"

use super::{CameraModel, CameraModelError, Intrinsics, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-3;

/// Extended Unified Camera Model with 6 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EucmCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl EucmCamera {
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        alpha: f64,
        beta: f64,
    ) -> Result<Self, CameraModelError> {
        let camera = Self {
            fx,
            fy,
            cx,
            cy,
            alpha,
            beta,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Checks the geometric condition for a valid projection.
    pub fn check_projection_condition(&self, z: f64, denom: f64) -> bool {
        let mut condition = true;
        if self.alpha > 0.5 {
            let c = (self.alpha - 1.0) / (2.0 * self.alpha - 1.0);
            if z < denom * c {
                condition = false;
            }
        }
        condition
    }

    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let mut condition = true;
        if self.alpha > 0.5 && r_squared > (1.0 / self.beta * (2.0 * self.alpha - 1.0)) {
            condition = false;
        }
        condition
    }
}

impl CameraModel for EucmCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d = √(β(x² + y²) + z²)
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
    /// - `None` - If denom < PRECISION
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d = (self.beta * r2 + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        if denom < PRECISION || !self.check_projection_condition(z, denom) {
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
    /// Algebraic solution using EUCM inverse equations with α and β parameters.
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

        let mx = (u - self.cx) / self.fx;
        let my = (v - self.cy) / self.fy;

        let r2 = mx * mx + my * my;
        let beta_r2 = self.beta * r2;

        let gamma = 1.0 - self.alpha;
        let gamma_sq = gamma * gamma;

        let discriminant = beta_r2 * gamma_sq + gamma_sq;
        if discriminant < 0.0 || !self.check_unprojection_condition(r2) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let sqrt_disc = discriminant.sqrt();
        let denom = beta_r2 + 1.0;

        if denom.abs() < PRECISION {
            return Err(CameraModelError::NumericalError(
                "Division by near-zero in EUCM unprojection".to_string(),
            ));
        }

        let mz = (gamma * sqrt_disc) / denom;

        let point3d = Vector3::new(mx, my, mz);
        Ok(point3d.normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - denom = α·d + (1-α)·z must be ≥ PRECISION
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d = (self.beta * r2 + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        denom >= PRECISION && self.check_projection_condition(z, denom)
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d = (self.beta * r2 + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        // ∂d/∂x = β·x/d, ∂d/∂y = β·y/d, ∂d/∂z = z/d
        let dd_dx = self.beta * x / d;
        let dd_dy = self.beta * y / d;
        let dd_dz = z / d;

        // ∂denom/∂x = α·∂d/∂x
        let ddenom_dx = self.alpha * dd_dx;
        let ddenom_dy = self.alpha * dd_dy;
        let ddenom_dz = self.alpha * dd_dz + (1.0 - self.alpha);

        let denom2 = denom * denom;

        // ∂(x/denom)/∂x = (denom - x·∂denom/∂x) / denom²
        let du_dx = self.fx * (denom - x * ddenom_dx) / denom2;
        let du_dy = self.fx * (-x * ddenom_dy) / denom2;
        let du_dz = self.fx * (-x * ddenom_dz) / denom2;

        let dv_dx = self.fy * (-y * ddenom_dx) / denom2;
        let dv_dy = self.fy * (denom - y * ddenom_dy) / denom2;
        let dv_dz = self.fy * (-y * ddenom_dz) / denom2;

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×6).
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d = (self.beta * r2 + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        let x_norm = x / denom;
        let y_norm = y / denom;

        // ∂u/∂fx = x/denom, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y/denom, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For α and β, need chain rule
        let ddenom_dalpha = d - z;

        let dd_dbeta = r2 / (2.0 * d);
        let ddenom_dbeta = self.alpha * dd_dbeta;

        let du_dalpha = -self.fx * x * ddenom_dalpha / (denom * denom);
        let dv_dalpha = -self.fy * y * ddenom_dalpha / (denom * denom);

        let du_dbeta = -self.fx * x * ddenom_dbeta / (denom * denom);
        let dv_dbeta = -self.fy * y * ddenom_dbeta / (denom * denom);

        SMatrix::<f64, 2, 6>::new(
            x_norm, 0.0, 1.0, 0.0, du_dalpha, du_dbeta, 0.0, y_norm, 0.0, 1.0, dv_dalpha, dv_dbeta,
        )
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta,
        ])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 6,
            "EucmCamera requires at least 6 parameters"
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            alpha: params[4],
            beta: params[5],
        }
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - α must be in [0, 1]
    /// - β must be positive (> 0)
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        if !self.alpha.is_finite() || self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(CameraModelError::InvalidParams(
                "alpha must be in [0, 1]".to_string(),
            ));
        }

        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(CameraModelError::InvalidParams(
                "beta must be positive".to_string(),
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
        vec![self.alpha, self.beta]
    }

    fn get_model_name(&self) -> &'static str {
        "eucm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_eucm_camera_creation() -> TestResult {
        let camera = EucmCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 1.0)?;
        assert_eq!(camera.fx, 300.0);
        assert_eq!(camera.alpha, 0.5);
        assert_eq!(camera.beta, 1.0);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = EucmCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 1.0)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert!((uv.x - 320.0).abs() < 1e-6);
        assert!((uv.y - 240.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = EucmCamera::new(300.0, 300.0, 320.0, 240.0, 0.6, 1.2)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);
        let eps = 1e-7;

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
                assert!(diff < 1e-5, "Mismatch at ({}, {})", r, i);
            }
        }
        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let camera = EucmCamera::new(300.0, 300.0, 320.0, 240.0, 0.6, 1.2)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = 1e-7;

        for i in 0..6 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = EucmCamera::from_params(params_plus.as_slice());
            let cam_minus = EucmCamera::from_params(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam).ok_or("Projection failed")?;
            let uv_minus = cam_minus.project(&p_cam).ok_or("Projection failed")?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(diff < 1e-5, "Mismatch at ({}, {})", r, i);
            }
        }
        Ok(())
    }
}
