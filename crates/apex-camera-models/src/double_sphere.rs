//! Double Sphere Camera Model
//!
//! A two-parameter fisheye model that provides improved accuracy over
//! the Unified Camera Model by using two sphere projections.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d₁ = √(x² + y² + z²)
//! d₂ = √(x² + y² + (ξ·d₁ + z)²)
//! denom = α·d₂ + (1-α)·(ξ·d₁ + z)
//! u = fx · (x/denom) + cx
//! v = fy · (y/denom) + cy
//! ```
//!
//! where:
//! - ξ (xi) is the first distortion parameter
//! - α (alpha) ∈ (0, 1] is the second distortion parameter
//!
//! ## Unprojection (2D → 3D)
//!
//! Algebraic solution using the double sphere inverse equations.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: ξ (xi), α (alpha) (6 parameters total)
//!
//! # Use Cases
//!
//! - High-quality fisheye calibration
//! - Wide field-of-view cameras
//! - More accurate than UCM for extreme wide-angle lenses
//!
//! # References
//!
//! - Usenko et al., "The Double Sphere Camera Model", 3DV 2018

use super::{CameraModel, CameraModelError, Intrinsics, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-6;

/// Double Sphere camera model with 6 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleSphereCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub xi: f64,
    pub alpha: f64,
}

impl DoubleSphereCamera {
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        xi: f64,
        alpha: f64,
    ) -> Result<Self, CameraModelError> {
        let model = Self {
            fx,
            fy,
            cx,
            cy,
            xi,
            alpha,
        };
        // Use validate_params to ensure consistency
        model.validate_params()?;
        Ok(model)
    }

    /// Checks the geometric condition for a valid projection.
    fn check_projection_condition(&self, z: f64, d1: f64) -> bool {
        let w1 = if self.alpha > 0.5 {
            (1.0 - self.alpha) / self.alpha
        } else {
            self.alpha / (1.0 - self.alpha)
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi + self.xi * self.xi + 1.0).sqrt();
        z > -w2 * d1
    }

    /// Checks the geometric condition for a valid unprojection.
    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        if self.alpha > 0.5 {
            if r_squared > 1.0 / (2.0 * self.alpha - 1.0) {
                return false;
            }
        }
        true
    }
}

impl CameraModel for DoubleSphereCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d₁ = √(x² + y² + z²)
    /// d₂ = √(x² + y² + (ξ·d₁ + z)²)
    /// denom = α·d₂ + (1-α)·(ξ·d₁ + z)
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
    /// - `None` - If projection condition fails
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();

        // Check projection condition using the helper
        if !self.check_projection_condition(z, d1) {
            return None;
        }

        let xi_d1_z = self.xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = self.alpha * d2 + (1.0 - self.alpha) * xi_d1_z;

        if denom < PRECISION {
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
    /// Algebraic solution for double sphere inverse projection.
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

        if !self.check_unprojection_condition(r2) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let xi = self.xi;
        let alpha = self.alpha;

        let mz_num = 1.0 - alpha * alpha * r2;
        let mz_denom = alpha * (1.0 - (2.0 * alpha - 1.0) * r2).sqrt() + (1.0 - alpha);
        let mz = mz_num / mz_denom;

        let mz2 = mz * mz;

        let num_term = mz * xi + (mz2 + (1.0 - xi * xi) * r2).sqrt();
        let denom_term = mz2 + r2;

        if denom_term < PRECISION {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let k = num_term / denom_term;

        let point3d = Vector3::new(k * mx, k * my, k * mz - xi);
        Ok(point3d.normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - d₁ > PRECISION
    /// - Projection condition: z > -w₂·d₁ where w₂ depends on α and ξ
    /// - denom = α·d₂ + (1-α)·(ξ·d₁ + z) must be ≥ PRECISION
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();

        if d1 < PRECISION {
            return false;
        }

        let w1 = if self.alpha > 0.5 {
            (1.0 - self.alpha) / self.alpha
        } else {
            self.alpha / (1.0 - self.alpha)
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi).sqrt();

        if z <= -w2 * d1 {
            return false;
        }

        let xi_d1_z = self.xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = self.alpha * d2 + (1.0 - self.alpha) * xi_d1_z;

        denom >= PRECISION
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// # Mathematical Derivatives
    ///
    /// Complex chain rule involving d₁ and d₂ derivatives.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let xi_d1_z = self.xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = self.alpha * d2 + (1.0 - self.alpha) * xi_d1_z;

        // ∂d₁/∂x = x/d₁, ∂d₁/∂y = y/d₁, ∂d₁/∂z = z/d₁
        let dd1_dx = x / d1;
        let dd1_dy = y / d1;
        let dd1_dz = z / d1;

        // ∂(ξ·d₁+z)/∂x = ξ·∂d₁/∂x
        let d_xi_d1_z_dx = self.xi * dd1_dx;
        let d_xi_d1_z_dy = self.xi * dd1_dy;
        let d_xi_d1_z_dz = self.xi * dd1_dz + 1.0;

        // ∂d₂/∂x = (x + (ξ·d₁+z)·∂(ξ·d₁+z)/∂x) / d₂
        let dd2_dx = (x + xi_d1_z * d_xi_d1_z_dx) / d2;
        let dd2_dy = (y + xi_d1_z * d_xi_d1_z_dy) / d2;
        let dd2_dz = (xi_d1_z * d_xi_d1_z_dz) / d2;

        // ∂denom/∂x = α·∂d₂/∂x + (1-α)·∂(ξ·d₁+z)/∂x
        let ddenom_dx = self.alpha * dd2_dx + (1.0 - self.alpha) * d_xi_d1_z_dx;
        let ddenom_dy = self.alpha * dd2_dy + (1.0 - self.alpha) * d_xi_d1_z_dy;
        let ddenom_dz = self.alpha * dd2_dz + (1.0 - self.alpha) * d_xi_d1_z_dz;

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
        let d1 = (r2 + z * z).sqrt();
        let xi_d1_z = self.xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = self.alpha * d2 + (1.0 - self.alpha) * xi_d1_z;

        let x_norm = x / denom;
        let y_norm = y / denom;

        // ∂u/∂fx = x/denom, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y/denom, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For ξ and α derivatives
        let d_xi_d1_z_dxi = d1;
        let dd2_dxi = (xi_d1_z * d_xi_d1_z_dxi) / d2;
        let ddenom_dxi = self.alpha * dd2_dxi + (1.0 - self.alpha) * d_xi_d1_z_dxi;

        let ddenom_dalpha = d2 - xi_d1_z;

        let du_dxi = -self.fx * x * ddenom_dxi / (denom * denom);
        let dv_dxi = -self.fy * y * ddenom_dxi / (denom * denom);

        let du_dalpha = -self.fx * x * ddenom_dalpha / (denom * denom);
        let dv_dalpha = -self.fy * y * ddenom_dalpha / (denom * denom);

        SMatrix::<f64, 2, 6>::new(
            x_norm, 0.0, 1.0, 0.0, du_dxi, du_dalpha, 0.0, y_norm, 0.0, 1.0, dv_dxi, dv_dalpha,
        )
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha,
        ])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 6,
            "DoubleSphereCamera requires at least 6 parameters"
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            xi: params[4],
            alpha: params[5],
        }
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - ξ must be finite
    /// - α must be in (0, 1]
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        if !self.xi.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "xi must be finite".to_string(),
            ));
        }

        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(CameraModelError::InvalidParams(
                "alpha must be in (0, 1]".to_string(),
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
        vec![self.xi, self.alpha]
    }

    fn get_model_name(&self) -> &'static str {
        "double_sphere"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_double_sphere_camera_creation() -> TestResult {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, -0.2, 0.6)?;
        assert_eq!(camera.fx, 300.0);
        assert_eq!(camera.xi, -0.2);
        assert_eq!(camera.alpha, 0.6);

        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, -0.2, 0.6)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert!((uv.x - 320.0).abs() < 1e-4);
        assert!((uv.y - 240.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, -0.2, 0.6)?;
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
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, -0.2, 0.6)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = 1e-7;

        for i in 0..6 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = DoubleSphereCamera::from_params(params_plus.as_slice());
            let cam_minus = DoubleSphereCamera::from_params(params_minus.as_slice());

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
