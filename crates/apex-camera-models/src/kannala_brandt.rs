//! Kannala-Brandt Fisheye Camera Model
//!
//! A widely-used fisheye camera model with polynomial radial distortion,
//! commonly implemented in OpenCV for fisheye lens calibration.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! r = √(x² + y²)
//! θ = atan2(r, z)
//! θ_d = θ·(1 + k₁·θ² + k₂·θ⁴ + k₃·θ⁶ + k₄·θ⁸)
//! u = fx · θ_d · (x/r) + cx
//! v = fy · θ_d · (y/r) + cy
//! ```
//!
//! Or equivalently: d(θ) = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
//!
//! ## Unprojection (2D → 3D)
//!
//! Uses Newton-Raphson iteration to solve for θ from θ_d, then recovers
//! the 3D ray direction.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: k₁, k₂, k₃, k₄ (8 parameters total)
//!
//! # Use Cases
//!
//! - Fisheye cameras with up to 180° field of view
//! - Wide-angle surveillance cameras
//! - Automotive and robotics applications
//! - OpenCV fisheye calibration
//!
//! # References
//!
//! - Kannala & Brandt, "A Generic Camera Model and Calibration Method for
//!   Conventional, Wide-Angle, and Fish-Eye Lenses", PAMI 2006

use super::{CameraModel, CameraModelError, Intrinsics, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-6;

/// Kannala-Brandt fisheye camera model with 8 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KannalaBrandtCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
}

impl KannalaBrandtCamera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    ) -> Result<Self, CameraModelError> {
        let model = Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            k4,
        };
        model.validate_params()?;
        Ok(model)
    }

    /// Checks the geometric condition for a valid projection.
    fn check_projection_condition(&self, z: f64) -> bool {
        z > f64::EPSILON
    }
}

impl CameraModel for KannalaBrandtCamera {
    const INTRINSIC_DIM: usize = 8;
    type IntrinsicJacobian = SMatrix<f64, 2, 8>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// r = √(x² + y²)
    /// θ = atan2(r, z)
    /// θ_d = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
    /// u = fx · θ_d · (x/r) + cx
    /// v = fy · θ_d · (y/r) + cy
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates if valid
    /// - `None` - If r < PRECISION (point on optical axis)
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        // Check if point is valid for projection (in front of camera)
        if !self.check_projection_condition(z) {
            return None;
        }

        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let theta = r.atan2(z);

        // Polynomial distortion: d(θ) = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d =
            theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

        if r < PRECISION {
            // Point near optical axis: x/r and y/r are unstable.
            // Limit approaches (fx * (theta_d/r) * x + cx)
            // theta ~ r/z (for small theta), theta_d ~ theta (for small theta)
            // theta_d/r ~ 1/z.
            // u = fx * x/z + cx, v = fy * y/z + cy.
            // Effectively pinhole close to center.
            let inv_z = 1.0 / z;
            return Some(Vector2::new(
                self.fx * x * inv_z + self.cx,
                self.fy * y * inv_z + self.cy,
            ));
        }

        let inv_r = 1.0 / r;
        Some(Vector2::new(
            self.fx * theta_d * x * inv_r + self.cx,
            self.fy * theta_d * y * inv_r + self.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Newton-Raphson iteration to solve for θ from θ_d:
    /// - f(θ) = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹ - θ_d = 0
    /// - f'(θ) = 1 + 3k₁·θ² + 5k₂·θ⁴ + 7k₃·θ⁶ + 9k₄·θ⁸
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - If Newton-Raphson fails to converge
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let mx = (u - self.cx) / self.fx;
        let my = (v - self.cy) / self.fy;

        let mut ru = (mx * mx + my * my).sqrt();

        // Clamp ru to avoid instability if extremely large (from C++ impl: min(ru, PI/2))
        ru = ru.min(std::f64::consts::PI / 2.0);

        if ru < PRECISION {
            return Ok(Vector3::new(0.0, 0.0, 1.0));
        }

        // Newton-Raphson
        let mut theta = ru; // Initial guess
        const MAX_ITER: usize = 10;
        const CONVERGENCE_THRESHOLD: f64 = 1e-6;

        for _ in 0..MAX_ITER {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;

            let k1_theta2 = self.k1 * theta2;
            let k2_theta4 = self.k2 * theta4;
            let k3_theta6 = self.k3 * theta6;
            let k4_theta8 = self.k4 * theta8;

            // f(θ)
            let f = theta * (1.0 + k1_theta2 + k2_theta4 + k3_theta6 + k4_theta8) - ru;

            // f'(θ)
            let f_prime =
                1.0 + 3.0 * k1_theta2 + 5.0 * k2_theta4 + 7.0 * k3_theta6 + 9.0 * k4_theta8;

            if f_prime.abs() < f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "Derivative too small in KB unprojection".to_string(),
                ));
            }

            let delta = f / f_prime;
            theta -= delta;

            if delta.abs() < CONVERGENCE_THRESHOLD {
                break;
            }
        }

        // Convert θ to 3D ray
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        // Direction in xy plane
        // if ru is small we returned already.
        // x = mx * sin(theta) / ru
        // y = my * sin(theta) / ru
        let scale = sin_theta / ru;
        let x = mx * scale;
        let y = my * scale;
        let z = cos_theta;

        Ok(Vector3::new(x, y, z).normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - Always returns true (KB model has wide acceptance range)
    fn is_valid_point(&self, _p_cam: &Vector3<f64>) -> bool {
        true
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// # Chain Rule Application
    ///
    /// Complex derivatives involving:
    /// - ∂θ/∂(x,y,z)
    /// - ∂θ_d/∂θ using polynomial derivative
    /// - ∂(u,v)/∂θ_d
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r = (x * x + y * y).sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d =
            theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

        // ∂θ_d/∂θ = 1 + 3k₁·θ² + 5k₂·θ⁴ + 7k₃·θ⁶ + 9k₄·θ⁸
        let dtheta_d_dtheta = 1.0
            + 3.0 * self.k1 * theta2
            + 5.0 * self.k2 * theta2 * theta2
            + 7.0 * self.k3 * theta2 * theta2 * theta2
            + 9.0 * self.k4 * theta2 * theta2 * theta2 * theta2;

        if r < PRECISION {
            // Near optical axis, use simplified Jacobian
            return SMatrix::<f64, 2, 3>::new(
                self.fx * dtheta_d_dtheta / z,
                0.0,
                0.0,
                0.0,
                self.fy * dtheta_d_dtheta / z,
                0.0,
            );
        }

        let inv_r = 1.0 / r;
        let r2 = r * r;
        let r_z2 = r2 + z * z;

        // ∂θ/∂x = z·x / (r·(r² + z²))
        // ∂θ/∂y = z·y / (r·(r² + z²))
        // ∂θ/∂z = -r / (r² + z²)
        let dtheta_dx = z * x / (r * r_z2);
        let dtheta_dy = z * y / (r * r_z2);
        let dtheta_dz = -r / r_z2;

        // ∂r/∂x = x/r, ∂r/∂y = y/r, ∂r/∂z = 0

        // Chain rule for u = fx · θ_d · (x/r) + cx
        let inv_r2 = inv_r * inv_r;

        let du_dx = self.fx
            * (dtheta_d_dtheta * dtheta_dx * x * inv_r
                + theta_d * (inv_r - x * x * inv_r2 * inv_r));
        let du_dy =
            self.fx * (dtheta_d_dtheta * dtheta_dy * x * inv_r - theta_d * x * y * inv_r2 * inv_r);
        let du_dz = self.fx * dtheta_d_dtheta * dtheta_dz * x * inv_r;

        let dv_dx =
            self.fy * (dtheta_d_dtheta * dtheta_dx * y * inv_r - theta_d * x * y * inv_r2 * inv_r);
        let dv_dy = self.fy
            * (dtheta_d_dtheta * dtheta_dy * y * inv_r
                + theta_d * (inv_r - y * y * inv_r2 * inv_r));
        let dv_dz = self.fy * dtheta_d_dtheta * dtheta_dz * y * inv_r;

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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×8).
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r = (x * x + y * y).sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d =
            theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

        if r < PRECISION {
            return SMatrix::<f64, 2, 8>::zeros();
        }

        let inv_r = 1.0 / r;
        let x_theta_d_r = x * theta_d * inv_r;
        let y_theta_d_r = y * theta_d * inv_r;

        // ∂u/∂fx = θ_d·x/r, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = θ_d·y/r, ∂v/∂cx = 0, ∂v/∂cy = 1

        // ∂u/∂k₁ = fx·θ³·x/r, ∂u/∂k₂ = fx·θ⁵·x/r, etc.
        let du_dk1 = self.fx * theta3 * x * inv_r;
        let du_dk2 = self.fx * theta5 * x * inv_r;
        let du_dk3 = self.fx * theta7 * x * inv_r;
        let du_dk4 = self.fx * theta9 * x * inv_r;

        let dv_dk1 = self.fy * theta3 * y * inv_r;
        let dv_dk2 = self.fy * theta5 * y * inv_r;
        let dv_dk3 = self.fy * theta7 * y * inv_r;
        let dv_dk4 = self.fy * theta9 * y * inv_r;

        SMatrix::<f64, 2, 8>::from_row_slice(&[
            x_theta_d_r,
            0.0,
            1.0,
            0.0,
            du_dk1,
            du_dk2,
            du_dk3,
            du_dk4,
            0.0,
            y_theta_d_r,
            0.0,
            1.0,
            dv_dk1,
            dv_dk2,
            dv_dk3,
            dv_dk4,
        ])
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4,
        ])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 8,
            "KannalaBrandtCamera requires at least 8 parameters"
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            k1: params[4],
            k2: params[5],
            k3: params[6],
            k4: params[7],
        }
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - k₁, k₂, k₃, k₄ must be finite
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        if !self.k1.is_finite()
            || !self.k2.is_finite()
            || !self.k3.is_finite()
            || !self.k4.is_finite()
        {
            return Err(CameraModelError::InvalidParams(
                "Distortion coefficients must be finite".to_string(),
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
        vec![self.k1, self.k2, self.k3, self.k4]
    }

    fn get_model_name(&self) -> &'static str {
        "kannala_brandt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_kb_camera_creation() -> TestResult {
        let camera =
            KannalaBrandtCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001)?;
        assert_eq!(camera.fx, 300.0);
        assert_eq!(camera.k1, 0.1);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera =
            KannalaBrandtCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert!((uv.x - 320.0).abs() < 1e-6);
        assert!((uv.y - 240.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera =
            KannalaBrandtCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001)?;
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
        let camera =
            KannalaBrandtCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = 1e-7;

        for i in 0..8 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = KannalaBrandtCamera::from_params(params_plus.as_slice());
            let cam_minus = KannalaBrandtCamera::from_params(params_minus.as_slice());

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
