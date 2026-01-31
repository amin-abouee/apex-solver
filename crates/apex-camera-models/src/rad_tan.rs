//! Radial-Tangential Distortion Camera Model
//!
//! The standard OpenCV camera model combining radial and tangential distortion.
//! Widely used for narrow to moderate field-of-view cameras.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! x' = x/z,  y' = y/z  (normalized coordinates)
//! r² = x'² + y'²
//!
//! Radial distortion:
//! r' = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
//!
//! Tangential distortion:
//! dx = 2·p₁·x'·y' + p₂·(r² + 2·x'²)
//! dy = p₁·(r² + 2·y'²) + 2·p₂·x'·y'
//!
//! Distorted coordinates:
//! x'' = r'·x' + dx
//! y'' = r'·y' + dy
//!
//! Final projection:
//! u = fx·x'' + cx
//! v = fy·y'' + cy
//! ```
//!
//! ## Unprojection (2D → 3D)
//!
//! Iterative Jacobian-based method to solve the non-linear inverse equations.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: k₁, k₂, p₁, p₂, k₃ (9 parameters total)
//!
//! # Use Cases
//!
//! - Standard narrow FOV cameras
//! - OpenCV-calibrated cameras
//! - Robotics and AR/VR applications
//! - Most conventional lenses
//!
//! # References
//!
//! - Brown, "Decentering Distortion of Lenses", 1966
//! - OpenCV Camera Calibration Documentation

use super::{CameraModel, CameraModelError, Intrinsics, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, Matrix2, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-6;

/// Radial-Tangential camera model with 9 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadTanCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl RadTanCamera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    ) -> Result<Self, super::CameraModelError> {
        let camera = Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Checks if a 3D point satisfies the projection condition (z >= PRECISION).
    fn check_projection_condition(&self, z: f64) -> bool {
        z >= PRECISION
    }
}

impl CameraModel for RadTanCamera {
    const INTRINSIC_DIM: usize = 9;
    type IntrinsicJacobian = SMatrix<f64, 2, 9>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// Combines radial distortion (k₁, k₂, k₃) and tangential distortion (p₁, p₂).
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates if z > PRECISION
    /// - `None` - If point is at or behind camera
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        if !self.check_projection_condition(p_cam.z) {
            return None;
        }

        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // Radial distortion: r' = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        // Tangential distortion
        let xy = x_prime * y_prime;
        let dx = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x_prime * x_prime);
        let dy = self.p1 * (r2 + 2.0 * y_prime * y_prime) + 2.0 * self.p2 * xy;

        // Distorted coordinates
        let x_distorted = radial * x_prime + dx;
        let y_distorted = radial * y_prime + dy;

        Some(Vector2::new(
            self.fx * x_distorted + self.cx,
            self.fy * y_distorted + self.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Iterative Newton-Raphson with Jacobian matrix:
    /// 1. Start with undistorted estimate
    /// 2. Compute distortion and Jacobian
    /// 3. Update estimate: p' = p' - J⁻¹·f(p')
    /// 4. Repeat until convergence
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - If iteration fails to converge
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        // Validate unprojection condition if needed (always true for RadTan generally)
        let u = point_2d.x;
        let v = point_2d.y;

        // Initial estimate (undistorted)
        let x_distorted = (u - self.cx) / self.fx;
        let y_distorted = (v - self.cy) / self.fy;
        let target_distorted_point = Vector2::new(x_distorted, y_distorted);

        let mut point = target_distorted_point;

        const EPS: f64 = 1e-6;
        const MAX_ITERATIONS: u32 = 100;

        for iteration in 0..MAX_ITERATIONS {
            let x = point.x;
            let y = point.y;

            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            // Radial distortion
            let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

            // Tangential distortion
            let xy = x * y;
            let dx = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * xy;

            // Distorted point
            let x_dist = radial * x + dx;
            let y_dist = radial * y + dy;

            // Residual
            let fx = x_dist - target_distorted_point.x;
            let fy = y_dist - target_distorted_point.y;

            if fx.abs() < EPS && fy.abs() < EPS {
                break;
            }

            // Jacobian matrix
            let dradial_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;

            // ∂(radial·x + dx)/∂x
            let dfx_dx =
                radial + 2.0 * x * dradial_dr2 * x + 2.0 * self.p1 * y + 2.0 * self.p2 * (3.0 * x);

            // ∂(radial·x + dx)/∂y
            let dfx_dy = 2.0 * x * dradial_dr2 * y + 2.0 * self.p1 * x + 2.0 * self.p2 * y;

            // ∂(radial·y + dy)/∂x
            let dfy_dx = 2.0 * y * dradial_dr2 * x + 2.0 * self.p1 * x + 2.0 * self.p2 * y;

            // ∂(radial·y + dy)/∂y
            let dfy_dy =
                radial + 2.0 * y * dradial_dr2 * y + 2.0 * self.p1 * (3.0 * y) + 2.0 * self.p2 * x;

            let jacobian = Matrix2::new(dfx_dx, dfx_dy, dfy_dx, dfy_dy);

            // Solve: J·Δp = -f
            let det = jacobian[(0, 0)] * jacobian[(1, 1)] - jacobian[(0, 1)] * jacobian[(1, 0)];

            if det.abs() < PRECISION {
                return Err(CameraModelError::NumericalError(
                    "Singular Jacobian in RadTan unprojection".to_string(),
                ));
            }

            let inv_det = 1.0 / det;
            let delta_x = inv_det * (jacobian[(1, 1)] * (-fx) - jacobian[(0, 1)] * (-fy));
            let delta_y = inv_det * (-jacobian[(1, 0)] * (-fx) + jacobian[(0, 0)] * (-fy));

            point.x += delta_x;
            point.y += delta_y;

            if iteration == MAX_ITERATIONS - 1 {
                return Err(CameraModelError::NumericalError(
                    "RadTan unprojection did not converge".to_string(),
                ));
            }
        }

        // Normalize to unit ray
        let r2 = point.x * point.x + point.y * point.y;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(
            point.x * norm_inv,
            point.y * norm_inv,
            norm_inv,
        ))
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - z ≥ PRECISION (point in front of camera)
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        self.check_projection_condition(p_cam.z)
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// # Chain Rule Application
    ///
    /// Most complex Jacobian due to combined radial + tangential distortion.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let dradial_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;

        // Derivatives of distorted coordinates w.r.t. normalized coordinates
        // x_dist = radial·x' + dx where dx = 2p₁x'y' + p₂(r² + 2x'²)
        // ∂x_dist/∂x' = radial + x'·∂radial/∂r²·∂r²/∂x' + ∂dx/∂x'
        //             = radial + x'·dradial_dr2·2x' + (2p₁y' + p₂·(2x' + 4x'))
        //             = radial + 2x'²·dradial_dr2 + 2p₁y' + 6p₂x'
        let dx_dist_dx_prime = radial
            + 2.0 * x_prime * x_prime * dradial_dr2
            + 2.0 * self.p1 * y_prime
            + 6.0 * self.p2 * x_prime;

        // ∂x_dist/∂y' = x'·∂radial/∂r²·∂r²/∂y' + ∂dx/∂y'
        //             = x'·dradial_dr2·2y' + (2p₁x' + 2p₂y')
        let dx_dist_dy_prime = 2.0 * x_prime * y_prime * dradial_dr2
            + 2.0 * self.p1 * x_prime
            + 2.0 * self.p2 * y_prime;

        // y_dist = radial·y' + dy where dy = p₁(r² + 2y'²) + 2p₂x'y'
        // ∂y_dist/∂x' = y'·∂radial/∂r²·∂r²/∂x' + ∂dy/∂x'
        //             = y'·dradial_dr2·2x' + (p₁·2x' + 2p₂y')
        let dy_dist_dx_prime = 2.0 * y_prime * x_prime * dradial_dr2
            + 2.0 * self.p1 * x_prime
            + 2.0 * self.p2 * y_prime;

        // ∂y_dist/∂y' = radial + y'·∂radial/∂r²·∂r²/∂y' + ∂dy/∂y'
        //             = radial + y'·dradial_dr2·2y' + (p₁·(2y' + 4y') + 2p₂x')
        //             = radial + 2y'²·dradial_dr2 + 6p₁y' + 2p₂x'
        let dy_dist_dy_prime = radial
            + 2.0 * y_prime * y_prime * dradial_dr2
            + 6.0 * self.p1 * y_prime
            + 2.0 * self.p2 * x_prime;

        // Derivatives of normalized coordinates w.r.t. camera coordinates
        // x' = x/z => ∂x'/∂x = 1/z, ∂x'/∂y = 0, ∂x'/∂z = -x/z²
        // y' = y/z => ∂y'/∂x = 0, ∂y'/∂y = 1/z, ∂y'/∂z = -y/z²

        // Chain rule: ∂(u,v)/∂(x,y,z) = ∂(u,v)/∂(x_dist,y_dist) · ∂(x_dist,y_dist)/∂(x',y') · ∂(x',y')/∂(x,y,z)

        let du_dx = self.fx * (dx_dist_dx_prime * inv_z);
        let du_dy = self.fx * (dx_dist_dy_prime * inv_z);
        let du_dz = self.fx
            * (dx_dist_dx_prime * (-x_prime * inv_z) + dx_dist_dy_prime * (-y_prime * inv_z));

        let dv_dx = self.fy * (dy_dist_dx_prime * inv_z);
        let dv_dy = self.fy * (dy_dist_dy_prime * inv_z);
        let dv_dz = self.fy
            * (dy_dist_dx_prime * (-x_prime * inv_z) + dy_dist_dy_prime * (-y_prime * inv_z));

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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×9).
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let xy = x_prime * y_prime;
        let dx = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x_prime * x_prime);
        let dy = self.p1 * (r2 + 2.0 * y_prime * y_prime) + 2.0 * self.p2 * xy;

        let x_distorted = radial * x_prime + dx;
        let y_distorted = radial * y_prime + dy;

        // ∂u/∂fx = x_distorted, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y_distorted, ∂v/∂cx = 0, ∂v/∂cy = 1

        // Distortion parameter derivatives
        let du_dk1 = self.fx * x_prime * r2;
        let du_dk2 = self.fx * x_prime * r4;
        let du_dp1 = self.fx * 2.0 * xy;
        let du_dp2 = self.fx * (r2 + 2.0 * x_prime * x_prime);
        let du_dk3 = self.fx * x_prime * r6;

        let dv_dk1 = self.fy * y_prime * r2;
        let dv_dk2 = self.fy * y_prime * r4;
        let dv_dp1 = self.fy * (r2 + 2.0 * y_prime * y_prime);
        let dv_dp2 = self.fy * 2.0 * xy;
        let dv_dk3 = self.fy * y_prime * r6;

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

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3,
        ])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 9,
            "RadTanCamera requires at least 9 parameters"
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            k1: params[4],
            k2: params[5],
            p1: params[6],
            p2: params[7],
            k3: params[8],
        }
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - k₁, k₂, p₁, p₂, k₃ must be finite
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        if !self.k1.is_finite()
            || !self.k2.is_finite()
            || !self.p1.is_finite()
            || !self.p2.is_finite()
            || !self.k3.is_finite()
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
        vec![self.k1, self.k2, self.p1, self.p2, self.k3]
    }

    fn get_model_name(&self) -> &'static str {
        "rad_tan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_radtan_camera_creation() -> TestResult {
        let camera = RadTanCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002, 0.001)?;
        assert_eq!(camera.fx, 300.0);
        assert_eq!(camera.k1, 0.1);
        assert_eq!(camera.p1, 0.001);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = RadTanCamera::new(300.0, 300.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert!((uv.x - 320.0).abs() < 1e-10);
        assert!((uv.y - 240.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = RadTanCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002, 0.001)?;
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
        let camera = RadTanCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002, 0.001)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = 1e-7;

        for i in 0..9 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = RadTanCamera::from_params(params_plus.as_slice());
            let cam_minus = RadTanCamera::from_params(params_minus.as_slice());

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
