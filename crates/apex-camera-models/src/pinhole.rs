//! Pinhole Camera Model
//!
//! The simplest perspective camera model with no lens distortion.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! u = fx · (x/z) + cx
//! v = fy · (y/z) + cy
//! ```
//!
//! where:
//! - (fx, fy) are focal lengths in pixels
//! - (cx, cy) is the principal point in pixels
//! - z > 0 (point in front of camera)
//!
//! ## Unprojection (2D → 3D)
//!
//! For a 2D point (u, v) in image coordinates, the unprojected ray is:
//!
//! ```text
//! mx = (u - cx) / fx
//! my = (v - cy) / fy
//! ray = normalize([mx, my, 1])
//! ```
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy (4 parameters)
//! - **Distortion**: None
//!
//! # Use Cases
//!
//! - Standard narrow field-of-view cameras
//! - Initial calibration estimates
//! - Testing and validation
//!
//! # References
//!
//! - Hartley & Zisserman, "Multiple View Geometry in Computer Vision"

use super::{CameraModel, CameraModelError, Intrinsics, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Pinhole camera model with 4 intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl PinholeCamera {
    #[must_use]
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }
}

impl CameraModel for PinholeCamera {
    const INTRINSIC_DIM: usize = 4;
    type IntrinsicJacobian = SMatrix<f64, 2, 4>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// u = fx · (x/z) + cx
    /// v = fy · (y/z) + cy
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z)
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates if z > MIN_DEPTH
    /// - `None` - If point is behind or at camera (z ≤ MIN_DEPTH)
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        const MIN_DEPTH: f64 = 1e-6;
        if p_cam.z < MIN_DEPTH {
            return None;
        }
        let inv_z = 1.0 / p_cam.z;
        Some(Vector2::new(
            self.fx * p_cam.x * inv_z + self.cx,
            self.fy * p_cam.y * inv_z + self.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray in camera frame.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// mx = (u - cx) / fx
    /// my = (v - cy) / fy
    /// ray = normalize([mx, my, 1])
    /// ```
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates (u, v)
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - Never fails for pinhole model
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let mx = (point_2d.x - self.cx) / self.fx;
        let my = (point_2d.y - self.cy) / self.fy;

        let r2 = mx * mx + my * my;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(mx * norm_inv, my * norm_inv, norm_inv))
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// `true` if the point satisfies projection constraints.
    ///
    /// # Validity Conditions
    ///
    /// - z ≥ MIN_DEPTH (point in front of camera)
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        const MIN_DEPTH: f64 = 1e-6;
        p_cam.z >= MIN_DEPTH
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// # Mathematical Derivatives
    ///
    /// ```text
    /// u = fx · x/z + cx  =>  ∂u/∂x = fx/z, ∂u/∂y = 0, ∂u/∂z = -fx·x/z²
    /// v = fy · y/z + cy  =>  ∂v/∂x = 0, ∂v/∂y = fy/z, ∂v/∂z = -fy·y/z²
    /// ```
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_norm = p_cam.x * inv_z;
        let y_norm = p_cam.y * inv_z;

        // Jacobian ∂(u,v)/∂(x,y,z) where (x,y,z) is point in camera frame
        SMatrix::<f64, 2, 3>::new(
            self.fx * inv_z,
            0.0,
            -self.fx * x_norm * inv_z,
            0.0,
            self.fy * inv_z,
            -self.fy * y_norm * inv_z,
        )
    }

    /// Jacobian of projection w.r.t. camera pose (SE3).
    ///
    /// Returns both the projection Jacobian and the pose Jacobian for chain rule.
    ///
    /// # SE3 Perturbation
    ///
    /// Using right perturbation δpose = [δt; δω]:
    /// - Translation: ∂p_cam/∂δt = -R^T
    /// - Rotation: ∂p_cam/∂δω = [p_cam]× (skew-symmetric matrix)
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        // Jacobian of transformed point w.r.t. pose
        // p_cam = R^T * (p_world - t)
        // ∂p_cam/∂[δt; δω] = [-R^T | [p_cam]×]

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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×4).
    ///
    /// # Mathematical Derivatives
    ///
    /// ```text
    /// u = fx · x/z + cx  =>  ∂u/∂fx = x/z, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
    /// v = fy · y/z + cy  =>  ∂v/∂fx = 0, ∂v/∂fy = y/z, ∂v/∂cx = 0, ∂v/∂cy = 1
    /// ```
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_norm = p_cam.x * inv_z;
        let y_norm = p_cam.y * inv_z;

        // Jacobian ∂(u,v)/∂(fx,fy,cx,cy)
        SMatrix::<f64, 2, 4>::new(x_norm, 0.0, 1.0, 0.0, 0.0, y_norm, 0.0, 1.0)
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.fx, self.fy, self.cx, self.cy])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 4,
            "PinholeCamera requires at least 4 parameters, got {}",
            params.len()
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
        }
    }

    /// Validates camera parameters.
    ///
    /// # Returns
    ///
    /// - `Ok(())` - All parameters are valid
    /// - `Err(CameraModelError)` - Invalid parameter detected
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite (not NaN or infinity)
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }
        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
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
        vec![]
    }

    fn get_model_name(&self) -> &'static str {
        "pinhole"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn assert_approx_eq(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "Values {} and {} differ by more than {}",
            a,
            b,
            eps
        );
    }

    #[test]
    fn test_pinhole_camera_creation() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(camera.fx, 500.0);
        assert_eq!(camera.fy, 500.0);
        assert_eq!(camera.cx, 320.0);
        assert_eq!(camera.cy, 240.0);
    }

    #[test]
    fn test_pinhole_from_params() {
        let params = vec![600.0, 600.0, 320.0, 240.0];
        let camera = PinholeCamera::from_params(&params);
        assert_eq!(camera.fx, 600.0);
        assert_eq!(camera.intrinsics_vec(), DVector::from_vec(params));
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.0, 0.0, 1.0);

        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert_approx_eq(uv.x, 320.0, 1e-10);
        assert_approx_eq(uv.y, 240.0, 1e-10);

        Ok(())
    }

    #[test]
    fn test_projection_off_axis() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        assert_approx_eq(uv.x, 370.0, 1e-10);
        assert_approx_eq(uv.y, 340.0, 1e-10);

        Ok(())
    }

    #[test]
    fn test_projection_behind_camera() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.0, 0.0, -1.0);

        let result = camera.project(&p_cam);
        assert!(result.is_none());
        assert!(!camera.is_valid_point(&p_cam));
    }

    #[test]
    fn test_jacobian_point_dimensions() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac = camera.jacobian_point(&p_cam);

        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 3);

        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_dimensions() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac = camera.jacobian_intrinsics(&p_cam);

        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 4);
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);

        let eps = 1e-7;
        for i in 0..3 {
            let mut p_plus = p_cam;
            let mut p_minus = p_cam;
            p_plus[i] += eps;
            p_minus[i] -= eps;

            let uv_plus = camera.project(&p_plus).ok_or("Projection+ failed")?;
            let uv_minus = camera.project(&p_minus).ok_or("Projection- failed")?;

            let numerical_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let analytical = jac_analytical[(r, i)];
                let numerical = numerical_jac[r];
                let rel_error = (analytical - numerical).abs() / (1.0 + numerical.abs());
                assert!(
                    rel_error < 1e-5,
                    "Jacobian mismatch at ({}, {}): analytical={}, numerical={}, rel_error={}",
                    r,
                    i,
                    analytical,
                    numerical,
                    rel_error
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);

        let eps = 1e-7;
        let params = vec![camera.fx, camera.fy, camera.cx, camera.cy];

        for i in 0..4 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = PinholeCamera::from_params(&params_plus);
            let cam_minus = PinholeCamera::from_params(&params_minus);

            let uv_plus = cam_plus.project(&p_cam).ok_or("Projection+ failed")?;
            let uv_minus = cam_minus.project(&p_cam).ok_or("Projection- failed")?;

            let numerical_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let analytical = jac_analytical[(r, i)];
                let numerical = numerical_jac[r];
                let rel_error = (analytical - numerical).abs() / (1.0 + numerical.abs());
                assert!(
                    rel_error < 1e-5,
                    "Intrinsics Jacobian mismatch at ({}, {}): analytical={}, numerical={}, rel_error={}",
                    r,
                    i,
                    analytical,
                    numerical,
                    rel_error
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_jacobian_pose() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let pose = SE3::identity();
        let p_world = Vector3::new(0.1, 0.2, 1.0);

        let (d_uv_d_pcam, d_pcam_d_pose) = camera.jacobian_pose(&p_world, &pose);

        assert_eq!(d_uv_d_pcam.nrows(), 2);
        assert_eq!(d_uv_d_pcam.ncols(), 3);
        assert_eq!(d_pcam_d_pose.nrows(), 3);
        assert_eq!(d_pcam_d_pose.ncols(), 6);

        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(&p_world, None, None);
        assert_approx_eq((p_cam - p_world).norm(), 0.0, 1e-10);

        Ok(())
    }
}
