//! Kannala-Brandt fisheye camera model.

use super::{CameraModel, skew_symmetric};
use crate::manifold::LieGroup;
use crate::manifold::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Kannala-Brandt fisheye camera model.
///
/// This model uses a polynomial approximation for the projection angle.
///
/// # Parameters
///
/// - `fx`, `fy`: Focal lengths in pixels
/// - `cx`, `cy`: Principal point coordinates in pixels
/// - `k1`, `k2`, `k3`, `k4`: Distortion coefficients
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KannalaBrandtCamera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Distortion coefficient k1
    pub k1: f64,
    /// Distortion coefficient k2
    pub k2: f64,
    /// Distortion coefficient k3
    pub k3: f64,
    /// Distortion coefficient k4
    pub k4: f64,
}

impl KannalaBrandtCamera {
    /// Create a new Kannala-Brandt camera.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            k4,
        }
    }
}

impl CameraModel for KannalaBrandtCamera {
    const INTRINSIC_DIM: usize = 8;
    type IntrinsicJacobian = SMatrix<f64, 2, 8>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        if z < f64::EPSILON {
            return None;
        }

        let r_squared = x * x + y * y;
        let r = r_squared.sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d =
            theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

        let (x_r, y_r) = if r < f64::EPSILON {
            (0.0, 0.0)
        } else {
            (x / r, y / r)
        };

        Some(Vector2::new(
            self.fx * theta_d * x_r + self.cx,
            self.fy * theta_d * y_r + self.cy,
        ))
    }

    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        p_cam[2] >= f64::EPSILON
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r_squared = x * x + y * y;
        let r = r_squared.sqrt();

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        if r < f64::EPSILON {
            // Degenerate case - use pinhole approximation
            jac[(0, 0)] = self.fx / z;
            jac[(0, 1)] = 0.0;
            jac[(0, 2)] = -self.fx * x / (z * z);

            jac[(1, 0)] = 0.0;
            jac[(1, 1)] = self.fy / z;
            jac[(1, 2)] = -self.fy * y / (z * z);
        } else {
            let theta = r.atan2(z);
            let theta2 = theta * theta;

            // Compute distorted radius
            let theta3 = theta2 * theta;
            let theta5 = theta3 * theta2;
            let theta7 = theta5 * theta2;
            let theta9 = theta7 * theta2;
            let r_theta =
                theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

            // Compute d_r_theta_d_theta = 1 + 3k1θ² + 5k2θ⁴ + 7k3θ⁶ + 9k4θ⁸
            let mut d_r_theta_d_theta = 9.0 * self.k4 * theta2;
            d_r_theta_d_theta += 7.0 * self.k3;
            d_r_theta_d_theta *= theta2;
            d_r_theta_d_theta += 5.0 * self.k2;
            d_r_theta_d_theta *= theta2;
            d_r_theta_d_theta += 3.0 * self.k1;
            d_r_theta_d_theta *= theta2;
            d_r_theta_d_theta += 1.0;

            // Derivatives of r and theta w.r.t. x, y, z
            let d_r_d_x = x / r;
            let d_r_d_y = y / r;

            let tmp = z * z + r_squared;
            let d_theta_d_x = d_r_d_x * z / tmp;
            let d_theta_d_y = d_r_d_y * z / tmp;
            let d_theta_d_z = -r / tmp;

            // Jacobian entries from granite-headers formula
            jac[(0, 0)] = self.fx
                * (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x - x * x * r_theta / r)
                / r_squared;
            jac[(1, 0)] =
                self.fy * y * (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta / r) / r_squared;

            jac[(0, 1)] =
                self.fx * x * (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta / r) / r_squared;
            jac[(1, 1)] = self.fy
                * (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y - y * y * r_theta / r)
                / r_squared;

            jac[(0, 2)] = self.fx * x * d_r_theta_d_theta * d_theta_d_z / r;
            jac[(1, 2)] = self.fy * y * d_r_theta_d_theta * d_theta_d_z / r;
        }

        jac
    }

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

    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r_squared = x * x + y * y;
        let r = r_squared.sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d =
            theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

        let (x_r, y_r) = if r < f64::EPSILON {
            (0.0, 0.0)
        } else {
            (x / r, y / r)
        };

        let mut jac = SMatrix::<f64, 2, 8>::zeros();

        // ∂u/∂fx = theta_d * x_r
        jac[(0, 0)] = theta_d * x_r;

        // ∂v/∂fy = theta_d * y_r
        jac[(1, 1)] = theta_d * y_r;

        // ∂u/∂cx = 1
        jac[(0, 2)] = 1.0;

        // ∂v/∂cy = 1
        jac[(1, 3)] = 1.0;

        // ∂u/∂k1 = fx * θ³ * x_r
        jac[(0, 4)] = self.fx * theta3 * x_r;
        jac[(1, 4)] = self.fy * theta3 * y_r;

        // ∂u/∂k2 = fx * θ⁵ * x_r
        jac[(0, 5)] = self.fx * theta5 * x_r;
        jac[(1, 5)] = self.fy * theta5 * y_r;

        // ∂u/∂k3 = fx * θ⁷ * x_r
        jac[(0, 6)] = self.fx * theta7 * x_r;
        jac[(1, 6)] = self.fy * theta7 * y_r;

        // ∂u/∂k4 = fx * θ⁹ * x_r
        jac[(0, 7)] = self.fx * theta9 * x_r;
        jac[(1, 7)] = self.fy * theta9 * y_r;

        jac
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
    fn test_kb_camera_creation() {
        let camera = KannalaBrandtCamera::new(460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04);
        assert_eq!(camera.fx, 460.0);
        assert_eq!(camera.k1, -0.01);
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = KannalaBrandtCamera::new(460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04);
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;
        assert_approx_eq(uv.x, 320.0, 1e-10);
        assert_approx_eq(uv.y, 240.0, 1e-10);
        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = KannalaBrandtCamera::new(460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04);
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
        let camera = KannalaBrandtCamera::new(460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04);
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
