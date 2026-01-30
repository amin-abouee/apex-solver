//! Extended Unified Camera Model (EUCM).

use super::{CameraModel, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-3;

/// Extended Unified Camera Model (EUCM).
///
/// This model is suitable for fisheye and wide-angle cameras.
///
/// # Parameters
///
/// - `fx`, `fy`: Focal lengths in pixels
/// - `cx`, `cy`: Principal point coordinates in pixels
/// - `alpha`: EUCM projection parameter (0 <= alpha <= 1)
/// - `beta`: EUCM distortion parameter (beta > 0)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EucmCamera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Projection parameter
    pub alpha: f64,
    /// Distortion parameter
    pub beta: f64,
}

impl EucmCamera {
    /// Create a new EUCM camera.
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64, alpha: f64, beta: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            alpha,
            beta,
        }
    }
}

impl CameraModel for EucmCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r_squared = x * x + y * y;
        let d = (self.beta * r_squared + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        // Check projection validity
        if self.alpha > 0.5 {
            let c = (self.alpha - 1.0) / (2.0 * self.alpha - 1.0);
            if z < denom * c {
                return None;
            }
        }

        if denom < PRECISION {
            return None;
        }

        Some(Vector2::new(
            self.fx * x / denom + self.cx,
            self.fy * y / denom + self.cy,
        ))
    }

    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r_squared = x * x + y * y;
        let d = (self.beta * r_squared + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        if self.alpha > 0.5 {
            let c = (self.alpha - 1.0) / (2.0 * self.alpha - 1.0);
            if z < denom * c {
                return false;
            }
        }

        denom >= PRECISION
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let rho = (self.beta * r2 + z * z).sqrt();
        let norm = self.alpha * rho + (1.0 - self.alpha) * z;
        let denom = norm * norm * rho;

        let mid = -(self.alpha * self.beta * x * y);
        let add = norm * rho;
        let addz = self.alpha * z + (1.0 - self.alpha) * rho;

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        // ∂u/∂x, ∂u/∂y, ∂u/∂z
        jac[(0, 0)] = self.fx * (add - x * x * self.alpha * self.beta) / denom;
        jac[(0, 1)] = self.fx * mid / denom;
        jac[(0, 2)] = -self.fx * x * addz / denom;

        // ∂v/∂x, ∂v/∂y, ∂v/∂z
        jac[(1, 0)] = self.fy * mid / denom;
        jac[(1, 1)] = self.fy * (add - y * y * self.alpha * self.beta) / denom;
        jac[(1, 2)] = -self.fy * y * addz / denom;

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
        let d = (self.beta * r_squared + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        // u = fx * x / denom + cx
        // v = fy * y / denom + cy

        let x_norm = x / denom;
        let y_norm = y / denom;

        let u_cx = self.fx * x_norm;
        let v_cy = self.fy * y_norm;

        let mut jac = SMatrix::<f64, 2, 6>::zeros();

        // ∂u/∂fx = x / denom
        jac[(0, 0)] = x_norm;

        // ∂v/∂fy = y / denom
        jac[(1, 1)] = y_norm;

        // ∂u/∂cx = 1
        jac[(0, 2)] = 1.0;

        // ∂v/∂cy = 1
        jac[(1, 3)] = 1.0;

        // ∂denom/∂alpha = d - z
        let d_denom_d_alpha = d - z;

        // ∂u/∂alpha = -fx * x / denom^2 * (d - z) = -u_cx * (d - z) / denom
        jac[(0, 4)] = -u_cx * d_denom_d_alpha / denom;
        jac[(1, 4)] = -v_cy * d_denom_d_alpha / denom;

        // ∂denom/∂beta = alpha * 0.5 / d * r_squared = alpha * r_squared / (2d)
        // ∂u/∂beta = -fx * x / denom^2 * (alpha * r_squared / 2d)
        //          = -u_cx * (alpha * r_squared) / (2 * d * denom)
        if d > PRECISION {
            let term = (self.alpha * r_squared) / (2.0 * d * denom);
            jac[(0, 5)] = -u_cx * term;
            jac[(1, 5)] = -v_cy * term;
        }

        jac
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
    fn test_eucm_camera_creation() {
        let camera = EucmCamera::new(1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5);
        assert_eq!(camera.fx, 1313.83);
        assert_eq!(camera.beta, 0.5);
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = EucmCamera::new(1000.0, 1000.0, 500.0, 500.0, 0.5, 1.0);
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;
        assert_approx_eq(uv.x, 500.0, 1e-10);
        assert_approx_eq(uv.y, 500.0, 1e-10);
        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = EucmCamera::new(1000.0, 1000.0, 500.0, 500.0, 0.6, 0.8);
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
        let camera = EucmCamera::new(1000.0, 1000.0, 500.0, 500.0, 0.6, 0.8);
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
