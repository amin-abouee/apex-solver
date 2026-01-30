//! Double Sphere camera model.

use super::{CameraModel, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const PRECISION: f64 = 1e-3;

/// Double Sphere camera model.
///
/// This model is suitable for wide-angle and fisheye cameras.
///
/// # Parameters
///
/// - `fx`, `fy`: Focal lengths in pixels
/// - `cx`, `cy`: Principal point coordinates in pixels
/// - `xi`: Sphere offset parameter
/// - `alpha`: Sphere distortion parameter
///
/// # Projection Model
///
/// See [`compute_residual_double_sphere`](crate::factors::double_sphere_factor) for details.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleSphereCamera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Sphere distortion parameter
    pub alpha: f64,
    /// Sphere offset parameter
    pub xi: f64,
}

impl DoubleSphereCamera {
    /// Create a new Double Sphere camera.
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64, xi: f64, alpha: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            alpha,
            xi,
        }
    }
}

impl CameraModel for DoubleSphereCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r_squared = x * x + y * y;
        let d1 = (r_squared + z * z).sqrt();
        let gamma = self.xi * d1 + z;
        let d2 = (r_squared + gamma * gamma).sqrt();
        let m_alpha = 1.0 - self.alpha;
        let denom = self.alpha * d2 + m_alpha * gamma;

        // Check projection validity
        let w1 = if self.alpha <= 0.5 {
            self.alpha / m_alpha
        } else {
            m_alpha / self.alpha
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi + self.xi * self.xi + 1.0).sqrt();
        let check_projection = z > -w2 * d1;

        if denom.abs() < PRECISION || !check_projection {
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
        let d1 = (r_squared + z * z).sqrt();
        let gamma = self.xi * d1 + z;
        let d2 = (r_squared + gamma * gamma).sqrt();
        let m_alpha = 1.0 - self.alpha;
        let denom = self.alpha * d2 + m_alpha * gamma;

        // Check projection validity
        let w1 = if self.alpha <= 0.5 {
            self.alpha / m_alpha
        } else {
            m_alpha / self.alpha
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi + self.xi * self.xi + 1.0).sqrt();
        let check_projection = z > -w2 * d1;

        denom.abs() >= PRECISION && check_projection
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let k = self.xi * d1 + z;
        let d2 = (r2 + k * k).sqrt();
        let norm = self.alpha * d2 + (1.0 - self.alpha) * k;
        let norm2 = norm * norm;

        // Intermediate terms from granite
        let d_norm_d_r2 = (self.xi * (1.0 - self.alpha) / d1
            + self.alpha * (self.xi * k / d1 + 1.0) / d2)
            / norm2;
        let tt2 = self.xi * z / d1 + 1.0;
        let tmp2 = ((1.0 - self.alpha) * tt2 + self.alpha * k * tt2 / d2) / norm2;

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        // ∂u/∂x, ∂u/∂y, ∂u/∂z
        jac[(0, 0)] = self.fx * (1.0 / norm - x * x * d_norm_d_r2);
        jac[(0, 1)] = -self.fx * x * y * d_norm_d_r2;
        jac[(0, 2)] = -self.fx * x * tmp2;

        // ∂v/∂x, ∂v/∂y, ∂v/∂z
        jac[(1, 0)] = -self.fy * x * y * d_norm_d_r2;
        jac[(1, 1)] = self.fy * (1.0 / norm - y * y * d_norm_d_r2);
        jac[(1, 2)] = -self.fy * y * tmp2;

        jac
    }

    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        // Transform point from world to camera frame
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        // Jacobian of projection w.r.t. point in camera frame
        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        // Jacobian of transformed point w.r.t. pose
        // p_cam = R^T * (p_world - t)
        // Using right perturbation on SE3: δpose = [δt; δω]
        // Combined: ∂p_cam/∂[δt; δω] = [-R^T | [p_cam]×]

        let r_transpose = pose_inv.rotation_so3().rotation_matrix();
        let p_cam_skew = skew_symmetric(&p_cam);

        let d_pcam_d_pose = SMatrix::<f64, 3, 6>::from_fn(|r, c| {
            if c < 3 {
                // Translation part: -R^T
                -r_transpose[(r, c)]
            } else {
                // Rotation part: [p_cam]×
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
        let d1 = (r_squared + z * z).sqrt();
        let gamma = self.xi * d1 + z;
        let d2 = (r_squared + gamma * gamma).sqrt();
        let m_alpha = 1.0 - self.alpha;
        let denom = self.alpha * d2 + m_alpha * gamma;

        // Used for derivatives
        let u_cx = self.fx * x / denom;
        let v_cy = self.fy * y / denom;

        let mut jac = SMatrix::<f64, 2, 6>::zeros();

        // ∂residual / ∂fx
        jac[(0, 0)] = x / denom;
        jac[(1, 0)] = 0.0;

        // ∂residual / ∂fy
        jac[(0, 1)] = 0.0;
        jac[(1, 1)] = y / denom;

        // ∂residual / ∂cx
        jac[(0, 2)] = 1.0;
        jac[(1, 2)] = 0.0;

        // ∂residual / ∂cy
        jac[(0, 3)] = 0.0;
        jac[(1, 3)] = 1.0;

        // ∂residual / ∂alpha
        jac[(0, 4)] = u_cx * (gamma - d2) / denom;
        jac[(1, 4)] = v_cy * (gamma - d2) / denom;

        // ∂residual / ∂xi
        let coeff = (self.alpha * d1 * gamma) / d2 + (m_alpha * d1);
        jac[(0, 5)] = -u_cx * coeff / denom;
        jac[(1, 5)] = -v_cy * coeff / denom;

        jac
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.xi,
        ])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 6,
            "DoubleSphereCamera requires at least 6 parameters, got {}",
            params.len()
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            alpha: params[4],
            xi: params[5],
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
    fn test_double_sphere_camera_creation() {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.5);
        assert_eq!(camera.fx, 300.0);
        assert_eq!(camera.xi, 0.1);
        assert_eq!(camera.alpha, 0.5);
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.5);
        let p_cam = Vector3::new(0.0, 0.0, 1.0);

        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        // Point on optical axis should project to principal point
        assert_approx_eq(uv.x, 320.0, 1e-10);
        assert_approx_eq(uv.y, 240.0, 1e-10);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.5);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);

        // Numerical differentiation
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
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.1, 0.5);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);

        // Numerical differentiation
        let eps = 1e-7;
        let params = vec![
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.alpha,
            camera.xi,
        ];

        for i in 0..6 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = DoubleSphereCamera::from_params(&params_plus);
            let cam_minus = DoubleSphereCamera::from_params(&params_minus);

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
}
