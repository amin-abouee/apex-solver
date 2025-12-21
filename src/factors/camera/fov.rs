//! Field-of-View (FOV) camera model.

use super::{CameraModel, skew_symmetric};
use crate::manifold::LieGroup;
use crate::manifold::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

const EPS_SQRT: f64 = 1e-7;

/// Field-of-View (FOV) camera model.
///
/// This model is suitable for fisheye cameras with radial distortion.
///
/// # Parameters
///
/// - `fx`, `fy`: Focal lengths in pixels
/// - `cx`, `cy`: Principal point coordinates in pixels
/// - `w`: Distortion parameter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovCamera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Distortion parameter
    pub w: f64,
}

impl FovCamera {
    /// Create a new FOV camera.
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64, w: f64) -> Self {
        Self { fx, fy, cx, cy, w }
    }
}

impl CameraModel for FovCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        if z < EPS_SQRT {
            return None;
        }

        let r2 = x * x + y * y;
        let r = r2.sqrt();

        let tan_w_half = (self.w / 2.0).tan();
        let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

        let rd = if r2 < EPS_SQRT {
            2.0 * tan_w_half / self.w
        } else {
            atan_wrd / (r * self.w)
        };

        let mx = x * rd;
        let my = y * rd;

        Some(Vector2::new(self.fx * mx + self.cx, self.fy * my + self.cy))
    }

    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        p_cam[2] >= EPS_SQRT
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r2 = x * x + y * y;
        let r = r2.sqrt();

        let tan_w_half = (self.w / 2.0).tan();
        let two_tan_w_half = 2.0 * tan_w_half;

        let rd = if r2 >= EPS_SQRT {
            let atan_wrd = (two_tan_w_half * r).atan2(z);
            atan_wrd / (r * self.w)
        } else {
            two_tan_w_half / self.w
        };

        let (d_rd_d_x, d_rd_d_y, d_rd_d_z) = if r2 >= EPS_SQRT {
            let denom_atan = z * z + 4.0 * tan_w_half * tan_w_half * r2;
            let term1 = two_tan_w_half * z / (self.w * denom_atan);

            let d_rd_d_x = x / r2 * (term1 - rd);
            let d_rd_d_y = y / r2 * (term1 - rd);
            let d_rd_d_z = -two_tan_w_half / (self.w * denom_atan);

            (d_rd_d_x, d_rd_d_y, d_rd_d_z)
        } else {
            (0.0, 0.0, 0.0)
        };

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        jac[(0, 0)] = self.fx * (d_rd_d_x * x + rd);
        jac[(0, 1)] = self.fx * d_rd_d_y * x;
        jac[(0, 2)] = self.fx * d_rd_d_z * x;

        jac[(1, 0)] = self.fy * d_rd_d_x * y;
        jac[(1, 1)] = self.fy * (d_rd_d_y * y + rd);
        jac[(1, 2)] = self.fy * d_rd_d_z * y;

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

        let r2 = x * x + y * y;
        let r = r2.sqrt();

        let tan_w_half = (self.w / 2.0).tan();
        let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

        let rd = if r2 >= EPS_SQRT {
            atan_wrd / (r * self.w)
        } else {
            2.0 * tan_w_half / self.w
        };

        let d_rd_d_w = if r2 >= EPS_SQRT {
            let tmp1 = 1.0 / (self.w / 2.0).cos();
            let d_tanwhalf_d_w = 0.5 * tmp1 * tmp1;
            let tmp = z * z + 4.0 * tan_w_half * tan_w_half * r2;
            let d_atan_wrd_d_w = 2.0 * r * d_tanwhalf_d_w * z / tmp;
            (d_atan_wrd_d_w * self.w - atan_wrd) / (r * self.w * self.w)
        } else {
            let tmp1 = 1.0 / (self.w / 2.0).cos();
            let d_tanwhalf_d_w = 0.5 * tmp1 * tmp1;
            2.0 * (d_tanwhalf_d_w * self.w - tan_w_half) / (self.w * self.w)
        };

        let mx = x * rd;
        let my = y * rd;

        let mut jac = SMatrix::<f64, 2, 5>::zeros();

        // ∂u/∂fx = mx
        jac[(0, 0)] = mx;

        // ∂v/∂fy = my
        jac[(1, 1)] = my;

        // ∂u/∂cx = 1
        jac[(0, 2)] = 1.0;

        // ∂v/∂cy = 1
        jac[(1, 3)] = 1.0;

        // ∂u/∂w = fx * x * d_rd_d_w
        jac[(0, 4)] = self.fx * x * d_rd_d_w;
        jac[(1, 4)] = self.fy * y * d_rd_d_w;

        jac
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.fx, self.fy, self.cx, self.cy, self.w])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 5,
            "FovCamera requires at least 5 parameters"
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            w: params[4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_fov_camera_creation() {
        let camera = FovCamera::new(400.0, 400.0, 376.0, 240.0, 1.0);
        assert_eq!(camera.fx, 400.0);
        assert_eq!(camera.w, 1.0);
    }

    #[test]
    fn test_projection_at_optical_axis() {
        let camera = FovCamera::new(400.0, 400.0, 376.0, 240.0, 1.0);
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).unwrap();
        assert_approx_eq(uv.x, 376.0, 1e-10);
        assert_approx_eq(uv.y, 240.0, 1e-10);
    }

    #[test]
    fn test_jacobian_point_numerical() {
        let camera = FovCamera::new(400.0, 400.0, 376.0, 240.0, 0.9);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);
        let eps = 1e-7;

        for i in 0..3 {
            let mut p_plus = p_cam;
            let mut p_minus = p_cam;
            p_plus[i] += eps;
            p_minus[i] -= eps;

            let uv_plus = camera.project(&p_plus).unwrap();
            let uv_minus = camera.project(&p_minus).unwrap();
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
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() {
        let camera = FovCamera::new(400.0, 400.0, 376.0, 240.0, 0.9);
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params = camera.intrinsics_vec();
        let eps = 1e-7;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = FovCamera::from_params(params_plus.as_slice());
            let cam_minus = FovCamera::from_params(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam).unwrap();
            let uv_minus = cam_minus.project(&p_cam).unwrap();
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
    }
}
