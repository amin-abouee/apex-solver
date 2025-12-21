//! Radial-Tangential (RadTan) camera model.
//!
//! ## Mathematical Formulation
//!
//! The RadTan model (also known as Brown-Conrady or Plumb Bob model) is the most widely
//! used camera distortion model, implemented in OpenCV, MATLAB, and most SfM/SLAM systems.
//!
//! ### Projection Model
//!
//! Given a 3D point **p** = (x, y, z) in camera frame, the projection to pixel coordinates is:
//!
//! ```text
//! 1. Normalize: x' = x/z, y' = y/z
//!
//! 2. Compute radial distance squared: r² = x'² + y'²
//!
//! 3. Radial distortion factor:
//!    d = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
//!
//! 4. Distorted coordinates (radial + tangential):
//!    x_dist = d·x' + 2p₁·x'y' + p₂·(r² + 2x'²)
//!    y_dist = d·y' + 2p₂·x'y' + p₁·(r² + 2y'²)
//!
//! 5. Pixel coordinates:
//!    u = fx·x_dist + cx
//!    v = fy·y_dist + cy
//! ```
//!
//! ### Jacobians
//!
//! **J_point** (∂uv/∂xyz): 2×3 Jacobian w.r.t. 3D point in camera frame
//!
//! Computed via chain rule through normalized coordinates (x', y') and distorted coordinates.
//! All derivatives are analytical (no numerical approximation).
//!
//! **J_intrinsics** (∂uv/∂params): 2×9 Jacobian w.r.t. intrinsic parameters
//!
//! Parameter order: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
//!
//! Key derivatives:
//! ```text
//! ∂u/∂fx = x_dist,  ∂u/∂cx = 1
//! ∂u/∂k₁ = fx·x'·r²,  ∂u/∂k₂ = fx·x'·r⁴,  ∂u/∂k₃ = fx·x'·r⁶
//! ∂u/∂p₁ = fx·2x'y',  ∂u/∂p₂ = fx·(r² + 2x'²)
//!
//! (similarly for v with fy)
//! ```
//!
//! **J_pose** (∂uv/∂pose): 2×6 Jacobian w.r.t. SE3 camera pose
//!
//! Returns tuple `(J_projection, J_transform)` for chain rule application:
//! - Uses **right perturbation** on SE3: δT = exp(ξ^) ∘ T
//! - J_transform = [-R^T | [p_cam]×] for tangent space [translation | rotation]
//!
//! ### References
//!
//! - **OpenCV Camera Calibration**:
//!   <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
//!
//! - **Brown, D.C. (1966)**:
//!   "Decentering Distortion of Lenses"
//!   Photogrammetric Engineering, Vol. 32, No. 3, pp. 444-462
//!
//! - **Verification**:
//!   - Symbolically verified against SymPy automatic differentiation
//!   - Cross-validated against fisheye-calib-adapter reference implementation
//!   - Numerically tested with tolerance < 1e-5
//!
//! ### Usage
//!
//! This model is compatible with:
//! - OpenCV `cv::calibrateCamera()` output
//! - MATLAB Camera Calibrator Toolbox
//! - ROS camera_calibration package
//! - Most structure-from-motion pipelines (COLMAP, ORB-SLAM, etc.)
//!
//! For fisheye cameras with extreme field-of-view (>180°), consider using
//! `KannalaBrandtCamera` or `DoubleSphereCamera` instead.

use super::{CameraModel, skew_symmetric};
use crate::manifold::LieGroup;
use crate::manifold::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Radial-Tangential (RadTan) camera model.
///
/// This model accounts for radial and tangential lens distortion.
///
/// # Parameters
///
/// - `fx`, `fy`: Focal lengths in pixels
/// - `cx`, `cy`: Principal point coordinates in pixels
/// - `k1`, `k2`, `k3`: Radial distortion coefficients
/// - `p1`, `p2`: Tangential distortion coefficients
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadTanCamera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Radial distortion k1
    pub k1: f64,
    /// Radial distortion k2
    pub k2: f64,
    /// Tangential distortion p1
    pub p1: f64,
    /// Tangential distortion p2
    pub p2: f64,
    /// Radial distortion k3
    pub k3: f64,
}

impl RadTanCamera {
    /// Create a new RadTan camera.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        }
    }
}

impl CameraModel for RadTanCamera {
    const INTRINSIC_DIM: usize = 9;
    type IntrinsicJacobian = SMatrix<f64, 2, 9>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        if z < f64::EPSILON.sqrt() {
            return None;
        }

        let x_prime = x / z;
        let y_prime = y / z;

        let r2 = x_prime.powi(2) + y_prime.powi(2);
        let r4 = r2.powi(2);
        let r6 = r4 * r2;

        let d = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let x_distorted = d * x_prime
            + 2.0 * self.p1 * x_prime * y_prime
            + self.p2 * (r2 + 2.0 * x_prime.powi(2));
        let y_distorted = d * y_prime
            + 2.0 * self.p2 * x_prime * y_prime
            + self.p1 * (r2 + 2.0 * y_prime.powi(2));

        Some(Vector2::new(
            self.fx * x_distorted + self.cx,
            self.fy * y_distorted + self.cy,
        ))
    }

    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        p_cam[2] >= f64::EPSILON.sqrt()
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let x_prime = x / z;
        let y_prime = y / z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let d = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let d_d_r2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;

        let d_r2_dx_prime = 2.0 * x_prime;
        let d_r2_dy_prime = 2.0 * y_prime;

        let d_xdist_dx_prime = d
            + x_prime * d_d_r2 * d_r2_dx_prime
            + 2.0 * self.p1 * y_prime
            + self.p2 * (d_r2_dx_prime + 4.0 * x_prime);
        let d_xdist_dy_prime =
            x_prime * d_d_r2 * d_r2_dy_prime + 2.0 * self.p1 * x_prime + self.p2 * d_r2_dy_prime;

        let d_ydist_dx_prime =
            y_prime * d_d_r2 * d_r2_dx_prime + 2.0 * self.p2 * y_prime + self.p1 * d_r2_dx_prime;
        let d_ydist_dy_prime = d
            + y_prime * d_d_r2 * d_r2_dy_prime
            + 2.0 * self.p2 * x_prime
            + self.p1 * (d_r2_dy_prime + 4.0 * y_prime);

        let d_xprime_dx = 1.0 / z;
        let d_xprime_dz = -x / (z * z);
        let d_yprime_dy = 1.0 / z;
        let d_yprime_dz = -y / (z * z);

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        jac[(0, 0)] = self.fx * d_xdist_dx_prime * d_xprime_dx;
        jac[(0, 1)] = self.fx * d_xdist_dy_prime * d_yprime_dy;
        jac[(0, 2)] = self.fx * (d_xdist_dx_prime * d_xprime_dz + d_xdist_dy_prime * d_yprime_dz);

        jac[(1, 0)] = self.fy * d_ydist_dx_prime * d_xprime_dx;
        jac[(1, 1)] = self.fy * d_ydist_dy_prime * d_yprime_dy;
        jac[(1, 2)] = self.fy * (d_ydist_dx_prime * d_xprime_dz + d_ydist_dy_prime * d_yprime_dz);

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

        let x_prime = x / z;
        let y_prime = y / z;

        let r2 = x_prime.powi(2) + y_prime.powi(2);
        let r4 = r2.powi(2);
        let r6 = r4 * r2;

        let d = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let x_distorted = d * x_prime
            + 2.0 * self.p1 * x_prime * y_prime
            + self.p2 * (r2 + 2.0 * x_prime.powi(2));
        let y_distorted = d * y_prime
            + 2.0 * self.p2 * x_prime * y_prime
            + self.p1 * (r2 + 2.0 * y_prime.powi(2));

        let mut jac = SMatrix::<f64, 2, 9>::zeros();

        // ∂residual / ∂fx
        jac[(0, 0)] = x_distorted;

        // ∂residual / ∂fy
        jac[(1, 1)] = y_distorted;

        // ∂residual / ∂cx
        jac[(0, 2)] = 1.0;

        // ∂residual / ∂cy
        jac[(1, 3)] = 1.0;

        // ∂residual / ∂k1
        jac[(0, 4)] = self.fx * x_prime * r2;
        jac[(1, 4)] = self.fy * y_prime * r2;

        // ∂residual / ∂k2
        jac[(0, 5)] = self.fx * x_prime * r4;
        jac[(1, 5)] = self.fy * y_prime * r4;

        // ∂residual / ∂p1
        jac[(0, 6)] = self.fx * 2.0 * x_prime * y_prime;
        jac[(1, 6)] = self.fy * (r2 + 2.0 * y_prime.powi(2));

        // ∂residual / ∂p2
        jac[(0, 7)] = self.fx * (r2 + 2.0 * x_prime.powi(2));
        jac[(1, 7)] = self.fy * 2.0 * x_prime * y_prime;

        // ∂residual / ∂k3
        jac[(0, 8)] = self.fx * x_prime * r6;
        jac[(1, 8)] = self.fy * y_prime * r6;

        jac
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
    fn test_radtan_camera_creation() {
        let camera = RadTanCamera::new(
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        );
        assert_eq!(camera.fx, 461.629);
        assert_eq!(camera.k1, -0.28340811);
    }

    #[test]
    fn test_projection_at_optical_axis() {
        let camera = RadTanCamera::new(
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        );
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam).unwrap();
        assert_approx_eq(uv.x, 362.680, 1e-10);
        assert_approx_eq(uv.y, 246.049, 1e-10);
    }

    #[test]
    fn test_jacobian_point_numerical() {
        let camera = RadTanCamera::new(
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        );
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
        let camera = RadTanCamera::new(
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        );
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
