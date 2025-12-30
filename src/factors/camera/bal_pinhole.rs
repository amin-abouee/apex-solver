//! BAL (Bundle Adjustment in the Large) pinhole camera model.
//!
//! This module implements a pinhole camera model that follows the BAL dataset convention
//! where cameras look down the -Z axis (negative Z in front of camera).

use super::{CameraModel, skew_symmetric};
use crate::manifold::LieGroup;
use crate::manifold::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// BAL Pinhole camera model with radial distortion.
///
/// This camera model follows the Bundle Adjustment in the Large (BAL) convention:
/// - Camera looks down -Z axis (negative Z values are in front of camera)
/// - This is also known as OpenGL convention
/// - Includes radial distortion with two coefficients (k1, k2)
///
/// # Parameters
///
/// - `fx`, `fy`: Focal lengths in pixels
/// - `cx`, `cy`: Principal point coordinates in pixels
/// - `k1`, `k2`: Radial distortion coefficients
///
/// # Projection Model
///
/// For a 3D point `p_cam = (x, y, z)` in camera frame where z < 0:
/// ```text
/// x_n = x / (-z)
/// y_n = y / (-z)
/// r² = x_n² + y_n²
/// distortion = 1 + k1*r² + k2*r⁴
/// x_d = x_n * distortion
/// y_d = y_n * distortion
/// u = fx * x_d + cx
/// v = fy * y_d + cy
/// ```
///
/// Note the negation of z in the denominator compared to standard pinhole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BALPinholeCamera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// First radial distortion coefficient
    pub k1: f64,
    /// Second radial distortion coefficient
    pub k2: f64,
}

impl BALPinholeCamera {
    /// Create a new BAL pinhole camera with distortion.
    #[must_use]
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
        }
    }

    /// Create a BAL pinhole camera without distortion (k1=0, k2=0).
    #[must_use]
    pub const fn new_no_distortion(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
        }
    }
}

impl CameraModel for BALPinholeCamera {
    const INTRINSIC_DIM: usize = 6; // fx, fy, cx, cy, k1, k2
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        const MIN_DEPTH: f64 = 1e-6;
        // BAL convention: negative Z is in front
        if p_cam.z > -MIN_DEPTH {
            return None;
        }
        let inv_neg_z = -1.0 / p_cam.z; // Note: negation for BAL convention

        // Normalized coordinates
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        // Radial distortion
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + self.k1 * r2 + self.k2 * r4;

        // Apply distortion
        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        Some(Vector2::new(
            self.fx * x_d + self.cx,
            self.fy * y_d + self.cy,
        ))
    }

    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        const MIN_DEPTH: f64 = 1e-6;
        // BAL convention: negative Z is in front
        p_cam.z < -MIN_DEPTH
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        // Radial distortion
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + self.k1 * r2 + self.k2 * r4;

        // Derivative of distortion w.r.t. r²
        let d_dist_dr2 = self.k1 + 2.0 * self.k2 * r2;

        // Jacobian of normalized coordinates w.r.t. camera point
        // x_n = x / (-z), y_n = y / (-z)
        // ∂x_n/∂x = 1/(-z) = inv_neg_z
        // ∂x_n/∂z = x * ∂(1/(-z))/∂z = x * (1/z²)
        //
        // With inv_neg_z = -1/z, we have:
        //   x_n = x * inv_neg_z, so x = x_n / inv_neg_z
        //   ∂x_n/∂z = x / z² = (x_n / inv_neg_z) / z²
        //
        // Since z² = 1/inv_neg_z² (because inv_neg_z = -1/z means z = -1/inv_neg_z):
        //   z² = (-1/inv_neg_z)² = 1/inv_neg_z²
        //   ∂x_n/∂z = (x_n / inv_neg_z) * inv_neg_z² = x_n * inv_neg_z
        //
        // Verified numerically: with z=-2, x=1: inv_neg_z=0.5, x_n=0.5
        //   ∂x_n/∂z = 1/4 = 0.25 = x_n * inv_neg_z = 0.5 * 0.5 ✓
        let dxn_dz = x_n * inv_neg_z;
        let dyn_dz = y_n * inv_neg_z;

        // Jacobian of distorted point w.r.t. normalized point
        // x_d = x_n * (1 + k1*r² + k2*r⁴)
        // ∂x_d/∂x_n = distortion + x_n * d_dist_dr2 * ∂r²/∂x_n
        // ∂r²/∂x_n = 2*x_n, ∂r²/∂y_n = 2*y_n
        let dx_d_dxn = distortion + x_n * d_dist_dr2 * 2.0 * x_n;
        let dx_d_dyn = x_n * d_dist_dr2 * 2.0 * y_n;
        let dy_d_dxn = y_n * d_dist_dr2 * 2.0 * x_n;
        let dy_d_dyn = distortion + y_n * d_dist_dr2 * 2.0 * y_n;

        // Chain rule: ∂(u,v)/∂(x,y,z) = ∂(u,v)/∂(x_d,y_d) * ∂(x_d,y_d)/∂(x_n,y_n) * ∂(x_n,y_n)/∂(x,y,z)
        let du_dx = self.fx * (dx_d_dxn * inv_neg_z);
        let du_dy = self.fx * (dx_d_dyn * inv_neg_z);
        let du_dz = self.fx * (dx_d_dxn * dxn_dz + dx_d_dyn * dyn_dz);

        let dv_dx = self.fy * (dy_d_dxn * inv_neg_z);
        let dv_dy = self.fy * (dy_d_dyn * inv_neg_z);
        let dv_dz = self.fy * (dy_d_dxn * dxn_dz + dy_d_dyn * dyn_dz);

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        // Transform point from world to camera frame
        // pose is camera-to-world (R, t), so we need pose^{-1} for world-to-camera
        // p_cam = R^T * (p_world - t) = R^T * p_world - R^T * t
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        // Jacobian of projection w.r.t. point in camera frame
        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        // Jacobian of transformed point w.r.t. pose
        // pose is camera-to-world: (R, t)
        // p_cam = R^T * (p_world - t)
        //
        // Using right perturbation on SE3: pose' = pose ∘ Exp([δρ; δθ])
        // For small perturbations:
        //   R' = R * Exp(δθ) ≈ R * (I + [δθ]×)
        //   t' ≈ t + R * δρ  (V(δθ) ≈ I for small δθ)
        //
        // Then:
        //   (R')^T = (I - [δθ]×) * R^T
        //   p_cam' = (R')^T * (p_world - t')
        //          = (I - [δθ]×) * R^T * (p_world - t - R * δρ)
        //          = (I - [δθ]×) * R^T * (p_world - t) - (I - [δθ]×) * δρ
        //          ≈ (I - [δθ]×) * p_cam - δρ
        //          ≈ p_cam - [δθ]× * p_cam - δρ
        //          = p_cam + p_cam × δθ - δρ
        //          = p_cam + [p_cam]× * δθ - δρ
        //
        // So: ∂p_cam/∂[δρ; δθ] = [-I | [p_cam]×]

        let p_cam_skew = skew_symmetric(&p_cam);

        let d_pcam_d_pose = SMatrix::<f64, 3, 6>::from_fn(|r, c| {
            if c < 3 {
                // Translation part: -I
                if r == c { -1.0 } else { 0.0 }
            } else {
                // Rotation part: [p_cam]×
                p_cam_skew[(r, c - 3)]
            }
        });

        (d_uv_d_pcam, d_pcam_d_pose)
    }

    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        // Radial distortion
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + self.k1 * r2 + self.k2 * r4;

        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        // Jacobian ∂(u,v)/∂(fx,fy,cx,cy,k1,k2)
        // u = fx * x_d + cx
        // v = fy * y_d + cy
        // ∂u/∂fx = x_d, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂u/∂k1 = fx * x_n * r², ∂u/∂k2 = fx * x_n * r⁴
        // ∂v/∂fx = 0, ∂v/∂fy = y_d, ∂v/∂cx = 0, ∂v/∂cy = 1
        // ∂v/∂k1 = fy * y_n * r², ∂v/∂k2 = fy * y_n * r⁴
        SMatrix::<f64, 2, 6>::new(
            x_d,
            0.0,
            1.0,
            0.0,
            self.fx * x_n * r2,
            self.fx * x_n * r4,
            0.0,
            y_d,
            0.0,
            1.0,
            self.fy * y_n * r2,
            self.fy * y_n * r4,
        )
    }

    fn intrinsics_vec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.fx, self.fy, self.cx, self.cy, self.k1, self.k2])
    }

    fn from_params(params: &[f64]) -> Self {
        assert!(
            params.len() >= 6,
            "BALPinholeCamera requires at least 6 parameters, got {}",
            params.len()
        );
        Self {
            fx: params[0],
            fy: params[1],
            cx: params[2],
            cy: params[3],
            k1: params[4],
            k2: params[5],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_bal_pinhole_camera_creation() {
        let camera = BALPinholeCamera::new(500.0, 500.0, 320.0, 240.0, 0.0, 0.0);
        assert_eq!(camera.fx, 500.0);
        assert_eq!(camera.fy, 500.0);
        assert_eq!(camera.cx, 320.0);
        assert_eq!(camera.cy, 240.0);
        assert_eq!(camera.k1, 0.0);
        assert_eq!(camera.k2, 0.0);
    }

    #[test]
    fn test_bal_convention_negative_z() {
        let camera = BALPinholeCamera::new_no_distortion(500.0, 500.0, 320.0, 240.0);

        // Point at negative Z should be valid (in front of camera)
        let p_cam = Vector3::new(0.0, 0.0, -1.0);
        assert!(camera.is_valid_point(&p_cam));

        // Point at positive Z should be invalid (behind camera)
        let p_cam_behind = Vector3::new(0.0, 0.0, 1.0);
        assert!(!camera.is_valid_point(&p_cam_behind));
    }

    #[test]
    fn test_bal_projection_at_optical_axis() -> TestResult {
        let camera = BALPinholeCamera::new_no_distortion(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.0, 0.0, -1.0); // Note: negative Z

        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        // Point on optical axis should project to principal point (no distortion on axis)
        assert!((uv.x - 320.0).abs() < 1e-10);
        assert!((uv.y - 240.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_bal_projection_off_axis_no_distortion() -> TestResult {
        let camera = BALPinholeCamera::new_no_distortion(500.0, 500.0, 320.0, 240.0);
        let p_cam = Vector3::new(0.1, 0.2, -1.0); // Note: negative Z

        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        // u = 500 * 0.1/1.0 + 320 = 370
        // v = 500 * 0.2/1.0 + 240 = 340
        assert!((uv.x - 370.0).abs() < 1e-10);
        assert!((uv.y - 340.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_bal_projection_with_distortion() -> TestResult {
        let camera = BALPinholeCamera::new(500.0, 500.0, 320.0, 240.0, 0.1, 0.01);
        let p_cam = Vector3::new(0.1, 0.2, -1.0);

        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        // Normalized: x_n = 0.1, y_n = 0.2
        // r² = 0.01 + 0.04 = 0.05
        // r⁴ = 0.0025
        // distortion = 1 + 0.1*0.05 + 0.01*0.0025 = 1.005025
        // x_d = 0.1 * 1.005025 = 0.1005025
        // y_d = 0.2 * 1.005025 = 0.201005
        // u = 500 * 0.1005025 + 320 ≈ 370.25125
        // v = 500 * 0.201005 + 240 ≈ 340.5025
        assert!((uv.x - 370.25125).abs() < 1e-4);
        assert!((uv.y - 340.5025).abs() < 1e-4);

        Ok(())
    }
}
