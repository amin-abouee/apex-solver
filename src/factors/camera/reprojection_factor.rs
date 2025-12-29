//! Single-observation reprojection factor for bundle adjustment.
//!
//! This factor connects one camera pose (SE3) to one 3D landmark (R3)
//! and computes the reprojection error for a single 2D observation.

use nalgebra::{DMatrix, DVector, Vector2, Vector3};

use super::CameraModel;
use crate::factors::Factor;
use crate::manifold::LieGroup;
use crate::manifold::se3::SE3;

/// Reprojection factor for bundle adjustment.
///
/// This factor computes the reprojection error between an observed 2D image point
/// and a projected 3D landmark. It connects one camera pose (SE3) to one 3D point (R3).
///
/// # Parameters (in linearize)
///
/// - `params[0]`: Camera pose as SE3 (7D: qw, qx, qy, qz, tx, ty, tz)
/// - `params[1]`: 3D landmark position (3D: x, y, z)
///
/// # Residual
///
/// 2D vector: (projected_u - observed_u, projected_v - observed_v)
///
/// # Jacobian
///
/// 2×9 matrix: [∂residual/∂pose (2×6), ∂residual/∂landmark (2×3)]
#[derive(Clone, Debug)]
pub struct ReprojectionFactor<CAM: CameraModel> {
    /// Observed 2D point in image coordinates
    pub observation: Vector2<f64>,
    /// Camera model with intrinsics
    pub camera: CAM,
}

impl<CAM: CameraModel> ReprojectionFactor<CAM> {
    /// Create a new reprojection factor.
    ///
    /// # Arguments
    ///
    /// * `observation` - 2D image point (u, v)
    /// * `camera` - Camera model with intrinsics
    pub fn new(observation: Vector2<f64>, camera: CAM) -> Self {
        Self {
            observation,
            camera,
        }
    }
}

impl<CAM: CameraModel> Factor for ReprojectionFactor<CAM> {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert!(
            params.len() >= 2,
            "ReprojectionFactor requires 2 parameters: pose (SE3) and landmark (R3)"
        );

        // Extract pose and landmark
        let pose = SE3::from(params[0].clone());
        let p_world = Vector3::new(params[1][0], params[1][1], params[1][2]);

        // Transform point to camera frame using pose.act()
        // BAL convention: P_cam = R * X_world + t (world-to-camera)
        // pose.act() computes: R * p_world + t = p_cam
        let p_cam = pose.act(&p_world, None, None);

        // Check validity
        if !self.camera.is_valid_point(&p_cam) {
            // Point behind camera or invalid
            let residual = DVector::from_vec(vec![1e6, 1e6]);
            let jacobian = if compute_jacobian {
                Some(DMatrix::zeros(2, 9))
            } else {
                None
            };
            return (residual, jacobian);
        }

        // Project point
        let uv = match self.camera.project(&p_cam) {
            Some(proj) => proj,
            None => {
                let residual = DVector::from_vec(vec![1e6, 1e6]);
                let jacobian = if compute_jacobian {
                    Some(DMatrix::zeros(2, 9))
                } else {
                    None
                };
                return (residual, jacobian);
            }
        };

        // Compute residual
        let residual =
            DVector::from_vec(vec![uv.x - self.observation.x, uv.y - self.observation.y]);

        // Compute Jacobian if requested
        let jacobian = if compute_jacobian {
            // Jacobian of projection w.r.t. point in camera frame
            let d_uv_d_pcam = self.camera.jacobian_point(&p_cam);

            // Jacobian of p_cam w.r.t. pose for world-to-camera convention
            // p_cam = R * p_world + t
            //
            // Using right perturbation: pose' = pose ∘ Exp([δρ; δθ])
            // For small perturbations:
            //   R' = R * Exp(δθ) ≈ R * (I + [δθ]×)
            //   t' = t + R * δρ
            //
            // So:
            //   p_cam' = R' * p_world + t' = R*(I+[δθ]×)*p_world + t + R*δρ
            //          = R*p_world + t + R*[δθ]×*p_world + R*δρ
            //          = p_cam + R*(δρ + [δθ]× * p_world)
            //          = p_cam + R*(δρ - [p_world]× * δθ)
            //          = p_cam + R*δρ - R*[p_world]×*δθ
            //
            // ∂p_cam/∂δρ = R
            // ∂p_cam/∂δθ = -R * [p_world]×
            let rotation = pose.rotation_so3().rotation_matrix();
            let p_world_skew = super::skew_symmetric(&p_world);

            let d_pcam_d_pose = nalgebra::SMatrix::<f64, 3, 6>::from_fn(|r, c| {
                if c < 3 {
                    // Translation part: R
                    rotation[(r, c)]
                } else {
                    // Rotation part: -R * [p_world]×
                    let col = c - 3;
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += rotation[(r, k)] * p_world_skew[(k, col)];
                    }
                    -sum
                }
            });

            // Chain rule: ∂uv/∂pose = ∂uv/∂p_cam * ∂p_cam/∂pose
            let d_uv_d_pose = d_uv_d_pcam.clone() * d_pcam_d_pose;

            // Jacobian w.r.t. landmark (2×3)
            // p_cam = R * p_world + t
            // ∂p_cam/∂p_world = R
            // ∂uv/∂p_world = ∂uv/∂p_cam * R
            let d_uv_d_landmark = d_uv_d_pcam * rotation;

            // Combine into full Jacobian (2×9)
            let mut full_jac = DMatrix::zeros(2, 9);
            for r in 0..2 {
                for c in 0..6 {
                    full_jac[(r, c)] = d_uv_d_pose[(r, c)];
                }
                for c in 0..3 {
                    full_jac[(r, c + 6)] = d_uv_d_landmark[(r, c)];
                }
            }

            Some(full_jac)
        } else {
            None
        };

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        2 // (u, v) residual
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::factors::camera::BALPinholeCamera;
    use crate::manifold::se3::SE3Tangent;

    #[test]
    fn test_reprojection_factor_residual() {
        // Create a simple camera
        let camera = BALPinholeCamera::new(500.0, 500.0, 0.0, 0.0, 0.0, 0.0);

        // Identity pose (world-to-camera: camera at origin, looking down -Z)
        let pose = SE3::identity();

        // Point in front of camera (negative Z for BAL convention)
        // With identity world-to-camera pose, p_cam = p_world
        let landmark = Vector3::new(0.0, 0.0, -2.0);

        // Expected projection: at principal point (0, 0)
        let observation = Vector2::new(0.0, 0.0);

        let factor = ReprojectionFactor::new(observation, camera);
        let params = vec![
            DVector::from(pose),
            DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]),
        ];

        let (residual, _) = factor.linearize(&params, false);

        // Residual should be zero for perfect projection
        assert!((residual[0]).abs() < 1e-10, "residual[0] = {}", residual[0]);
        assert!((residual[1]).abs() < 1e-10, "residual[1] = {}", residual[1]);
    }

    #[test]
    fn test_reprojection_factor_jacobian_numerical() {
        // Create a simple camera
        let camera = BALPinholeCamera::new(500.0, 500.0, 0.0, 0.0, 0.0, 0.0);

        // Random pose (world-to-camera)
        let pose = SE3::from_translation_euler(0.1, 0.2, 0.3, 0.1, 0.2, 0.3);

        // Point in world coordinates that ends up in front of camera
        // p_cam = R * p_world + t, we need p_cam.z < 0
        let landmark = Vector3::new(1.0, 0.5, -5.0);

        // Project to get observation (world-to-camera convention)
        let p_cam = pose.act(&landmark, None, None);
        let observation = camera
            .project(&p_cam)
            .expect("Test: Camera projection should succeed");

        let factor = ReprojectionFactor::new(observation, camera);

        // Compute analytical Jacobian
        let params = vec![
            DVector::from(pose.clone()),
            DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]),
        ];
        let (_, analytical_jac) = factor.linearize(&params, true);
        let analytical_jac = analytical_jac.expect("Test: Linearize should return Jacobian");

        // Compute numerical Jacobian
        let eps = 1e-7;
        let mut numerical_jac = DMatrix::zeros(2, 9);

        // Pose Jacobian (6 DOF)
        // SE3Tangent uses [rho (translation), theta (rotation)]
        for i in 0..6 {
            let mut rho = Vector3::zeros();
            let mut theta = Vector3::zeros();
            if i < 3 {
                rho[i] = eps;
            } else {
                theta[i - 3] = eps;
            }
            let delta = SE3Tangent::new(rho, theta);

            let pose_plus = pose.plus(&delta, None, None);
            let delta_neg = SE3Tangent::new(-rho, -theta);
            let pose_minus = pose.plus(&delta_neg, None, None);

            let params_plus = vec![
                DVector::from(pose_plus),
                DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]),
            ];
            let params_minus = vec![
                DVector::from(pose_minus),
                DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]),
            ];

            let (res_plus, _) = factor.linearize(&params_plus, false);
            let (res_minus, _) = factor.linearize(&params_minus, false);

            for j in 0..2 {
                numerical_jac[(j, i)] = (res_plus[j] - res_minus[j]) / (2.0 * eps);
            }
        }

        // Landmark Jacobian (3 DOF)
        for i in 0..3 {
            let mut lm_plus = landmark;
            let mut lm_minus = landmark;
            lm_plus[i] += eps;
            lm_minus[i] -= eps;

            let params_plus = vec![
                DVector::from(pose.clone()),
                DVector::from_vec(vec![lm_plus.x, lm_plus.y, lm_plus.z]),
            ];
            let params_minus = vec![
                DVector::from(pose.clone()),
                DVector::from_vec(vec![lm_minus.x, lm_minus.y, lm_minus.z]),
            ];

            let (res_plus, _) = factor.linearize(&params_plus, false);
            let (res_minus, _) = factor.linearize(&params_minus, false);

            for j in 0..2 {
                numerical_jac[(j, i + 6)] = (res_plus[j] - res_minus[j]) / (2.0 * eps);
            }
        }

        // Compare
        for r in 0..2 {
            for c in 0..9 {
                let diff = (analytical_jac[(r, c)] - numerical_jac[(r, c)]).abs();
                let max_val = analytical_jac[(r, c)]
                    .abs()
                    .max(numerical_jac[(r, c)].abs())
                    .max(1.0);
                assert!(
                    diff < 1e-4 * max_val,
                    "Jacobian mismatch at ({}, {}): analytical={}, numerical={}, diff={}",
                    r,
                    c,
                    analytical_jac[(r, c)],
                    numerical_jac[(r, c)],
                    diff
                );
            }
        }
    }
}
