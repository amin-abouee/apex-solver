//! Reprojection with the camera-to-body extrinsics as a variable.
//!
//! [`ProjectionFactor`](super::projection::ProjectionFactor) takes a
//! world-to-camera pose directly, which folds the rig calibration into the
//! quantity being estimated. On a real platform the extrinsics are the least
//! trustworthy part of the setup, and they are shared by every observation from
//! that camera — so they belong in the graph as their own variable, where many
//! observations can determine them.

use apex_camera_models::{CameraModel, CameraModelError};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::cheirality::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

/// `∂(T⁻¹·p)/∂(T's right tangent)` and `∂(T⁻¹·p)/∂p`.
///
/// For `q = T⁻¹·p` and a right perturbation `T ← T·Exp(δ)`,
/// `q ← Exp(−δ)·q = q − δρ + [q]ₓ·δθ`, so the pose block is `[−I | [q]ₓ]` and
/// the point block is `Rᵀ`.
pub(crate) fn inverse_act_jacobians(
    pose: &SE3,
    point_world: &Vector3<f64>,
) -> (Vector3<f64>, SMatrix<f64, 3, 6>, Matrix3<f64>) {
    let rotation = pose.rotation_so3().rotation_matrix();
    let q = rotation.transpose() * (point_world - pose.translation());

    let mut d_pose = SMatrix::<f64, 3, 6>::zeros();
    d_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-Matrix3::identity()));
    d_pose.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(&q));

    (q, d_pose, rotation.transpose())
}

/// Monocular reprojection over `[body pose, camera extrinsics, landmark]`.
///
/// ```text
/// T_WC  = T_WB ∘ T_BC                    camera pose in the world
/// p_cam = T_WC⁻¹ · p_world
/// r     = project(p_cam) − uv_measured   (2D)
/// ```
///
/// Poses follow the crate's body-in-world convention: `T_WB` places the body in
/// the world and `T_BC` the camera in the body, so both read the way a
/// calibration file does.
///
/// Cheirality violations produce the same bounded, gradient-carrying penalty as
/// the other camera factors rather than a hard failure.
///
/// # Parameter layout (3 blocks, 15 DOF)
///
/// ```text
/// params[0]: T_WB      — SE3 body pose in world, 7D / 6 DOF
/// params[1]: T_BC      — SE3 camera in body (the extrinsics), 7D / 6 DOF
/// params[2]: p_world   — 3D landmark
/// ```
pub struct ExtrinsicProjectionFactor<CAM: CameraModel> {
    camera: CAM,
    measurement: Vector2<f64>,
    verbose_cheirality: bool,
}

impl<CAM: CameraModel> ExtrinsicProjectionFactor<CAM> {
    /// Create the factor from a pixel observation and a fixed intrinsic model.
    pub fn new(camera: CAM, measurement: Vector2<f64>) -> Self {
        Self {
            camera,
            measurement,
            verbose_cheirality: false,
        }
    }

    /// Log a warning whenever the landmark falls behind the camera.
    pub fn with_verbose_cheirality(mut self) -> Self {
        self.verbose_cheirality = true;
        self
    }
}

impl<CAM: CameraModel + Send + Sync + 'static> Factor for ExtrinsicProjectionFactor<CAM> {
    /// Columns: `[T_WB(6) | T_BC(6) | landmark(3)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let t_wb = SE3::from_param_slice(params[0]);
        let t_bc = SE3::from_param_slice(params[1]);
        let p_world = Vector3::new(params[2][0], params[2][1], params[2][2]);

        // ∂T_WC/∂T_WB = Ad(T_BC⁻¹) and ∂T_WC/∂T_BC = I, from `compose`.
        let mut d_wc_d_wb = SE3::zero_jacobian();
        let t_wc = t_wb.compose(&t_bc, Some(&mut d_wc_d_wb), None);

        let (p_cam, d_cam_d_wc, d_cam_d_point) = inverse_act_jacobians(&t_wc, &p_world);

        let projection = match self.camera.project(&p_cam) {
            Ok(uv) => uv,
            Err(CameraModelError::PointBehindCamera { z, min_z }) => {
                if self.verbose_cheirality {
                    warn!("ExtrinsicProjectionFactor: point behind camera (z={z}, min_z={min_z})");
                }
                let penalty =
                    CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * (min_z - z).max(0.0);
                residual[0] = penalty;
                residual[1] = penalty;
                if let Some(mut jac) = jacobian {
                    // Gradient of the penalty through the camera-frame depth.
                    let d_pose = -CHEIRALITY_DEPTH_SCALE * (d_cam_d_wc.row(2) * d_wc_d_wb);
                    let d_ext = -CHEIRALITY_DEPTH_SCALE * d_cam_d_wc.row(2);
                    let d_point = -CHEIRALITY_DEPTH_SCALE * d_cam_d_point.row(2);
                    for row in 0..2 {
                        for col in 0..6 {
                            *jac.rb_mut().get_mut(row, col) = d_pose[(0, col)];
                            *jac.rb_mut().get_mut(row, 6 + col) = d_ext[(0, col)];
                        }
                        for col in 0..3 {
                            *jac.rb_mut().get_mut(row, 12 + col) = d_point[(0, col)];
                        }
                    }
                }
                return;
            }
            Err(err) => {
                if self.verbose_cheirality {
                    warn!("ExtrinsicProjectionFactor: invalid projection: {err}");
                }
                residual[0] = CHEIRALITY_BASE_PENALTY;
                residual[1] = CHEIRALITY_BASE_PENALTY;
                return;
            }
        };

        residual[0] = projection.x - self.measurement.x;
        residual[1] = projection.y - self.measurement.y;

        let Some(mut jac) = jacobian else { return };

        let d_uv_d_cam = self.camera.jacobian_point(&p_cam);
        let d_uv_d_wc = d_uv_d_cam.clone() * d_cam_d_wc;
        let d_uv_d_body = d_uv_d_wc * d_wc_d_wb;
        let d_uv_d_point = d_uv_d_cam * d_cam_d_point;

        for row in 0..2 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = d_uv_d_body[(row, col)];
                *jac.rb_mut().get_mut(row, 6 + col) = d_uv_d_wc[(row, col)];
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, 12 + col) = d_uv_d_point[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 15)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE, SE3::REP_SIZE, 3],
            "ExtrinsicProjectionFactor expects [SE3 T_WB, SE3 T_BC, 3D landmark]",
        )
    }
}
