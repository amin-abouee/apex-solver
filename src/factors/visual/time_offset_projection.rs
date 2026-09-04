//! Reprojection with the camera-to-IMU **time offset** as a variable.
//!
//! Camera and IMU timestamps are rarely aligned: the two run off different
//! clocks, and the image timestamp usually lags the true exposure instant by a
//! constant `t_d` that no calibration file records reliably. Left unmodelled it
//! is indistinguishable from an extrinsic rotation error at constant angular
//! rate, and it corrupts every reprojection on a moving platform — a 20 ms
//! offset at 1 m/s is 2 cm of position error, and at 1 rad/s it is 0.02 rad of
//! orientation error.
//!
//! The state is an [`SE23`] navigation state, which already carries the
//! velocity this correction needs. The angular rate comes from the gyro sample
//! at that instant and is treated as a measurement, not a variable — the graph
//! has no `ω` to estimate:
//!
//! ```text
//! p(t+t_d) = p + v·t_d
//! R(t+t_d) = R · Exp(ω·t_d)
//! ```
//!
//! a first-order expansion about the state, which is what makes `t_d`
//! observable at all: the correction is proportional to how fast the platform
//! is moving, so a stationary platform determines nothing.

use apex_camera_models::{CameraModel, CameraModelError};
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;
use apex_manifolds::so3::SO3Tangent;
use apex_manifolds::{LieGroup, Tangent};
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::cheirality::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::common::validate::expect_block_sizes;
use crate::factors::visual::extrinsic_projection::inverse_act_jacobians;

/// Monocular reprojection over `[SE23 state, extrinsics, landmark, time offset]`.
///
/// # Parameter layout (4 blocks, 19 DOF)
///
/// ```text
/// params[0]: SE23 state at the IMU timestamp — 10D, 9 DOF
/// params[1]: T_BC — SE3 camera in body (the extrinsics), 7D / 6 DOF
/// params[2]: p_world — 3D landmark
/// params[3]: t_d — 1D camera-to-IMU time offset [s]
/// ```
pub struct TimeOffsetProjectionFactor<CAM: CameraModel> {
    camera: CAM,
    measurement: Vector2<f64>,
    angular_velocity: Vector3<f64>,
    verbose_cheirality: bool,
}

impl<CAM: CameraModel> TimeOffsetProjectionFactor<CAM> {
    /// Create the factor.
    ///
    /// `angular_velocity` is the bias-corrected body rate at the state's
    /// timestamp, in the body frame [rad/s].
    pub fn new(camera: CAM, measurement: Vector2<f64>, angular_velocity: Vector3<f64>) -> Self {
        Self {
            camera,
            measurement,
            angular_velocity,
            verbose_cheirality: false,
        }
    }

    /// Log a warning whenever the landmark falls behind the camera.
    pub fn with_verbose_cheirality(mut self) -> Self {
        self.verbose_cheirality = true;
        self
    }

    /// The body pose at exposure time, and the derivatives of its right tangent
    /// with respect to the `SE23` state and to `t_d`.
    fn body_pose_at_exposure(
        &self,
        state: &SE23,
        t_d: f64,
    ) -> (SE3, SMatrix<f64, 6, 9>, SMatrix<f64, 6, 1>) {
        let rotation = state.rotation_matrix();
        let delta_rot = SO3Tangent::new(self.angular_velocity * t_d).exp(None);
        let r_d = delta_rot.rotation_matrix();

        let pose = SE3::new(
            state.translation() + state.velocity() * t_d,
            (state.rotation_so3().compose(&delta_rot, None, None)).quaternion(),
        );

        // Push the SE23 tangent through to the exposure pose's SE3 tangent.
        //
        // Position:  p_c ← p_c + R·δρ + Δt·R·δν, and p_c ← p_c + R_c·δρ' with
        //            R_c = R·R_d, so δρ' = R_dᵀ(δρ + t_d·δν).
        // Rotation:  R·Exp(δθ)·R_d = R_c·Exp(R_dᵀ·δθ), so δθ' = R_dᵀ·δθ.
        let r_d_t = r_d.transpose();
        let mut d_state = SMatrix::<f64, 6, 9>::zeros();
        d_state.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_d_t);
        d_state
            .fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(r_d_t * t_d));
        d_state.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_d_t);

        // t_d moves the body by v per second and rotates it at ω, expressed in
        // the exposure pose's own right tangent.
        let mut d_offset = SMatrix::<f64, 6, 1>::zeros();
        let r_c_t: Matrix3<f64> = (rotation * r_d).transpose();
        d_offset
            .fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(r_c_t * state.velocity()));
        d_offset
            .fixed_view_mut::<3, 1>(3, 0)
            .copy_from(&self.angular_velocity);

        (pose, d_state, d_offset)
    }
}

impl<CAM: CameraModel + Send + Sync + 'static> Factor for TimeOffsetProjectionFactor<CAM> {
    /// Columns: `[state(9) | T_BC(6) | landmark(3) | t_d(1)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = SE23::from_param_slice(params[0]);
        let t_bc = SE3::from_param_slice(params[1]);
        let p_world = Vector3::new(params[2][0], params[2][1], params[2][2]);
        let t_d = params[3][0];

        let (t_wb, d_body_d_state, d_body_d_offset) = self.body_pose_at_exposure(&state, t_d);

        let mut d_wc_d_wb = SE3::zero_jacobian();
        let t_wc = t_wb.compose(&t_bc, Some(&mut d_wc_d_wb), None);
        let (p_cam, d_cam_d_wc, d_cam_d_point) = inverse_act_jacobians(&t_wc, &p_world);

        let projection = match self.camera.project(&p_cam) {
            Ok(uv) => uv,
            Err(CameraModelError::PointBehindCamera { z, min_z }) => {
                if self.verbose_cheirality {
                    warn!("TimeOffsetProjectionFactor: point behind camera (z={z}, min_z={min_z})");
                }
                let penalty =
                    CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * (min_z - z).max(0.0);
                residual[0] = penalty;
                residual[1] = penalty;
                if let Some(mut jac) = jacobian {
                    for row in 0..2 {
                        for col in 0..19 {
                            *jac.rb_mut().get_mut(row, col) = 0.0;
                        }
                    }
                }
                return;
            }
            Err(err) => {
                if self.verbose_cheirality {
                    warn!("TimeOffsetProjectionFactor: invalid projection: {err}");
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
        let d_uv_d_wb = d_uv_d_wc * d_wc_d_wb;

        let d_uv_d_state = d_uv_d_wb * d_body_d_state;
        let d_uv_d_offset = d_uv_d_wb * d_body_d_offset;
        let d_uv_d_point = d_uv_d_cam * d_cam_d_point;

        for row in 0..2 {
            for col in 0..9 {
                *jac.rb_mut().get_mut(row, col) = d_uv_d_state[(row, col)];
            }
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, 9 + col) = d_uv_d_wc[(row, col)];
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, 15 + col) = d_uv_d_point[(row, col)];
            }
            *jac.rb_mut().get_mut(row, 18) = d_uv_d_offset[(row, 0)];
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 19)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, SE3::REP_SIZE, 3, 1],
            "TimeOffsetProjectionFactor expects \
             [SE23 state, SE3 T_BC, 3D landmark, time offset (1D)]",
        )
    }
}
