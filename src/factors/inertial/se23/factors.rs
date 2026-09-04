//! The four SE_2(3) IMU factors.
//!
//! Each is a thin wrapper over [`se23_kinematics`]: unpack parameter blocks,
//! lift the shared tangent-space derivatives into this factor's column layout,
//! weight, and write out. They come in two families, matching GTSAM:
//!
//! | Factor | Residual | Bias handling |
//! |---|---|---|
//! | [`ImuFactor`] | 9D `[p, q, v]` | one shared bias variable |
//! | [`Se23ImuFactor`] | 9D `[p, q, v]` | one shared bias variable |
//! | [`CombinedImuFactor`] | 15D `[p, q, v, bg, ba]` | random walk in the residual |
//! | [`CombinedSe23ImuFactor`] | 15D `[p, q, v, bg, ba]` | random walk in the residual |
//!
//! The 9D factors leave bias evolution to an explicit edge — build one with
//! [`bias_random_walk`](crate::factors::inertial::bias::bias_random_walk).
//! The 15D factors embed it, so they need no such edge. Mixing the two (a 15D
//! factor *and* a bias edge) counts the random walk twice.
//!
//! Within each family the two variants differ only in state parameterization:
//! `(SE3 pose, R³ velocity)` blocks versus a single native `SE23` state.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use super::kinematics::{
    Se23Kinematics, gravity_corrected_state_from_se23, gravity_corrected_state_i,
    pose_to_se23_tangent, se23_kinematics, state_to_gc_tangent, velocity_to_gc_tangent,
    velocity_to_se23_tangent,
};
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::inertial::preintegration::ImuPreintegration;

/// Copy a fixed-size weighted residual into the output buffer.
fn write_residual<const N: usize>(weighted: &SVector<f64, N>, residual: &mut [f64]) {
    residual[..N].copy_from_slice(weighted.as_slice());
}

/// Copy a fixed-size weighted Jacobian into the caller's column-major buffer.
fn write_jacobian<const R: usize, const C: usize>(
    weighted: &SMatrix<f64, R, C>,
    jac: &mut faer::mat::MatMut<'_, f64>,
) {
    for row in 0..R {
        for col in 0..C {
            *jac.rb_mut().get_mut(row, col) = weighted[(row, col)];
        }
    }
}

/// Stack the two bias-correction blocks into `∂r_kin/∂[b_g, b_a]` (9×6).
fn bias_block(kin: &Se23Kinematics) -> SMatrix<f64, 9, 6> {
    let mut j = SMatrix::<f64, 9, 6>::zeros();
    j.fixed_view_mut::<9, 3>(0, 0).copy_from(&kin.d_bias_gyro);
    j.fixed_view_mut::<9, 3>(0, 3).copy_from(&kin.d_bias_accel);
    j
}

/// Read a 6D bias block as `(b_g, b_a)`.
fn split_bias(block: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(block[0], block[1], block[2]),
        Vector3::new(block[3], block[4], block[5]),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// ImuFactor — 9D residual, pose/velocity blocks, one shared bias
// ─────────────────────────────────────────────────────────────────────────────

/// Non-combined IMU factor over `(pose, velocity)` blocks — GTSAM `ImuFactor`.
///
/// # Residual (9D)
///
/// ```text
/// r = sqrt_info · [ r_p (3)   position
///                   r_q (3)   rotation
///                   r_v (3) ] velocity
/// ```
///
/// Bias appears only through the first-order correction of the preintegrated
/// delta; its evolution between keyframes is a separate edge, so a single bias
/// variable is shared by both frames. Weighting uses the measurement-noise-only
/// 9×9 information — see
/// [`kinematic_square_root_information`](ImuPreintegration::kinematic_square_root_information).
///
/// # Parameter layout (5 blocks, 24 minimal DOF)
///
/// ```text
/// params[0]: SE3 pose i  — 7D, 6 DOF
/// params[1]: velocity i  — 3D
/// params[2]: SE3 pose j  — 7D, 6 DOF
/// params[3]: velocity j  — 3D
/// params[4]: imu bias    — 6D [bg, ba], shared by both frames
/// ```
pub struct ImuFactor {
    preintegration: ImuPreintegration,
}

impl ImuFactor {
    /// Create a non-combined IMU factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for ImuFactor {
    /// Columns: `[pose_i(6) | vel_i(3) | pose_j(6) | vel_j(3) | bias(6)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let preint = &self.preintegration;

        let pose_i = SE3::from_param_slice(params[0]);
        let v_i = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let pose_j = SE3::from_param_slice(params[2]);
        let v_j = Vector3::new(params[3][0], params[3][1], params[3][2]);
        let (b_g, b_a) = split_bias(params[4]);

        let gc_state_i = gravity_corrected_state_i(preint, &pose_i, v_i);
        let state_j = SE23::new(pose_j.translation(), v_j, pose_j.rotation_quaternion());
        let kin = se23_kinematics(preint, &gc_state_i, &state_j, b_g, b_a);

        let sqrt_info = preint.kinematic_square_root_information();
        write_residual(&(sqrt_info * kin.r_kin), residual);

        let Some(mut jac) = jacobian else { return };

        let dt = preint.delta_t();
        let r_i = pose_i.rotation_so3().rotation_matrix();
        let r_j = pose_j.rotation_so3().rotation_matrix();

        let mut j_full = SMatrix::<f64, 9, 24>::zeros();
        j_full
            .fixed_view_mut::<9, 6>(0, 0)
            .copy_from(&(kin.d_gc_state_i * pose_to_se23_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 6)
            .copy_from(&(kin.d_gc_state_i * velocity_to_gc_tangent(&r_i, dt)));
        j_full
            .fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&(kin.d_state_j * pose_to_se23_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 15)
            .copy_from(&(kin.d_state_j * velocity_to_se23_tangent(&r_j)));
        j_full
            .fixed_view_mut::<9, 6>(0, 18)
            .copy_from(&bias_block(&kin));

        write_jacobian(&(sqrt_info * j_full), &mut jac);
    }

    fn residual_dim(&self) -> usize {
        9
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (9, 24)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE, 3, SE3::REP_SIZE, 3, 6],
            "ImuFactor expects [SE3 pose_i, vel_i, SE3 pose_j, vel_j, bias]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Se23ImuFactor — 9D residual, native SE23 states, one shared bias
// ─────────────────────────────────────────────────────────────────────────────

/// Non-combined IMU factor over native `SE23` states — GTSAM `ImuFactor2`.
///
/// Same 9D residual as [`ImuFactor`]; the pose and velocity of each keyframe
/// are a single `SE23` variable rather than two blocks, which keeps the
/// optimizer's update on the manifold the preintegration actually lives on.
///
/// # Parameter layout (3 blocks, 24 minimal DOF)
///
/// ```text
/// params[0]: SE23 state i — 10D, 9 DOF
/// params[1]: SE23 state j — 10D, 9 DOF
/// params[2]: imu bias     — 6D [bg, ba], shared by both frames
/// ```
pub struct Se23ImuFactor {
    preintegration: ImuPreintegration,
}

impl Se23ImuFactor {
    /// Create a native-`SE23` non-combined IMU factor.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for Se23ImuFactor {
    /// Columns: `[state_i(9) | state_j(9) | bias(6)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let preint = &self.preintegration;

        let state_i = SE23::from_param_slice(params[0]);
        let state_j = SE23::from_param_slice(params[1]);
        let (b_g, b_a) = split_bias(params[2]);

        let gc_state_i = gravity_corrected_state_from_se23(preint, &state_i);
        let kin = se23_kinematics(preint, &gc_state_i, &state_j, b_g, b_a);

        let sqrt_info = preint.kinematic_square_root_information();
        write_residual(&(sqrt_info * kin.r_kin), residual);

        let Some(mut jac) = jacobian else { return };

        let mut j_full = SMatrix::<f64, 9, 24>::zeros();
        j_full
            .fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&(kin.d_gc_state_i * state_to_gc_tangent(preint.delta_t())));
        j_full
            .fixed_view_mut::<9, 9>(0, 9)
            .copy_from(&kin.d_state_j);
        j_full
            .fixed_view_mut::<9, 6>(0, 18)
            .copy_from(&bias_block(&kin));

        write_jacobian(&(sqrt_info * j_full), &mut jac);
    }

    fn residual_dim(&self) -> usize {
        9
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (9, 24)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, SE23::REP_SIZE, 6],
            "Se23ImuFactor expects [SE23 state_i, SE23 state_j, bias]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CombinedImuFactor — 15D residual, pose/velocity blocks, per-frame bias
// ─────────────────────────────────────────────────────────────────────────────

/// Combined IMU factor over `(pose, velocity, bias)` blocks — GTSAM
/// `CombinedImuFactor`.
///
/// # Residual (15D)
///
/// ```text
/// r = sqrt_info · [ r_p (3)    position
///                   r_q (3)    rotation
///                   r_v (3)    velocity
///                   r_bg (3)   gyro bias
///                   r_ba (3) ] accel bias
/// ```
///
/// The trailing six rows are the Gauss–Markov bias random walk, so no separate
/// bias edge is needed — and adding one would count that uncertainty twice.
///
/// # Parameter layout (6 blocks, 30 minimal DOF)
///
/// ```text
/// params[0]: SE3 pose i  — 7D, 6 DOF     params[3]: SE3 pose j  — 7D, 6 DOF
/// params[1]: velocity i  — 3D            params[4]: velocity j  — 3D
/// params[2]: imu bias i  — 6D [bg, ba]   params[5]: imu bias j  — 6D
/// ```
pub struct CombinedImuFactor {
    preintegration: ImuPreintegration,
}

impl CombinedImuFactor {
    /// Create a combined IMU factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for CombinedImuFactor {
    /// Columns:
    /// `[pose_i(6) | vel_i(3) | bias_i(6) | pose_j(6) | vel_j(3) | bias_j(6)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let preint = &self.preintegration;

        let pose_i = SE3::from_param_slice(params[0]);
        let v_i = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let (b_g_i, b_a_i) = split_bias(params[2]);
        let pose_j = SE3::from_param_slice(params[3]);
        let v_j = Vector3::new(params[4][0], params[4][1], params[4][2]);
        let (b_g_j, b_a_j) = split_bias(params[5]);

        let gc_state_i = gravity_corrected_state_i(preint, &pose_i, v_i);
        let state_j = SE23::new(pose_j.translation(), v_j, pose_j.rotation_quaternion());
        let kin = se23_kinematics(preint, &gc_state_i, &state_j, b_g_i, b_a_i);

        let residual_raw = combined_residual(&kin, b_g_i, b_a_i, b_g_j, b_a_j);
        let sqrt_info = preint.square_root_information();
        write_residual(&(sqrt_info * residual_raw), residual);

        let Some(mut jac) = jacobian else { return };

        let dt = preint.delta_t();
        let r_i = pose_i.rotation_so3().rotation_matrix();
        let r_j = pose_j.rotation_so3().rotation_matrix();

        let mut j_full = SMatrix::<f64, 15, 30>::zeros();
        j_full
            .fixed_view_mut::<9, 6>(0, 0)
            .copy_from(&(kin.d_gc_state_i * pose_to_se23_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 6)
            .copy_from(&(kin.d_gc_state_i * velocity_to_gc_tangent(&r_i, dt)));
        j_full
            .fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&bias_block(&kin));
        j_full
            .fixed_view_mut::<9, 6>(0, 15)
            .copy_from(&(kin.d_state_j * pose_to_se23_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 21)
            .copy_from(&(kin.d_state_j * velocity_to_se23_tangent(&r_j)));
        // bias_j does not enter the kinematic rows.
        write_bias_walk_rows(&mut j_full, 9, 24);

        write_jacobian(&(sqrt_info * j_full), &mut jac);
    }

    fn residual_dim(&self) -> usize {
        15
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (15, 30)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE, 3, 6, SE3::REP_SIZE, 3, 6],
            "CombinedImuFactor expects [SE3 pose_i, vel_i, bias_i, SE3 pose_j, vel_j, bias_j]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CombinedSe23ImuFactor — 15D residual, native SE23 states, per-frame bias
// ─────────────────────────────────────────────────────────────────────────────

/// Combined IMU factor over native `SE23` states.
///
/// Same 15D residual as [`CombinedImuFactor`], with pose and velocity fused
/// into one `SE23` variable per keyframe.
///
/// # Parameter layout (4 blocks, 30 minimal DOF)
///
/// ```text
/// params[0]: SE23 state i — 10D, 9 DOF    params[2]: SE23 state j — 10D, 9 DOF
/// params[1]: imu bias i   — 6D [bg, ba]   params[3]: imu bias j   — 6D
/// ```
pub struct CombinedSe23ImuFactor {
    preintegration: ImuPreintegration,
}

impl CombinedSe23ImuFactor {
    /// Create a native-`SE23` combined IMU factor.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for CombinedSe23ImuFactor {
    /// Columns: `[state_i(9) | bias_i(6) | state_j(9) | bias_j(6)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let preint = &self.preintegration;

        let state_i = SE23::from_param_slice(params[0]);
        let (b_g_i, b_a_i) = split_bias(params[1]);
        let state_j = SE23::from_param_slice(params[2]);
        let (b_g_j, b_a_j) = split_bias(params[3]);

        let gc_state_i = gravity_corrected_state_from_se23(preint, &state_i);
        let kin = se23_kinematics(preint, &gc_state_i, &state_j, b_g_i, b_a_i);

        let residual_raw = combined_residual(&kin, b_g_i, b_a_i, b_g_j, b_a_j);
        let sqrt_info = preint.square_root_information();
        write_residual(&(sqrt_info * residual_raw), residual);

        let Some(mut jac) = jacobian else { return };

        let mut j_full = SMatrix::<f64, 15, 30>::zeros();
        j_full
            .fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&(kin.d_gc_state_i * state_to_gc_tangent(preint.delta_t())));
        j_full
            .fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&bias_block(&kin));
        j_full
            .fixed_view_mut::<9, 9>(0, 15)
            .copy_from(&kin.d_state_j);
        write_bias_walk_rows(&mut j_full, 9, 24);

        write_jacobian(&(sqrt_info * j_full), &mut jac);
    }

    fn residual_dim(&self) -> usize {
        15
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (15, 30)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, 6, SE23::REP_SIZE, 6],
            "CombinedSe23ImuFactor expects [SE23 state_i, bias_i, SE23 state_j, bias_j]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared pieces of the 15D (combined) form
// ─────────────────────────────────────────────────────────────────────────────

/// Stack the 9D kinematic residual on the 6D bias-difference rows.
fn combined_residual(
    kin: &Se23Kinematics,
    b_g_i: Vector3<f64>,
    b_a_i: Vector3<f64>,
    b_g_j: Vector3<f64>,
    b_a_j: Vector3<f64>,
) -> SVector<f64, 15> {
    let mut r = SVector::<f64, 15>::zeros();
    r.fixed_rows_mut::<9>(0).copy_from(&kin.r_kin);
    r.fixed_rows_mut::<3>(9).copy_from(&(b_g_i - b_g_j));
    r.fixed_rows_mut::<3>(12).copy_from(&(b_a_i - b_a_j));
    r
}

/// Write the bias random-walk rows `∂(b_i − b_j)/∂(b_i, b_j)`, where the frame-i
/// bias block starts at column `col_i` and the frame-j block at `col_j`.
fn write_bias_walk_rows(j_full: &mut SMatrix<f64, 15, 30>, col_i: usize, col_j: usize) {
    let id = Matrix3::identity();
    j_full.fixed_view_mut::<3, 3>(9, col_i).copy_from(&id);
    j_full.fixed_view_mut::<3, 3>(9, col_j).copy_from(&(-id));
    j_full.fixed_view_mut::<3, 3>(12, col_i + 3).copy_from(&id);
    j_full
        .fixed_view_mut::<3, 3>(12, col_j + 3)
        .copy_from(&(-id));
}

/// Shared `validate_variables` body: check block count and per-block sizes.
fn expect_block_sizes(
    variables: &[&dyn ManifoldVariable],
    expected: &[usize],
    message: &str,
) -> Result<(), String> {
    if variables.len() != expected.len()
        || variables
            .iter()
            .zip(expected)
            .any(|(v, &n)| v.as_param_slice().len() != n)
    {
        return Err(message.into());
    }
    Ok(())
}
