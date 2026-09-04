//! The four SGal(3) IMU factors.
//!
//! Same two axes as the SE_2(3) family (combined or not; split blocks or a
//! native state), but on the Special Galilean group, whose tangent carries a
//! time coordinate:
//!
//! | Factor | Residual | Blocks |
//! |---|---|---|
//! | [`Sgal3ImuFactor`] | 9D | `(SE3, vel, SE3, vel, bias)` |
//! | [`Sgal3StateImuFactor`] | **10D** | `(SGal3, SGal3, bias)` |
//! | [`Sgal3CombinedImuFactor`] | 15D | `(SE3, vel, bias) × 2` |
//! | [`Sgal3CombinedStateImuFactor`] | **16D** | `(SGal3, bias) × 2` |
//!
//! The extra row in the native-state factors is the time constraint
//! `(t_j − t_i) − Δt`. The split-block factors have no time variable, so that
//! row is identically zero for them and is dropped rather than carried as a
//! null direction in the linear system.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::sgal3::SGal3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use super::kinematics::{
    Sgal3Kinematics, gravity_corrected_from_pose, gravity_corrected_from_state, kinematic_rows,
    pose_to_sgal3_tangent, sgal3_kinematics, state_j_from_pose, state_to_gc_tangent, time_row,
    velocity_to_gc_tangent, velocity_to_state_j_tangent,
};
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::inertial::preintegration::ImuPreintegration;

/// Default standard deviation on the inter-keyframe time constraint [s].
///
/// 100 µs is the scale of timestamp jitter on a typical synchronized IMU. It is
/// only used by the native-`SGal3` factors, and only for the time row; override
/// it with `with_time_sigma` when calibrating a sensor time offset, where the
/// interval is genuinely uncertain.
pub const DEFAULT_TIME_SIGMA: f64 = 1.0e-4;

fn split_bias(block: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(block[0], block[1], block[2]),
        Vector3::new(block[3], block[4], block[5]),
    )
}

fn write_residual<const N: usize>(weighted: &SVector<f64, N>, residual: &mut [f64]) {
    residual[..N].copy_from_slice(weighted.as_slice());
}

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

/// Stack the bias-correction blocks into `∂r/∂[b_g, b_a]` (10×6).
fn bias_block(kin: &Sgal3Kinematics) -> SMatrix<f64, 10, 6> {
    let mut j = SMatrix::<f64, 10, 6>::zeros();
    j.fixed_view_mut::<10, 3>(0, 0).copy_from(&kin.d_bias_gyro);
    j.fixed_view_mut::<10, 3>(0, 3).copy_from(&kin.d_bias_accel);
    j
}

/// The 6 bias-difference rows shared by the combined factors.
fn bias_walk_residual(
    b_g_i: Vector3<f64>,
    b_a_i: Vector3<f64>,
    b_g_j: Vector3<f64>,
    b_a_j: Vector3<f64>,
) -> SVector<f64, 6> {
    let mut r = SVector::<f64, 6>::zeros();
    r.fixed_rows_mut::<3>(0).copy_from(&(b_g_i - b_g_j));
    r.fixed_rows_mut::<3>(3).copy_from(&(b_a_i - b_a_j));
    r
}

// ─────────────────────────────────────────────────────────────────────────────
// Split-block factors — no time variable, 9D / 15D
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! preintegration_accessors {
    ($t:ty) => {
        impl $t {
            /// Access the underlying preintegration.
            pub fn preintegration(&self) -> &ImuPreintegration {
                &self.preintegration
            }
        }
    };
}

/// Non-combined SGal(3) IMU factor over `(pose, velocity)` blocks.
///
/// 9D residual `[p, q, v]` with one bias variable shared across the interval;
/// pair it with [`bias_random_walk`](crate::factors::inertial::bias::bias_random_walk).
///
/// ```text
/// params[0]: SE3 pose i — 7D    params[2]: SE3 pose j — 7D
/// params[1]: velocity i — 3D    params[3]: velocity j — 3D
/// params[4]: imu bias   — 6D, shared
/// ```
pub struct Sgal3ImuFactor {
    preintegration: ImuPreintegration,
}
preintegration_accessors!(Sgal3ImuFactor);

impl Sgal3ImuFactor {
    /// Create the factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }
}

impl Factor for Sgal3ImuFactor {
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

        let gc = gravity_corrected_from_pose(preint, &pose_i, v_i);
        let sj = state_j_from_pose(preint, &pose_j, v_j);
        let kin = sgal3_kinematics(preint, &gc, &sj, b_g, b_a);

        let p = kinematic_rows();
        let sqrt_info = preint.kinematic_square_root_information();
        write_residual(&(sqrt_info * (p * kin.r_tangent)), residual);

        let Some(mut jac) = jacobian else { return };

        let dt = preint.delta_t();
        let r_i = pose_i.rotation_so3().rotation_matrix();
        let r_j = pose_j.rotation_so3().rotation_matrix();

        let mut j_full = SMatrix::<f64, 9, 24>::zeros();
        j_full
            .fixed_view_mut::<9, 6>(0, 0)
            .copy_from(&(p * kin.d_gc_state_i * pose_to_sgal3_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 6)
            .copy_from(&(p * kin.d_gc_state_i * velocity_to_gc_tangent(&r_i, dt)));
        j_full
            .fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&(p * kin.d_state_j * pose_to_sgal3_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 15)
            .copy_from(&(p * kin.d_state_j * velocity_to_state_j_tangent(&r_j, dt)));
        j_full
            .fixed_view_mut::<9, 6>(0, 18)
            .copy_from(&(p * bias_block(&kin)));

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
            "Sgal3ImuFactor expects [SE3 pose_i, vel_i, SE3 pose_j, vel_j, bias]",
        )
    }
}

/// Combined SGal(3) IMU factor over `(pose, velocity, bias)` blocks.
///
/// 15D residual `[p, q, v, bg, ba]`; the bias random walk is embedded, so no
/// separate bias edge is needed.
///
/// ```text
/// params[0..3]: SE3 pose i, velocity i, imu bias i
/// params[3..6]: SE3 pose j, velocity j, imu bias j
/// ```
pub struct Sgal3CombinedImuFactor {
    preintegration: ImuPreintegration,
}
preintegration_accessors!(Sgal3CombinedImuFactor);

impl Sgal3CombinedImuFactor {
    /// Create the factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }
}

impl Factor for Sgal3CombinedImuFactor {
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

        let gc = gravity_corrected_from_pose(preint, &pose_i, v_i);
        let sj = state_j_from_pose(preint, &pose_j, v_j);
        let kin = sgal3_kinematics(preint, &gc, &sj, b_g_i, b_a_i);

        let p = kinematic_rows();
        let mut raw = SVector::<f64, 15>::zeros();
        raw.fixed_rows_mut::<9>(0).copy_from(&(p * kin.r_tangent));
        raw.fixed_rows_mut::<6>(9)
            .copy_from(&bias_walk_residual(b_g_i, b_a_i, b_g_j, b_a_j));

        let sqrt_info = preint.square_root_information();
        write_residual(&(sqrt_info * raw), residual);

        let Some(mut jac) = jacobian else { return };

        let dt = preint.delta_t();
        let r_i = pose_i.rotation_so3().rotation_matrix();
        let r_j = pose_j.rotation_so3().rotation_matrix();
        let id = Matrix3::identity();

        let mut j_full = SMatrix::<f64, 15, 30>::zeros();
        j_full
            .fixed_view_mut::<9, 6>(0, 0)
            .copy_from(&(p * kin.d_gc_state_i * pose_to_sgal3_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 6)
            .copy_from(&(p * kin.d_gc_state_i * velocity_to_gc_tangent(&r_i, dt)));
        j_full
            .fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&(p * bias_block(&kin)));
        j_full
            .fixed_view_mut::<9, 6>(0, 15)
            .copy_from(&(p * kin.d_state_j * pose_to_sgal3_tangent()));
        j_full
            .fixed_view_mut::<9, 3>(0, 21)
            .copy_from(&(p * kin.d_state_j * velocity_to_state_j_tangent(&r_j, dt)));

        j_full.fixed_view_mut::<3, 3>(9, 9).copy_from(&id);
        j_full.fixed_view_mut::<3, 3>(9, 24).copy_from(&(-id));
        j_full.fixed_view_mut::<3, 3>(12, 12).copy_from(&id);
        j_full.fixed_view_mut::<3, 3>(12, 27).copy_from(&(-id));

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
            "Sgal3CombinedImuFactor expects \
             [SE3 pose_i, vel_i, bias_i, SE3 pose_j, vel_j, bias_j]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Native-state factors — time is estimated, 10D / 16D
// ─────────────────────────────────────────────────────────────────────────────

/// Non-combined SGal(3) IMU factor over native `SGal3` states.
///
/// # Residual (10D)
///
/// ```text
/// rows 0..9 : [p, q, v]           kinematics, as elsewhere
/// row  9    : (t_j − t_i) − Δt    inter-keyframe time constraint
/// ```
///
/// The time row is what distinguishes this from
/// [`Sgal3ImuFactor`]: because each keyframe's `SGal3` variable carries its own
/// time coordinate, that row constrains a quantity the optimizer can actually
/// move, which is what makes SGal(3) worth using — sensor time-offset and
/// rolling-shutter calibration. Its weight is `1/σ_t`, defaulting to
/// [`DEFAULT_TIME_SIGMA`].
///
/// ```text
/// params[0]: SGal3 state i — 11D, 10 DOF
/// params[1]: SGal3 state j — 11D, 10 DOF
/// params[2]: imu bias      — 6D, shared
/// ```
pub struct Sgal3StateImuFactor {
    preintegration: ImuPreintegration,
    time_sigma: f64,
}
preintegration_accessors!(Sgal3StateImuFactor);

impl Sgal3StateImuFactor {
    /// Create the factor with the default time-row weight.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self {
            preintegration,
            time_sigma: DEFAULT_TIME_SIGMA,
        }
    }

    /// Override the standard deviation of the inter-keyframe time constraint.
    ///
    /// Loosen it when the interval itself is being estimated (time-offset
    /// calibration); tighten it when timestamps are trusted.
    pub fn with_time_sigma(mut self, sigma: f64) -> Self {
        self.time_sigma = sigma;
        self
    }
}

impl Factor for Sgal3StateImuFactor {
    /// Columns: `[state_i(10) | state_j(10) | bias(6)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let preint = &self.preintegration;
        let state_i = SGal3::from_param_slice(params[0]);
        let state_j = SGal3::from_param_slice(params[1]);
        let (b_g, b_a) = split_bias(params[2]);

        let gc = gravity_corrected_from_state(preint, &state_i);
        let kin = sgal3_kinematics(preint, &gc, &state_j, b_g, b_a);

        let p = kinematic_rows();
        let t = time_row();
        let w_t = 1.0 / self.time_sigma;
        let sqrt_info = preint.kinematic_square_root_information();

        let mut out = SVector::<f64, 10>::zeros();
        out.fixed_rows_mut::<9>(0)
            .copy_from(&(sqrt_info * (p * kin.r_tangent)));
        out[9] = w_t * (t * kin.r_tangent)[0];
        write_residual(&out, residual);

        let Some(mut jac) = jacobian else { return };

        let lift_i = kin.d_gc_state_i * state_to_gc_tangent(preint.delta_t());
        let bias = bias_block(&kin);

        let mut j_full = SMatrix::<f64, 10, 26>::zeros();
        j_full
            .fixed_view_mut::<9, 10>(0, 0)
            .copy_from(&(sqrt_info * (p * lift_i)));
        j_full
            .fixed_view_mut::<9, 10>(0, 10)
            .copy_from(&(sqrt_info * (p * kin.d_state_j)));
        j_full
            .fixed_view_mut::<9, 6>(0, 20)
            .copy_from(&(sqrt_info * (p * bias)));
        j_full
            .fixed_view_mut::<1, 10>(9, 0)
            .copy_from(&(w_t * (t * lift_i)));
        j_full
            .fixed_view_mut::<1, 10>(9, 10)
            .copy_from(&(w_t * (t * kin.d_state_j)));
        j_full
            .fixed_view_mut::<1, 6>(9, 20)
            .copy_from(&(w_t * (t * bias)));

        write_jacobian(&j_full, &mut jac);
    }

    fn residual_dim(&self) -> usize {
        10
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (10, 26)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SGal3::REP_SIZE, SGal3::REP_SIZE, 6],
            "Sgal3StateImuFactor expects [SGal3 state_i, SGal3 state_j, bias]",
        )
    }
}

/// Combined SGal(3) IMU factor over native `SGal3` states.
///
/// # Residual (16D)
///
/// ```text
/// rows  0..9  : [p, q, v]           kinematics
/// rows  9..15 : [bg, ba]            bias random walk
/// row  15     : (t_j − t_i) − Δt    inter-keyframe time constraint
/// ```
///
/// The time row is appended rather than interleaved so the leading 15 rows keep
/// the preintegration's own `[p, q, v, bg, ba]` covariance layout and can be
/// weighted with it directly.
///
/// ```text
/// params[0]: SGal3 state i — 11D    params[2]: SGal3 state j — 11D
/// params[1]: imu bias i    — 6D     params[3]: imu bias j    — 6D
/// ```
pub struct Sgal3CombinedStateImuFactor {
    preintegration: ImuPreintegration,
    time_sigma: f64,
}
preintegration_accessors!(Sgal3CombinedStateImuFactor);

impl Sgal3CombinedStateImuFactor {
    /// Create the factor with the default time-row weight.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self {
            preintegration,
            time_sigma: DEFAULT_TIME_SIGMA,
        }
    }

    /// Override the standard deviation of the inter-keyframe time constraint.
    pub fn with_time_sigma(mut self, sigma: f64) -> Self {
        self.time_sigma = sigma;
        self
    }
}

impl Factor for Sgal3CombinedStateImuFactor {
    /// Columns: `[state_i(10) | bias_i(6) | state_j(10) | bias_j(6)]`.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let preint = &self.preintegration;
        let state_i = SGal3::from_param_slice(params[0]);
        let (b_g_i, b_a_i) = split_bias(params[1]);
        let state_j = SGal3::from_param_slice(params[2]);
        let (b_g_j, b_a_j) = split_bias(params[3]);

        let gc = gravity_corrected_from_state(preint, &state_i);
        let kin = sgal3_kinematics(preint, &gc, &state_j, b_g_i, b_a_i);

        let p = kinematic_rows();
        let t = time_row();
        let w_t = 1.0 / self.time_sigma;

        let mut raw = SVector::<f64, 15>::zeros();
        raw.fixed_rows_mut::<9>(0).copy_from(&(p * kin.r_tangent));
        raw.fixed_rows_mut::<6>(9)
            .copy_from(&bias_walk_residual(b_g_i, b_a_i, b_g_j, b_a_j));
        let sqrt_info = preint.square_root_information();

        let mut out = SVector::<f64, 16>::zeros();
        out.fixed_rows_mut::<15>(0).copy_from(&(sqrt_info * raw));
        out[15] = w_t * (t * kin.r_tangent)[0];
        write_residual(&out, residual);

        let Some(mut jac) = jacobian else { return };

        let lift_i = kin.d_gc_state_i * state_to_gc_tangent(preint.delta_t());
        let bias = bias_block(&kin);
        let id = Matrix3::identity();

        let mut raw_j = SMatrix::<f64, 15, 32>::zeros();
        raw_j.fixed_view_mut::<9, 10>(0, 0).copy_from(&(p * lift_i));
        raw_j.fixed_view_mut::<9, 6>(0, 10).copy_from(&(p * bias));
        raw_j
            .fixed_view_mut::<9, 10>(0, 16)
            .copy_from(&(p * kin.d_state_j));
        raw_j.fixed_view_mut::<3, 3>(9, 10).copy_from(&id);
        raw_j.fixed_view_mut::<3, 3>(9, 26).copy_from(&(-id));
        raw_j.fixed_view_mut::<3, 3>(12, 13).copy_from(&id);
        raw_j.fixed_view_mut::<3, 3>(12, 29).copy_from(&(-id));

        let mut j_full = SMatrix::<f64, 16, 32>::zeros();
        j_full
            .fixed_view_mut::<15, 32>(0, 0)
            .copy_from(&(sqrt_info * raw_j));
        j_full
            .fixed_view_mut::<1, 10>(15, 0)
            .copy_from(&(w_t * (t * lift_i)));
        j_full
            .fixed_view_mut::<1, 6>(15, 10)
            .copy_from(&(w_t * (t * bias)));
        j_full
            .fixed_view_mut::<1, 10>(15, 16)
            .copy_from(&(w_t * (t * kin.d_state_j)));

        write_jacobian(&j_full, &mut jac);
    }

    fn residual_dim(&self) -> usize {
        16
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (16, 32)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SGal3::REP_SIZE, 6, SGal3::REP_SIZE, 6],
            "Sgal3CombinedStateImuFactor expects \
             [SGal3 state_i, bias_i, SGal3 state_j, bias_j]",
        )
    }
}
