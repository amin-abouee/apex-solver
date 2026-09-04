//! The two SGal(3) IMU factors.
//!
//! The same two factors as [`se23`](crate::factors::imu::se23), expressed
//! on the Special Galilean group. An [`SGal3`] element is `(R, t, v, s)` — a
//! navigation state *plus a time coordinate* — so a keyframe carries its own
//! timestamp as an estimated quantity and the residual gains a tenth row,
//! `(t_j − t_i) − Δt`.
//!
//! | Factor | Residual | Blocks |
//! |---|---|---|
//! | [`ImuFactor`] | 10D `[ρ, ν, θ, s]` | `(SGal3, SGal3, bias)` |
//! | [`CombinedImuFactor`] | 16D `[ρ, ν, θ, s, bg, ba]` | `(SGal3, bias, SGal3, bias)` |
//!
//! That extra row is the reason to pick SGal(3) over SE_2(3): it makes the
//! inter-keyframe interval something the optimizer can move, which is what
//! sensor time-offset and rolling-shutter calibration need. Weight it through
//! [`ImuFactor::with_time_sigma`]; the default assumes trusted timestamps.
//!
//! Bias handling matches SE_2(3): [`ImuFactor`] shares one bias and needs a
//! companion [`bias_random_walk`](crate::factors::imu::bias::bias_random_walk)
//! edge, [`CombinedImuFactor`] embeds the walk and needs none.
//!
//! # Known limitation: timestamps must be interval-relative
//!
//! SGal(3)'s group law is `t = R₁·(t₂ + s₁·v₂) + t₁` — the **left** operand's
//! time coordinate couples the right operand's velocity into translation. The
//! residual composes `gc_i⁻¹ ∘ state_j`, so it depends on the absolute `s_i`
//! and not only on `s_j − s_i`, while the preintegrated delta it is compared
//! against corresponds to `s_i = 0`.
//!
//! In practice: a **single interval** with `s_i = 0` and `s_j = Δt` is correct
//! (that is what the tests cover), but chaining these factors across keyframes
//! carrying a common absolute clock is **not** — the residual picks up a
//! spurious `s_i`-dependent translation term that grows with the timestamp.
//! Making the residual origin-invariant requires re-deriving it in terms of
//! `Δs = s_j − s_i`; until then prefer the [`se23`](super::super::se23) factors
//! for multi-keyframe chains.

use apex_manifolds::sgal3::{SGal3, SGal3Tangent};
use apex_manifolds::{LieGroup, Tangent};
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

/// A 10×10 SGal(3) Jacobian.
type Matrix10 = SMatrix<f64, 10, 10>;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_block_sizes;
use crate::factors::imu::preintegration::ImuPreintegration;
use crate::factors::imu::types::SpeedAndBiasExt;

/// Default standard deviation on the inter-keyframe time constraint [s].
///
/// 100 µs is the scale of timestamp jitter on a synchronized IMU. Loosen it
/// when the interval itself is being estimated.
pub const DEFAULT_TIME_SIGMA: f64 = 1.0e-4;

/// One evaluation of the interval, in `SGal3` tangent space `[ρ, ν, θ, s]`.
struct Interval {
    /// Unweighted 10D residual.
    residual: SVector<f64, 10>,
    /// `∂r/∂state_i`.
    d_state_i: Matrix10,
    /// `∂r/∂state_j`.
    d_state_j: Matrix10,
    /// `∂r/∂[b_g, b_a]`, through the first-order bias correction.
    d_bias: SMatrix<f64, 10, 6>,
}

/// Evaluate the interval residual and its `SGal3`-tangent derivatives.
///
/// Identical in shape to the SE_2(3) case, with the gravity-corrected state
/// keeping `state_i`'s own time coordinate so the time row stays informative.
fn evaluate(
    preint: &ImuPreintegration,
    state_i: &SGal3,
    state_j: &SGal3,
    b_g: Vector3<f64>,
    b_a: Vector3<f64>,
) -> Interval {
    let dt = preint.delta_t();
    let gravity = Vector3::new(0.0, 0.0, preint.imu_params().g);
    let v_i = state_i.velocity();

    let gc_i = SGal3::new(
        state_i.translation() + v_i * dt - 0.5 * gravity * dt * dt,
        v_i - gravity * dt,
        state_i.rotation_quaternion(),
        state_i.time(),
    );
    let predicted = gc_i.inverse(None).compose(state_j, None, None);

    let reference = preint.speed_and_biases_ref();
    let db_g = b_g - reference.gyro_bias();
    let db_a = b_a - reference.accel_bias();
    let correction = SGal3Tangent::new(
        preint.dp_db_g() * db_g - preint.c_doubleintegral() * db_a,
        preint.dv_db_g() * db_g - preint.c_integral() * db_a,
        -preint.dalpha_db_g() * db_g,
        0.0,
    );

    let mut d_delta_d_correction = Matrix10::zeros();
    let delta = preint
        .delta_sgal3()
        .right_plus(&correction, None, Some(&mut d_delta_d_correction));

    let mut d_r_d_predicted = Matrix10::zeros();
    let mut d_r_d_delta = Matrix10::zeros();
    let tangent = predicted.right_minus(&delta, Some(&mut d_r_d_predicted), Some(&mut d_r_d_delta));

    let d_predicted_d_gc = -predicted.inverse(None).adjoint();

    // (δρ, δν, δθ, δs) ↦ (δρ + Δt·δν, δν, δθ, δs): velocity feeds position,
    // everything else — the time coordinate included — passes through.
    let mut d_gc_d_state_i = Matrix10::identity();
    d_gc_d_state_i
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(Matrix3::identity() * dt));

    // The correction enters as [ρ, ν, θ] blocks; it does not touch time.
    let mut d_correction_d_bias = SMatrix::<f64, 10, 6>::zeros();
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(preint.dp_db_g());
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(preint.dv_db_g());
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(-preint.dalpha_db_g()));
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-preint.c_doubleintegral()));
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(-preint.c_integral()));

    Interval {
        residual: SVector::<f64, 10>::from_column_slice(tangent.as_slice()),
        d_state_i: d_r_d_predicted * d_predicted_d_gc * d_gc_d_state_i,
        d_state_j: d_r_d_predicted,
        d_bias: d_r_d_delta * d_delta_d_correction * d_correction_d_bias,
    }
}

/// Reorder an `SGal3` tangent `[ρ, ν, θ, s]` into the crate's kinematic row
/// order `[ρ, θ, ν]`, which is how the preintegration covariance is laid out.
fn kinematic_rows() -> SMatrix<f64, 9, 10> {
    let mut p = SMatrix::<f64, 9, 10>::zeros();
    let id = Matrix3::identity();
    p.fixed_view_mut::<3, 3>(0, 0).copy_from(&id); // ρ
    p.fixed_view_mut::<3, 3>(3, 6).copy_from(&id); // θ
    p.fixed_view_mut::<3, 3>(6, 3).copy_from(&id); // ν
    p
}

fn split_bias(block: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(block[0], block[1], block[2]),
        Vector3::new(block[3], block[4], block[5]),
    )
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

// ─────────────────────────────────────────────────────────────────────────────

/// IMU factor over two `SGal3` states with a shared bias.
///
/// # Residual (10D)
///
/// ```text
/// rows 0..9 : [ρ, θ, ν]           kinematics, weighted by the 9×9 information
/// row  9    : (t_j − t_i) − Δt    time constraint, weighted by 1/σ_t
/// ```
///
/// # Parameter layout (3 blocks, 26 minimal DOF)
///
/// ```text
/// params[0]: SGal3 state i — 11D, 10 DOF
/// params[1]: SGal3 state j — 11D, 10 DOF
/// params[2]: imu bias      — 6D [bg, ba], shared by both keyframes
/// ```
pub struct ImuFactor {
    preintegration: ImuPreintegration,
    time_sigma: f64,
}

impl ImuFactor {
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

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for ImuFactor {
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

        let interval = evaluate(preint, &state_i, &state_j, b_g, b_a);
        let rows = kinematic_rows();
        let sqrt_info = preint.kinematic_square_root_information();
        let w_time = 1.0 / self.time_sigma;

        let mut out = SVector::<f64, 10>::zeros();
        out.fixed_rows_mut::<9>(0)
            .copy_from(&(sqrt_info * (rows * interval.residual)));
        out[9] = w_time * interval.residual[9];
        residual.copy_from_slice(out.as_slice());

        let Some(mut jac) = jacobian else { return };

        let mut full = SMatrix::<f64, 10, 26>::zeros();
        full.fixed_view_mut::<9, 10>(0, 0)
            .copy_from(&(sqrt_info * (rows * interval.d_state_i)));
        full.fixed_view_mut::<9, 10>(0, 10)
            .copy_from(&(sqrt_info * (rows * interval.d_state_j)));
        full.fixed_view_mut::<9, 6>(0, 20)
            .copy_from(&(sqrt_info * (rows * interval.d_bias)));
        full.fixed_view_mut::<1, 10>(9, 0)
            .copy_from(&(interval.d_state_i.row(9) * w_time));
        full.fixed_view_mut::<1, 10>(9, 10)
            .copy_from(&(interval.d_state_j.row(9) * w_time));
        full.fixed_view_mut::<1, 6>(9, 20)
            .copy_from(&(interval.d_bias.row(9) * w_time));

        write_jacobian(&full, &mut jac);
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
            "ImuFactor expects [SGal3 state_i, SGal3 state_j, bias]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// IMU factor over two `SGal3` states with a bias per keyframe.
///
/// # Residual (16D)
///
/// ```text
/// rows  0..15 : [ρ, θ, ν, bg, ba]   kinematics + bias walk, 15×15 information
/// row  15     : (t_j − t_i) − Δt    time constraint, weighted by 1/σ_t
/// ```
///
/// The time row is appended rather than interleaved so the leading fifteen rows
/// keep the preintegration's own `[p, q, v, bg, ba]` layout and can be weighted
/// with its information matrix directly.
///
/// # Parameter layout (4 blocks, 32 minimal DOF)
///
/// ```text
/// params[0]: SGal3 state i — 11D, 10 DOF   params[2]: SGal3 state j — 11D
/// params[1]: imu bias i    — 6D            params[3]: imu bias j    — 6D
/// ```
pub struct CombinedImuFactor {
    preintegration: ImuPreintegration,
    time_sigma: f64,
}

impl CombinedImuFactor {
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

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for CombinedImuFactor {
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

        let interval = evaluate(preint, &state_i, &state_j, b_g_i, b_a_i);
        let rows = kinematic_rows();
        let sqrt_info = preint.square_root_information();
        let w_time = 1.0 / self.time_sigma;

        let mut raw = SVector::<f64, 15>::zeros();
        raw.fixed_rows_mut::<9>(0)
            .copy_from(&(rows * interval.residual));
        raw.fixed_rows_mut::<3>(9).copy_from(&(b_g_i - b_g_j));
        raw.fixed_rows_mut::<3>(12).copy_from(&(b_a_i - b_a_j));

        let mut out = SVector::<f64, 16>::zeros();
        out.fixed_rows_mut::<15>(0).copy_from(&(sqrt_info * raw));
        out[15] = w_time * interval.residual[9];
        residual.copy_from_slice(out.as_slice());

        let Some(mut jac) = jacobian else { return };

        let identity = Matrix3::identity();
        let mut raw_jac = SMatrix::<f64, 15, 32>::zeros();
        raw_jac
            .fixed_view_mut::<9, 10>(0, 0)
            .copy_from(&(rows * interval.d_state_i));
        raw_jac
            .fixed_view_mut::<9, 6>(0, 10)
            .copy_from(&(rows * interval.d_bias));
        raw_jac
            .fixed_view_mut::<9, 10>(0, 16)
            .copy_from(&(rows * interval.d_state_j));
        raw_jac.fixed_view_mut::<3, 3>(9, 10).copy_from(&identity);
        raw_jac
            .fixed_view_mut::<3, 3>(9, 26)
            .copy_from(&(-identity));
        raw_jac.fixed_view_mut::<3, 3>(12, 13).copy_from(&identity);
        raw_jac
            .fixed_view_mut::<3, 3>(12, 29)
            .copy_from(&(-identity));

        let mut full = SMatrix::<f64, 16, 32>::zeros();
        full.fixed_view_mut::<15, 32>(0, 0)
            .copy_from(&(sqrt_info * raw_jac));
        full.fixed_view_mut::<1, 10>(15, 0)
            .copy_from(&(interval.d_state_i.row(9) * w_time));
        full.fixed_view_mut::<1, 6>(15, 10)
            .copy_from(&(interval.d_bias.row(9) * w_time));
        full.fixed_view_mut::<1, 10>(15, 16)
            .copy_from(&(interval.d_state_j.row(9) * w_time));

        write_jacobian(&full, &mut jac);
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
            "CombinedImuFactor expects [SGal3 state_i, bias_i, SGal3 state_j, bias_j]",
        )
    }
}
