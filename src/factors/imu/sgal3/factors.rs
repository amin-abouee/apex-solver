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
//! # Absolute timestamps are supported (formerly ISSUE-0009)
//!
//! `SGal(3)`'s group law is `t = R₁·t₂ + t₁ + v₁·s₂` — the **left** operand's
//! velocity couples the right operand's time into translation. `evaluate`
//! builds its gravity-corrected `gc_i` from `state_i` alone and composes
//! `gc_i⁻¹ ∘ state_j`, so a naive construction would pick up a coupling
//! artifact of `gc_i⁻¹`'s velocity times `state_j.time() − state_i.time()` —
//! real per the group law, but not part of the standard kinematic
//! prediction, and it does not vanish just because `state_i.time() ≈ 0`
//! (only `state_j.time() = state_i.time()` would zero it, which is never
//! true for a real interval). An explicit `correction_element`, composed on
//! the left of `gc_i`, cancels it exactly for any `state_i.time()` — not
//! only near zero — so keyframes on a shared absolute clock (not just
//! interval-relative timestamps starting at `state_i.time() = 0`) are a
//! valid registration. See `residuals_vanish_at_ground_truth_with_nonzero_
//! absolute_time` and the Jacobian-FD counterpart in `tests.rs`.

use apex_manifolds::sgal3::{SGal3, SGal3Tangent};
use apex_manifolds::{LieGroup, Tangent};
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};

/// A 10×10 SGal(3) Jacobian.
type Matrix10 = SMatrix<f64, 10, 10>;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;
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
    let v_gc = v_i - gravity * dt;
    let delta_s = state_j.time() - state_i.time();

    // `gc_i_uncorrected` folds gravity into state_i's own translation/
    // velocity so the comparison against the preintegrated delta is a plain
    // right-minus (see the SE_2(3) sibling). On `SGal3`, `compose`'s `s·v`
    // coupling is real: composing `gc_i.inverse()` (time = state_i.time(), so
    // its own velocity `v_gc` is nonzero) against `state_j` (time =
    // state_j.time(), generally ≠ state_i.time()) would pick up a spurious
    // `v_gc·Δs` term on top of the standard kinematic prediction, where
    // `Δs = state_j.time() - state_i.time()`.
    //
    // `correction` (translation `-v_gc·Δs`, everything else identity) cancels
    // it exactly — composed on the *left*, unrotated, it shifts
    // `gc_i_uncorrected`'s translation by `-v_gc·Δs` with no `R_i` factor,
    // matching the sign of the artifact. Left composition also keeps
    // `correction`'s own rotation at `I`, which is what makes its Jacobian
    // w.r.t. `state_i`/`state_j` a plain (unrotated) vector derivative below,
    // instead of needing a hand-derived raw-to-tangent conversion through
    // `R_i` (an earlier version of this fix got exactly that conversion
    // wrong: it forgot `gc_i`'s own time coordinate shifts at the same time
    // as its translation under a `state_i` time-DOF perturbation, so `R_i^T`
    // alone wasn't the right map — using `compose`'s own Jacobian convention
    // here instead sidesteps re-deriving it by hand).
    //
    // The time row is untouched: composing `gc_i` (time = state_i.time())
    // with `state_j` (time = state_j.time()) still gives
    // `state_j.time() - state_i.time()`, compared against `delta_sgal3()`'s
    // `Δt`.
    //
    // `gc_i_uncorrected` is itself built as `state_i.compose(&flow)` rather
    // than a raw `SGal3::new` with a hand-added translation/velocity offset.
    // The two give numerically identical results (verified), but only the
    // composition form has a Jacobian obtainable from `compose`'s own
    // machinery: `state_i`'s own `s`-tangent shifts its raw translation by
    // `v_i·δs` (the same real `SGal3` time coupling this file works around
    // above, now in the *variable's* own manifold structure) *while its own
    // time also shifts by `δs`* — with `flow`'s velocity component (`-g·Δt`,
    // large relative to the orbital scale here) nonzero, those two
    // simultaneous shifts interact through `right_minus`'s nonlinear log map
    // in a way a hand-derived "raw shift, converted by `R_i^T`" formula does
    // not capture (an earlier version of this fix got exactly that wrong).
    // `flow` has rotation `I` and time `0`, so its own dependence on
    // `state_i` (through `R_i`, `v_i`) needs no such conversion.
    let r_i = state_i.rotation_matrix();
    let flow = SGal3::new(
        r_i.transpose() * (v_i * dt - 0.5 * gravity * dt * dt),
        r_i.transpose() * (-gravity * dt),
        UnitQuaternion::identity(),
        0.0,
    );
    let mut jac_state_i_direct = Matrix10::zeros();
    let mut jac_flow = Matrix10::zeros();
    let gc_i_uncorrected =
        state_i.compose(&flow, Some(&mut jac_state_i_direct), Some(&mut jac_flow));

    let correction_element = SGal3::new(
        -v_gc * delta_s,
        Vector3::zeros(),
        UnitQuaternion::identity(),
        0.0,
    );
    let mut jac_correction = Matrix10::zeros();
    let mut jac_gc_uncorrected = Matrix10::zeros();
    let gc_i = correction_element.compose(
        &gc_i_uncorrected,
        Some(&mut jac_correction),
        Some(&mut jac_gc_uncorrected),
    );
    let gc_i_inv = gc_i.inverse(None);
    let predicted = gc_i_inv.compose(state_j, None, None);

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

    // `flow`'s translation/velocity are `R_i^T·(v_i·Δt − ½g·Δt²)` and
    // `-R_i^T·g·Δt` — plain vector formulas in `state_i`'s raw `R_i`, `v_i`,
    // with `flow`'s own rotation fixed at `I` (no further rotation
    // conversion needed for *its* tangent, same reasoning as
    // `correction_element` below). `state_i`'s raw `v_i` shifts by `R_i·δν_i`
    // under its own ν-tangent (the `R_i`/`R_i^T` cancel, leaving `Δt·I`); its
    // raw `R_i` shifts by `R_i·Exp(δθ_i)` under its own θ-tangent, giving the
    // standard `skew(R_i^T·X)` derivative of `R_i^T·X` for each of `flow`'s
    // two vector fields.
    let mut d_flow_d_state_i = Matrix10::zeros();
    d_flow_d_state_i
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(Matrix3::identity() * dt));
    d_flow_d_state_i
        .fixed_view_mut::<3, 3>(0, 6)
        .copy_from(&skew(&flow.translation()));
    d_flow_d_state_i
        .fixed_view_mut::<3, 3>(3, 6)
        .copy_from(&skew(&flow.velocity()));
    let d_gc_uncorrected_d_state_i = jac_state_i_direct + jac_flow * d_flow_d_state_i;

    // `correction_element`'s translation is `-v_gc·(s_j - s_i)`, a plain
    // vector formula in the *raw* (v_i, s_i, s_j), with `correction_element`'s
    // own rotation fixed at `I` — so unlike `gc_i`/`flow`, no rotation
    // conversion is needed between a raw-parameter shift and its own
    // tangent-ρ. `state_i`'s raw `v_i` shifts by `R_i·δν_i` under its own
    // ν-tangent, and its raw `s_i` shifts by `δs_i` directly under its own
    // s-tangent (ordinary `SGal3` manifold coupling, independent of this
    // fix); `state_j`'s raw `s_j` shifts by `δs_j` directly under its own
    // s-tangent.
    let mut d_correction_d_state_i = Matrix10::zeros();
    d_correction_d_state_i
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-delta_s * r_i));
    d_correction_d_state_i
        .fixed_view_mut::<3, 1>(0, 9)
        .copy_from(&v_gc);

    let mut d_correction_d_state_j = Matrix10::zeros();
    d_correction_d_state_j
        .fixed_view_mut::<3, 1>(0, 9)
        .copy_from(&(-v_gc));

    let d_gc_d_state_i =
        jac_gc_uncorrected * d_gc_uncorrected_d_state_i + jac_correction * d_correction_d_state_i;
    let d_gc_d_state_j = jac_correction * d_correction_d_state_j;

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
        // `state_j` enters both directly (as `compose`'s right operand, the
        // existing `d_r_d_predicted` term) and indirectly through `gc_i`'s
        // new dependency on `state_j.time()`.
        d_state_j: d_r_d_predicted + d_r_d_predicted * d_predicted_d_gc * d_gc_d_state_j,
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
