//! The two SE_2(3) IMU factors.
//!
//! Both express an inertial interval directly on [`SE23`], the group whose
//! element *is* a navigation state `(R, t, v)`. A keyframe is one variable, and
//! the optimizer's update is an `SE23` right-plus, so the pose/velocity
//! coupling the IMU actually produces is handled by the group rather than by
//! bookkeeping across separate pose and velocity blocks.
//!
//! | Factor | Residual | Blocks |
//! |---|---|---|
//! | [`ImuFactor`] | 9D `[ρ, θ, ν]` | `(SE23, SE23, bias)` |
//! | [`CombinedImuFactor`] | 15D `[ρ, θ, ν, bg, ba]` | `(SE23, bias, SE23, bias)` |
//!
//! They differ in how bias evolution is modelled. [`ImuFactor`] shares one bias
//! variable across the interval and leaves the random walk to a separate edge
//! ([`bias_random_walk`](crate::factors::imu::bias::bias_random_walk));
//! [`CombinedImuFactor`] takes a bias per keyframe and puts the walk in its
//! trailing six rows, so it needs no companion edge. Using both at once would
//! count that uncertainty twice.
//!
//! Every derivative below comes from `SE23`'s own operations — `right_plus`,
//! `right_minus`, `compose`, `inverse`/`adjoint` all report their Jacobians in
//! the group's right convention.

use apex_manifolds::se23::{Matrix9, SE23, SE23Tangent};
use apex_manifolds::{LieGroup, Tangent};
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_block_sizes;
use crate::factors::imu::preintegration::ImuPreintegration;
use crate::factors::imu::types::SpeedAndBiasExt;

/// One evaluation of the interval, in `SE23` tangent space.
///
/// Shared by both factors: they differ only in how they weight it and which
/// columns they write it into.
struct Interval {
    /// Unweighted 9D residual `[ρ, θ, ν]`.
    residual: SVector<f64, 9>,
    /// `∂r/∂state_i`.
    d_state_i: Matrix9<f64>,
    /// `∂r/∂state_j`.
    d_state_j: Matrix9<f64>,
    /// `∂r/∂[b_g, b_a]`, through the first-order bias correction.
    d_bias: SMatrix<f64, 9, 6>,
}

/// Evaluate the interval residual and its `SE23`-tangent derivatives.
///
/// ```text
/// gc_i = SE23(t_i + v_i·Δt − ½g·Δt², v_i − g·Δt, R_i)   gravity-corrected
/// r    = (gc_i⁻¹ ∘ state_j) ⊟ (Δ ⊞ correction(b))
/// ```
///
/// Folding gravity into the state (rather than the residual) is what lets the
/// comparison against the preintegrated delta be a plain `SE23` right-minus.
fn evaluate(
    preint: &ImuPreintegration,
    state_i: &SE23,
    state_j: &SE23,
    b_g: Vector3<f64>,
    b_a: Vector3<f64>,
) -> Interval {
    let dt = preint.delta_t();
    let gravity = Vector3::new(0.0, 0.0, preint.imu_params().g);
    let v_i = state_i.velocity();

    let gc_i = SE23::new(
        state_i.translation() + v_i * dt - 0.5 * gravity * dt * dt,
        v_i - gravity * dt,
        state_i.rotation_quaternion(),
    );
    let predicted = gc_i.inverse(None).compose(state_j, None, None);

    // First-order correction of the preintegrated delta for the current bias.
    let reference = preint.speed_and_biases_ref();
    let db_g = b_g - reference.gyro_bias();
    let db_a = b_a - reference.accel_bias();
    let correction = SE23Tangent::new(
        preint.dp_db_g() * db_g - preint.c_doubleintegral() * db_a,
        -preint.dalpha_db_g() * db_g,
        preint.dv_db_g() * db_g - preint.c_integral() * db_a,
    );

    let mut d_delta_d_correction = Matrix9::zeros();
    let delta = preint
        .delta_se23()
        .right_plus(&correction, None, Some(&mut d_delta_d_correction));

    let mut d_r_d_predicted = Matrix9::zeros();
    let mut d_r_d_delta = Matrix9::zeros();
    let tangent = predicted.right_minus(&delta, Some(&mut d_r_d_predicted), Some(&mut d_r_d_delta));

    // ∂predicted/∂gc_i for the left argument of a between-style composition.
    let d_predicted_d_gc = -predicted.inverse(None).adjoint();

    // gc_i shares state_i's rotation and shifts translation/velocity by gravity
    // terms only, so to first order (δρ, δθ, δν) ↦ (δρ + Δt·δν, δθ, δν) — a
    // constant lift, independent of the linearization point.
    let mut d_gc_d_state_i = Matrix9::identity();
    d_gc_d_state_i
        .fixed_view_mut::<3, 3>(0, 6)
        .copy_from(&(Matrix3::identity() * dt));

    // The correction enters as [ρ, θ, ν] blocks per the preintegration.
    let mut d_correction_d_bias = SMatrix::<f64, 9, 6>::zeros();
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(preint.dp_db_g());
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(-preint.dalpha_db_g()));
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(preint.dv_db_g());
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-preint.c_doubleintegral()));
    d_correction_d_bias
        .fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(-preint.c_integral()));

    Interval {
        residual: SVector::<f64, 9>::from_column_slice(tangent.as_slice()),
        d_state_i: d_r_d_predicted * d_predicted_d_gc * d_gc_d_state_i,
        d_state_j: d_r_d_predicted,
        d_bias: d_r_d_delta * d_delta_d_correction * d_correction_d_bias,
    }
}

/// Read a 6D bias block as `(b_g, b_a)`.
fn split_bias(block: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(block[0], block[1], block[2]),
        Vector3::new(block[3], block[4], block[5]),
    )
}

/// Copy a weighted Jacobian into the caller's column-major buffer.
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

/// IMU factor over two `SE23` states with a shared bias.
///
/// # Residual (9D)
///
/// ```text
/// r = sqrt_info · [ ρ (3)   position
///                   θ (3)   rotation
///                   ν (3) ] velocity
/// ```
///
/// Bias enters only through the first-order correction of the preintegrated
/// delta, so one bias variable covers the interval; its evolution between
/// keyframes belongs to a
/// [`bias_random_walk`](crate::factors::imu::bias::bias_random_walk) edge.
/// Weighting therefore uses the measurement-noise-only 9×9 information — the
/// random-walk terms live in that edge instead.
///
/// # Parameter layout (3 blocks, 24 minimal DOF)
///
/// ```text
/// params[0]: SE23 state i — 10D, 9 DOF
/// params[1]: SE23 state j — 10D, 9 DOF
/// params[2]: imu bias     — 6D [bg, ba], shared by both keyframes
/// ```
pub struct ImuFactor {
    preintegration: ImuPreintegration,
}

impl ImuFactor {
    /// Create the factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for ImuFactor {
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

        let interval = evaluate(preint, &state_i, &state_j, b_g, b_a);
        let sqrt_info = preint.kinematic_square_root_information();
        residual.copy_from_slice((sqrt_info * interval.residual).as_slice());

        let Some(mut jac) = jacobian else { return };

        let mut full = SMatrix::<f64, 9, 24>::zeros();
        full.fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&interval.d_state_i);
        full.fixed_view_mut::<9, 9>(0, 9)
            .copy_from(&interval.d_state_j);
        full.fixed_view_mut::<9, 6>(0, 18)
            .copy_from(&interval.d_bias);

        write_jacobian(&(sqrt_info * full), &mut jac);
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
            "ImuFactor expects [SE23 state_i, SE23 state_j, bias]",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// IMU factor over two `SE23` states with a bias per keyframe.
///
/// # Residual (15D)
///
/// ```text
/// r = sqrt_info · [ ρ (3)     position
///                   θ (3)     rotation
///                   ν (3)     velocity
///                   r_bg (3)  gyro bias walk
///                   r_ba (3) ] accel bias walk
/// ```
///
/// The trailing six rows are the Gauss–Markov bias random walk, so this factor
/// needs no companion bias edge — and adding one would count that uncertainty
/// twice. Weighting uses the full 15×15 information, which includes the
/// random-walk covariance.
///
/// # Parameter layout (4 blocks, 30 minimal DOF)
///
/// ```text
/// params[0]: SE23 state i — 10D, 9 DOF    params[2]: SE23 state j — 10D, 9 DOF
/// params[1]: imu bias i   — 6D [bg, ba]   params[3]: imu bias j   — 6D
/// ```
pub struct CombinedImuFactor {
    preintegration: ImuPreintegration,
}

impl CombinedImuFactor {
    /// Create the factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for CombinedImuFactor {
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

        let interval = evaluate(preint, &state_i, &state_j, b_g_i, b_a_i);

        let mut raw = SVector::<f64, 15>::zeros();
        raw.fixed_rows_mut::<9>(0).copy_from(&interval.residual);
        raw.fixed_rows_mut::<3>(9).copy_from(&(b_g_i - b_g_j));
        raw.fixed_rows_mut::<3>(12).copy_from(&(b_a_i - b_a_j));

        let sqrt_info = preint.square_root_information();
        residual.copy_from_slice((sqrt_info * raw).as_slice());

        let Some(mut jac) = jacobian else { return };

        let identity = Matrix3::identity();
        let mut full = SMatrix::<f64, 15, 30>::zeros();
        full.fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&interval.d_state_i);
        full.fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&interval.d_bias);
        full.fixed_view_mut::<9, 9>(0, 15)
            .copy_from(&interval.d_state_j);
        // The kinematic rows do not depend on bias_j.
        full.fixed_view_mut::<3, 3>(9, 9).copy_from(&identity);
        full.fixed_view_mut::<3, 3>(9, 24).copy_from(&(-identity));
        full.fixed_view_mut::<3, 3>(12, 12).copy_from(&identity);
        full.fixed_view_mut::<3, 3>(12, 27).copy_from(&(-identity));

        write_jacobian(&(sqrt_info * full), &mut jac);
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
            "CombinedImuFactor expects [SE23 state_i, bias_i, SE23 state_j, bias_j]",
        )
    }
}
