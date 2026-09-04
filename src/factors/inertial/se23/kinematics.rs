//! The SE_2(3) preintegration kernel shared by every SE_2(3) IMU factor.
//!
//! All four factors in [`super::factors`] evaluate the *same* on-manifold
//! residual (Forster et al. 2017) and differ only in how the state is split
//! across parameter blocks. That shared part lives here, expressed purely in
//! SE_2(3) tangent space:
//!
//! ```text
//! gc_state_i = SE23(p_i + v_i·Δt − ½g·Δt², v_i − g·Δt, q_i)   (gravity-corrected)
//! predicted  = gc_state_i⁻¹ ∘ state_j
//! r_kin      = predicted ⊟ (Δ ⊞ correction(b_i))              ∈ R⁹  [ρ, θ, ν]
//! ```
//!
//! Each factor then lifts [`Se23Kinematics`]'s tangent-space derivatives into
//! its own parameter blocks — a per-factor concern, because a factor over
//! `(SE3 pose, R³ velocity)` and one over a native `SE23` state reach the same
//! tangent through different chains.

use apex_manifolds::LieGroup;
use apex_manifolds::se23::{SE23, SE23Tangent};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::factors::inertial::preintegration::ImuPreintegration;

/// The shared SE_2(3) residual and its tangent-space derivatives.
///
/// Every derivative is with respect to an SE_2(3) *tangent*, not a parameter
/// block, so it is independent of how a factor groups its variables.
pub(crate) struct Se23Kinematics {
    /// Unweighted 9D kinematic residual `[ρ(3), θ(3), ν(3)]`.
    pub r_kin: SVector<f64, 9>,
    /// `∂r_kin / ∂(gravity-corrected state i)`, in SE_2(3) tangent space.
    pub d_gc_state_i: SMatrix<f64, 9, 9>,
    /// `∂r_kin / ∂(state j)`, in SE_2(3) tangent space.
    pub d_state_j: SMatrix<f64, 9, 9>,
    /// `∂r_kin / ∂b_g` through the first-order bias correction.
    pub d_bias_gyro: SMatrix<f64, 9, 3>,
    /// `∂r_kin / ∂b_a` through the first-order bias correction.
    pub d_bias_accel: SMatrix<f64, 9, 3>,
}

/// Build the gravity-corrected state at frame `i`.
///
/// Folding gravity into the state (rather than the residual) is what lets the
/// comparison against the preintegrated delta be a plain SE_2(3) right-minus.
pub(crate) fn gravity_corrected_state_i(
    preint: &ImuPreintegration,
    pose_i: &apex_manifolds::se3::SE3,
    v_i: Vector3<f64>,
) -> SE23 {
    let dt = preint.delta_t();
    let g_vec = Vector3::new(0.0, 0.0, preint.imu_params().g);
    let gc_p = pose_i.translation() + v_i * dt - 0.5 * g_vec * dt * dt;
    let gc_v = v_i - g_vec * dt;
    SE23::new(gc_p, gc_v, pose_i.rotation_quaternion())
}

/// Build the gravity-corrected state at frame `i` from a native `SE23` state.
pub(crate) fn gravity_corrected_state_from_se23(
    preint: &ImuPreintegration,
    state_i: &SE23,
) -> SE23 {
    let dt = preint.delta_t();
    let g_vec = Vector3::new(0.0, 0.0, preint.imu_params().g);
    let v_i = state_i.velocity();
    SE23::new(
        state_i.translation() + v_i * dt - 0.5 * g_vec * dt * dt,
        v_i - g_vec * dt,
        state_i.rotation_quaternion(),
    )
}

/// Lift an `SE23` state-i tangent into the gravity-corrected state's tangent.
///
/// `gc_state_i` shares `state_i`'s rotation and shifts translation/velocity by
/// gravity terms only, so to first order `(δρ, δθ, δν) ↦ (δρ + Δt·δν, δθ, δν)`
/// — a constant matrix, independent of the linearization point.
pub(crate) fn state_to_gc_tangent(dt: f64) -> SMatrix<f64, 9, 9> {
    let mut j = SMatrix::<f64, 9, 9>::zeros();
    let id = Matrix3::identity();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&id);
    j.fixed_view_mut::<3, 3>(0, 6).copy_from(&(id * dt));
    j.fixed_view_mut::<3, 3>(3, 3).copy_from(&id);
    j.fixed_view_mut::<3, 3>(6, 6).copy_from(&id);
    j
}

/// Reference biases at the preintegration linearization point, `(b_g, b_a)`.
pub(crate) fn reference_biases(preint: &ImuPreintegration) -> (Vector3<f64>, Vector3<f64>) {
    let sb = preint.speed_and_biases_ref();
    (
        Vector3::new(sb[3], sb[4], sb[5]),
        Vector3::new(sb[6], sb[7], sb[8]),
    )
}

/// Evaluate the shared SE_2(3) kinematic residual and its tangent derivatives.
pub(crate) fn se23_kinematics(
    preint: &ImuPreintegration,
    gc_state_i: &SE23,
    state_j: &SE23,
    b_g_i: Vector3<f64>,
    b_a_i: Vector3<f64>,
) -> Se23Kinematics {
    let predicted = gc_state_i.inverse(None).compose(state_j, None, None);

    // First-order correction of the preintegrated delta for the current bias.
    let (b_g_ref, b_a_ref) = reference_biases(preint);
    let db_g = b_g_i - b_g_ref;
    let db_a = b_a_i - b_a_ref;

    let corr_rho = preint.dp_db_g() * db_g - preint.c_doubleintegral() * db_a;
    let corr_theta = -preint.dalpha_db_g() * db_g;
    let corr_nu = preint.dv_db_g() * db_g - preint.c_integral() * db_a;
    let correction = SE23Tangent::new(corr_rho, corr_theta, corr_nu);

    let mut jac_rp_tangent = SMatrix::<f64, 9, 9>::zeros();
    let delta_corrected =
        preint
            .delta_se23()
            .right_plus(&correction, None, Some(&mut jac_rp_tangent));

    let mut jac_rm_pred = SMatrix::<f64, 9, 9>::zeros();
    let mut jac_rm_dc = SMatrix::<f64, 9, 9>::zeros();
    let r_kin_tangent = predicted.right_minus(
        &delta_corrected,
        Some(&mut jac_rm_pred),
        Some(&mut jac_rm_dc),
    );

    // ∂predicted/∂(gc_state_i) in tangent space.
    let jac_pred_wrt_gc: SMatrix<f64, 9, 9> = -predicted.inverse(None).adjoint();
    let j_rkin_dc = jac_rm_dc * jac_rp_tangent;

    let mut d_bias_gyro = SMatrix::<f64, 9, 3>::zeros();
    d_bias_gyro
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(preint.dp_db_g());
    d_bias_gyro
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(-preint.dalpha_db_g()));
    d_bias_gyro
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(preint.dv_db_g());

    let mut d_bias_accel = SMatrix::<f64, 9, 3>::zeros();
    d_bias_accel
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-preint.c_doubleintegral()));
    d_bias_accel
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(-preint.c_integral()));

    Se23Kinematics {
        r_kin: se23_tangent_data(&r_kin_tangent),
        d_gc_state_i: jac_rm_pred * jac_pred_wrt_gc,
        d_state_j: jac_rm_pred,
        d_bias_gyro: j_rkin_dc * d_bias_gyro,
        d_bias_accel: j_rkin_dc * d_bias_accel,
    }
}

/// Lift an SE(3) pose tangent `(δρ, δθ)` into the SE_2(3) tangent at a state
/// whose position and rotation come from that pose: `[ρ→ρ, θ→θ, ν untouched]`.
pub(crate) fn pose_to_se23_tangent() -> SMatrix<f64, 9, 6> {
    let mut j = SMatrix::<f64, 9, 6>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&Matrix3::identity());
    j.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&Matrix3::identity());
    j
}

/// Lift a world-frame velocity perturbation into the SE_2(3) tangent at the
/// **gravity-corrected** state i, where `δv` moves both position (through
/// `v·Δt`) and velocity.
pub(crate) fn velocity_to_gc_tangent(rotation_i: &Matrix3<f64>, dt: f64) -> SMatrix<f64, 9, 3> {
    let r_t = rotation_i.transpose();
    let mut j = SMatrix::<f64, 9, 3>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&(r_t * dt));
    j.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_t);
    j
}

/// Lift a world-frame velocity perturbation into the SE_2(3) tangent at state
/// j, where it only moves velocity.
pub(crate) fn velocity_to_se23_tangent(rotation_j: &Matrix3<f64>) -> SMatrix<f64, 9, 3> {
    let mut j = SMatrix::<f64, 9, 3>::zeros();
    j.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&rotation_j.transpose());
    j
}

/// Extract raw 9D data from an [`SE23Tangent`] as `[ρ(3), θ(3), ν(3)]`.
fn se23_tangent_data(t: &SE23Tangent) -> SVector<f64, 9> {
    let rho = t.rho();
    let theta = t.theta();
    let nu = t.nu();
    SVector::<f64, 9>::from_iterator(rho.iter().chain(theta.iter()).chain(nu.iter()).copied())
}
