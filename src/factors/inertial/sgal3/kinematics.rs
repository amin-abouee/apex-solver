//! The SGal(3) preintegration kernel shared by every SGal(3) IMU factor.
//!
//! Mirrors [`se23::kinematics`](crate::factors::inertial::se23::kinematics),
//! but expresses the interval on the Special Galilean group, whose tangent
//! `[ρ, ν, θ, s]` carries a **time** coordinate alongside position, velocity
//! and rotation:
//!
//! ```text
//! gc_state_i = SGal3(p_i + v_i·Δt − ½g·Δt², v_i − g·Δt, q_i, s_i)
//! predicted  = gc_state_i⁻¹ ∘ state_j              (its s is s_j − s_i)
//! r          = predicted ⊟ (Δ ⊞ correction(b_i))   ∈ R¹⁰
//! ```
//!
//! The time row is `(s_j − s_i) − Δt`. Whether it carries information depends
//! entirely on the factor's variables: with `(SE3 pose, R³ velocity)` blocks
//! nothing represents time, so `s_i` and `s_j` are constants and the row is
//! identically zero — those factors drop it. The native-`SGal3` factors take
//! the time coordinate as part of an estimated state, so for them the row is a
//! genuine constraint on the inter-keyframe interval.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::sgal3::{SGal3, SGal3Tangent};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::factors::inertial::preintegration::ImuPreintegration;
use crate::factors::inertial::types::SpeedAndBiasExt;

/// A 10×10 SGal(3) Jacobian.
pub(crate) type Matrix10 = SMatrix<f64, 10, 10>;

/// The shared SGal(3) residual and its tangent-space derivatives.
pub(crate) struct Sgal3Kinematics {
    /// Unweighted residual in SGal(3) tangent order `[ρ(3), ν(3), θ(3), s(1)]`.
    pub r_tangent: SVector<f64, 10>,
    /// `∂r / ∂(gravity-corrected state i)`, in SGal(3) tangent space.
    pub d_gc_state_i: Matrix10,
    /// `∂r / ∂(state j)`, in SGal(3) tangent space.
    pub d_state_j: Matrix10,
    /// `∂r / ∂b_g` through the first-order bias correction.
    pub d_bias_gyro: SMatrix<f64, 10, 3>,
    /// `∂r / ∂b_a` through the first-order bias correction.
    pub d_bias_accel: SMatrix<f64, 10, 3>,
}

/// Reorder an SGal(3) tangent `[ρ, ν, θ, s]` into the crate's kinematic row
/// order `[ρ, θ, ν]`, which is what the preintegration covariance is laid out
/// in (`[p, q, v]`). Dropping the time row is the caller's choice.
pub(crate) fn kinematic_rows() -> SMatrix<f64, 9, 10> {
    let mut p = SMatrix::<f64, 9, 10>::zeros();
    let id = Matrix3::identity();
    p.fixed_view_mut::<3, 3>(0, 0).copy_from(&id); // ρ → rows 0..3
    p.fixed_view_mut::<3, 3>(3, 6).copy_from(&id); // θ → rows 3..6
    p.fixed_view_mut::<3, 3>(6, 3).copy_from(&id); // ν → rows 6..9
    p
}

/// Extract the time row of an SGal(3) tangent.
pub(crate) fn time_row() -> SMatrix<f64, 1, 10> {
    let mut p = SMatrix::<f64, 1, 10>::zeros();
    p[(0, 9)] = 1.0;
    p
}

/// Gravity-corrected state at frame `i` from `(SE3 pose, R³ velocity)` blocks.
///
/// The time coordinate is 0 here and `Δt` at frame j, matching the
/// preintegrated delta so the time row cancels — these blocks carry no time
/// variable to constrain.
pub(crate) fn gravity_corrected_from_pose(
    preint: &ImuPreintegration,
    pose_i: &SE3,
    v_i: Vector3<f64>,
) -> SGal3 {
    let dt = preint.delta_t();
    let g = Vector3::new(0.0, 0.0, preint.imu_params().g);
    SGal3::new(
        pose_i.translation() + v_i * dt - 0.5 * g * dt * dt,
        v_i - g * dt,
        pose_i.rotation_quaternion(),
        0.0,
    )
}

/// State at frame `j` from `(SE3 pose, R³ velocity)` blocks, carrying `s = Δt`.
pub(crate) fn state_j_from_pose(
    preint: &ImuPreintegration,
    pose_j: &SE3,
    v_j: Vector3<f64>,
) -> SGal3 {
    SGal3::new(
        pose_j.translation(),
        v_j,
        pose_j.rotation_quaternion(),
        preint.delta_t(),
    )
}

/// Gravity-corrected state at frame `i` from a native `SGal3` state, preserving
/// the state's own time coordinate so the time row stays informative.
pub(crate) fn gravity_corrected_from_state(preint: &ImuPreintegration, state_i: &SGal3) -> SGal3 {
    let dt = preint.delta_t();
    let g = Vector3::new(0.0, 0.0, preint.imu_params().g);
    let v_i = state_i.velocity();
    SGal3::new(
        state_i.translation() + v_i * dt - 0.5 * g * dt * dt,
        v_i - g * dt,
        state_i.rotation_quaternion(),
        state_i.time(),
    )
}

/// Evaluate the shared SGal(3) residual and its tangent derivatives.
pub(crate) fn sgal3_kinematics(
    preint: &ImuPreintegration,
    gc_state_i: &SGal3,
    state_j: &SGal3,
    b_g_i: Vector3<f64>,
    b_a_i: Vector3<f64>,
) -> Sgal3Kinematics {
    let predicted = gc_state_i.inverse(None).compose(state_j, None, None);

    let b_ref = preint.speed_and_biases_ref();
    let db_g = b_g_i - b_ref.gyro_bias();
    let db_a = b_a_i - b_ref.accel_bias();

    let corr_rho = preint.dp_db_g() * db_g - preint.c_doubleintegral() * db_a;
    let corr_nu = preint.dv_db_g() * db_g - preint.c_integral() * db_a;
    let corr_theta = -preint.dalpha_db_g() * db_g;
    let correction = SGal3Tangent::new(corr_rho, corr_nu, corr_theta, 0.0);

    let mut jac_rp_tangent = Matrix10::zeros();
    let delta_corrected =
        preint
            .delta_sgal3()
            .right_plus(&correction, None, Some(&mut jac_rp_tangent));

    let mut jac_rm_pred = Matrix10::zeros();
    let mut jac_rm_dc = Matrix10::zeros();
    let r = predicted.right_minus(
        &delta_corrected,
        Some(&mut jac_rm_pred),
        Some(&mut jac_rm_dc),
    );

    let j_rkin_dc = jac_rm_dc * jac_rp_tangent;

    // Bias-correction embeddings, in SGal(3) tangent order [ρ, ν, θ, s].
    let mut e_bg = SMatrix::<f64, 10, 3>::zeros();
    e_bg.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(preint.dp_db_g());
    e_bg.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(preint.dv_db_g());
    e_bg.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(-preint.dalpha_db_g()));

    let mut e_ba = SMatrix::<f64, 10, 3>::zeros();
    e_ba.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-preint.c_doubleintegral()));
    e_ba.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(-preint.c_integral()));

    Sgal3Kinematics {
        r_tangent: SVector::<f64, 10>::from_iterator(
            r.rho()
                .iter()
                .chain(r.nu().iter())
                .chain(r.theta().iter())
                .copied()
                .chain(std::iter::once(r.s())),
        ),
        d_gc_state_i: jac_rm_pred * (-predicted.inverse(None).adjoint()),
        d_state_j: jac_rm_pred,
        d_bias_gyro: j_rkin_dc * e_bg,
        d_bias_accel: j_rkin_dc * e_ba,
    }
}

/// Lift an SE(3) pose tangent into the SGal(3) tangent: `ρ → ρ`, `θ → θ`.
pub(crate) fn pose_to_sgal3_tangent() -> SMatrix<f64, 10, 6> {
    let mut j = SMatrix::<f64, 10, 6>::zeros();
    let id = Matrix3::identity();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&id);
    j.fixed_view_mut::<3, 3>(6, 3).copy_from(&id);
    j
}

/// Lift a world-frame velocity perturbation into the SGal(3) tangent at the
/// gravity-corrected state i, where `δv` moves position through `v·Δt`.
pub(crate) fn velocity_to_gc_tangent(rotation_i: &Matrix3<f64>, dt: f64) -> SMatrix<f64, 10, 3> {
    let r_t = rotation_i.transpose();
    let mut j = SMatrix::<f64, 10, 3>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&(r_t * dt));
    j.fixed_view_mut::<3, 3>(3, 0).copy_from(&r_t);
    j
}

/// Lift a world-frame velocity perturbation into the SGal(3) tangent at state
/// j.
///
/// SGal(3)-specific: state j carries `s = Δt`, and the group law couples time
/// into translation (`x ↦ Rx + t + s·v`), so a velocity perturbation must be
/// offset by `−Δt·R_jᵀδv` in `ρ` to leave the composition's translation fixed.
pub(crate) fn velocity_to_state_j_tangent(
    rotation_j: &Matrix3<f64>,
    dt: f64,
) -> SMatrix<f64, 10, 3> {
    let r_t = rotation_j.transpose();
    let mut j = SMatrix::<f64, 10, 3>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-dt * r_t));
    j.fixed_view_mut::<3, 3>(3, 0).copy_from(&r_t);
    j
}

/// Lift a native `SGal3` state-i tangent into the gravity-corrected state's
/// tangent: velocity feeds position through `Δt`, everything else passes
/// through, including the time coordinate.
pub(crate) fn state_to_gc_tangent(dt: f64) -> Matrix10 {
    let mut j = Matrix10::identity();
    j.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(Matrix3::identity() * dt));
    j
}
