//! Combined IMU factor — GTSAM-style 6-block layout.
//!
//! Unlike [`ImuFactor`] which packs velocity + bias into a single 9D
//! `SpeedAndBias` block per frame, `CombinedImuFactor` uses **separate**
//! parameter blocks for velocity (3D) and bias (6D).  This matches GTSAM's
//! `CombinedImuFactor` interface and is more convenient when:
//! - Velocity and bias are estimated by different subsystems.
//! - The bias covariance needs to be decoupled from velocity uncertainty.
//!
//! # Residual layout (15D)
//!
//! ```text
//! r = sqrt_info · [ r_p (3)    position
//!                   r_q (3)    rotation
//!                   r_v (3)    velocity
//!                   r_bg (3)   gyro bias
//!                   r_ba (3) ] accel bias
//! ```
//!
//! # Parameter layout (6 blocks, 30 minimal DOF)
//!
//! ```text
//! params[0]: SE3 pose i   — 7D [tx,ty,tz,qw,qx,qy,qz], 6 DOF
//! params[1]: velocity i   — 3D [vx,vy,vz]
//! params[2]: imu bias i   — 6D [bgx,bgy,bgz,bax,bay,baz]
//! params[3]: SE3 pose j   — 7D, 6 DOF
//! params[4]: velocity j   — 3D
//! params[5]: imu bias j   — 6D
//! ```

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::{SE23, SE23Tangent};
use faer::prelude::ReborrowMut;
use nalgebra::{DVector, Matrix3, SMatrix, Vector3};

use super::preintegration::ImuPreintegration;
use crate::factors::Factor;

/// Combined IMU factor with separate velocity and bias parameter blocks.
///
/// Identical math to [`ImuFactor`] — same 15D residual and same analytical
/// Jacobians — but with 6 parameter blocks instead of 4, matching GTSAM's
/// `CombinedImuFactor` convention.
///
/// [`ImuFactor`]: super::imu_factor::ImuFactor
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
    /// Compute the weighted 15D residual and optional 15×30 Jacobian.
    ///
    /// # Residual
    ///
    /// Same as `ImuFactor::linearize` — SE_2(3) right-minus formulation.
    /// See [`ImuFactor`] for the full derivation.
    ///
    /// # Jacobian column layout (30 total minimal DOF)
    ///
    /// ```text
    /// 0..6   — pose_i   (SE3, 6 DOF)
    /// 6..9   — vel_i    (R³,  3 DOF)
    /// 9..15  — bias_i   (R⁶,  6 DOF: [bg, ba])
    /// 15..21 — pose_j   (SE3, 6 DOF)
    /// 21..24 — vel_j    (R³,  3 DOF)
    /// 24..30 — bias_j   (R⁶,  6 DOF)
    /// ```
    ///
    /// [`ImuFactor`]: super::imu_factor::ImuFactor
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            6,
            "CombinedImuFactor expects 6 parameter blocks"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be velocity (3D)");
        debug_assert_eq!(params[2].len(), 6, "params[2] must be IMU bias (6D)");
        debug_assert_eq!(params[3].len(), 7, "params[3] must be SE3 (7D)");
        debug_assert_eq!(params[4].len(), 3, "params[4] must be velocity (3D)");
        debug_assert_eq!(params[5].len(), 6, "params[5] must be IMU bias (6D)");

        let preint = &self.preintegration;

        // ── Parse parameters ──────────────────────────────────────────────
        let pose_i = SE3::from_param_slice(params[0]);
        let v_i = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let b_g_i = Vector3::new(params[2][0], params[2][1], params[2][2]);
        let b_a_i = Vector3::new(params[2][3], params[2][4], params[2][5]);

        let pose_j = SE3::from_param_slice(params[3]);
        let v_j = Vector3::new(params[4][0], params[4][1], params[4][2]);
        let b_g_j = Vector3::new(params[5][0], params[5][1], params[5][2]);
        let b_a_j = Vector3::new(params[5][3], params[5][4], params[5][5]);

        let p_i = pose_i.translation();
        let q_i = pose_i.rotation_quaternion();
        let r_i = pose_i.rotation_so3().rotation_matrix();

        let p_j = pose_j.translation();
        let q_j = pose_j.rotation_quaternion();
        let r_j = pose_j.rotation_so3().rotation_matrix();

        // Reference biases from preintegration linearization point
        let b_g_ref = Vector3::new(
            preint.speed_and_biases_ref()[3],
            preint.speed_and_biases_ref()[4],
            preint.speed_and_biases_ref()[5],
        );
        let b_a_ref = Vector3::new(
            preint.speed_and_biases_ref()[6],
            preint.speed_and_biases_ref()[7],
            preint.speed_and_biases_ref()[8],
        );

        let dt = preint.delta_t();
        let g_vec = Vector3::new(0.0, 0.0, preint.imu_params().g);

        // ── Gravity-corrected state at i ──────────────────────────────────
        let gc_p = p_i + v_i * dt - 0.5 * g_vec * dt * dt;
        let gc_v = v_i - g_vec * dt;
        let gc_state_i = SE23::new(gc_p, gc_v, q_i);
        let state_j = SE23::new(p_j, v_j, q_j);

        let predicted = gc_state_i.inverse(None).compose(&state_j, None, None);

        // ── First-order bias correction ───────────────────────────────────
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

        // ── 9D kinematic residual ─────────────────────────────────────────
        let mut jac_rm_pred = SMatrix::<f64, 9, 9>::zeros();
        let mut jac_rm_dc = SMatrix::<f64, 9, 9>::zeros();
        let r_kin_tangent = predicted.right_minus(
            &delta_corrected,
            Some(&mut jac_rm_pred),
            Some(&mut jac_rm_dc),
        );
        let r_kin = se23_tangent_data(&r_kin_tangent);

        // ── Bias residuals ────────────────────────────────────────────────
        let r_bg = b_g_i - b_g_j;
        let r_ba = b_a_i - b_a_j;

        // ── Full 15D residual ─────────────────────────────────────────────
        let mut residual_raw = DVector::zeros(15);
        residual_raw.rows_mut(0, 9).copy_from(&r_kin);
        residual_raw[9] = r_bg.x;
        residual_raw[10] = r_bg.y;
        residual_raw[11] = r_bg.z;
        residual_raw[12] = r_ba.x;
        residual_raw[13] = r_ba.y;
        residual_raw[14] = r_ba.z;

        let sqrt_info = preint.square_root_information();
        let weighted =
            sqrt_info * nalgebra::SVector::<f64, 15>::from_iterator(residual_raw.iter().copied());
        for i in 0..15 {
            residual[i] = weighted[i];
        }

        let Some(mut jac) = jacobian else {
            return;
        };

        // ── Jacobians ─────────────────────────────────────────────────────
        let jac_pred_wrt_gc: SMatrix<f64, 9, 9> = -predicted.inverse(None).adjoint();

        // J_gc_pose (9×6): SE3 tangent → SE23 tangent at gc_state_i
        let mut j_gc_pose = SMatrix::<f64, 9, 6>::zeros();
        j_gc_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity()); // ρ → ρ
        j_gc_pose
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::identity()); // θ → θ

        // J_gc_vel (9×3): world-frame δv_i → SE23 tangent at gc_state_i
        let r_i_t = r_i.transpose();
        let r_j_t = r_j.transpose();
        let mut j_gc_vel = SMatrix::<f64, 9, 3>::zeros();
        j_gc_vel
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(r_i_t * dt)); // ρ = R_iᵀ δv dt
        j_gc_vel.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_i_t); // ν = R_iᵀ δv

        // J_stj_pose (9×6): SE3 tangent → SE23 tangent at state_j
        let mut j_stj_pose = SMatrix::<f64, 9, 6>::zeros();
        j_stj_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        j_stj_pose
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::identity());

        // J_stj_vel (9×3): world-frame δv_j → SE23 tangent at state_j
        let mut j_stj_vel = SMatrix::<f64, 9, 3>::zeros();
        j_stj_vel.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_j_t);

        // Bias correction Jacobians (9×3 each)
        let mut j_corr_bg = SMatrix::<f64, 9, 3>::zeros();
        j_corr_bg
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(preint.dp_db_g());
        j_corr_bg
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-preint.dalpha_db_g()));
        j_corr_bg
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(preint.dv_db_g());

        let mut j_corr_ba = SMatrix::<f64, 9, 3>::zeros();
        j_corr_ba
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-preint.c_doubleintegral()));
        j_corr_ba
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-preint.c_integral()));

        let j_rkin_dc = jac_rm_dc * jac_rp_tangent;

        // ── Assemble 9×30 kinematic Jacobian ──────────────────────────────
        // Columns: [pose_i(6) | vel_i(3) | bg_i(3) | ba_i(3) | pose_j(6) | vel_j(3) | bg_j(3) | ba_j(3)]
        //           0          6           9          12         15          21          24         27
        let mut j_kin = SMatrix::<f64, 9, 30>::zeros();

        let d_pred_gc = jac_rm_pred * jac_pred_wrt_gc;

        j_kin
            .fixed_view_mut::<9, 6>(0, 0)
            .copy_from(&(d_pred_gc * j_gc_pose)); // pose_i
        j_kin
            .fixed_view_mut::<9, 3>(0, 6)
            .copy_from(&(d_pred_gc * j_gc_vel)); // vel_i
        j_kin
            .fixed_view_mut::<9, 3>(0, 9)
            .copy_from(&(j_rkin_dc * j_corr_bg)); // bg_i
        j_kin
            .fixed_view_mut::<9, 3>(0, 12)
            .copy_from(&(j_rkin_dc * j_corr_ba)); // ba_i
        j_kin
            .fixed_view_mut::<9, 6>(0, 15)
            .copy_from(&(jac_rm_pred * j_stj_pose)); // pose_j
        j_kin
            .fixed_view_mut::<9, 3>(0, 21)
            .copy_from(&(jac_rm_pred * j_stj_vel)); // vel_j
        // bg_j, ba_j: r_kin does not depend on sb_j (cols 24..30 = 0)

        // ── Assemble full 15×30 Jacobian ──────────────────────────────────
        let mut j_full = SMatrix::<f64, 15, 30>::zeros();
        j_full.fixed_view_mut::<9, 30>(0, 0).copy_from(&j_kin);

        // r_bg = b_g_i − b_g_j   (rows 9..12)
        j_full
            .fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&Matrix3::identity()); // d/d(bg_i)
        j_full
            .fixed_view_mut::<3, 3>(9, 24)
            .copy_from(&(-Matrix3::identity())); // d/d(bg_j)

        // r_ba = b_a_i − b_a_j   (rows 12..15)
        j_full
            .fixed_view_mut::<3, 3>(12, 12)
            .copy_from(&Matrix3::identity()); // d/d(ba_i)
        j_full
            .fixed_view_mut::<3, 3>(12, 27)
            .copy_from(&(-Matrix3::identity())); // d/d(ba_j)

        let j_weighted = sqrt_info * j_full;
        for row in 0..15 {
            for col in 0..30 {
                *jac.rb_mut().get_mut(row, col) = j_weighted[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        15
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (15, 30)
    }
}

/// Extract raw 9D data from SE23Tangent as `[ρ(3), θ(3), ν(3)]`.
fn se23_tangent_data(t: &SE23Tangent) -> nalgebra::SVector<f64, 9> {
    let rho = t.rho();
    let theta = t.theta();
    let nu = t.nu();
    nalgebra::SVector::<f64, 9>::from_iterator(
        rho.iter().chain(theta.iter()).chain(nu.iter()).copied(),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::{SE3, SE3Tangent};
    use nalgebra::{DMatrix, Vector3};

    use super::super::preintegration::ImuPreintegration;
    use super::super::types::{
        ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt,
    };

    fn euroc_params() -> ImuParameters {
        ImuParameters {
            sigma_g_c: 1.6968e-04,
            sigma_a_c: 2.0000e-03,
            sigma_gw_c: 1.9393e-05,
            sigma_aw_c: 3.0000e-03,
            g: 9.81,
            ..ImuParameters::default()
        }
    }

    fn make_meas(t: f64, gyr: Vector3<f64>, acc: Vector3<f64>) -> ImuMeasurement {
        ImuMeasurement::new(
            t,
            ImuSensorReadings {
                gyroscopes: gyr,
                accelerometers: acc,
            },
        )
    }

    /// Build SpeedAndBias and split into the three CombinedImuFactor blocks.
    fn sb_to_blocks(sb: &SpeedAndBias) -> (DVector<f64>, DVector<f64>) {
        let vel = DVector::from_vec(vec![sb[0], sb[1], sb[2]]);
        let bias = DVector::from_vec(vec![sb[3], sb[4], sb[5], sb[6], sb[7], sb[8]]);
        (vel, bias)
    }

    fn identity_blocks() -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let vel = DVector::zeros(3);
        let bias = DVector::zeros(6);
        (pose, vel, bias)
    }

    fn perturb_se3(pose: &[f64], tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from_param_slice(pose);
        let tan = SE3Tangent::from_slice(tangent);
        DVector::from_column_slice(se3.right_plus(&tan, None, None).as_param_slice())
    }

    fn perturb_vec(v: &[f64], idx: usize, eps: f64) -> DVector<f64> {
        let mut out = DVector::from_column_slice(v);
        out[idx] += eps;
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_residual(
        factor: &CombinedImuFactor,
        pose_i: &[f64],
        vel_i: &[f64],
        bias_i: &[f64],
        pose_j: &[f64],
        vel_j: &[f64],
        bias_j: &[f64],
    ) -> Vec<f64> {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(
            &[pose_i, vel_i, bias_i, pose_j, vel_j, bias_j],
            &mut residual,
            None,
        );
        residual
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_with_jacobian(
        factor: &CombinedImuFactor,
        pose_i: &[f64],
        vel_i: &[f64],
        bias_i: &[f64],
        pose_j: &[f64],
        vel_j: &[f64],
        bias_j: &[f64],
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(
            &[pose_i, vel_i, bias_i, pose_j, vel_j, bias_j],
            &mut residual,
            Some(jac_mut),
        );
        let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
        (residual, jacobian)
    }

    // ── Test 1: zero residual for stationary ground truth ─────────────────

    #[test]
    fn zero_residual_stationary() {
        let params_imu = euroc_params();
        let g = params_imu.g;
        let sb = SpeedAndBias::zeros();

        let dt_step = 0.005_f64;
        let n = 201_usize;
        let t1 = (n - 1) as f64 * dt_step;

        let measurements: Vec<_> = (0..n)
            .map(|i| {
                make_meas(
                    i as f64 * dt_step,
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, g),
                )
            })
            .collect();

        let preint = ImuPreintegration::new(measurements, params_imu, 0.0, t1, &sb);
        let factor = CombinedImuFactor::new(preint);

        let (pose, vel, bias) = identity_blocks();
        let residual = compute_residual(
            &factor,
            pose.as_slice(),
            vel.as_slice(),
            bias.as_slice(),
            pose.as_slice(),
            vel.as_slice(),
            bias.as_slice(),
        );

        for (i, ri) in residual.iter().enumerate().take(15).skip(9) {
            assert!(ri.abs() < 1e-12, "bias residual[{i}] nonzero");
        }

        let sqrt_info = factor.preintegration().square_root_information();
        assert!(
            sqrt_info.iter().all(|v| v.is_finite()),
            "sqrt_info non-finite"
        );
    }

    // ── Test 2: near-zero residual for propagated states ─────────────────

    #[test]
    fn zero_residual_propagated_states() {
        let params_imu = euroc_params();
        let g = params_imu.g;
        let omega = 0.1_f64;
        let dt_step = 0.005_f64;
        let n = 201_usize;
        let t0 = 0.0;
        let t1 = (n - 1) as f64 * dt_step;

        let measurements: Vec<_> = (0..n)
            .map(|i| {
                make_meas(
                    i as f64 * dt_step,
                    Vector3::new(0.0, 0.0, omega),
                    Vector3::new(0.0, 0.0, g),
                )
            })
            .collect();

        let sb_zero = SpeedAndBias::zeros();
        let preint =
            ImuPreintegration::new(measurements.clone(), params_imu.clone(), t0, t1, &sb_zero);

        let mut t_ws_j = SE3::identity();
        let mut sb_j = SpeedAndBias::zeros();
        ImuPreintegration::propagation(&measurements, &params_imu, &mut t_ws_j, &mut sb_j, t0, t1);

        let factor = CombinedImuFactor::new(preint);

        let pose_i = DVector::from_column_slice(SE3::identity().as_param_slice());
        let pose_j = DVector::from_column_slice(t_ws_j.as_param_slice());
        let vj = sb_j.velocity();

        let (_, sb_j_bias) = sb_to_blocks(&sb_j);
        let vel_j = DVector::from_vec(vec![vj.x, vj.y, vj.z]);
        let (_, zero_bias) = sb_to_blocks(&sb_zero);
        let vel_i = DVector::zeros(3);

        let residual = compute_residual(
            &factor,
            pose_i.as_slice(),
            vel_i.as_slice(),
            zero_bias.as_slice(),
            pose_j.as_slice(),
            vel_j.as_slice(),
            sb_j_bias.as_slice(),
        );

        for (i, ri) in residual.iter().enumerate().take(15).skip(9) {
            assert!(ri.abs() < 1e-14, "bias residual[{i}] nonzero");
        }

        let kin_norm: f64 = residual[0..9].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            kin_norm < 1.0,
            "kinematic residual too large: {kin_norm:.4}"
        );
    }

    // ── Test 3: finite-difference Jacobian verification ───────────────────

    #[test]
    fn finite_difference_jacobians() {
        let params_imu = euroc_params();
        let g = params_imu.g;
        let omega = 0.1_f64;
        let dt_step = 0.005_f64;
        let n = 101_usize;
        let t0 = 0.0;
        let t1 = (n - 1) as f64 * dt_step;

        let measurements: Vec<_> = (0..n)
            .map(|i| {
                make_meas(
                    i as f64 * dt_step,
                    Vector3::new(0.0, 0.0, omega),
                    Vector3::new(0.0, 0.0, g),
                )
            })
            .collect();

        let sb_zero = SpeedAndBias::zeros();
        let preint =
            ImuPreintegration::new(measurements.clone(), params_imu.clone(), t0, t1, &sb_zero);

        let mut t_ws_j = SE3::identity();
        let mut sb_j = SpeedAndBias::zeros();
        ImuPreintegration::propagation(&measurements, &params_imu, &mut t_ws_j, &mut sb_j, t0, t1);

        let factor = CombinedImuFactor::new(preint);

        let pose_i_vec = DVector::from_column_slice(SE3::identity().as_param_slice());
        let pose_j_vec = DVector::from_column_slice(t_ws_j.as_param_slice());
        let vj = sb_j.velocity();
        let vel_i_vec = DVector::zeros(3);
        let vel_j_vec = DVector::from_vec(vec![vj.x, vj.y, vj.z]);
        let bias_i_vec = DVector::zeros(6);
        let bias_j_vec = DVector::zeros(6);

        let (r0, jac) = compute_with_jacobian(
            &factor,
            pose_i_vec.as_slice(),
            vel_i_vec.as_slice(),
            bias_i_vec.as_slice(),
            pose_j_vec.as_slice(),
            vel_j_vec.as_slice(),
            bias_j_vec.as_slice(),
        );

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-3;

        // Block 0: pose_i (SE3, 6 DOF, cols 0..6)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_i_p = perturb_se3(pose_i_vec.as_slice(), &tan);
            let r_pert = compute_residual(
                &factor,
                pose_i_p.as_slice(),
                vel_i_vec.as_slice(),
                bias_i_vec.as_slice(),
                pose_j_vec.as_slice(),
                vel_j_vec.as_slice(),
                bias_j_vec.as_slice(),
            );
            for row in 0..15 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J_pose_i[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, col)],
                    fd
                );
            }
        }

        // Block 1: vel_i (3 DOF, cols 6..9)
        for col in 0..3 {
            let vel_i_p = perturb_vec(vel_i_vec.as_slice(), col, EPS);
            let r_pert = compute_residual(
                &factor,
                pose_i_vec.as_slice(),
                vel_i_p.as_slice(),
                bias_i_vec.as_slice(),
                pose_j_vec.as_slice(),
                vel_j_vec.as_slice(),
                bias_j_vec.as_slice(),
            );
            for row in 0..15 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_vel_i[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd
                );
            }
        }

        // Block 2: bias_i (6 DOF, cols 9..15)
        for col in 0..6 {
            let bias_i_p = perturb_vec(bias_i_vec.as_slice(), col, EPS);
            let r_pert = compute_residual(
                &factor,
                pose_i_vec.as_slice(),
                vel_i_vec.as_slice(),
                bias_i_p.as_slice(),
                pose_j_vec.as_slice(),
                vel_j_vec.as_slice(),
                bias_j_vec.as_slice(),
            );
            for row in 0..15 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 9 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_bias_i[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 9 + col)],
                    fd
                );
            }
        }

        // Block 3: pose_j (SE3, 6 DOF, cols 15..21)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_j_p = perturb_se3(pose_j_vec.as_slice(), &tan);
            let r_pert = compute_residual(
                &factor,
                pose_i_vec.as_slice(),
                vel_i_vec.as_slice(),
                bias_i_vec.as_slice(),
                pose_j_p.as_slice(),
                vel_j_vec.as_slice(),
                bias_j_vec.as_slice(),
            );
            for row in 0..15 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 15 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_pose_j[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 15 + col)],
                    fd
                );
            }
        }

        // Block 4: vel_j (3 DOF, cols 21..24)
        for col in 0..3 {
            let vel_j_p = perturb_vec(vel_j_vec.as_slice(), col, EPS);
            let r_pert = compute_residual(
                &factor,
                pose_i_vec.as_slice(),
                vel_i_vec.as_slice(),
                bias_i_vec.as_slice(),
                pose_j_vec.as_slice(),
                vel_j_p.as_slice(),
                bias_j_vec.as_slice(),
            );
            for row in 0..15 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 21 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_vel_j[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 21 + col)],
                    fd
                );
            }
        }

        // Block 5: bias_j (6 DOF, cols 24..30)
        for col in 0..6 {
            let bias_j_p = perturb_vec(bias_j_vec.as_slice(), col, EPS);
            let r_pert = compute_residual(
                &factor,
                pose_i_vec.as_slice(),
                vel_i_vec.as_slice(),
                bias_i_vec.as_slice(),
                pose_j_vec.as_slice(),
                vel_j_vec.as_slice(),
                bias_j_p.as_slice(),
            );
            for row in 0..15 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 24 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_bias_j[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 24 + col)],
                    fd
                );
            }
        }
    }
}
