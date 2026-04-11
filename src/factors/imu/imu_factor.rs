//! SE_2(3)-based IMU factor for nonlinear optimization.
//!
//! Implements the [`Factor`] trait using the Forster et al. (2017) manifold
//! preintegration approach.  The 9D kinematic residual is computed via
//! `SE23::right_minus()` and analytical Jacobians are derived through the
//! group adjoint and right/left Jacobian inverses provided by `apex-manifolds`.
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
//! # Parameter layout (4 blocks, 30 minimal DOF)
//!
//! ```text
//! params[0]: SE3 pose i  — 7D [tx,ty,tz,qw,qx,qy,qz], 6 DOF
//! params[1]: SpeedAndBias i — 9D [vx,vy,vz,bgx,bgy,bgz,bax,bay,baz]
//! params[2]: SE3 pose j  — 7D, 6 DOF
//! params[3]: SpeedAndBias j — 9D
//! ```

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::{SE23, SE23Tangent};
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

use super::preintegration::ImuPreintegration;
use super::types::{SpeedAndBias, SpeedAndBiasExt};
use crate::factors::Factor;

/// IMU between-frames factor using SE_2(3) manifold preintegration.
///
/// Connects two SE3 poses with their associated velocity and IMU bias states.
/// The preintegration is performed externally and passed in at construction.
pub struct ImuFactor {
    preintegration: ImuPreintegration,
}

impl ImuFactor {
    /// Create an IMU factor from a completed preintegration.
    pub fn new(preintegration: ImuPreintegration) -> Self {
        Self { preintegration }
    }

    /// Access the underlying preintegration.
    pub fn preintegration(&self) -> &ImuPreintegration {
        &self.preintegration
    }
}

impl Factor for ImuFactor {
    fn get_dimension(&self) -> usize {
        15
    }

    /// Compute the weighted 15D residual and optional 15×30 Jacobian.
    ///
    /// # Residual
    ///
    /// Define `gc_state_i = SE23(p_i + v_i·dt − ½g·dt², v_i − g·dt, R_i)` to
    /// remove the gravity contribution, then:
    ///
    /// ```text
    /// predicted  = inv(gc_state_i) ∘ state_j
    /// r_kin      = right_minus(predicted, delta_corrected)
    ///            = Log(delta_corrected⁻¹ ∘ predicted)
    /// r_bg       = b_g_i − b_g_j
    /// r_ba       = b_a_i − b_a_j
    /// ```
    ///
    /// # Jacobians
    ///
    /// ```text
    /// d(predicted)/d(gc_state_i) = −Ad(predicted⁻¹)   (9×9)
    /// d(predicted)/d(state_j)   =  I                   (9×9)
    /// ```
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        debug_assert_eq!(params.len(), 4, "ImuFactor expects 4 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 9, "params[1] must be SpeedAndBias (9D)");
        debug_assert_eq!(params[2].len(), 7, "params[2] must be SE3 (7D)");
        debug_assert_eq!(params[3].len(), 9, "params[3] must be SpeedAndBias (9D)");

        let preint = &self.preintegration;

        // ── Parse parameters ──────────────────────────────────────────────
        let pose_i = SE3::from(params[0].clone());
        let sb_i = SpeedAndBias::from_iterator(params[1].iter().copied());
        let pose_j = SE3::from(params[2].clone());
        let sb_j = SpeedAndBias::from_iterator(params[3].iter().copied());

        let p_i = pose_i.translation();
        let q_i = pose_i.rotation_quaternion();
        let r_i = pose_i.rotation_so3().rotation_matrix();

        let p_j = pose_j.translation();
        let q_j = pose_j.rotation_quaternion();
        let r_j = pose_j.rotation_so3().rotation_matrix();

        let v_i = sb_i.velocity();
        let b_g_i = sb_i.gyro_bias();
        let b_a_i = sb_i.accel_bias();

        let v_j = sb_j.velocity();
        let b_g_j = sb_j.gyro_bias();
        let b_a_j = sb_j.accel_bias();

        let dt = preint.delta_t();
        let g_vec = Vector3::new(0.0, 0.0, preint.imu_params().g);

        // ── Gravity-corrected state at i ──────────────────────────────────
        let gc_p = p_i + v_i * dt - 0.5 * g_vec * dt * dt;
        let gc_v = v_i - g_vec * dt;
        let gc_state_i = SE23::new(gc_p, gc_v, q_i);
        let state_j = SE23::new(p_j, v_j, q_j);

        let predicted = gc_state_i.inverse(None).compose(&state_j, None, None);

        // ── First-order bias correction ───────────────────────────────────
        let db_g = b_g_i - preint.speed_and_biases_ref().gyro_bias();
        let db_a = b_a_i - preint.speed_and_biases_ref().accel_bias();

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

        // ── Bias residuals ─────────────────────────────────────────────────
        let r_bg = b_g_i - b_g_j;
        let r_ba = b_a_i - b_a_j;

        // ── Full 15D residual ──────────────────────────────────────────────
        let mut residual_raw = DVector::zeros(15);
        residual_raw.rows_mut(0, 9).copy_from(&r_kin);
        residual_raw[9] = r_bg.x;
        residual_raw[10] = r_bg.y;
        residual_raw[11] = r_bg.z;
        residual_raw[12] = r_ba.x;
        residual_raw[13] = r_ba.y;
        residual_raw[14] = r_ba.z;

        let sqrt_info = preint.square_root_information();
        let weighted = DVector::from_iterator(
            15,
            (sqrt_info * nalgebra::SVector::<f64, 15>::from_iterator(residual_raw.iter().copied()))
                .iter()
                .copied(),
        );

        if !compute_jacobian {
            return (weighted, None);
        }

        // ── Jacobians ─────────────────────────────────────────────────────
        let jac_pred_wrt_gc: SMatrix<f64, 9, 9> = -predicted.inverse(None).adjoint();

        // J_gc_pose (9×6): SE3 tangent (ρ,θ) → SE23 tangent at gc_state_i
        let mut j_gc_pose = SMatrix::<f64, 9, 6>::zeros();
        j_gc_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity()); // ρ → ρ
        j_gc_pose
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::identity()); // θ → θ
        // ν = 0

        // J_gc_vel (9×3): world-frame δv_i → SE23 tangent at gc_state_i
        let r_i_t = r_i.transpose();
        let r_j_t = r_j.transpose();
        let mut j_gc_vel = SMatrix::<f64, 9, 3>::zeros();
        j_gc_vel
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(r_i_t * dt)); // ρ = R_iᵀ δv dt
        j_gc_vel.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_i_t); // ν = R_iᵀ δv

        // J_stj_pose (9×6): SE3 tangent (ρ,θ) → SE23 tangent at state_j
        let mut j_stj_pose = SMatrix::<f64, 9, 6>::zeros();
        j_stj_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        j_stj_pose
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::identity());

        // J_stj_vel (9×3): world-frame δv_j → SE23 tangent at state_j
        let mut j_stj_vel = SMatrix::<f64, 9, 3>::zeros();
        j_stj_vel.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_j_t); // ν = R_jᵀ δv_j

        // Bias correction Jacobians
        let mut j_corr_bg = SMatrix::<f64, 9, 3>::zeros();
        j_corr_bg
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(preint.dp_db_g()); // ρ
        j_corr_bg
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-preint.dalpha_db_g())); // θ
        j_corr_bg
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(preint.dv_db_g()); // ν

        let mut j_corr_ba = SMatrix::<f64, 9, 3>::zeros();
        j_corr_ba
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-preint.c_doubleintegral())); // ρ
        j_corr_ba
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-preint.c_integral())); // ν

        // d(r_kin)/d(correction) = jac_rm_dc · jac_rp_tangent
        let j_rkin_dc = jac_rm_dc * jac_rp_tangent;

        // ── Assemble 9×30 kinematic Jacobian ──────────────────────────────
        // Columns: [pose_i(6) | v_i(3) | bg_i(3) | ba_i(3) | pose_j(6) | v_j(3) | bg_j(3) | ba_j(3)]
        //           0          6         9          12         15          21        24         27
        let mut j_kin = SMatrix::<f64, 9, 30>::zeros();

        let d_pred_gc = jac_rm_pred * jac_pred_wrt_gc;

        j_kin
            .fixed_view_mut::<9, 6>(0, 0)
            .copy_from(&(d_pred_gc * j_gc_pose)); // pose_i
        j_kin
            .fixed_view_mut::<9, 3>(0, 6)
            .copy_from(&(d_pred_gc * j_gc_vel)); // v_i
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
            .copy_from(&(jac_rm_pred * j_stj_vel)); // v_j
        // bg_j, ba_j: r_kin does not depend on sb_j (columns 24..30 = 0)

        // ── Assemble full 15×30 Jacobian ──────────────────────────────────
        let mut j_full = SMatrix::<f64, 15, 30>::zeros();
        j_full.fixed_view_mut::<9, 30>(0, 0).copy_from(&j_kin);

        // r_bg = b_g_i − b_g_j  (rows 9..12)
        j_full
            .fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&Matrix3::identity()); // d/d(bg_i)
        j_full
            .fixed_view_mut::<3, 3>(9, 24)
            .copy_from(&(-Matrix3::identity())); // d/d(bg_j)

        // r_ba = b_a_i − b_a_j  (rows 12..15)
        j_full
            .fixed_view_mut::<3, 3>(12, 12)
            .copy_from(&Matrix3::identity()); // d/d(ba_i)
        j_full
            .fixed_view_mut::<3, 3>(12, 27)
            .copy_from(&(-Matrix3::identity())); // d/d(ba_j)

        let j_weighted = sqrt_info * j_full;
        let jac_dmat = DMatrix::from_iterator(15, 30, j_weighted.iter().copied());

        (weighted, Some(jac_dmat))
    }
}

/// Extract the raw 9D data from SE23Tangent as `[ρ(3), θ(3), ν(3)]`.
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
    use apex_manifolds::se3::{SE3, SE3Tangent};
    use nalgebra::Vector3;

    fn euroc_params() -> super::super::types::ImuParameters {
        super::super::types::ImuParameters {
            sigma_g_c: 1.6968e-04,
            sigma_a_c: 2.0000e-03,
            sigma_gw_c: 1.9393e-05,
            sigma_aw_c: 3.0000e-03,
            g: 9.81,
            ..super::super::types::ImuParameters::default()
        }
    }

    fn make_meas(
        t: f64,
        gyr: Vector3<f64>,
        acc: Vector3<f64>,
    ) -> super::super::types::ImuMeasurement {
        super::super::types::ImuMeasurement::new(
            t,
            super::super::types::ImuSensorReadings {
                gyroscopes: gyr,
                accelerometers: acc,
            },
        )
    }

    fn stationary_factor(dt_step: f64, n: usize) -> (ImuFactor, DVector<f64>, DVector<f64>) {
        let params = euroc_params();
        let g = params.g;
        let sb = SpeedAndBias::zeros();

        let measurements: Vec<_> = (0..n)
            .map(|i| {
                make_meas(
                    i as f64 * dt_step,
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, g),
                )
            })
            .collect();

        let t0 = 0.0;
        let t1 = (n - 1) as f64 * dt_step;
        let preint = ImuPreintegration::new(measurements, params, t0, t1, &sb);
        let factor = ImuFactor::new(preint);

        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let sb_vec = DVector::zeros(9);
        (factor, pose, sb_vec)
    }

    fn perturb_se3(pose: &DVector<f64>, tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from(pose.clone());
        let tan = SE3Tangent::from(DVector::from_vec(tangent.to_vec()));
        let perturbed = se3.right_plus(&tan, None, None);
        DVector::from(perturbed)
    }

    fn perturb_sb(sb: &DVector<f64>, idx: usize, eps: f64) -> DVector<f64> {
        let mut out = sb.clone();
        out[idx] += eps;
        out
    }

    // ── Test 1: zero residual for stationary ground truth ─────────────────

    #[test]
    fn zero_residual_for_exact_states() {
        let (factor, pose, sb) = stationary_factor(0.005, 201);
        let params = vec![pose.clone(), sb.clone(), pose.clone(), sb.clone()];
        let (residual, _) = factor.linearize(&params, false);

        // Bias rows must be exactly zero (same bias at i and j)
        for i in 9..15 {
            assert!(
                residual[i].abs() < 1e-12,
                "bias residual[{i}] = {} should be zero",
                residual[i]
            );
        }

        // sqrt_info must be finite (basic sanity)
        let sqrt_info = factor.preintegration().square_root_information();
        assert!(
            sqrt_info.iter().all(|v| v.is_finite()),
            "sqrt_info has non-finite entries"
        );
    }

    // ── Test 2: near-zero residual for propagated ground-truth states ─────

    #[test]
    fn zero_residual_for_propagated_states() {
        let params_imu = euroc_params();
        let g = params_imu.g;
        let dt_step = 0.005_f64;
        let n = 201_usize;
        let t0 = 0.0;
        let t1 = (n - 1) as f64 * dt_step;
        let omega = 0.1_f64;

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

        let factor = ImuFactor::new(preint);

        let pose_i = DVector::from(SE3::identity());
        let sb_i_vec = DVector::zeros(9);
        let pose_j = DVector::from(t_ws_j);
        let vj = sb_j.velocity();
        let mut sb_j_vec = DVector::zeros(9);
        sb_j_vec[0] = vj.x;
        sb_j_vec[1] = vj.y;
        sb_j_vec[2] = vj.z;

        let (residual, _) = factor.linearize(&[pose_i, sb_i_vec, pose_j, sb_j_vec], false);

        for i in 9..15 {
            assert!(residual[i].abs() < 1e-14, "bias residual[{i}] nonzero");
        }

        let kin_norm = residual.rows(0, 9).norm();
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
        let dt_step = 0.005_f64;
        let n = 101_usize;
        let t0 = 0.0;
        let t1 = (n - 1) as f64 * dt_step;
        let omega = 0.1_f64;

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

        let factor = ImuFactor::new(preint);

        let pose_i_vec = DVector::from(SE3::identity());
        let sb_i_vec = DVector::zeros(9);
        let pose_j_vec = DVector::from(t_ws_j);
        let vj = sb_j.velocity();
        let mut sb_j_vec = DVector::zeros(9);
        sb_j_vec[0] = vj.x;
        sb_j_vec[1] = vj.y;
        sb_j_vec[2] = vj.z;

        let nominal = vec![
            pose_i_vec.clone(),
            sb_i_vec.clone(),
            pose_j_vec.clone(),
            sb_j_vec.clone(),
        ];

        let (r0, jac_opt) = factor.linearize(&nominal, true);
        let jac = jac_opt.expect("Jacobian should be computed");

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-3;

        // Block 0: pose_i (6 DOF)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[0] = perturb_se3(&pose_i_vec, &tan);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..15 {
                let err = (fd[row] - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J_pose_i[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, col)],
                    fd[row]
                );
            }
        }

        // Block 1: sb_i (9 DOF)
        for col in 0..9 {
            let mut p = nominal.clone();
            p[1] = perturb_sb(&sb_i_vec, col, EPS);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..15 {
                let err = (fd[row] - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_sb_i[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd[row]
                );
            }
        }

        // Block 2: pose_j (6 DOF)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[2] = perturb_se3(&pose_j_vec, &tan);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..15 {
                let err = (fd[row] - jac[(row, 15 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_pose_j[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 15 + col)],
                    fd[row]
                );
            }
        }

        // Block 3: sb_j (9 DOF)
        for col in 0..9 {
            let mut p = nominal.clone();
            p[3] = perturb_sb(&sb_j_vec, col, EPS);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..15 {
                let err = (fd[row] - jac[(row, 21 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_sb_j[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 21 + col)],
                    fd[row]
                );
            }
        }
    }
}
