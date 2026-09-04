//! GPS measurement factor for pose-graph optimization.
//!
//! Implements a synchronous GPS error term connecting a robot pose `T_WS`
//! and a GPS-to-world frame transform `T_GW` to a 3D GPS position measurement.
//!
//! # Mathematical Formulation
//!
//! The GPS antenna is displaced from the sensor origin by a known lever arm
//! `r_SA` (sensor-to-antenna in sensor frame). The predicted antenna position
//! in the GPS frame is:
//!
//! ```text
//! t_antenna_W = t_WS + C_WS · r_SA
//! predicted   = C_GW · t_antenna_W + t_GW
//! e           = z_GPS − predicted
//! r           = sqrt_info · e
//! ```
//!
//! # Parameter Layout (2 blocks, 12 DOF total)
//!
//! ```text
//! params[0]: T_WS — robot pose in world frame (7D [tx,ty,tz,qw,qx,qy,qz], 6 DOF)
//! params[1]: T_GW — GPS-to-world frame transform (7D, 6 DOF)
//! ```
//!
//! # Jacobians (3×12)
//!
//! Using **right** SE3 perturbation `T ⊕ [δρ,δθ] → t + C·δρ`, `C → C·Exp(δθ)`:
//!
//! ```text
//! ∂e/∂δρ_WS = −C_GW · C_WS
//! ∂e/∂δθ_WS =  C_GW · C_WS · [r_SA]×
//! ∂e/∂δρ_GW = −C_GW
//! ∂e/∂δθ_GW =  C_GW · [t_antenna_W]×
//! ```

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector3};

use crate::factors::Factor;
use crate::factors::common::math::skew;
use crate::factors::inertial::preintegration::ImuPreintegration;
use crate::factors::inertial::types::{SpeedAndBias, SpeedAndBiasExt};

/// GPS synchronous factor: connects `T_WS` (robot pose) and `T_GW`
/// (GPS-to-world) to a 3D GPS position measurement.
pub struct GpsFactor {
    /// GPS measurement in the GPS frame [m].
    measurement: Vector3<f64>,
    /// Sensor-to-antenna lever arm in the sensor frame [m].
    antenna_offset: Vector3<f64>,
    /// Square-root information matrix (3×3 upper-triangular, `W` s.t. `Wᵀ W = Σ⁻¹`).
    sqrt_information: SMatrix<f64, 3, 3>,
}

impl GpsFactor {
    /// Create a GPS factor.
    ///
    /// # Arguments
    /// * `measurement` — GPS position in GPS frame [m]
    /// * `antenna_offset` — sensor-to-antenna lever arm in sensor frame [m]
    /// * `sqrt_information` — 3×3 square-root information matrix `W` so that
    ///   the noise covariance satisfies `W^T W = Σ⁻¹`
    pub fn new(
        measurement: Vector3<f64>,
        antenna_offset: Vector3<f64>,
        sqrt_information: SMatrix<f64, 3, 3>,
    ) -> Self {
        Self {
            measurement,
            antenna_offset,
            sqrt_information,
        }
    }

    /// Create a GPS factor with isotropic noise (scalar standard deviation).
    ///
    /// The information matrix is `(1/σ²)·I` and the square-root is `(1/σ)·I`.
    pub fn new_isotropic(
        measurement: Vector3<f64>,
        antenna_offset: Vector3<f64>,
        sigma: f64,
    ) -> Self {
        let sqrt_info = SMatrix::<f64, 3, 3>::identity() * (1.0 / sigma);
        Self::new(measurement, antenna_offset, sqrt_info)
    }
}

impl Factor for GpsFactor {
    /// Compute the weighted 3D residual and optional 3×12 Jacobian.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "GpsFactor expects 2 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 7, "params[1] must be SE3 (7D)");

        let t_ws = SE3::from_param_slice(params[0]);
        let t_gw = SE3::from_param_slice(params[1]);

        let c_ws: Matrix3<f64> = t_ws.rotation_so3().rotation_matrix();
        let c_gw: Matrix3<f64> = t_gw.rotation_so3().rotation_matrix();
        let pos_ws = t_ws.translation();
        let pos_gw = t_gw.translation();

        // Antenna position in world frame
        let t_antenna_w: Vector3<f64> = pos_ws + c_ws * self.antenna_offset;

        // Predicted GPS measurement
        let predicted: Vector3<f64> = c_gw * t_antenna_w + pos_gw;

        // Unweighted residual
        let raw_err: Vector3<f64> = self.measurement - predicted;

        // Weighted residual (apply sqrt_info)
        let weighted_err = self.sqrt_information * raw_err;
        for i in 0..3 {
            residual[i] = weighted_err[i];
        }

        let Some(mut jac) = jacobian else {
            return;
        };

        // ── Jacobians ─────────────────────────────────────────────────────────
        //
        // Right SE3 perturbation:  T ⊕ [δρ, δθ]
        //   t → t + C·δρ  (to first order)
        //   C → C·(I + [δθ]×)
        //
        // For T_WS:
        //   t_antenna_W → (t_WS + C_WS·δρ_WS) + C_WS·(I+[δθ_WS]×)·r_SA
        //                = t_antenna_W + C_WS·δρ_WS - C_WS·[r_SA]×·δθ_WS
        //   predicted → predicted + C_GW·(C_WS·δρ_WS - C_WS·[r_SA]×·δθ_WS)
        //   ∂e/∂δρ_WS = −C_GW·C_WS
        //   ∂e/∂δθ_WS =  C_GW·C_WS·[r_SA]×
        //
        // For T_GW:
        //   predicted → C_GW·(I+[δθ_GW]×)·t_antenna_W + t_GW + C_GW·δρ_GW
        //             = predicted − C_GW·[t_antenna_W]×·δθ_GW + C_GW·δρ_GW
        //   ∂e/∂δρ_GW = −C_GW
        //   ∂e/∂δθ_GW =  C_GW·[t_antenna_W]×
        //
        // Each block is 3×3; full unweighted Jacobian is 3×12.
        // After weighting: J_weighted = sqrt_info · J_raw.

        let r_sa_hat = skew(&self.antenna_offset);
        let t_ant_hat = skew(&t_antenna_w);

        let de_drho_ws = -(c_gw * c_ws);
        let de_dtheta_ws = c_gw * c_ws * r_sa_hat;
        let de_drho_gw = -c_gw;
        let de_dtheta_gw = c_gw * t_ant_hat;

        let mut j_raw = SMatrix::<f64, 3, 12>::zeros();
        j_raw.fixed_view_mut::<3, 3>(0, 0).copy_from(&de_drho_ws);
        j_raw.fixed_view_mut::<3, 3>(0, 3).copy_from(&de_dtheta_ws);
        j_raw.fixed_view_mut::<3, 3>(0, 6).copy_from(&de_drho_gw);
        j_raw.fixed_view_mut::<3, 3>(0, 9).copy_from(&de_dtheta_gw);

        let j_weighted = self.sqrt_information * j_raw;
        for row in 0..3 {
            for col in 0..12 {
                *jac.rb_mut().get_mut(row, col) = j_weighted[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 12)
    }
}

// ── GpsAsyncFactor ────────────────────────────────────────────────────────────

/// GPS asynchronous factor: extends [`GpsFactor`] for GPS measurements that
/// arrive at a time `t_g ≠ t_k` from the keyframe.
///
/// An IMU preintegration from `t_k` to `t_g` is used to propagate the robot
/// pose forward before evaluating the same 3D position error.
///
/// # Mathematical Formulation
///
/// ```text
/// Δt           = t_g − t_k
/// p_propagated = p_k + v_k · Δt + C_WS_k · Δp_body
/// C_WS_g       = C_WS_k · ΔR                          (preintegrated rotation)
/// t_antenna_W  = p_propagated + C_WS_g · r_SA
/// predicted    = C_GW · t_antenna_W + t_GW
/// e            = z_GPS − predicted
/// r            = sqrt_info · e
/// ```
///
/// Velocity `v_k` is in **world frame** (SpeedAndBias layout [0..2]).
/// `Δp_body` = `preintegration.acc_doubleintegral()` (body-frame position increment).
///
/// # Parameter Layout (3 blocks, 21 DOF)
///
/// ```text
/// params[0]: T_WS  — SE3 at keyframe t_k (7D, 6 DOF)
/// params[1]: sb_k  — SpeedAndBias at t_k  (9D, 9 DOF)
/// params[2]: T_GW  — GPS-to-world         (7D, 6 DOF)
/// ```
///
/// # Jacobians (3×21)
///
/// ```text
/// ∂e/∂δρ_WS = −C_GW · C_WS_k
/// ∂e/∂δθ_WS =  C_GW · C_WS_k · [Δp_body + ΔR · r_SA]×
/// ∂e/∂v_k   = −C_GW · Δt · I₃
/// ∂e/∂b_g   = −C_GW · C_WS_k · dp_db_g
/// ∂e/∂b_a   = −C_GW · C_WS_k · c_doubleintegral
/// ∂e/∂δρ_GW = −C_GW
/// ∂e/∂δθ_GW =  C_GW · [t_antenna_W]×
/// ```
pub struct GpsAsyncFactor {
    /// GPS measurement in the GPS frame [m].
    measurement: Vector3<f64>,
    /// Sensor-to-antenna lever arm in the sensor frame [m].
    antenna_offset: Vector3<f64>,
    /// Square-root information matrix (3×3, `W` s.t. `Wᵀ W = Σ⁻¹`).
    sqrt_information: SMatrix<f64, 3, 3>,
    /// IMU preintegration from keyframe time `t_k` to GPS timestamp `t_g`.
    preintegration: ImuPreintegration,
}

impl GpsAsyncFactor {
    /// Create an asynchronous GPS factor.
    ///
    /// # Arguments
    /// * `measurement`      — GPS position in GPS frame [m]
    /// * `antenna_offset`   — sensor-to-antenna lever arm in sensor frame [m]
    /// * `sqrt_information` — 3×3 square-root information matrix
    /// * `preintegration`   — IMU preintegration from `t_k` to `t_g`
    pub fn new(
        measurement: Vector3<f64>,
        antenna_offset: Vector3<f64>,
        sqrt_information: SMatrix<f64, 3, 3>,
        preintegration: ImuPreintegration,
    ) -> Self {
        Self {
            measurement,
            antenna_offset,
            sqrt_information,
            preintegration,
        }
    }

    /// Create with isotropic noise (scalar standard deviation).
    pub fn new_isotropic(
        measurement: Vector3<f64>,
        antenna_offset: Vector3<f64>,
        sigma: f64,
        preintegration: ImuPreintegration,
    ) -> Self {
        let sqrt_info = SMatrix::<f64, 3, 3>::identity() * (1.0 / sigma);
        Self::new(measurement, antenna_offset, sqrt_info, preintegration)
    }
}

impl Factor for GpsAsyncFactor {
    /// Compute the weighted 3D residual and optional 3×21 Jacobian.
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 3, "GpsAsyncFactor expects 3 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 9, "params[1] must be SpeedAndBias (9D)");
        debug_assert_eq!(params[2].len(), 7, "params[2] must be SE3 (7D)");

        let preint = &self.preintegration;

        let t_ws_k = SE3::from_param_slice(params[0]);
        let sb_k = SpeedAndBias::from_column_slice(params[1]);
        let t_gw = SE3::from_param_slice(params[2]);

        let c_ws_k: Matrix3<f64> = t_ws_k.rotation_so3().rotation_matrix();
        let c_gw: Matrix3<f64> = t_gw.rotation_so3().rotation_matrix();
        let p_k = t_ws_k.translation();
        let t_gw_pos = t_gw.translation();

        let v_k = sb_k.velocity(); // world frame

        let delta_t = preint.t1() - preint.t0();
        let delta_r: Matrix3<f64> = preint.delta_q().to_rotation_matrix().into_inner();

        // First-order bias correction: Δp_body ≈ Δp_nom + dp_db_g·δb_g + c_dbl·δb_a
        let b_g_ref = preint.speed_and_biases_ref().gyro_bias();
        let b_a_ref = preint.speed_and_biases_ref().accel_bias();
        let db_g = sb_k.gyro_bias() - b_g_ref;
        let db_a = sb_k.accel_bias() - b_a_ref;
        let delta_p_body = preint.acc_doubleintegral()
            + preint.dp_db_g() * db_g
            + preint.c_doubleintegral() * db_a;

        // Propagate position to GPS time
        let p_propagated = p_k + v_k * delta_t + c_ws_k * delta_p_body;

        // Rotation at GPS time (first-order)
        let c_ws_g = c_ws_k * delta_r;

        // Antenna position in world frame
        let t_antenna_w = p_propagated + c_ws_g * self.antenna_offset;

        // Predicted GPS measurement
        let predicted = c_gw * t_antenna_w + t_gw_pos;

        // Unweighted residual
        let raw_err = self.measurement - predicted;

        // Weighted residual
        let weighted_err = self.sqrt_information * raw_err;
        for i in 0..3 {
            residual[i] = weighted_err[i];
        }

        let Some(mut jac) = jacobian else {
            return;
        };

        // ── Jacobians ─────────────────────────────────────────────────────────
        //
        // Right SE3 perturbation for T_WS(t_k): T → T ⊕ [δρ, δθ]
        //   p_k → p_k + C_WS_k · δρ
        //   C_WS_k → C_WS_k · (I + [δθ]×)
        //
        // Under this perturbation:
        //   p_propagated → p_propagated + C_WS_k·δρ − C_WS_k·[Δp_body]×·δθ
        //   C_WS_g·r_SA → C_WS_g·r_SA − C_WS_k·[ΔR·r_SA]×·δθ
        //   δt_antenna_W = C_WS_k·δρ − C_WS_k·([Δp_body]× + [ΔR·r_SA]×)·δθ
        //
        //   ∂e/∂δρ_WS = −C_GW·C_WS_k
        //   ∂e/∂δθ_WS =  C_GW·C_WS_k·[Δp_body + ΔR·r_SA]×
        //
        // For sb_k = [v, b_g, b_a]:
        //   p_propagated += v·Δt + C_WS_k·dp_db_g·δb_g + C_WS_k·c_dbl·δb_a
        //
        //   ∂e/∂v_k  = −C_GW · Δt · I₃
        //   ∂e/∂b_g  = −C_GW · C_WS_k · dp_db_g
        //   ∂e/∂b_a  = −C_GW · C_WS_k · c_doubleintegral
        //
        // For T_GW: same as GpsFactor, evaluated at propagated t_antenna_W:
        //   ∂e/∂δρ_GW = −C_GW
        //   ∂e/∂δθ_GW =  C_GW · [t_antenna_W]×

        let delta_r_r_sa = delta_r * self.antenna_offset;
        let theta_arm = skew(&(delta_p_body + delta_r_r_sa));
        let t_ant_hat = skew(&t_antenna_w);

        // Block 0: T_WS (cols 0–5)
        let de_drho_ws = -(c_gw * c_ws_k);
        let de_dtheta_ws = c_gw * c_ws_k * theta_arm;

        // Block 1: sb_k (cols 6–14)  [v(3) | b_g(3) | b_a(3)]
        let de_dv = -(c_gw * delta_t);
        let de_dbg = -(c_gw * c_ws_k * preint.dp_db_g());
        let de_dba = -(c_gw * c_ws_k * preint.c_doubleintegral());

        // Block 2: T_GW (cols 15–20)
        let de_drho_gw = -c_gw;
        let de_dtheta_gw = c_gw * t_ant_hat;

        let mut j_raw = SMatrix::<f64, 3, 21>::zeros();
        // T_WS block
        j_raw.fixed_view_mut::<3, 3>(0, 0).copy_from(&de_drho_ws);
        j_raw.fixed_view_mut::<3, 3>(0, 3).copy_from(&de_dtheta_ws);
        // sb_k block
        j_raw.fixed_view_mut::<3, 3>(0, 6).copy_from(&de_dv);
        j_raw.fixed_view_mut::<3, 3>(0, 9).copy_from(&de_dbg);
        j_raw.fixed_view_mut::<3, 3>(0, 12).copy_from(&de_dba);
        // T_GW block
        j_raw.fixed_view_mut::<3, 3>(0, 15).copy_from(&de_drho_gw);
        j_raw.fixed_view_mut::<3, 3>(0, 18).copy_from(&de_dtheta_gw);

        let j_weighted = self.sqrt_information * j_raw;
        for row in 0..3 {
            for col in 0..21 {
                *jac.rb_mut().get_mut(row, col) = j_weighted[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 21)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::{SE3, SE3Tangent};
    use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};

    use crate::factors::inertial::preintegration::ImuPreintegration;
    use crate::factors::inertial::types::{
        ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias,
    };

    fn identity_pose() -> DVector<f64> {
        DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    }

    fn make_pose(tx: f64, ty: f64, tz: f64, q: UnitQuaternion<f64>) -> DVector<f64> {
        let q = q.quaternion();
        DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k])
    }

    fn perturb_se3(pose: &[f64], tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from_param_slice(pose);
        let tan = SE3Tangent::from_slice(tangent);
        DVector::from_column_slice(se3.right_plus(&tan, None, None).as_param_slice())
    }

    fn compute_residual_gps(factor: &GpsFactor, pose_ws: &[f64], pose_gw: &[f64]) -> Vec<f64> {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(&[pose_ws, pose_gw], &mut residual, None);
        residual
    }

    fn compute_with_jacobian_gps(
        factor: &GpsFactor,
        pose_ws: &[f64],
        pose_gw: &[f64],
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[pose_ws, pose_gw], &mut residual, Some(jac_mut));
        let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
        (residual, jacobian)
    }

    fn compute_residual_async(
        factor: &GpsAsyncFactor,
        pose_ws: &[f64],
        sb: &[f64],
        pose_gw: &[f64],
    ) -> Vec<f64> {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(&[pose_ws, sb, pose_gw], &mut residual, None);
        residual
    }

    fn compute_with_jacobian_async(
        factor: &GpsAsyncFactor,
        pose_ws: &[f64],
        sb: &[f64],
        pose_gw: &[f64],
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[pose_ws, sb, pose_gw], &mut residual, Some(jac_mut));
        let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
        (residual, jacobian)
    }

    // ── Test 1: zero residual when measurement equals prediction (no offset) ──

    #[test]
    fn zero_residual_when_antenna_at_origin() {
        // Identity poses, antenna at origin, measure exactly t_WS
        let t_ws_pos = Vector3::new(1.0, 2.0, 3.0);
        let pose_ws = make_pose(
            t_ws_pos.x,
            t_ws_pos.y,
            t_ws_pos.z,
            UnitQuaternion::identity(),
        );
        let pose_gw = identity_pose(); // C_GW=I, t_GW=0 → GPS frame == world frame

        let measurement = t_ws_pos; // No offset, no T_GW rotation
        let factor = GpsFactor::new_isotropic(measurement, Vector3::zeros(), 1.0);
        let r = compute_residual_gps(&factor, pose_ws.as_slice(), pose_gw.as_slice());

        for (i, ri) in r.iter().enumerate().take(3) {
            assert!(ri.abs() < 1e-12, "residual[{i}] = {} should be zero", ri);
        }
    }

    // ── Test 2: zero residual with nonzero antenna offset ─────────────────────

    #[test]
    fn zero_residual_with_antenna_offset() {
        let t_ws = Vector3::new(0.5, -1.0, 2.0);
        let r_sa = Vector3::new(0.1, 0.0, 0.05);

        // Build a small yaw rotation for T_WS
        let q_ws = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            0.3,
        );
        let c_ws = q_ws.to_rotation_matrix().into_inner();

        // T_GW: small rotation + translation
        let q_gw = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            0.1,
        );
        let t_gw_pos = Vector3::new(100.0, 200.0, 0.0);
        let c_gw = q_gw.to_rotation_matrix().into_inner();

        let t_antenna_w = t_ws + c_ws * r_sa;
        let measurement = c_gw * t_antenna_w + t_gw_pos;

        let pose_ws = make_pose(t_ws.x, t_ws.y, t_ws.z, q_ws);
        let pose_gw = make_pose(t_gw_pos.x, t_gw_pos.y, t_gw_pos.z, q_gw);

        let factor = GpsFactor::new_isotropic(measurement, r_sa, 1.0);
        let r = compute_residual_gps(&factor, pose_ws.as_slice(), pose_gw.as_slice());

        for (i, ri) in r.iter().enumerate().take(3) {
            assert!(ri.abs() < 1e-10, "residual[{i}] = {} should be zero", ri);
        }
    }

    // ── Test 3: finite-difference Jacobian check ──────────────────────────────

    #[test]
    fn finite_difference_jacobians() {
        let t_ws = Vector3::new(1.0, -0.5, 0.3);
        let r_sa = Vector3::new(0.15, 0.05, -0.02);

        let q_ws = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.25,
        );
        let c_ws = q_ws.to_rotation_matrix().into_inner();

        let q_gw = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            -0.15,
        );
        let t_gw_pos = Vector3::new(50.0, 30.0, -5.0);
        let c_gw = q_gw.to_rotation_matrix().into_inner();

        let t_antenna_w = t_ws + c_ws * r_sa;
        let measurement = c_gw * t_antenna_w + t_gw_pos;

        let pose_ws = make_pose(t_ws.x, t_ws.y, t_ws.z, q_ws);
        let pose_gw = make_pose(t_gw_pos.x, t_gw_pos.y, t_gw_pos.z, q_gw);

        let factor = GpsFactor::new_isotropic(measurement, r_sa, 0.5);
        let (r0, jac) = compute_with_jacobian_gps(&factor, pose_ws.as_slice(), pose_gw.as_slice());

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;

        // Block 0: T_WS (6 DOF)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_ws_p = perturb_se3(pose_ws.as_slice(), &tan);
            let r_pert = compute_residual_gps(&factor, pose_ws_p.as_slice(), pose_gw.as_slice());
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J_T_WS[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, col)],
                    fd
                );
            }
        }

        // Block 1: T_GW (6 DOF)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_gw_p = perturb_se3(pose_gw.as_slice(), &tan);
            let r_pert = compute_residual_gps(&factor, pose_ws.as_slice(), pose_gw_p.as_slice());
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_T_GW[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd
                );
            }
        }
    }

    // ── Helpers for async tests ───────────────────────────────────────────────

    /// Build a trivial preintegration with near-identity IMU (gravity-only accel,
    /// zero gyro) so the bias Jacobians are non-trivial but the result is known.
    fn make_preint(dt: f64, v_world: Vector3<f64>) -> ImuPreintegration {
        let params = ImuParameters::default();
        let g = params.g;
        let n = 21usize;
        let dt_step = dt / (n - 1) as f64;

        let measurements: Vec<ImuMeasurement> = (0..n)
            .map(|i| ImuMeasurement {
                timestamp: i as f64 * dt_step,
                measurement: ImuSensorReadings {
                    gyroscopes: Vector3::zeros(),
                    accelerometers: Vector3::new(0.0, 0.0, g), // gravity compensation
                },
            })
            .collect();

        // Use sb = [v_world, 0, 0] as linearization point
        let mut sb = SpeedAndBias::zeros();
        sb[0] = v_world.x;
        sb[1] = v_world.y;
        sb[2] = v_world.z;

        ImuPreintegration::new(measurements, params, 0.0, dt, &sb)
    }

    fn perturb_sb(sb: &[f64], idx: usize, eps: f64) -> DVector<f64> {
        let mut out = DVector::from_column_slice(sb);
        out[idx] += eps;
        out
    }

    // ── Test 4: zero residual for async factor with trivial preintegration ────

    #[test]
    fn async_zero_residual() {
        let dt = 0.1;
        let v_k = Vector3::new(0.5, -0.2, 0.0);
        let preint = make_preint(dt, v_k);

        let t_ws_pos = Vector3::new(1.0, 2.0, 0.5);
        let q_ws = UnitQuaternion::identity();
        let r_sa = Vector3::zeros(); // no antenna offset for simplicity

        // Propagated position (gravity-only IMU ⟹ Δp_body ≈ 0, delta_R = I)
        let delta_p_body = *preint.acc_doubleintegral();
        let delta_r: nalgebra::Matrix3<f64> = preint.delta_q().to_rotation_matrix().into_inner();
        let c_ws_k: nalgebra::Matrix3<f64> = q_ws.to_rotation_matrix().into_inner();
        let p_propagated = t_ws_pos + v_k * dt + c_ws_k * delta_p_body;
        let c_ws_g = c_ws_k * delta_r;
        let t_antenna_w = p_propagated + c_ws_g * r_sa;

        // GPS frame = world frame (identity T_GW)
        let measurement = t_antenna_w; // perfect measurement

        let pose_ws = make_pose(t_ws_pos.x, t_ws_pos.y, t_ws_pos.z, q_ws);
        let pose_gw = identity_pose();
        let mut sb_vec = DVector::zeros(9);
        sb_vec[0] = v_k.x;
        sb_vec[1] = v_k.y;
        sb_vec[2] = v_k.z;

        let factor = GpsAsyncFactor::new_isotropic(measurement, r_sa, 1.0, preint);
        let r = compute_residual_async(
            &factor,
            pose_ws.as_slice(),
            sb_vec.as_slice(),
            pose_gw.as_slice(),
        );

        for (i, ri) in r.iter().enumerate().take(3) {
            assert!(
                ri.abs() < 1e-6,
                "async residual[{i}] = {} should be near zero",
                ri
            );
        }
    }

    // ── Test 5: finite-difference Jacobian check for async factor ─────────────

    #[test]
    fn async_finite_difference_jacobians() {
        let dt = 0.05;
        let v_k = Vector3::new(0.3, -0.1, 0.05);
        let r_sa = Vector3::new(0.1, 0.0, 0.05);

        let q_ws = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.2,
        );
        let t_ws_pos = Vector3::new(1.0, -0.5, 0.3);
        let c_ws_k: nalgebra::Matrix3<f64> = q_ws.to_rotation_matrix().into_inner();

        let q_gw = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            -0.1,
        );
        let t_gw_pos = Vector3::new(50.0, 30.0, -2.0);
        let c_gw: nalgebra::Matrix3<f64> = q_gw.to_rotation_matrix().into_inner();

        let preint = make_preint(dt, v_k);

        // Build measurement from ground truth
        let delta_p_body = *preint.acc_doubleintegral();
        let delta_r: nalgebra::Matrix3<f64> = preint.delta_q().to_rotation_matrix().into_inner();
        let p_propagated = t_ws_pos + v_k * dt + c_ws_k * delta_p_body;
        let c_ws_g = c_ws_k * delta_r;
        let t_antenna_w = p_propagated + c_ws_g * r_sa;
        let measurement = c_gw * t_antenna_w + t_gw_pos;

        let pose_ws = make_pose(t_ws_pos.x, t_ws_pos.y, t_ws_pos.z, q_ws);
        let pose_gw = make_pose(t_gw_pos.x, t_gw_pos.y, t_gw_pos.z, q_gw);
        let mut sb_vec = DVector::zeros(9);
        sb_vec[0] = v_k.x;
        sb_vec[1] = v_k.y;
        sb_vec[2] = v_k.z;

        let make_factor =
            || GpsAsyncFactor::new_isotropic(measurement, r_sa, 0.5, make_preint(dt, v_k));

        let factor = make_factor();
        let (r0, jac) = compute_with_jacobian_async(
            &factor,
            pose_ws.as_slice(),
            sb_vec.as_slice(),
            pose_gw.as_slice(),
        );

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;

        // Block 0: T_WS (6 DOF, cols 0–5)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_ws_p = perturb_se3(pose_ws.as_slice(), &tan);
            let r_pert = compute_residual_async(
                &make_factor(),
                pose_ws_p.as_slice(),
                sb_vec.as_slice(),
                pose_gw.as_slice(),
            );
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "async J_T_WS[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, col)],
                    fd
                );
            }
        }

        // Block 1: sb_k (9 DOF, cols 6–14)
        for col in 0..9 {
            let sb_p = perturb_sb(sb_vec.as_slice(), col, EPS);
            let r_pert = compute_residual_async(
                &make_factor(),
                pose_ws.as_slice(),
                sb_p.as_slice(),
                pose_gw.as_slice(),
            );
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "async J_sb[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd
                );
            }
        }

        // Block 2: T_GW (6 DOF, cols 15–20)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_gw_p = perturb_se3(pose_gw.as_slice(), &tan);
            let r_pert = compute_residual_async(
                &make_factor(),
                pose_ws.as_slice(),
                sb_vec.as_slice(),
                pose_gw_p.as_slice(),
            );
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 15 + col)]).abs();
                assert!(
                    err < TOL,
                    "async J_T_GW[{row},{col}]: analytical={:.6} fd={:.6} err={err:.2e}",
                    jac[(row, 15 + col)],
                    fd
                );
            }
        }
    }
}
