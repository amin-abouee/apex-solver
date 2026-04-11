//! IMU preintegration using midpoint integration.
//!
//! Accumulates IMU measurements between two keyframes into compact relative-motion
//! constraints, following the OKVIS2-X formulation.  The output is an SE_2(3)
//! group element (via `delta_se23()`) used by `ImuFactor` and `CombinedImuFactor`.
//!
//! All accumulated quantities are expressed in the **body frame at `t0`**.
//! Covariance is propagated in the 4-component form
//! `P = σ_g²·P₀ + σ_a²·P₁ + σ_gw²·P₂ + σ_aw²·P₃`,
//! matching the `dPdsigma_` decomposition of OKVIS2-X exactly.

use apex_manifolds::Tangent;
use apex_manifolds::se23::SE23;
use apex_manifolds::so3::SO3Tangent;
use nalgebra::{Matrix3, SMatrix, UnitQuaternion, Vector3};

use super::helpers::{cross_matrix, sinc, symm_sqrt_inverse};
use super::types::{ImuMeasurement, ImuParameters, SpeedAndBias, SpeedAndBiasExt};

/// Accumulated preintegration state between two keyframes.
#[derive(Clone)]
pub struct ImuPreintegration {
    // Configuration
    imu_params: ImuParameters,
    measurements: Vec<ImuMeasurement>,
    t0: f64,
    t1: f64,

    // Preintegrated increments (body frame at t0)
    /// ΔR: preintegrated rotation
    delta_q: UnitQuaternion<f64>,
    /// ∫ R(τ) dτ
    c_integral: Matrix3<f64>,
    /// ∫∫ R(τ) dτ²
    c_doubleintegral: Matrix3<f64>,
    /// Δv: ∫ R(τ) a(τ) dτ
    acc_integral: Vector3<f64>,
    /// Δp: ∫∫ R(τ) a(τ) dτ²
    acc_doubleintegral: Vector3<f64>,

    // Cross-term for bias Jacobians
    cross: Matrix3<f64>,

    // First-order bias correction sub-Jacobians
    dalpha_db_g: Matrix3<f64>,
    dv_db_g: Matrix3<f64>,
    dp_db_g: Matrix3<f64>,

    // 4-component covariance decomposition (OKVIS dPdsigma_)
    dp_dsigma: [SMatrix<f64, 15, 15>; 4],

    // Derived
    p_delta: SMatrix<f64, 15, 15>,
    square_root_information: SMatrix<f64, 15, 15>,

    // Linearization point
    speed_and_biases_ref: SpeedAndBias,
}

impl ImuPreintegration {
    /// Create and immediately integrate measurements over `[t0, t1]`.
    pub fn new(
        measurements: Vec<ImuMeasurement>,
        imu_params: ImuParameters,
        t0: f64,
        t1: f64,
        speed_and_biases: &SpeedAndBias,
    ) -> Self {
        let mut preint = Self {
            imu_params,
            measurements,
            t0,
            t1,
            delta_q: UnitQuaternion::identity(),
            c_integral: Matrix3::zeros(),
            c_doubleintegral: Matrix3::zeros(),
            acc_integral: Vector3::zeros(),
            acc_doubleintegral: Vector3::zeros(),
            cross: Matrix3::zeros(),
            dalpha_db_g: Matrix3::zeros(),
            dv_db_g: Matrix3::zeros(),
            dp_db_g: Matrix3::zeros(),
            dp_dsigma: [SMatrix::zeros(); 4],
            p_delta: SMatrix::zeros(),
            square_root_information: SMatrix::zeros(),
            speed_and_biases_ref: *speed_and_biases,
        };
        preint.redo_preintegration(speed_and_biases);
        preint
    }

    /// Full re-integration from scratch at the given bias linearization point.
    pub fn redo_preintegration(&mut self, speed_and_biases: &SpeedAndBias) -> usize {
        self.delta_q = UnitQuaternion::identity();
        self.c_integral = Matrix3::zeros();
        self.c_doubleintegral = Matrix3::zeros();
        self.acc_integral = Vector3::zeros();
        self.acc_doubleintegral = Vector3::zeros();
        self.cross = Matrix3::zeros();
        self.dalpha_db_g = Matrix3::zeros();
        self.dv_db_g = Matrix3::zeros();
        self.dp_db_g = Matrix3::zeros();
        self.dp_dsigma = [SMatrix::zeros(); 4];
        self.speed_and_biases_ref = *speed_and_biases;
        self.integrate_from(0, speed_and_biases)
    }

    /// Append new measurements and extend to `t1_new` without resetting.
    pub fn append(
        &mut self,
        new_measurements: &[ImuMeasurement],
        t1_new: f64,
        speed_and_biases: &SpeedAndBias,
    ) -> usize {
        let start_idx = self.measurements.len();
        for m in new_measurements {
            if m.timestamp > self.t1 + 1e-12 {
                self.measurements.push(m.clone());
            }
        }
        self.t1 = t1_new;
        if start_idx > 0 {
            self.integrate_from(start_idx - 1, speed_and_biases)
        } else {
            0
        }
    }

    /// Core integration loop starting from measurement-pair index `start_idx`.
    ///
    /// Formulas match `ImuError::redoPreintegration` in OKVIS2-X.
    fn integrate_from(&mut self, start_idx: usize, speed_and_biases: &SpeedAndBias) -> usize {
        let b_g = speed_and_biases.gyro_bias();
        let b_a = speed_and_biases.accel_bias();

        let sigma_g_c = self.imu_params.sigma_g_c;
        let sigma_a_c = self.imu_params.sigma_a_c;
        let g_max = self.imu_params.g_max;
        let a_max = self.imu_params.a_max;

        let n = self.measurements.len();
        if n < 2 {
            return 0;
        }

        let mut num_steps = 0;
        let mut has_started = false;
        let mut time = self.t0;

        for i in start_idx..(n - 1) {
            let mut omega_s_0 = self.measurements[i].measurement.gyroscopes;
            let mut omega_s_1 = self.measurements[i + 1].measurement.gyroscopes;
            let mut acc_s_0 = self.measurements[i].measurement.accelerometers;
            let mut acc_s_1 = self.measurements[i + 1].measurement.accelerometers;

            let t0_meas = self.measurements[i].timestamp;
            let t1_meas = self.measurements[i + 1].timestamp;

            if t1_meas <= self.t0 {
                continue;
            }

            // Interpolate at start boundary
            if !has_started {
                has_started = true;
                let interval = t1_meas - t0_meas;
                if interval > 1e-12 {
                    let r = (self.t0 - t0_meas) / interval;
                    if r > 0.0 {
                        omega_s_0 = (1.0 - r) * omega_s_0 + r * omega_s_1;
                        acc_s_0 = (1.0 - r) * acc_s_0 + r * acc_s_1;
                    }
                }
                time = if t0_meas > self.t0 { t0_meas } else { self.t0 };
            }

            // Interpolate at end boundary
            let next_time = t1_meas;
            let end = self.t1;
            let dt;
            if next_time > end {
                let interval = next_time - t0_meas;
                if interval > 1e-12 {
                    let r = (end - t0_meas) / interval;
                    omega_s_1 = (1.0 - r) * omega_s_0 + r * omega_s_1;
                    acc_s_1 = (1.0 - r) * acc_s_0 + r * acc_s_1;
                }
                dt = end - time;
            } else {
                dt = next_time - time;
            }

            if dt < 1e-12 {
                time = next_time;
                continue;
            }

            // Saturation multipliers (match OKVIS gyr_sat_mult / acc_sat_mult)
            let gyr_sat_mult = if omega_s_0
                .iter()
                .chain(omega_s_1.iter())
                .any(|&v| v.abs() > g_max)
            {
                100.0_f64
            } else {
                1.0_f64
            };
            let acc_sat_mult = if acc_s_0
                .iter()
                .chain(acc_s_1.iter())
                .any(|&v| v.abs() > a_max)
            {
                100.0_f64
            } else {
                1.0_f64
            };

            // Midpoint bias-corrected measurements
            let omega_true = 0.5 * (omega_s_0 + omega_s_1) - b_g;
            let acc_true = 0.5 * (acc_s_0 + acc_s_1) - b_a;

            // ── Quaternion propagation ──────────────────────────────────────
            let theta_half = omega_true.norm() * 0.5 * dt;
            let dq = {
                let w = theta_half.cos();
                let s = sinc(theta_half);
                let v = s * omega_true * 0.5 * dt;
                UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, v.x, v.y, v.z))
            };
            let c_before = self.delta_q.to_rotation_matrix().into_inner();
            let delta_q_new = self.delta_q * dq;
            let c_after = delta_q_new.to_rotation_matrix().into_inner();
            let c_mid = 0.5 * (c_before + c_after);

            // Save old state (OKVIS memory-shift pattern)
            let acc_integral_old = self.acc_integral;
            let c_integral_old = self.c_integral;
            let dv_db_g_old = self.dv_db_g;
            let cross_old = self.cross;

            // ── Accumulate integrals ───────────────────────────────────────
            let c_integral_1 = c_integral_old + c_mid * dt;
            let acc_integral_1 = acc_integral_old + c_mid * acc_true * dt;

            self.c_doubleintegral += c_integral_old * dt + 0.25 * c_mid * dt * dt;
            self.acc_doubleintegral += acc_integral_old * dt + 0.25 * c_mid * acc_true * dt * dt;
            self.c_integral = c_integral_1;
            self.acc_integral = acc_integral_1;

            // ── Sub-Jacobian propagation ───────────────────────────────────
            let omega_dt = omega_true * dt;
            let jr = SO3Tangent::new(omega_dt).right_jacobian();

            let dq_inv_rot = dq.inverse().to_rotation_matrix().into_inner();
            let cross_new = dq_inv_rot * cross_old + jr * dt;

            self.dalpha_db_g += c_after * jr * dt;

            let acc_skew = cross_matrix(&acc_true);
            let dv_db_g_step =
                0.5 * dt * (c_before * acc_skew * cross_old + c_after * acc_skew * cross_new);
            let dv_db_g_1 = dv_db_g_old + dv_db_g_step;

            let dp_db_g_step = dv_db_g_old * dt + 0.5 * dt * dv_db_g_step;
            self.dp_db_g += dp_db_g_step;
            self.dv_db_g = dv_db_g_1;
            self.cross = cross_new;

            // ── Covariance propagation (OKVIS F_delta and K matrices) ──────
            let acc_doubleintegral_step = acc_integral_old * dt + 0.25 * c_mid * acc_true * dt * dt;
            let acc_integral_step = c_mid * acc_true * dt;

            let mut f_delta = SMatrix::<f64, 15, 15>::identity();
            f_delta
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-cross_matrix(&acc_doubleintegral_step)));
            f_delta
                .fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(dt * Matrix3::identity()));
            f_delta
                .fixed_view_mut::<3, 3>(0, 9)
                .copy_from(&dp_db_g_step);
            f_delta
                .fixed_view_mut::<3, 3>(0, 12)
                .copy_from(&(-c_integral_old * dt + 0.25 * c_mid * dt * dt));
            f_delta
                .fixed_view_mut::<3, 3>(3, 9)
                .copy_from(&(-dt * c_after));
            f_delta
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(-cross_matrix(&acc_integral_step)));
            f_delta
                .fixed_view_mut::<3, 3>(6, 9)
                .copy_from(&dv_db_g_step);
            f_delta
                .fixed_view_mut::<3, 3>(6, 12)
                .copy_from(&(-c_mid * dt));

            // K matrices (per-sigma² noise)
            let mut k0 = SMatrix::<f64, 15, 15>::zeros();
            k0.fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(gyr_sat_mult * dt * Matrix3::identity()));

            let mut k1 = SMatrix::<f64, 15, 15>::zeros();
            k1.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(0.5 * dt * dt * dt * acc_sat_mult.powi(3) * Matrix3::identity()));
            k1.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&(acc_sat_mult * dt * Matrix3::identity()));

            let mut k2 = SMatrix::<f64, 15, 15>::zeros();
            k2.fixed_view_mut::<3, 3>(9, 9)
                .copy_from(&(dt * Matrix3::identity()));

            let mut k3 = SMatrix::<f64, 15, 15>::zeros();
            k3.fixed_view_mut::<3, 3>(12, 12)
                .copy_from(&(dt * Matrix3::identity()));

            let f_t = f_delta.transpose();
            self.dp_dsigma[0] = f_delta * self.dp_dsigma[0] * f_t + k0;
            self.dp_dsigma[1] = f_delta * self.dp_dsigma[1] * f_t + k1;
            self.dp_dsigma[2] = f_delta * self.dp_dsigma[2] * f_t + k2;
            self.dp_dsigma[3] = f_delta * self.dp_dsigma[3] * f_t + k3;
            for j in 0..4 {
                self.dp_dsigma[j] = 0.5 * (self.dp_dsigma[j] + self.dp_dsigma[j].transpose());
            }

            self.delta_q = delta_q_new;
            time = next_time;
            num_steps += 1;

            if next_time >= end {
                break;
            }
        }

        // Combine covariance components
        let sg2 = sigma_g_c * sigma_g_c;
        let sa2 = sigma_a_c * sigma_a_c;
        let sgw2 = self.imu_params.sigma_gw_c * self.imu_params.sigma_gw_c;
        let saw2 = self.imu_params.sigma_aw_c * self.imu_params.sigma_aw_c;

        self.p_delta = sg2 * self.dp_dsigma[0]
            + sa2 * self.dp_dsigma[1]
            + sgw2 * self.dp_dsigma[2]
            + saw2 * self.dp_dsigma[3];
        self.p_delta = 0.5 * (self.p_delta + self.p_delta.transpose());
        self.square_root_information = symm_sqrt_inverse(&self.p_delta);

        num_steps
    }

    /// Propagate state forward using IMU measurements (static utility).
    ///
    /// Integrates over `[t_start, t_end]`, updating `t_ws` and `speed_and_biases`
    /// in-place.  Gravity is `[0, 0, g]` in world frame.
    pub fn propagation(
        measurements: &[ImuMeasurement],
        imu_params: &ImuParameters,
        t_ws: &mut apex_manifolds::se3::SE3,
        speed_and_biases: &mut SpeedAndBias,
        t_start: f64,
        t_end: f64,
    ) -> usize {
        let b_g = speed_and_biases.gyro_bias();
        let b_a = speed_and_biases.accel_bias();
        let g_w = Vector3::new(0.0, 0.0, imu_params.g);

        let mut delta_q = UnitQuaternion::identity();
        let mut acc_integral = Vector3::zeros();
        let mut acc_doubleintegral = Vector3::zeros();

        let mut num_steps = 0;
        let mut has_started = false;
        let mut time = t_start;

        let n = measurements.len();
        if n < 2 {
            return 0;
        }

        for i in 0..(n - 1) {
            let mut omega_s_0 = measurements[i].measurement.gyroscopes;
            let mut omega_s_1 = measurements[i + 1].measurement.gyroscopes;
            let mut acc_s_0 = measurements[i].measurement.accelerometers;
            let mut acc_s_1 = measurements[i + 1].measurement.accelerometers;

            let t0_meas = measurements[i].timestamp;
            let t1_meas = measurements[i + 1].timestamp;

            if t1_meas <= t_start {
                continue;
            }

            if !has_started {
                has_started = true;
                let interval = t1_meas - t0_meas;
                if interval > 1e-12 {
                    let r = (t_start - t0_meas) / interval;
                    if r > 0.0 {
                        omega_s_0 = (1.0 - r) * omega_s_0 + r * omega_s_1;
                        acc_s_0 = (1.0 - r) * acc_s_0 + r * acc_s_1;
                    }
                }
                time = t_start;
            }

            let next_time = t1_meas;
            let dt;
            if next_time > t_end {
                let interval = next_time - t0_meas;
                if interval > 1e-12 {
                    let r = (t_end - t0_meas) / interval;
                    omega_s_1 = (1.0 - r) * omega_s_0 + r * omega_s_1;
                    acc_s_1 = (1.0 - r) * acc_s_0 + r * acc_s_1;
                }
                dt = t_end - time;
            } else {
                dt = next_time - time;
            }

            if dt < 1e-12 {
                time = next_time;
                continue;
            }

            let omega_true = 0.5 * (omega_s_0 + omega_s_1) - b_g;
            let acc_true = 0.5 * (acc_s_0 + acc_s_1) - b_a;

            let theta_half = omega_true.norm() * 0.5 * dt;
            let dq = {
                let w = theta_half.cos();
                let s = sinc(theta_half);
                let v = s * omega_true * 0.5 * dt;
                UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, v.x, v.y, v.z))
            };

            let c_before = delta_q.to_rotation_matrix().into_inner();
            let delta_q_new = delta_q * dq;
            let c_after = delta_q_new.to_rotation_matrix().into_inner();
            let c_mid = 0.5 * (c_before + c_after);

            let acc_integral_old = acc_integral;
            acc_doubleintegral += acc_integral_old * dt + 0.25 * c_mid * acc_true * dt * dt;
            acc_integral += c_mid * acc_true * dt;
            delta_q = delta_q_new;
            time = next_time;
            num_steps += 1;

            if next_time >= t_end {
                break;
            }
        }

        let c_ws = t_ws.rotation_quaternion().to_rotation_matrix().into_inner();
        let dt_total = t_end - t_start;
        let v = speed_and_biases.velocity();

        let new_pos = t_ws.translation() + v * dt_total + c_ws * acc_doubleintegral
            - 0.5 * g_w * dt_total * dt_total;
        let new_vel = v + c_ws * acc_integral - g_w * dt_total;
        let new_q = t_ws.rotation_quaternion() * delta_q;

        *t_ws = apex_manifolds::se3::SE3::new(new_pos, new_q);
        speed_and_biases[0] = new_vel.x;
        speed_and_biases[1] = new_vel.y;
        speed_and_biases[2] = new_vel.z;

        num_steps
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Total integration time span `t1 − t0`.
    pub fn delta_t(&self) -> f64 {
        self.t1 - self.t0
    }

    /// Start time.
    pub fn t0(&self) -> f64 {
        self.t0
    }

    /// End time.
    pub fn t1(&self) -> f64 {
        self.t1
    }

    /// Preintegrated rotation ΔR as unit quaternion.
    pub fn delta_q(&self) -> &UnitQuaternion<f64> {
        &self.delta_q
    }

    /// Δv — velocity increment (acc_integral).
    pub fn acc_integral(&self) -> &Vector3<f64> {
        &self.acc_integral
    }

    /// Δp — position increment (acc_doubleintegral).
    pub fn acc_doubleintegral(&self) -> &Vector3<f64> {
        &self.acc_doubleintegral
    }

    /// ∫ R(τ) dτ — for accel bias correction.
    pub fn c_integral(&self) -> &Matrix3<f64> {
        &self.c_integral
    }

    /// ∫∫ R(τ) dτ² — for accel bias correction.
    pub fn c_doubleintegral(&self) -> &Matrix3<f64> {
        &self.c_doubleintegral
    }

    /// d(Δα)/d(b_g).
    pub fn dalpha_db_g(&self) -> &Matrix3<f64> {
        &self.dalpha_db_g
    }

    /// d(Δv)/d(b_g).
    pub fn dv_db_g(&self) -> &Matrix3<f64> {
        &self.dv_db_g
    }

    /// d(Δp)/d(b_g).
    pub fn dp_db_g(&self) -> &Matrix3<f64> {
        &self.dp_db_g
    }

    /// 15×15 square-root information matrix (Cholesky of P_delta⁻¹).
    pub fn square_root_information(&self) -> &SMatrix<f64, 15, 15> {
        &self.square_root_information
    }

    /// Full 15×15 covariance P_delta.
    pub fn p_delta(&self) -> &SMatrix<f64, 15, 15> {
        &self.p_delta
    }

    /// Bias linearization point.
    pub fn speed_and_biases_ref(&self) -> &SpeedAndBias {
        &self.speed_and_biases_ref
    }

    /// IMU sensor parameters.
    pub fn imu_params(&self) -> &ImuParameters {
        &self.imu_params
    }

    /// Number of stored measurements.
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }

    /// Preintegrated delta as an SE_2(3) element: `SE23(Δp, Δv, ΔR)`.
    pub fn delta_se23(&self) -> SE23 {
        SE23::new(self.acc_doubleintegral, self.acc_integral, self.delta_q)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3;

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
            super::super::types::ImuSensorReadings {
                gyroscopes: gyr,
                accelerometers: acc,
            },
        )
    }

    #[test]
    fn zero_motion_identity_rotation() {
        let params = euroc_params();
        let sb = SpeedAndBias::zeros();
        let gyr = Vector3::zeros();
        let acc = Vector3::new(0.0, 0.0, params.g);

        let dt = 0.005_f64;
        let n = 200_usize;
        let measurements: Vec<_> = (0..n).map(|i| make_meas(i as f64 * dt, gyr, acc)).collect();
        let preint = ImuPreintegration::new(measurements, params, 0.0, (n - 1) as f64 * dt, &sb);

        assert!(
            preint.delta_q().angle() < 1e-10,
            "expected identity rotation"
        );
    }

    #[test]
    fn constant_angular_velocity() {
        let params = euroc_params();
        let sb = SpeedAndBias::zeros();
        let omega = 0.1_f64;
        let gyr = Vector3::new(0.0, 0.0, omega);
        let acc = Vector3::new(0.0, 0.0, params.g);

        let dt = 0.005_f64;
        let n = 200_usize;
        let t1 = (n - 1) as f64 * dt;
        let measurements: Vec<_> = (0..n).map(|i| make_meas(i as f64 * dt, gyr, acc)).collect();
        let preint = ImuPreintegration::new(measurements, params, 0.0, t1, &sb);

        let angle = preint.delta_q().angle();
        assert!(
            (angle - omega * t1).abs() < 1e-4,
            "angle mismatch: expected {:.6}, got {:.6}",
            omega * t1,
            angle
        );
    }

    #[test]
    fn propagation_stationary() {
        let params = euroc_params();
        let dt = 0.005_f64;
        let n = 200_usize;
        let t1 = (n - 1) as f64 * dt;
        let gyr = Vector3::zeros();
        let acc = Vector3::new(0.0, 0.0, params.g);
        let measurements: Vec<_> = (0..n).map(|i| make_meas(i as f64 * dt, gyr, acc)).collect();

        let mut t_ws = SE3::identity();
        let mut sb = SpeedAndBias::zeros();
        ImuPreintegration::propagation(&measurements, &params, &mut t_ws, &mut sb, 0.0, t1);

        assert!(t_ws.translation().norm() < 0.02, "position should be ~0");
        assert!(sb.velocity().norm() < 0.02, "velocity should be ~0");
    }

    #[test]
    fn delta_se23_matches_integrals() {
        let params = euroc_params();
        let sb = SpeedAndBias::zeros();
        let gyr = Vector3::new(0.05, 0.0, 0.0);
        let acc = Vector3::new(0.0, 0.0, params.g);

        let dt = 0.005_f64;
        let n = 100_usize;
        let t1 = (n - 1) as f64 * dt;
        let measurements: Vec<_> = (0..n).map(|i| make_meas(i as f64 * dt, gyr, acc)).collect();
        let preint = ImuPreintegration::new(measurements, params, 0.0, t1, &sb);

        let se23 = preint.delta_se23();
        assert!(
            (se23.translation() - preint.acc_doubleintegral()).norm() < 1e-14,
            "SE23 translation should equal acc_doubleintegral"
        );
        assert!(
            (se23.velocity() - preint.acc_integral()).norm() < 1e-14,
            "SE23 velocity should equal acc_integral"
        );
        let q_diff = se23.rotation_quaternion().inverse() * preint.delta_q();
        assert!(q_diff.angle() < 1e-14, "SE23 rotation should equal delta_q");
    }

    #[test]
    fn append_consistency() {
        let params = euroc_params();
        let sb = SpeedAndBias::zeros();
        let gyr = Vector3::new(0.0, 0.05, 0.0);
        let acc = Vector3::new(0.0, 0.0, params.g);

        let dt = 0.005_f64;
        let n = 200_usize;
        let t1 = (n - 1) as f64 * dt;
        let t_mid = (n / 2) as f64 * dt;
        let measurements: Vec<_> = (0..n).map(|i| make_meas(i as f64 * dt, gyr, acc)).collect();

        let one_shot = ImuPreintegration::new(measurements.clone(), params.clone(), 0.0, t1, &sb);

        let first_half: Vec<_> = measurements.iter().take(n / 2 + 1).cloned().collect();
        let mut split = ImuPreintegration::new(first_half, params.clone(), 0.0, t_mid, &sb);
        let second_half: Vec<_> = measurements.iter().skip(n / 2).cloned().collect();
        split.append(&second_half, t1, &sb);

        let angle_diff = (one_shot.delta_q().inverse() * split.delta_q()).angle();
        assert!(angle_diff < 1e-10, "rotation mismatch after append: {angle_diff}");

        let p_diff = (one_shot.acc_doubleintegral() - split.acc_doubleintegral()).norm();
        assert!(p_diff < 1e-12, "position integral mismatch: {p_diff}");
    }
}
