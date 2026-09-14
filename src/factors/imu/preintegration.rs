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
use apex_manifolds::sgal3::SGal3;
use apex_manifolds::so3::SO3Tangent;
use nalgebra::{Matrix3, SMatrix, UnitQuaternion, Vector3};

use super::types::{
    ImuMeasurement, ImuParameters, PreintegrationError, SpeedAndBias, SpeedAndBiasExt,
};
use crate::factors::common::math::{sinc, skew, symm_sqrt_inverse};

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
    kinematic_square_root_information: SMatrix<f64, 9, 9>,

    // Linearization point
    speed_and_biases_ref: SpeedAndBias,
}

impl ImuPreintegration {
    /// Create and immediately integrate measurements over `[t0, t1]`.
    /// Create and integrate, rejecting inputs that would integrate to a
    /// misleading result rather than to an error.
    ///
    /// Two conditions are checked, and neither is implied by the other:
    ///
    /// * the interval must be finite and run forwards, because the IMU factors
    ///   apply a gravity correction over `delta_t()` and a non-positive span
    ///   silently flips or zeroes it;
    /// * the samples must overlap the interval, because both boundaries
    ///   zero-order-hold the nearest reading, and a buffer lying wholly outside
    ///   `[t0, t1]` would be extrapolated across the whole span and returned as
    ///   though it had been measured.
    ///
    /// # Errors
    ///
    /// [`PreintegrationError::NonPositiveInterval`] or
    /// [`PreintegrationError::EmptyIntegration`].
    pub fn try_new(
        measurements: Vec<ImuMeasurement>,
        imu_params: ImuParameters,
        t0: f64,
        t1: f64,
        speed_and_biases: &SpeedAndBias,
    ) -> Result<Self, PreintegrationError> {
        if !t0.is_finite() || !t1.is_finite() || t1 <= t0 {
            return Err(PreintegrationError::NonPositiveInterval { t0, t1 });
        }
        if let (Some(first), Some(last)) = (measurements.first(), measurements.last())
            && (last.timestamp < t0 || first.timestamp > t1)
        {
            return Err(PreintegrationError::EmptyIntegration {
                got: measurements.len(),
                first: first.timestamp,
                last: last.timestamp,
                t0,
                t1,
            });
        }
        Ok(Self::new(
            measurements,
            imu_params,
            t0,
            t1,
            speed_and_biases,
        ))
    }

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
            kinematic_square_root_information: SMatrix::zeros(),
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
            let dt = if next_time > end {
                let interval = next_time - t0_meas;
                if interval > 1e-12 {
                    let r = (end - t0_meas) / interval;
                    omega_s_1 = (1.0 - r) * omega_s_0 + r * omega_s_1;
                    acc_s_1 = (1.0 - r) * acc_s_0 + r * acc_s_1;
                }
                end - time
            } else {
                next_time - time
            };

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

            // ── Sensor → body frame, then bias and scale correction ────────
            //
            // Bias and scale are **sensor-frame** corrections, so they apply to
            // the raw reading *before* the extrinsic rotation, not after:
            //
            // ```text
            // correct:  omega = R_bs · (omega_meas − b_g)
            // wrong:    omega = R_bs ·  omega_meas − b_g
            // ```
            //
            // The two differ by `(R_bs − I)·b_g`, identically zero while `t_bs`
            // is the identity and of order the bias itself once a real
            // extrinsic is supplied — a visual-inertial front end that
            // integrates in the camera frame passes exactly such a rotation.
            let r_bs = self.imu_params.rotation_body_from_sensor();
            let omega_true = r_bs * (0.5 * (omega_s_0 + omega_s_1) - b_g);
            let acc_true =
                r_bs * (0.5 * (acc_s_0 + acc_s_1) - b_a).component_div(&self.imu_params.s_a);

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

            // 0.5, not OKVIS's 0.25. `redoPreintegration` writes
            //
            //     acc_doubleintegral += acc_integral*dt
            //                         + 0.25*(C*a + C_1*a_1)*dt*dt
            //
            // where the bracket is a **sum** of the two rotated accelerations,
            // so its 0.25 is 0.5 (the trapezoid) times 0.5 (the `½at²`).
            // `c_mid` above is already the average, so carrying 0.25 across
            // halves the quadratic term. The consequence is not noise: the
            // position increment comes out short by 0.25*a*dt per second of
            // integration — 5 % of Δp at 200 Hz over a 20 Hz frame interval —
            // in the same direction every interval, which the estimator can
            // only absorb by inflating the accelerometer bias.
            self.c_doubleintegral += c_integral_old * dt + 0.5 * c_mid * dt * dt;
            self.acc_doubleintegral += acc_integral_old * dt + 0.5 * c_mid * acc_true * dt * dt;
            self.c_integral = c_integral_1;
            self.acc_integral = acc_integral_1;

            // ── Sub-Jacobian propagation ───────────────────────────────────
            let omega_dt = omega_true * dt;
            let jr = SO3Tangent::new(omega_dt).right_jacobian();

            let dq_inv_rot = dq.inverse().to_rotation_matrix().into_inner();
            let cross_new = dq_inv_rot * cross_old + jr * r_bs * dt;

            self.dalpha_db_g += c_after * jr * r_bs * dt;

            let acc_skew = skew(&acc_true);
            let dv_db_g_step =
                0.5 * dt * (c_before * acc_skew * cross_old + c_after * acc_skew * cross_new);
            let dv_db_g_1 = dv_db_g_old + dv_db_g_step;

            let dp_db_g_step = dv_db_g_old * dt + 0.5 * dt * dv_db_g_step;
            self.dp_db_g += dp_db_g_step;
            self.dv_db_g = dv_db_g_1;
            self.cross = cross_new;

            // ── Covariance propagation (OKVIS F_delta and K matrices) ──────
            let acc_doubleintegral_step = acc_integral_old * dt + 0.5 * c_mid * acc_true * dt * dt;
            let acc_integral_step = c_mid * acc_true * dt;

            let mut f_delta = SMatrix::<f64, 15, 15>::identity();
            f_delta
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-skew(&acc_doubleintegral_step)));
            f_delta
                .fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(dt * Matrix3::identity()));
            f_delta
                .fixed_view_mut::<3, 3>(0, 9)
                .copy_from(&dp_db_g_step);
            f_delta
                .fixed_view_mut::<3, 3>(0, 12)
                // OKVIS's `-C_integral*dt + 0.25*(C + C_1)*dt*dt`, with the
                // same sum-to-average correction as the integrals above.
                .copy_from(&(-c_integral_old * dt + 0.5 * c_mid * dt * dt));
            f_delta
                .fixed_view_mut::<3, 3>(3, 9)
                .copy_from(&(-dt * c_after));
            f_delta
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(-skew(&acc_integral_step)));
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

        // Kinematic-only weighting for the non-combined factors: measurement
        // noise alone, with the two random-walk terms deliberately excluded.
        let p_measurement = sg2 * self.dp_dsigma[0] + sa2 * self.dp_dsigma[1];
        let kinematic = p_measurement.fixed_view::<9, 9>(0, 0).into_owned();
        let kinematic = 0.5 * (kinematic + kinematic.transpose());
        self.kinematic_square_root_information = symm_sqrt_inverse(&kinematic);

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
            let dt = if next_time > t_end {
                let interval = next_time - t0_meas;
                if interval > 1e-12 {
                    let r = (t_end - t0_meas) / interval;
                    omega_s_1 = (1.0 - r) * omega_s_0 + r * omega_s_1;
                    acc_s_1 = (1.0 - r) * acc_s_0 + r * acc_s_1;
                }
                t_end - time
            } else {
                next_time - time
            };

            if dt < 1e-12 {
                time = next_time;
                continue;
            }

            let r_bs = imu_params.rotation_body_from_sensor();
            let omega_true = r_bs * (0.5 * (omega_s_0 + omega_s_1) - b_g);
            let acc_true = r_bs * (0.5 * (acc_s_0 + acc_s_1) - b_a).component_div(&imu_params.s_a);

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
            // See `integrate_from`: `c_mid` is an average, so the quadratic
            // term carries 0.5 rather than OKVIS's sum-form 0.25.
            acc_doubleintegral += acc_integral_old * dt + 0.5 * c_mid * acc_true * dt * dt;
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
    /// `R_bs · diag(1/s_a)` — the map from a sensor-frame accelerometer bias
    /// to its effect on the integrated, body-frame specific force.
    ///
    /// `acc_true = R_bs · ((a_meas − b_a) ⊘ s_a)`, so `∂acc_true/∂b_a` is
    /// `−R_bs · diag(1/s_a)`; the sign lives at the call site.
    fn bias_to_target(&self) -> Matrix3<f64> {
        let inverse_scale = Matrix3::from_diagonal(&Vector3::new(
            1.0 / self.imu_params.s_a.x,
            1.0 / self.imu_params.s_a.y,
            1.0 / self.imu_params.s_a.z,
        ));
        self.imu_params.rotation_body_from_sensor() * inverse_scale
    }

    /// `∂Δv/∂b_a`, up to sign.
    pub fn dv_db_a(&self) -> Matrix3<f64> {
        self.c_integral * self.bias_to_target()
    }

    /// `∂Δp/∂b_a`, up to sign. See [`Self::dv_db_a`].
    pub fn dp_db_a(&self) -> Matrix3<f64> {
        self.c_doubleintegral * self.bias_to_target()
    }

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

    /// 9×9 square-root information for the **kinematic** residual `[p, q, v]`,
    /// used by the non-combined factors.
    ///
    /// Built from measurement noise only. The gyro/accel random-walk terms that
    /// [`square_root_information`](Self::square_root_information) includes are
    /// deliberately left out: the non-combined factors model bias evolution
    /// with an explicit bias edge
    /// ([`bias_random_walk`](super::bias::bias_random_walk)), so folding that
    /// uncertainty into the kinematic weighting too would count it twice. This
    /// mirrors GTSAM's split between `PreintegratedImuMeasurements` (9×9) and
    /// `PreintegratedCombinedMeasurements` (15×15).
    pub fn kinematic_square_root_information(&self) -> &SMatrix<f64, 9, 9> {
        &self.kinematic_square_root_information
    }

    /// 15×15 square-root information matrix (Cholesky of P_delta⁻¹).
    ///
    /// Includes the bias random walk, matching the combined factors' 15D
    /// residual.
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

    /// Preintegrated delta as an SGal(3) element: `SGal3(Δp, Δv, ΔR, Δt)`.
    ///
    /// With frame-i states carried at `s = 0` and frame-j states at
    /// `s = Δt`, the composition `gc_state_i ∘ delta_sgal3()` reproduces the
    /// propagated frame-j state exactly (see [`SGal3`](apex_manifolds::sgal3::SGal3)).
    pub fn delta_sgal3(&self) -> SGal3 {
        SGal3::new(
            self.acc_doubleintegral,
            self.acc_integral,
            self.delta_q,
            self.t1 - self.t0,
        )
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

    /// Constant specific force with no rotation and no gravity: the position
    /// increment must be exactly `½aT²` and the velocity increment exactly
    /// `aT`.
    ///
    /// This is a closed form, not an approximation. With `C ≡ I` the midpoint
    /// scheme is *exact* for a constant integrand — summing
    /// `v_k·dt + ½a·dt²` over `n` steps telescopes to `½a·(n·dt)²` — so any
    /// disagreement is a coefficient error rather than discretization.
    ///
    /// # The bug this pins
    ///
    /// The double-integral lines carried OKVIS's `0.25*(C + C_1)` coefficient
    /// while multiplying by `c_mid`, which is already `½(C + C_1)`. That halves
    /// the quadratic term, so `Δp` comes out short by `¼·a·dt·T` — a
    /// *systematic* deficit proportional to the sample period, in the same
    /// direction on every interval. At 200 Hz over a 20 Hz frame interval it is
    /// 5 % of `Δp`, which an estimator can only absorb by inflating the
    /// accelerometer bias.
    #[test]
    fn constant_acceleration_matches_the_closed_form() -> Result<(), PreintegrationError> {
        let params = ImuParameters {
            g: 0.0,
            ..euroc_params()
        };
        let acceleration = Vector3::new(0.3, -0.7, 1.1);
        let dt = 0.005_f64;
        let span = 0.1_f64;
        let steps = (span / dt).round() as usize;

        let measurements: Vec<ImuMeasurement> = (0..=steps)
            .map(|k| make_meas(k as f64 * dt, Vector3::zeros(), acceleration))
            .collect();

        let preint =
            ImuPreintegration::try_new(measurements, params, 0.0, span, &SpeedAndBias::zeros())?;

        let expected_velocity = acceleration * span;
        let expected_position = 0.5 * acceleration * span * span;
        assert!(
            (preint.acc_integral() - expected_velocity).norm() < 1e-12,
            "Δv {} against {expected_velocity}",
            preint.acc_integral()
        );
        assert!(
            (preint.acc_doubleintegral() - expected_position).norm() < 1e-12,
            "Δp {} against {expected_position}; a deficit of ~{:.2e} is the \
             halved quadratic term",
            preint.acc_doubleintegral(),
            0.25 * acceleration.norm() * dt * span
        );
        Ok(())
    }

    /// The same closed form through the static propagator, which shares the
    /// coefficient and therefore shared the bug.
    ///
    /// It matters separately: this is what predicts the next frame's pose, so
    /// a deficit here biases every initial guess the estimator hands the
    /// solver, and PnP then has to pull it back.
    #[test]
    fn propagation_matches_the_closed_form() {
        let params = ImuParameters {
            g: 0.0,
            ..euroc_params()
        };
        let acceleration = Vector3::new(0.0, 0.0, 2.0);
        let dt = 0.005_f64;
        let span = 0.2_f64;
        let steps = (span / dt).round() as usize;

        let measurements: Vec<ImuMeasurement> = (0..=steps)
            .map(|k| make_meas(k as f64 * dt, Vector3::zeros(), acceleration))
            .collect();

        let mut pose = SE3::identity();
        let mut sb = SpeedAndBias::zeros();
        ImuPreintegration::propagation(&measurements, &params, &mut pose, &mut sb, 0.0, span);

        let expected_position = 0.5 * acceleration * span * span;
        assert!(
            (pose.translation() - expected_position).norm() < 1e-12,
            "position {} against {expected_position}",
            pose.translation()
        );
        assert!((sb.velocity() - acceleration * span).norm() < 1e-12);
    }

    /// A stationary, level platform must stay put. Velocity was always right
    /// here — position was not, which is what made the defect easy to miss:
    /// every velocity-based check passed.
    #[test]
    fn a_stationary_platform_does_not_drift() {
        let params = euroc_params();
        let dt = 0.005_f64;
        let span = 0.5_f64;
        let steps = (span / dt).round() as usize;
        let reading = Vector3::new(0.0, 0.0, params.g);

        let measurements: Vec<ImuMeasurement> = (0..=steps)
            .map(|k| make_meas(k as f64 * dt, Vector3::zeros(), reading))
            .collect();

        let mut pose = SE3::identity();
        let mut sb = SpeedAndBias::zeros();
        ImuPreintegration::propagation(&measurements, &params, &mut pose, &mut sb, 0.0, span);

        assert!(
            pose.translation().norm() < 1e-12,
            "a stationary platform moved {} m over {span} s",
            pose.translation().norm()
        );
        assert!(sb.velocity().norm() < 1e-12);
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
        assert!(
            angle_diff < 1e-10,
            "rotation mismatch after append: {angle_diff}"
        );

        let p_diff = (one_shot.acc_doubleintegral() - split.acc_doubleintegral()).norm();
        assert!(p_diff < 1e-12, "position integral mismatch: {p_diff}");
    }
    // ── try_new validation ──────────────────────────────────────────────────

    fn steady_samples(t0: f64, t1: f64, n: usize) -> Vec<ImuMeasurement> {
        (0..n)
            .map(|k| {
                let t = t0 + (t1 - t0) * k as f64 / (n - 1).max(1) as f64;
                make_meas(t, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81))
            })
            .collect()
    }

    /// The preintegrated covariance must match the closed form it is defined by.
    ///
    /// For a stationary, non-rotating platform the orientation increment is a
    /// random walk driven by white gyro noise of continuous density
    /// `sigma_g_c` \[rad/s/sqrt(Hz)\], so after `T` seconds
    ///
    /// ```text
    /// sigma_theta = sigma_g_c * sqrt(T)
    /// ```
    ///
    /// exactly — independent of the sample rate, which is the whole point of a
    /// *density*. Likewise the velocity increment integrates white accelerometer
    /// noise, giving `sigma_v = sigma_a_c * sqrt(T)`, and position
    /// double-integrates it:
    ///
    /// ```text
    /// var_p = sigma_a_c^2 * int_0^T (T-s)^2 ds = sigma_a_c^2 * T^3 / 3
    /// ```
    ///
    /// This pins the units. Nothing else in the suite checks the covariance's
    /// magnitude: every other test reads the mean, and the mean is blind to a
    /// noise model that is off by a factor of `dt` or of the sample rate. A
    /// covariance that is too small by such a factor does not fail any of them,
    /// it just silently over-weights every IMU factor in every graph.
    #[test]
    fn the_preintegrated_covariance_matches_its_closed_form() -> Result<(), PreintegrationError> {
        let params = euroc_params();
        // Two rates over the same span: a density-based model must agree.
        for (span, samples) in [(0.05, 11usize), (0.05, 21), (0.2, 41)] {
            let preint = ImuPreintegration::try_new(
                steady_samples(0.0, span, samples),
                params.clone(),
                0.0,
                span,
                &SpeedAndBias::zeros(),
            )?;
            let p = preint.p_delta();
            // Tangent order is [position, orientation, velocity, b_g, b_a].
            let sigma_p = p.fixed_view::<3, 3>(0, 0)[(0, 0)].sqrt();
            let sigma_theta = p.fixed_view::<3, 3>(3, 3)[(0, 0)].sqrt();
            let sigma_v = p.fixed_view::<3, 3>(6, 6)[(0, 0)].sqrt();
            let expect_theta = params.sigma_g_c * span.sqrt();
            let expect_v = params.sigma_a_c * span.sqrt();
            // Position double-integrates the same white noise:
            // var = sigma_a^2 * int_0^T (T-s)^2 ds = sigma_a^2 * T^3 / 3.
            let expect_p = params.sigma_a_c * span.powf(1.5) / 3.0_f64.sqrt();
            assert!(
                (sigma_p / expect_p - 1.0).abs() < 0.07,
                "span {span} n {samples}: sigma_p {sigma_p:.6e} \
                 should be {expect_p:.6e} (ratio {:.4})",
                sigma_p / expect_p
            );
            assert!(
                (sigma_theta / expect_theta - 1.0).abs() < 0.02,
                "span {span} n {samples}: sigma_theta {sigma_theta:.6e} \
                 should be {expect_theta:.6e} (ratio {:.4})",
                sigma_theta / expect_theta
            );
            assert!(
                (sigma_v / expect_v - 1.0).abs() < 0.05,
                "span {span} n {samples}: sigma_v {sigma_v:.6e} \
                 should be {expect_v:.6e} (ratio {:.4})",
                sigma_v / expect_v
            );
        }
        Ok(())
    }

    #[test]
    fn try_new_accepts_a_well_formed_interval() -> Result<(), PreintegrationError> {
        let preint = ImuPreintegration::try_new(
            steady_samples(0.0, 0.1, 11),
            euroc_params(),
            0.0,
            0.1,
            &SpeedAndBias::zeros(),
        )?;
        assert!((preint.delta_t() - 0.1).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn try_new_rejects_a_backwards_interval() {
        let err = ImuPreintegration::try_new(
            steady_samples(0.0, 0.1, 11),
            euroc_params(),
            0.1,
            0.0,
            &SpeedAndBias::zeros(),
        );
        let Err(PreintegrationError::NonPositiveInterval { t0, t1 }) = err else {
            panic!("expected NonPositiveInterval");
        };
        assert_eq!((t0, t1), (0.1, 0.0));
    }

    /// A buffer sitting wholly outside the interval satisfies every count-based
    /// proxy and still integrates nothing but extrapolation.
    #[test]
    fn try_new_rejects_samples_that_do_not_overlap() {
        let err = ImuPreintegration::try_new(
            steady_samples(1.0, 1.1, 11),
            euroc_params(),
            0.0,
            0.1,
            &SpeedAndBias::zeros(),
        );
        let Err(PreintegrationError::EmptyIntegration { first, last, .. }) = err else {
            panic!("expected EmptyIntegration");
        };
        assert!((first - 1.0).abs() < 1e-12 && (last - 1.1).abs() < 1e-12);
    }

    #[test]
    fn try_new_accepts_an_empty_buffer_like_new_does() -> Result<(), PreintegrationError> {
        // No samples is not the same failure as samples in the wrong place:
        // an empty window integrates to identity, which is a legitimate result
        // for a caller that has not received IMU data yet.
        ImuPreintegration::try_new(Vec::new(), euroc_params(), 0.0, 0.1, &SpeedAndBias::zeros())?;
        Ok(())
    }
    // ── t_bs and s_a are live parameters ────────────────────────────────────

    fn rotating_samples() -> Vec<ImuMeasurement> {
        (0..11)
            .map(|k| {
                make_meas(
                    k as f64 * 0.01,
                    Vector3::new(0.0, 0.0, 0.7),
                    Vector3::new(0.3, -0.2, 9.81),
                )
            })
            .collect()
    }

    fn integrate(params: ImuParameters) -> ImuPreintegration {
        ImuPreintegration::new(rotating_samples(), params, 0.0, 0.1, &SpeedAndBias::zeros())
    }

    /// `t_bs` and `s_a` were declared, documented and read by nothing: any
    /// caller supplying an extrinsic got silently unrotated results. These two
    /// tests are what stop that recurring — they fail if either parameter goes
    /// back to being ignored.
    #[test]
    fn a_non_identity_extrinsic_rotates_the_integrated_delta() {
        let identity = integrate(euroc_params());
        let rotated = integrate(ImuParameters {
            t_bs: SE3::new(
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            ),
            ..euroc_params()
        });

        let dv = (identity.acc_integral() - rotated.acc_integral()).norm();
        assert!(
            dv > 1e-3,
            "a 90 degree extrinsic left the velocity integral unchanged ({dv:.3e}); \
             t_bs is being ignored"
        );
    }

    #[test]
    fn accelerometer_scale_changes_the_integrated_velocity() {
        let unit = integrate(euroc_params());
        let scaled = integrate(ImuParameters {
            s_a: Vector3::new(2.0, 2.0, 2.0),
            ..euroc_params()
        });

        let dv = (unit.acc_integral() - scaled.acc_integral()).norm();
        assert!(
            dv > 1e-3,
            "doubling s_a left the velocity integral unchanged ({dv:.3e}); \
             s_a is being ignored"
        );
    }

    /// The accel-bias Jacobians carry the same sensor-to-body map, so they move
    /// with the extrinsic too.
    #[test]
    fn accel_bias_jacobians_follow_the_extrinsic() {
        let identity = integrate(euroc_params());
        let rotated = integrate(ImuParameters {
            t_bs: SE3::new(
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            ),
            ..euroc_params()
        });
        let diff = (identity.dv_db_a() - rotated.dv_db_a()).norm();
        assert!(diff > 1e-6, "dv_db_a ignored the extrinsic ({diff:.3e})");
    }
}
