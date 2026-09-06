//! IMU-specific type definitions for apex-solver.
//!
//! These mirror the types in `apex-vio-types` so the IMU factors are
//! self-contained inside apex-solver with no cross-workspace dependency.

use apex_manifolds::se3::SE3;
use nalgebra::{Matrix3, SVector, Vector3};

// ── SpeedAndBias ─────────────────────────────────────────────────────────────

/// 9D velocity + IMU bias vector.
///
/// Layout:
/// ```text
/// [0..2]: v   — velocity in world frame [m/s]
/// [3..5]: b_g — gyroscope bias [rad/s]
/// [6..8]: b_a — accelerometer bias [m/s²]
/// ```
pub type SpeedAndBias = SVector<f64, 9>;

/// Extension methods for [`SpeedAndBias`].
pub trait SpeedAndBiasExt {
    /// World-frame velocity (indices 0–2).
    fn velocity(&self) -> Vector3<f64>;
    /// Gyroscope bias (indices 3–5).
    fn gyro_bias(&self) -> Vector3<f64>;
    /// Accelerometer bias (indices 6–8).
    fn accel_bias(&self) -> Vector3<f64>;
}

impl SpeedAndBiasExt for SpeedAndBias {
    fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self[0], self[1], self[2])
    }
    fn gyro_bias(&self) -> Vector3<f64> {
        Vector3::new(self[3], self[4], self[5])
    }
    fn accel_bias(&self) -> Vector3<f64> {
        Vector3::new(self[6], self[7], self[8])
    }
}

// ── ImuParameters ─────────────────────────────────────────────────────────────

/// IMU sensor parameters.
#[derive(Clone)]
pub struct ImuParameters {
    /// Enable IMU in the optimizer.
    pub use_imu: bool,
    /// IMU-to-body extrinsics (body-in-sensor frame).
    pub t_bs: SE3,
    /// Accelerometer saturation threshold [m/s²].
    pub a_max: f64,
    /// Gyroscope saturation threshold [rad/s].
    pub g_max: f64,
    /// Gyroscope noise spectral density [rad/s/√Hz].
    pub sigma_g_c: f64,
    /// Accelerometer noise spectral density [m/s²/√Hz].
    pub sigma_a_c: f64,
    /// Gyroscope bias random walk [rad/s²/√Hz].
    pub sigma_gw_c: f64,
    /// Accelerometer bias random walk [m/s³/√Hz].
    pub sigma_aw_c: f64,
    /// Initial gyroscope bias uncertainty [rad/s].
    pub sigma_bg: f64,
    /// Initial accelerometer bias uncertainty [m/s²].
    pub sigma_ba: f64,
    /// Gravity magnitude [m/s²].
    pub g: f64,
    /// Initial gyroscope bias estimate [rad/s].
    pub g0: Vector3<f64>,
    /// Initial accelerometer bias estimate [m/s²].
    pub a0: Vector3<f64>,
    /// Accelerometer scale factors (diagonal of scale matrix).
    pub s_a: Vector3<f64>,
}

impl ImuParameters {
    /// `R_bs`, the rotation taking a sensor-frame vector to the body frame.
    ///
    /// `t_bs` is **body-from-sensor**, which is the opposite of what its name
    /// suggests to most readers; the accessor exists so the convention is
    /// stated once instead of at every call site.
    pub fn rotation_body_from_sensor(&self) -> Matrix3<f64> {
        self.t_bs
            .rotation_quaternion()
            .to_rotation_matrix()
            .into_inner()
    }
}

// Hand-written because `SE3` has no `Debug`.
impl std::fmt::Debug for ImuParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImuParameters")
            .field("use_imu", &self.use_imu)
            .field("t_bs_translation", &self.t_bs.translation())
            .field("a_max", &self.a_max)
            .field("g_max", &self.g_max)
            .field("sigma_g_c", &self.sigma_g_c)
            .field("sigma_a_c", &self.sigma_a_c)
            .field("sigma_gw_c", &self.sigma_gw_c)
            .field("sigma_aw_c", &self.sigma_aw_c)
            .field("sigma_bg", &self.sigma_bg)
            .field("sigma_ba", &self.sigma_ba)
            .field("g", &self.g)
            .field("g0", &self.g0)
            .field("a0", &self.a0)
            .field("s_a", &self.s_a)
            .finish()
    }
}

impl Default for ImuParameters {
    fn default() -> Self {
        Self {
            use_imu: true,
            t_bs: SE3::identity(),
            a_max: 176.0,
            g_max: 7.8,
            sigma_g_c: 1.7e-4,
            sigma_a_c: 2.0e-3,
            sigma_gw_c: 1.9e-5,
            sigma_aw_c: 3.0e-3,
            sigma_bg: 0.03,
            sigma_ba: 0.1,
            g: 9.81,
            g0: Vector3::zeros(),
            a0: Vector3::zeros(),
            s_a: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

// ── IMU measurements ──────────────────────────────────────────────────────────

/// Raw accelerometer + gyroscope reading.
#[derive(Clone, Debug)]
pub struct ImuSensorReadings {
    /// Angular rate [rad/s].
    pub gyroscopes: Vector3<f64>,
    /// Specific force [m/s²].
    pub accelerometers: Vector3<f64>,
}

/// Timestamped sensor measurement.
#[derive(Clone, Debug)]
pub struct ImuMeasurement {
    /// Measurement timestamp [s].
    pub timestamp: f64,
    /// Raw IMU readings.
    pub measurement: ImuSensorReadings,
}

impl ImuMeasurement {
    /// Create a new timestamped measurement.
    pub fn new(timestamp: f64, measurement: ImuSensorReadings) -> Self {
        Self {
            timestamp,
            measurement,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_and_bias_ext_splits_velocity_gyro_accel() {
        let sb: SpeedAndBias = SVector::from_row_slice(&[
            1.0, 2.0, 3.0, // velocity
            4.0, 5.0, 6.0, // gyro bias
            7.0, 8.0, 9.0, // accel bias
        ]);

        assert_eq!(sb.velocity(), Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(sb.gyro_bias(), Vector3::new(4.0, 5.0, 6.0));
        assert_eq!(sb.accel_bias(), Vector3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn imu_parameters_default_matches_documented_values() {
        let params = ImuParameters::default();

        assert!(params.use_imu);
        assert_eq!(params.a_max, 176.0);
        assert_eq!(params.g_max, 7.8);
        assert_eq!(params.g, 9.81);
        assert_eq!(params.g0, Vector3::zeros());
        assert_eq!(params.a0, Vector3::zeros());
        assert_eq!(params.s_a, Vector3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn imu_measurement_new_stores_timestamp_and_readings() {
        let readings = ImuSensorReadings {
            gyroscopes: Vector3::new(0.1, 0.2, 0.3),
            accelerometers: Vector3::new(0.4, 0.5, 0.6),
        };
        let measurement = ImuMeasurement::new(1.5, readings.clone());

        assert_eq!(measurement.timestamp, 1.5);
        assert_eq!(measurement.measurement.gyroscopes, readings.gyroscopes);
        assert_eq!(
            measurement.measurement.accelerometers,
            readings.accelerometers
        );
    }
}

/// Why a preintegration could not be built over an interval.
///
/// [`ImuPreintegration::new`](crate::factors::imu::ImuPreintegration::new)
/// integrates whatever it is given; `try_new` rejects the inputs that integrate
/// to something misleading rather than to an error.
// No `PartialEq`: `NonPositiveInterval` deliberately carries NaN bounds, and
// `err == err` is false for those — a non-reflexive comparison is a trap.
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum PreintegrationError {
    /// No measurement pair overlapped `[t0, t1]`, so nothing was integrated.
    ///
    /// Both boundaries zero-order-hold the nearest reading when no sample
    /// brackets them, which is exact for constant specific force and degrades
    /// with jerk. That is acceptable across the sub-IMU-period gap a missing
    /// bracket leaves, and unacceptable across an arbitrary one: a buffer lying
    /// wholly outside the interval would otherwise be extrapolated over the
    /// entire span and returned as though it had been measured. Overlap is what
    /// bounds the extrapolation.
    #[error(
        "no IMU measurement pair overlaps [{t0}, {t1}] s; \
         {got} sample(s) span [{first}, {last}] s"
    )]
    EmptyIntegration {
        /// Number of samples supplied.
        got: usize,
        /// Timestamp of the first sample \[s\].
        first: f64,
        /// Timestamp of the last sample \[s\].
        last: f64,
        /// Interval start \[s\].
        t0: f64,
        /// Interval end \[s\].
        t1: f64,
    },

    /// The interval is empty or runs backwards.
    ///
    /// The IMU factors compensate gravity over `delta_t()`, so a non-positive
    /// span does not merely integrate to zero — it applies a zero or
    /// sign-flipped gravity correction to a residual that is still evaluated.
    #[error("preintegration interval must satisfy t1 > t0, got t0 = {t0} s, t1 = {t1} s")]
    NonPositiveInterval {
        /// Interval start \[s\].
        t0: f64,
        /// Interval end \[s\].
        t1: f64,
    },
}
