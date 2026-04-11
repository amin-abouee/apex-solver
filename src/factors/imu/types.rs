//! IMU-specific type definitions for apex-solver.
//!
//! These mirror the types in `apex-vio-types` so the IMU factors are
//! self-contained inside apex-solver with no cross-workspace dependency.

use apex_manifolds::se3::SE3;
use nalgebra::{SVector, Vector3};

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
