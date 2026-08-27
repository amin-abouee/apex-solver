//! Trajectory sample types.

use apex_manifolds::se3::SE3;
use apex_manifolds::so3::SO3;
use nalgebra::Vector3;

/// Nanoseconds in one second, for `f64` conversion paths.
pub(crate) const NANOS_PER_SECOND: f64 = 1e9;

/// Nanoseconds in one second, for exact integer paths.
pub(crate) const NANOS_PER_SEC: u64 = 1_000_000_000;

/// One pose sample from a trajectory file.
///
/// Position and orientation are stored separately rather than as an [`SE3`],
/// for two reasons:
///
/// 1. `SE3` derives neither `Debug` nor `Default`, so a sample holding one
///    could not `#[derive(Debug)]`, and nor could any consumer struct
///    embedding a sample — a failing assert would print nothing.
/// 2. Building through `SE3` would silently repair exactly the malformed
///    quaternions these readers exist to reject.
///
/// `SE3` is available at the boundary through [`Self::se3`] and
/// [`Self::from_se3`] — free for consumers that want it, mandatory for none.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryPose {
    /// Timestamp \[ns\] since the Unix epoch, as ASL datasets store it.
    pub timestamp_ns: u64,
    /// Body position in the world frame \[m\].
    pub position: Vector3<f64>,
    /// Body orientation in the world frame.
    pub orientation: SO3,
}

impl TrajectoryPose {
    /// Build a sample from its parts.
    pub fn new(timestamp_ns: u64, position: Vector3<f64>, orientation: SO3) -> Self {
        Self {
            timestamp_ns,
            position,
            orientation,
        }
    }

    /// Build a sample from an [`SE3`] pose.
    pub fn from_se3(timestamp_ns: u64, pose: &SE3) -> Self {
        Self {
            timestamp_ns,
            position: pose.translation(),
            orientation: pose.rotation_so3(),
        }
    }

    /// This sample as an [`SE3`].
    pub fn se3(&self) -> SE3 {
        SE3::from_translation_so3(self.position, self.orientation.clone())
    }

    /// Timestamp in seconds.
    ///
    /// Lossy at epoch magnitudes: an `f64` second has an ulp of ~476 ns near
    /// 1.4e9 s, so this does not round-trip back to the same `u64`. Use it for
    /// arithmetic against other `f64` times, never as a key or for equality.
    pub fn timestamp_seconds(&self) -> f64 {
        self.timestamp_ns as f64 / NANOS_PER_SECOND
    }
}

/// The inertial columns EuRoC's `state_groundtruth_estimate0` ships alongside
/// the pose. Absent from TUM files and from TUM VI's `mocap0`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InertialState {
    /// Body velocity in the world frame \[m/s\].
    pub velocity: Vector3<f64>,
    /// Gyroscope bias in the IMU frame \[rad/s\].
    pub gyro_bias: Vector3<f64>,
    /// Accelerometer bias in the IMU frame \[m/s²\].
    pub accel_bias: Vector3<f64>,
}

/// Which file format a trajectory is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryFormat {
    /// TUM: whitespace-separated, `timestamp tx ty tz qx qy qz qw`.
    Tum,
    /// ASL/EuRoC CSV, either layout of [`AslLayout`](crate::asl::AslLayout).
    Asl,
}
