//! IMU factors for visual-inertial and inertial odometry.
//!
//! This module provides three SE_2(3)-based IMU factors implementing the
//! Forster et al. (2017) on-manifold preintegration approach:
//!
//! - [`ImuFactor`]: 4-block layout `(pose_i, sb_i, pose_j, sb_j)` where
//!   `sb = [v, bg, ba]` is the combined 9D speed-and-bias vector.
//! - [`CombinedImuFactor`]: 6-block layout `(pose_i, vel_i, bias_i, pose_j,
//!   vel_j, bias_j)` matching GTSAM's `CombinedImuFactor` convention.
//! - [`CombinedSe23ImuFactor`]: 4-block layout `(state_i, bias_i, state_j,
//!   bias_j)` where `state` is a single `SE23` element fusing pose and
//!   velocity, and `bias` is the 6D `[bg, ba]` block.
//!
//! All three factors share the same [`ImuPreintegration`] accumulator and
//! produce an identical 15D weighted residual.
//!
//! # Quick start
//!
//! ```ignore
//! use apex_solver::factors::imu::{
//!     ImuFactor, CombinedImuFactor, CombinedSe23ImuFactor, ImuPreintegration,
//!     ImuParameters, ImuMeasurement, ImuSensorReadings, SpeedAndBias,
//! };
//! ```

pub mod combined_imu_factor;
pub mod combined_se23_imu_factor;
pub mod helpers;
pub mod imu_factor;
pub mod preintegration;
pub mod types;

pub use combined_imu_factor::CombinedImuFactor;
pub use combined_se23_imu_factor::CombinedSe23ImuFactor;
pub use imu_factor::ImuFactor;
pub use preintegration::ImuPreintegration;
pub use types::{ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt};
