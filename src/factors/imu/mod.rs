//! IMU factors for visual-inertial and inertial odometry.
//!
//! This module provides two SE_2(3)-based IMU factors implementing the
//! Forster et al. (2017) on-manifold preintegration approach:
//!
//! - [`ImuFactor`]: 4-block layout `(pose_i, sb_i, pose_j, sb_j)` where
//!   `sb = [v, bg, ba]` is the combined 9D speed-and-bias vector.
//! - [`CombinedImuFactor`]: 6-block layout `(pose_i, vel_i, bias_i, pose_j,
//!   vel_j, bias_j)` matching GTSAM's `CombinedImuFactor` convention.
//!
//! Both factors share the same [`ImuPreintegration`] accumulator and produce
//! an identical 15D weighted residual.
//!
//! # Quick start
//!
//! ```ignore
//! use apex_solver::factors::imu::{
//!     ImuFactor, CombinedImuFactor, ImuPreintegration,
//!     ImuParameters, ImuMeasurement, ImuSensorReadings, SpeedAndBias,
//! };
//! ```

pub mod combined_imu_factor;
pub mod helpers;
pub mod imu_factor;
pub mod preintegration;
pub mod types;

pub use combined_imu_factor::CombinedImuFactor;
pub use imu_factor::ImuFactor;
pub use preintegration::ImuPreintegration;
pub use types::{ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt};
