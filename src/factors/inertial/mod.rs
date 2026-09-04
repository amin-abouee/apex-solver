//! IMU factors for visual-inertial and inertial odometry.
//!
//! Five factors implement the Forster et al. (2017) on-manifold preintegration
//! approach over two rotation formulations, all sharing the same
//! [`ImuPreintegration`] accumulator and producing an identical 15D weighted
//! residual `[p, q, v, bg, ba]`:
//!
//! SE_2(3) formulation:
//! - [`ImuFactor`]: 4-block layout `(pose_i, sb_i, pose_j, sb_j)` where
//!   `sb = [v, bg, ba]` is the combined 9D speed-and-bias vector.
//! - [`CombinedImuFactor`]: 6-block layout `(pose_i, vel_i, bias_i, pose_j,
//!   vel_j, bias_j)` matching GTSAM's `CombinedImuFactor` convention.
//! - [`CombinedSe23ImuFactor`]: 4-block layout `(state_i, bias_i, state_j,
//!   bias_j)` where `state` is a single `SE23` element fusing pose and
//!   velocity, and `bias` is the 6D `[bg, ba]` block.
//!
//! SGal(3) formulation:
//! - [`Sgal3ImuFactor`]: 4-block layout, kinematic constraint expressed
//!   through the Special Galilean group.
//! - [`Sgal3CombinedImuFactor`]: 6-block layout, SGal(3) kinematics.
//!
//! # Quick start
//!
//! ```ignore
//! use apex_solver::factors::inertial::{
//!     ImuFactor, CombinedImuFactor, CombinedSe23ImuFactor, Sgal3ImuFactor,
//!     Sgal3CombinedImuFactor, ImuPreintegration, ImuParameters, ImuMeasurement,
//!     ImuSensorReadings, SpeedAndBias,
//! };
//! ```

pub mod combined_imu_se23_factors;
pub mod combined_imu_sgal3_factors;
pub mod combined_se23_imu_factor;
pub mod imu_se23_factors;
pub mod imu_sgal3_factors;
pub mod preintegration;
pub mod types;

pub use combined_imu_se23_factors::CombinedImuFactor;
pub use combined_imu_sgal3_factors::Sgal3CombinedImuFactor;
pub use combined_se23_imu_factor::CombinedSe23ImuFactor;
pub use imu_se23_factors::ImuFactor;
pub use imu_sgal3_factors::Sgal3ImuFactor;
pub use preintegration::ImuPreintegration;
pub use types::{ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt};
