//! IMU factors for visual-inertial and inertial odometry.
//!
//! Each group gets two factors and nothing more, so a choice here is two
//! questions rather than a catalogue:
//!
//! **Which group?** [`se23`] models a keyframe as `(R, t, v)`; [`sgal3`] adds a
//! time coordinate, `(R, t, v, s)`, making the inter-keyframe interval an
//! estimated quantity — pick it for sensor time-offset or rolling-shutter
//! calibration, and `se23` otherwise.
//!
//! **Combined or not?** `ImuFactor` shares one bias variable across the
//! interval and leaves its evolution to a
//! [`bias_random_walk`](bias::bias_random_walk) edge. `CombinedImuFactor` takes
//! a bias per keyframe and embeds the random walk in its trailing six residual
//! rows, needing no such edge. Doing both counts that uncertainty twice.
//!
//! | | `ImuFactor` | `CombinedImuFactor` |
//! |---|---|---|
//! | [`se23`] | 9D, `(SE23, SE23, bias)` | 15D, `(SE23, bias, SE23, bias)` |
//! | [`sgal3`] | 10D, `(SGal3, SGal3, bias)` | 16D, `(SGal3, bias, SGal3, bias)` |
//!
//! Both names exist in both modules, so they are addressed by their group:
//! `imu::se23::ImuFactor`, `imu::sgal3::CombinedImuFactor`.
//!
//! A keyframe is a **single** state variable on the group, not separate pose
//! and velocity blocks — the optimizer's update is then a group right-plus, and
//! the pose/velocity coupling inertial integration produces is the group's job
//! rather than bookkeeping across blocks. All four share one
//! [`ImuPreintegration`] accumulator, and every derivative comes from the
//! group's own operation Jacobians.
//!
//! An initial bias prior (an `EuclideanPriorFactor` on R⁶) is required for
//! observability in every configuration.
//!
//! # Quick start
//!
//! ```ignore
//! use apex_solver::factors::imu::{
//!     ImuPreintegration, ImuParameters, ImuMeasurement,
//!     bias_random_walk, bias_random_walk_noise, se23,
//! };
//!
//! let factor = se23::ImuFactor::new(preintegration);
//! ```

pub mod bias;
pub mod preintegration;
pub mod se23;
pub mod sgal3;
pub mod types;

pub use bias::{bias_random_walk, bias_random_walk_noise};
pub use preintegration::ImuPreintegration;
pub use types::{ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt};
