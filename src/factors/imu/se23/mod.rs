//! SE_2(3) IMU preintegration factors.

pub mod factors;
pub mod gravity;

pub use factors::{CombinedImuFactor, ImuFactor};
pub use gravity::{
    GravityFrameImuFactor, ScaleGravityImuFactor, into_gravity_frame, into_metric_frame,
};

#[cfg(test)]
mod tests;
