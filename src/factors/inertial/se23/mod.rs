//! SE_2(3) ("extended pose") IMU preintegration factors.

pub mod factors;
pub(crate) mod kinematics;

pub use factors::{CombinedImuFactor, CombinedSe23ImuFactor, ImuFactor, Se23ImuFactor};

#[cfg(test)]
mod tests;
