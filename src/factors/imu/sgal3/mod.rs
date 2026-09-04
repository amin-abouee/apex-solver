//! SGal(3) IMU preintegration factors.

pub mod factors;

pub use factors::{CombinedImuFactor, DEFAULT_TIME_SIGMA, ImuFactor};

#[cfg(test)]
mod tests;
