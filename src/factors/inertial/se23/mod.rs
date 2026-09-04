//! SE_2(3) IMU preintegration factors.

pub mod factors;

pub use factors::{CombinedImuFactor, ImuFactor};

#[cfg(test)]
mod tests;
