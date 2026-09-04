//! SGal(3) (Special Galilean) IMU preintegration factors.

pub mod factors;
pub(crate) mod kinematics;

pub use factors::{
    DEFAULT_TIME_SIGMA, Sgal3CombinedImuFactor, Sgal3CombinedStateImuFactor, Sgal3ImuFactor,
    Sgal3StateImuFactor,
};

#[cfg(test)]
mod tests;
