//! Motion-model constraints: what the platform's own dynamics assert, with no
//! sensor involved.
//!
//! These carry no measurement of their own (beyond a stationary gyro reading),
//! which makes them unusually cheap — and unusually easy to misapply. Each is
//! only valid while its assumption holds, so they are added by whatever detects
//! that condition: a stationarity detector for [`ZeroVelocityFactor`] and
//! [`ZeroAngularRateFactor`], vehicle knowledge for [`NonholonomicFactor`] and
//! [`PlanarMotionFactor`]. Applied where the assumption is violated they are
//! confidently wrong, in the way only a zero-noise constraint can be.

pub mod nonholonomic;
pub mod planar;
pub mod zero_velocity;

pub use nonholonomic::NonholonomicFactor;
pub use planar::PlanarMotionFactor;
pub use zero_velocity::{ZeroAngularRateFactor, ZeroVelocityFactor};

#[cfg(test)]
mod tests;
