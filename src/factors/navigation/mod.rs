//! Navigation factors: GNSS (velocity, pseudorange, Doppler range-rate) and
//! auxiliary inertial aids (barometric altitude, attitude from
//! gravity/magnetometer).

pub mod barometric_attitude_factor;
pub mod gps_velocity_factor;
pub mod pseudorange_doppler_factor;

pub use barometric_attitude_factor::{AttitudeFactor, BarometricFactor};
pub use gps_velocity_factor::GpsVelocityFactor;
pub use pseudorange_doppler_factor::{DopplerFactor, PseudorangeFactor};
