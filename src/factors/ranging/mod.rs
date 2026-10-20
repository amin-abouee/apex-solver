//! Range and bearing factors (UWB, radar, sonar, acoustic AoA).
//!
//! Bearing-only measurements are not camera-specific, so they live here rather
//! than under [`visual`](super::visual); keeping [`bearing`] next to
//! [`bearing_range`] also keeps the two bearing residuals on a single SE(3)
//! convention.

pub mod bearing;
pub mod range;

pub use bearing::BearingFactor;
pub use range::{BearingRangeFactor, PosePointRangeFactor, PosePoseRangeFactor};
