//! Cheirality (points-in-front-of-camera) penalty shared by the camera factors.
//!
//! A projection factor cannot evaluate a landmark behind the camera, but
//! returning zero there would make an invalid configuration look optimal. All
//! camera factors instead emit a bounded, strictly-decreasing-toward-validity
//! penalty so the optimizer is pushed back into the valid region.

/// Baseline cheirality-violation penalty (pixels-equivalent). Chosen to
/// comfortably dominate any plausible in-image residual — an observation is
/// always a finite pixel coordinate, and even a badly-fit-but-valid
/// projection is bounded by roughly the image extent, so a small multiple
/// of a typical image diagonal (a few thousand pixels) is already an
/// unreachable residual for a valid projection — so becoming invalid is
/// never a cheaper escape hatch than fitting the data. Deliberately *not*
/// orders of magnitude larger than that: mixing genuinely huge and
/// pixel-scale residuals/Jacobian entries in the same least-squares problem
/// ill-conditions the normal equations (observed as Cholesky/linear-solve
/// failures in practice).
pub const CHEIRALITY_BASE_PENALTY: f64 = 1.0e4;

/// Penalty growth rate per metre of depth violation, added on top of
/// [`CHEIRALITY_BASE_PENALTY`] so the penalty keeps increasing — and keeps
/// providing a gradient pointing back toward validity — the further behind
/// the camera a point ends up. Kept modest for the same conditioning reason
/// as [`CHEIRALITY_BASE_PENALTY`].
pub const CHEIRALITY_DEPTH_SCALE: f64 = 1.0e3;
