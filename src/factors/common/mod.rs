//! Kernels shared across factor families.
//!
//! Anything used by more than one factor domain lives here rather than being
//! re-derived per module: skew-symmetric and matrix-root helpers, the
//! pose-times-point Jacobian block, and the cheirality penalty.

pub mod cheirality;
pub mod jacobians;
pub mod math;

#[cfg(test)]
pub(crate) mod test_utils;
