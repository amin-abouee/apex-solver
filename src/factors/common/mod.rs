//! Kernels shared across factor families.
//!
//! Anything used by more than one factor domain lives here rather than being
//! re-derived per module: skew-symmetric and matrix-root helpers, the
//! cheirality penalty, and parameter-block validation.
//!
//! Manifold derivatives are deliberately **not** here. Every group in
//! `apex-manifolds` already reports the Jacobians of its own operations
//! (`act`, `compose`, `log`, `right_plus`, …) through their optional output
//! arguments, using that group's own right/left conventions. Factors ask the
//! manifold rather than re-deriving blocks like `[R | −R·[p]ₓ]` by hand — a
//! hand-rolled copy is a second convention that can silently drift from the
//! group's.

pub mod cheirality;
pub mod math;
pub(crate) mod validate;

#[cfg(test)]
pub(crate) mod test_utils;
