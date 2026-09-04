//! Marginalization support: the Gaussian marginal prior over eliminated
//! variables (GTSAM iSAM2 `LinearContainerFactor` analogue).
//!
//! The partial-pose priors used by loop-closure initialization live in
//! [`pose::partial_prior`](super::pose::partial_prior) — they anchor a pose
//! rather than summarize eliminated variables.

pub mod marginal_prior;

pub use marginal_prior::{LocalLogFn, MarginalPriorFactor};
