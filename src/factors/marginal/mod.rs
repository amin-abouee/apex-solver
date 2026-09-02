//! Marginalization support: the Gaussian marginal prior over eliminated
//! variables (GTSAM iSAM2 `LinearContainerFactor` analogue) and partial-pose
//! priors used by loop-closure initialization.

pub mod marginal_prior_factor;
pub mod partial_pose_prior;

pub use marginal_prior_factor::MarginalPriorFactor;
pub use partial_pose_prior::{PoseRotationPrior, PoseTranslationPrior};
