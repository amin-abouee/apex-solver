//! Pose-graph core factors: relative constraints between two variables and
//! unary priors anchoring a single one.
//!
//! These carry no sensor model — they express graph topology and are shared by
//! every front end (odometry, loop closure, marginalization anchoring).

pub mod between;
pub mod partial_prior;
pub mod prior;

pub use between::BetweenFactor;
pub use partial_prior::{PoseRotationPrior, PoseTranslationPrior};
pub use prior::{EuclideanPriorFactor, PriorFactor};
