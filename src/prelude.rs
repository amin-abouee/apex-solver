//! The common set, for `use apex_solver::prelude::*;`.
//!
//! Factors live under a module named for their sensor modality
//! (`factors::visual::StereoFactor`, `factors::imu::ImuFactor`, …) so the
//! import says which domain a measurement comes from. This prelude carries only
//! the pieces almost every program needs — the problem, the optimizers, the
//! manifolds, and the three factors that appear in nearly any graph. Reach into
//! `factors::<domain>` for anything else.

pub use crate::core::noise::NoiseModel;
pub use crate::core::problem::Problem;
pub use crate::core::variable::Variable;
pub use crate::error::{ApexSolverError, ApexSolverResult};

pub use crate::factors::pose::{BetweenFactor, EuclideanPriorFactor, PriorFactor};
pub use crate::factors::visual::ProjectionFactor;
pub use crate::factors::{
    BundleAdjustment, Factor, LandmarksAndIntrinsics, OnlyIntrinsics, OnlyLandmarks, OnlyPose,
    OptimizeParams, PoseAndIntrinsics, SelfCalibration,
};

pub use crate::linalg::{JacobianMode, LinearSolverType};
pub use crate::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
pub use crate::optimizer::{DogLeg, GaussNewton, Optimizer};

pub use apex_manifolds::{
    LieGroup, ManifoldType, Tangent, rn::Rn, se2::SE2, se3::SE3, se23::SE23, sgal3::SGal3,
    so2::SO2, so3::SO3,
};
