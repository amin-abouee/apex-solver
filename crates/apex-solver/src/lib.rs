//! # Apex Solver
//!
//! A comprehensive Rust library for nonlinear least squares optimization, specifically designed
//! for computer vision applications such as bundle adjustment, graph-based pose optimization, and SLAM.
//!
//! ## Features
//!
//! - **Multiple Optimization Algorithms**: Gauss-Newton, Levenberg-Marquardt, and Dog Leg solvers
//! - **Flexible Linear Algebra Backend**: Support for both Sparse Cholesky and Sparse QR decomposition
//! - **Configurable Solver System**: Easy-to-use configuration system for algorithm and backend selection
//! - **High Performance**: Built on the faer linear algebra library for optimal performance
//! - **Comprehensive Testing**: Extensive test suite ensuring correctness and reliability
//!
//!
//! ## Solver Types
//!
//! - **Gauss-Newton**: Fast convergence for well-conditioned problems
//! - **Levenberg-Marquardt**: Robust algorithm with adaptive damping
//! - **Dog Leg**: Trust region method combining Gauss-Newton and steepest descent
//!
//! ## Linear Algebra Backends
//!
//! - **Sparse Cholesky**: Efficient for positive definite systems
//! - **Sparse QR**: More robust for rank-deficient or ill-conditioned systems

// Re-export workspace crates
pub use apex_camera_models;
pub use apex_io;
pub use apex_manifolds;

// Create module aliases for backward compatibility
pub mod manifold {
    pub use apex_manifolds::*;
}

pub mod camera_models {
    pub use apex_camera_models::*;
}

// Re-export commonly used types from workspace crates
pub use apex_camera_models::{
    BundleAdjustment, CameraModel, DoubleSphereCamera, EucmCamera, FovCamera, KannalaBrandtCamera,
    LandmarksAndIntrinsics, OnlyIntrinsics, OnlyLandmarks, OnlyPose, OptimizeParams, PinholeCamera,
    PoseAndIntrinsics, RadTanCamera, SelfCalibration, UcmCamera,
};
pub use apex_io::{BalLoader, G2oLoader, Graph, ToroLoader};
pub use apex_manifolds::{
    Interpolatable, LieGroup, ManifoldType, Tangent, rn::Rn, se2::SE2, se3::SE3, so2::SO2, so3::SO3,
};

// Local modules
pub mod core;
pub mod error;
pub mod factors;
pub mod io_utils;
pub mod linalg;
pub mod logger;
pub mod observers;
pub mod optimizer;

// Re-export core types
pub use core::variable::Variable;
pub use error::{ApexSolverError, ApexSolverResult};

// Re-export factor types
pub use factors::{BetweenFactor, Factor, PriorFactor, ProjectionFactor};

// Re-export linear algebra types
pub use linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};

// Re-export logger
pub use logger::{init_logger, init_logger_with_level};

// Re-export optimizer types
pub use optimizer::{
    LevenbergMarquardt, OptObserver, OptObserverVec, OptimizerType, Solver,
    levenberg_marquardt::LevenbergMarquardtConfig,
};

#[cfg(feature = "visualization")]
pub use observers::{RerunObserver, VisualizationConfig};
