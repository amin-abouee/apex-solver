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

pub mod core;
pub mod error;
pub mod factors;
pub mod io;
pub mod linalg;
pub mod manifold;
pub mod optimizer;

// Re-export core types
pub use core::variable::Variable;
pub use error::{ApexError, ApexResult};

// Re-export factor types
pub use factors::{
    BetweenFactorSE2, BetweenFactorSE3, DoubleSphereProjectionFactor, EucmProjectionFactor, Factor,
    KannalaBrandtProjectionFactor, PriorFactor, RadTanProjectionFactor, UcmProjectionFactor,
};

pub use linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
pub use optimizer::{
    LevenbergMarquardt, OptimizerType, Solver, levenberg_marquardt::LevenbergMarquardtConfig,
};
