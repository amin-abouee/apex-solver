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
//! ## Quick Start
//!
//! ```rust
//! use apex_solver::{
//!     solvers::{SolverConfig, SolverType, AnySolver},
//!     linalg::LinearSolverType,
//!     core::Optimizable,
//! };
//!
//! // Configure the solver
//! let config = SolverConfig::new()
//!     .with_solver_type(SolverType::LevenbergMarquardt)
//!     .with_linear_solver_type(LinearSolverType::SparseCholesky)
//!     .with_max_iterations(100)
//!     .with_cost_tolerance(1e-8);
//!
//! // Create the solver
//! let mut solver = AnySolver::new(config);
//!
//! // Solve your optimization problem
//! // let result = solver.solve(&problem, initial_parameters)?;
//! ```
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
pub mod factors;
pub mod io;
pub mod linalg;
pub mod manifold;
pub mod solvers;

pub use core::*;
pub use linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
pub use solvers::{SolverConfig, SolverType, AnySolver, SolverFactory};
