//! Error types for the apex-solver library
//!
//! This module provides the main error and result types used throughout the library.
//! All errors use the `thiserror` crate for automatic trait implementations.
//!
//! # Error Hierarchy
//!
//! The library uses a strict three-layer error hierarchy with bubble-up propagation
//! using the `?` operator:
//!
//! - **Layer A (Top/API)**: `ApexSolverError` — exposed to end-users, wraps all module errors
//! - **Layer B (Logic)**: `OptimizerError`, `ObserverError` — wrap Layer C errors with context
//! - **Layer C (Deep/Math)**: `CoreError`, `LinAlgError`, `ManifoldError`, `FactorError` —
//!   module-specific errors that must **never** return `ApexSolverError` directly
//!
//! # Error Propagation Convention
//!
//! Deep modules (Layer C) must return their own module-specific error type
//! (e.g., `Result<T, CoreError>`, not `Result<T, ApexSolverError>`). The `?` operator
//! and `#[from]` attributes handle automatic conversions at each layer boundary.
//!
//! Example error chain (Layer C → B → A):
//!
//! ```text
//! LinAlgError::SingularMatrix
//!   → OptimizerError::LinAlg (via #[from] at Layer B)
//!     → ApexSolverError::Optimizer (via #[from] at Layer A)
//! ```
//!
//! # Dual-Path Convention for LinAlgError
//!
//! `LinAlgError` can convert to `ApexSolverError` via two paths:
//! 1. Direct: `LinAlgError → ApexSolverError::LinearAlgebra(...)` — for standalone linalg usage
//! 2. Through optimizer: `LinAlgError → OptimizerError::LinAlg → ApexSolverError::Optimizer(...)` — during optimization
//!
//! When a `LinAlgError` occurs inside the optimizer, it should propagate through the
//! optimizer layer (path 2) to preserve optimization context. Standalone linalg usage
//! should use path 1 directly.

use crate::{
    core::CoreError, factors::FactorError, linalg::LinAlgError, linearizer::LinearizerError,
    observers::ObserverError, optimizer::OptimizerError,
};
use apex_camera_models::CameraModelError;
use apex_io::IoError;
use apex_manifolds::ManifoldError;
use std::error::Error as StdError;
use thiserror::Error;

/// Main result type used throughout the apex-solver library
pub type ApexSolverResult<T> = Result<T, ApexSolverError>;

/// Main error type for the apex-solver library
///
/// This is the top-level error type (Layer A) exposed by public APIs. It wraps
/// module-specific errors while preserving the full error chain for debugging.
///
/// # Error Chain Access
///
/// You can access the full error chain using the `chain()` method:
///
/// ```no_run
/// # use apex_solver::error::ApexSolverError;
/// # use tracing::warn;
/// # fn solver_optimize() -> Result<(), ApexSolverError> { Ok(()) }
/// # fn example() {
/// if let Err(e) = solver_optimize() {
///     warn!("Error: {}", e);
///     warn!("Full chain: {}", e.chain());
/// }
/// # }
/// ```
#[derive(Debug, Error)]
pub enum ApexSolverError {
    /// Core module errors (problem construction, factors, variables, loss functions)
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Optimization algorithm errors
    #[error(transparent)]
    Optimizer(#[from] OptimizerError),

    /// Linear algebra errors
    #[error(transparent)]
    LinearAlgebra(#[from] LinAlgError),

    /// Manifold operation errors
    #[error(transparent)]
    Manifold(#[from] ManifoldError),

    /// I/O and file parsing errors
    #[error(transparent)]
    Io(#[from] IoError),

    /// Observer/visualization errors
    #[error(transparent)]
    Observer(#[from] ObserverError),

    /// Factor computation errors (projection, between factors, etc.)
    #[error(transparent)]
    Factor(#[from] FactorError),

    /// Linearizer errors (Jacobian assembly, symbolic structure)
    #[error(transparent)]
    Linearizer(#[from] LinearizerError),

    /// Camera model errors (projection, parameter validation, etc.)
    #[error(transparent)]
    Camera(#[from] CameraModelError),
}

// Module-specific errors are automatically converted via #[from] attributes above
// No manual From implementations needed - thiserror handles it!

/// Trait for error logging with chaining support.
///
/// Provides `log()` and `log_with_source()` methods for all error types in the
/// apex-solver library. Implemented as a blanket trait for any type that implements
/// `Display`, so all error enums (`CoreError`, `LinAlgError`, `OptimizerError`,
/// `FactorError`, `LinearizerError`, `ObserverError`, `ApexSolverError`) get these
/// methods automatically without per-type boilerplate.
///
/// # Example
///
/// ```no_run
/// use apex_solver::error::ErrorLogging;
/// use apex_solver::core::CoreError;
///
/// fn operation() -> Result<(), CoreError> { Ok(()) }
///
/// fn example() -> Result<(), CoreError> {
///     // Log and propagate — .log() returns self for chaining with ?
///     operation()
///         .map_err(|e| e.log())?;
///     Ok(())
/// }
/// ```
///
/// # Logging with source context
///
/// ```no_run
/// use apex_solver::error::ErrorLogging;
/// use apex_solver::linalg::LinAlgError;
///
/// fn matrix_op() -> Result<(), std::io::Error> { Ok(()) }
///
/// fn example() -> Result<(), LinAlgError> {
///     matrix_op()
///         .map_err(|e| {
///             LinAlgError::SingularMatrix("matrix is singular".to_string())
///                 .log_with_source(e)
///         })?;
///     Ok(())
/// }
/// ```
pub trait ErrorLogging: Sized + std::fmt::Display {
    /// Log the error with `tracing::error` and return self for chaining.
    ///
    /// This is equivalent to `tracing::error!("{}", self); self` but allows
    /// method chaining with `?` via `.map_err(|e| e.log())`.
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::CoreError;
    /// use apex_solver::error::ErrorLogging;
    ///
    /// let e = CoreError::Variable("missing key".to_string());
    /// let returned = e.log();
    /// assert_eq!(returned.to_string(), "Variable error: missing key");
    /// ```
    fn log(self) -> Self {
        tracing::error!("{}", self);
        self
    }

    /// Log the error with an additional source error for debugging context.
    ///
    /// Logs both the error and the underlying source error (from a third-party
    /// library or internal operation), providing full debugging context.
    ///
    /// # Arguments
    ///
    /// * `source_error` — The original error (must implement `Debug`)
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::linalg::LinAlgError;
    /// use apex_solver::error::ErrorLogging;
    ///
    /// let source = std::io::Error::other("disk full");
    /// let e = LinAlgError::FactorizationFailed("LU decomposition failed".to_string());
    /// let returned = e.log_with_source(source);
    /// assert_eq!(returned.to_string(), "Matrix factorization failed: LU decomposition failed");
    /// ```
    fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        tracing::error!("{} | Source: {:?}", self, source_error);
        self
    }
}

// Blanket implementation: any type that implements Display gets ErrorLogging for free.
// This eliminates the need for per-error-type log()/log_with_source() methods.
impl<T: std::fmt::Display> ErrorLogging for T {}

impl ApexSolverError {
    /// Get the full error chain as a string for logging and debugging.
    ///
    /// This method traverses the error source chain and returns a formatted string
    /// showing the hierarchy of errors from the top-level ApexSolverError down to the
    /// root cause.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use apex_solver::error::ApexSolverError;
    /// # use tracing::warn;
    /// # fn solver_optimize() -> Result<(), ApexSolverError> { Ok(()) }
    /// # fn example() {
    /// match solver_optimize() {
    ///     Ok(result) => { /* ... */ }
    ///     Err(e) => {
    ///         warn!("Optimization failed!");
    ///         warn!("Error chain: {}", e.chain());
    ///         // Output: "Optimizer error: Linear system solve failed →
    ///         //          Linear algebra error: Singular matrix detected"
    ///     }
    /// }
    /// # }
    /// ```
    pub fn chain(&self) -> String {
        let mut chain = vec![self.to_string()];
        let mut source = self.source();

        while let Some(err) = source {
            chain.push(format!("  → {}", err));
            source = err.source();
        }

        chain.join("\n")
    }

    /// Get a compact single-line error chain for logging
    ///
    /// Similar to `chain()` but formats as a single line with arrow separators.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use apex_solver::error::ApexSolverError;
    /// # use apex_solver::core::CoreError;
    /// # use tracing::error;
    /// # fn example() {
    /// # let apex_err = ApexSolverError::Core(CoreError::InvalidInput("test".to_string()));
    /// error!("Operation failed: {}", apex_err.chain_compact());
    /// // Output: "Optimizer error → Linear algebra error → Singular matrix"
    /// # }
    /// ```
    pub fn chain_compact(&self) -> String {
        let mut chain = vec![self.to_string()];
        let mut source = self.source();

        while let Some(err) = source {
            chain.push(err.to_string());
            source = err.source();
        }

        chain.join(" → ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::FactorError;
    use faer::Mat;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    // ========================================================================
    // Layer C (Deep/Math): Simulates failures at the deepest layer
    // ========================================================================

    /// Layer C: Simulates a singular matrix failure at the deepest module layer.
    /// Returns `Result<T, LinAlgError>` — the module-specific error type.
    fn solve_linear_system() -> Result<Mat<f64>, LinAlgError> {
        Err(LinAlgError::SingularMatrix(
            "Simulated singular matrix in solve_linear_system".to_string(),
        ))
    }

    /// Layer C: Simulates a symbolic structure failure in the core module.
    /// Returns `Result<T, CoreError>` — the module-specific error type.
    fn build_structure() -> Result<(), CoreError> {
        Err(CoreError::SymbolicStructure(
            "Simulated duplicate variable index".to_string(),
        ))
    }

    /// Layer C: Simulates a factor dimension mismatch.
    /// Returns `Result<T, FactorError>` — the module-specific error type.
    fn compute_projection() -> Result<(), FactorError> {
        Err(FactorError::InvalidDimension {
            expected: 3,
            actual: 2,
        })
    }

    // ========================================================================
    // Layer B (Logic/Optimization): Calls Layer C using `?` operator
    // ========================================================================

    /// Layer B: Calls the linear solver using `?`.
    /// The `?` operator auto-converts `LinAlgError` → `OptimizerError::LinAlg`
    /// because `OptimizerError` implements `#[from] LinAlgError`.
    fn run_optimization_step() -> Result<Mat<f64>, OptimizerError> {
        let result = solve_linear_system()?;
        Ok(result)
    }

    /// Layer B: Calls the core structure builder using `?`.
    /// The `?` operator auto-converts `CoreError` → `OptimizerError::Core`
    /// because `OptimizerError` implements `#[from] CoreError`.
    fn initialize_optimization() -> Result<(), OptimizerError> {
        build_structure()?;
        Ok(())
    }

    // ========================================================================
    // Layer A (Top/API): The public API function returning `ApexSolverResult`.
    // The `?` operator auto-converts module errors → `ApexSolverError`.
    // ========================================================================

    /// Layer A: Public API function. The `?` operator auto-converts
    /// `OptimizerError` → `ApexSolverError::Optimizer(...)` via `#[from]`.
    fn solver_optimize() -> ApexSolverResult<()> {
        let _ = run_optimization_step()?;
        Ok(())
    }

    /// Layer A: Public API function with core error propagation.
    fn solver_optimize_with_core_error() -> ApexSolverResult<()> {
        initialize_optimization()?;
        Ok(())
    }

    /// Layer A: Public API function with factor error propagation.
    fn solver_optimize_with_factor_error() -> ApexSolverResult<()> {
        compute_projection()?;
        Ok(())
    }

    // ========================================================================
    // Existing tests
    // ========================================================================

    #[test]
    fn test_apex_solver_error_display() {
        let linalg_error = LinAlgError::SingularMatrix("test singular matrix".to_string());
        let error = ApexSolverError::from(linalg_error);
        assert!(error.to_string().contains("Singular matrix"));
    }

    #[test]
    fn test_apex_solver_error_chain() {
        let linalg_error =
            LinAlgError::FactorizationFailed("Cholesky factorization failed".to_string());
        let error = ApexSolverError::from(linalg_error);

        let chain = error.chain();
        assert!(chain.contains("factorization"));
        assert!(chain.contains("Cholesky"));
    }

    #[test]
    fn test_apex_solver_error_chain_compact() {
        let core_error = CoreError::Variable("Invalid variable index".to_string());
        let error = ApexSolverError::from(core_error);

        let chain_compact = error.chain_compact();
        assert!(chain_compact.contains("Invalid variable index"));
    }

    #[test]
    fn test_apex_solver_result_ok() {
        let result: ApexSolverResult<i32> = Ok(42);
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 42);
        }
    }

    #[test]
    fn test_apex_solver_result_err() {
        let core_error = CoreError::ResidualBlock("Test error".to_string());
        let result: ApexSolverResult<i32> = Err(ApexSolverError::from(core_error));
        assert!(result.is_err());
    }

    #[test]
    fn test_transparent_error_conversion() {
        let manifold_error = ManifoldError::DimensionMismatch {
            expected: 3,
            actual: 2,
        };

        let apex_error: ApexSolverError = manifold_error.into();
        assert!(
            matches!(apex_error, ApexSolverError::Manifold(_)),
            "Expected Manifold variant"
        );
    }

    // ========================================================================
    // New tests: Bubble-up error propagation (A → B → C)
    // ========================================================================

    #[test]
    fn test_error_chain_linalg_through_optimizer() -> TestResult {
        let result = solver_optimize();
        let err = result.expect_err("solver_optimize should fail with LinAlgError");

        assert!(
            matches!(err, ApexSolverError::Optimizer(OptimizerError::LinAlg(_))),
            "Expected Optimizer::LinAlg, got {:?}",
            err
        );

        let chain = err.chain();
        assert!(
            chain.contains("Linear algebra error") || chain.contains("Singular matrix"),
            "chain should contain error details: {}",
            chain
        );

        let compact = err.chain_compact();
        assert!(compact.contains("→"), "compact chain should contain →: {}", compact);
        Ok(())
    }

    #[test]
    fn test_error_chain_core_through_optimizer() -> TestResult {
        let result = solver_optimize_with_core_error();
        let err = result.expect_err("should fail with CoreError");

        assert!(
            matches!(err, ApexSolverError::Optimizer(OptimizerError::Core(_))),
            "Expected Optimizer::Core, got {:?}",
            err
        );

        let chain = err.chain();
        assert!(
            chain.contains("Symbolic structure") || chain.contains("duplicate"),
            "chain should contain error details: {}",
            chain
        );
        Ok(())
    }

    #[test]
    fn test_error_chain_factor_direct() -> TestResult {
        let result = solver_optimize_with_factor_error();
        let err = result.expect_err("should fail with FactorError");

        assert!(
            matches!(err, ApexSolverError::Factor(FactorError::InvalidDimension { .. })),
            "Expected Factor::InvalidDimension, got {:?}",
            err
        );

        let compact = err.chain_compact();
        assert!(compact.contains("expected 3"), "compact: {}", compact);
        assert!(compact.contains("got 2"), "compact: {}", compact);
        Ok(())
    }

    #[test]
    fn test_linalg_error_direct_to_apex() -> TestResult {
        let linalg_err = LinAlgError::SingularMatrix("test_direct".to_string());
        let apex_err: ApexSolverError = linalg_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::LinearAlgebra(_)),
            "Expected LinearAlgebra variant for direct LinAlgError conversion"
        );
        Ok(())
    }

    #[test]
    fn test_core_error_direct_to_apex() -> TestResult {
        let core_err = CoreError::SymbolicStructure("test_direct".to_string());
        let apex_err: ApexSolverError = core_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Core(_)),
            "Expected Core variant for direct CoreError conversion"
        );
        Ok(())
    }

    #[test]
    fn test_core_error_through_optimizer_to_apex() -> TestResult {
        let core_err = CoreError::InvalidInput("bad input".to_string());
        let opt_err: OptimizerError = core_err.into();
        let apex_err: ApexSolverError = opt_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Optimizer(OptimizerError::Core(_))),
            "Expected Optimizer::Core variant for CoreError through OptimizerError"
        );
        Ok(())
    }

    #[test]
    fn test_linalg_error_through_optimizer_preserves_context() -> TestResult {
        let linalg_err = LinAlgError::FactorizationFailed("LU decomposition failed".to_string());
        let opt_err: OptimizerError = linalg_err.into();
        let apex_err: ApexSolverError = opt_err.into();

        let chain = apex_err.chain();
        assert!(chain.contains("Linear algebra error"), "chain: {}", chain);
        assert!(chain.contains("LU decomposition"), "chain: {}", chain);
        Ok(())
    }

    #[test]
    fn test_observer_error_to_apex() -> TestResult {
        let obs_err = ObserverError::RerunInitialization("connect failed".to_string());
        let apex_err: ApexSolverError = obs_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Observer(_)),
            "Expected Observer variant"
        );

        let compact = apex_err.chain_compact();
        assert!(compact.contains("Rerun") || compact.contains("connect failed"), "compact: {}", compact);
        Ok(())
    }

    #[test]
    fn test_factor_error_to_apex() -> TestResult {
        let factor_err = FactorError::InvalidProjection("point behind camera".to_string());
        let apex_err: ApexSolverError = factor_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Factor(_)),
            "Expected Factor variant"
        );

        let compact = apex_err.chain_compact();
        assert!(compact.contains("behind camera"), "compact: {}", compact);
        Ok(())
    }

    #[test]
    fn test_all_error_variants_are_accessible() -> TestResult {
        let errors: Vec<ApexSolverError> = vec![
            CoreError::Variable("var".into()).into(),
            OptimizerError::EmptyProblem.into(),
            LinAlgError::SingularMatrix("sing".into()).into(),
            ManifoldError::DimensionMismatch { expected: 1, actual: 2 }.into(),
            ObserverError::InvalidState("bad".into()).into(),
            FactorError::InvalidDimension { expected: 3, actual: 2 }.into(),
            LinearizerError::SymbolicStructure("sym_err".into()).into(),
            CameraModelError::PointBehindCamera { z: -0.5, min_z: 1e-6 }.into(),
        ];

        for err in &errors {
            assert!(!err.to_string().is_empty(), "Error Display should not be empty");
            assert!(!err.chain_compact().is_empty(), "chain_compact should not be empty");
        }
        Ok(())
    }

    #[test]
    fn test_linearizer_error_direct_to_apex() -> TestResult {
        let lin_err = LinearizerError::SymbolicStructure("sparse build failed".to_string());
        let apex_err: ApexSolverError = lin_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Linearizer(_)),
            "Expected Linearizer variant for direct LinearizerError conversion"
        );

        let compact = apex_err.chain_compact();
        assert!(compact.contains("sparse build failed"), "compact: {}", compact);
        Ok(())
    }

    #[test]
    fn test_linearizer_error_through_core_to_apex() -> TestResult {
        let lin_err = LinearizerError::ParallelComputation("lock failure".to_string());
        let core_err: CoreError = lin_err.into();
        let apex_err: ApexSolverError = core_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Core(CoreError::ParallelComputation(_))),
            "Expected Core::ParallelComputation variant for LinearizerError through CoreError, got {:?}",
            apex_err
        );
        Ok(())
    }

    #[test]
    fn test_linearizer_error_through_optimizer_to_apex() -> TestResult {
        let lin_err = LinearizerError::Variable("missing key".to_string());
        let opt_err: OptimizerError = lin_err.into();
        let apex_err: ApexSolverError = opt_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Optimizer(OptimizerError::Linearizer(_))),
            "Expected Optimizer::Linearizer variant for LinearizerError through OptimizerError, got {:?}",
            apex_err
        );
        Ok(())
    }

    #[test]
    fn test_bubble_up_from_linalg_to_optimizer_to_api() -> TestResult {
        let result = solver_optimize();
        let err = result.expect_err("should propagate LinAlgError through OptimizerError");

        assert!(
            matches!(err, ApexSolverError::Optimizer(OptimizerError::LinAlg(_))),
            "Expected LinAlgError wrapped in OptimizerError, got {:?}",
            err
        );

        let source_chain = err.chain();
        assert!(
            source_chain.contains("Singular matrix"),
            "Chain should contain root cause: {}",
            source_chain
        );
        Ok(())
    }

    #[test]
    fn test_bubble_up_from_core_to_optimizer_to_api() -> TestResult {
        let result = solver_optimize_with_core_error();
        let err = result.expect_err("should propagate CoreError through OptimizerError");

        assert!(
            matches!(err, ApexSolverError::Optimizer(OptimizerError::Core(_))),
            "Expected CoreError wrapped in OptimizerError, got {:?}",
            err
        );

        let source_chain = err.chain();
        assert!(
            source_chain.contains("Symbolic structure"),
            "Chain should contain root cause: {}",
            source_chain
        );
        Ok(())
    }

    // ========================================================================
    // CameraModelError → ApexSolverError::Camera (transparent wrap)
    // ========================================================================

    #[test]
    fn test_camera_error_point_behind_camera_direct() -> TestResult {
        let cam_err = CameraModelError::PointBehindCamera { z: -0.5, min_z: 1e-6 };
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let compact = apex_err.chain_compact();
        assert!(compact.contains("behind camera"), "compact: {}", compact);
        assert!(compact.contains("z=-0.5"), "compact should preserve structured field z: {}", compact);
        Ok(())
    }

    #[test]
    fn test_camera_error_focal_length_preserves_fields() -> TestResult {
        let cam_err = CameraModelError::FocalLengthNotPositive { fx: -1.0, fy: 500.0 };
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let msg = apex_err.to_string();
        assert!(msg.contains("fx=-1"), "msg should contain fx: {}", msg);
        assert!(msg.contains("fy=500"), "msg should contain fy: {}", msg);
        Ok(())
    }

    #[test]
    fn test_camera_error_numerical_preserves_fields() -> TestResult {
        let cam_err = CameraModelError::DenominatorTooSmall { denom: 1e-15, threshold: 1e-6 };
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let msg = apex_err.to_string();
        assert!(msg.contains("denom"), "msg: {}", msg);
        assert!(msg.contains("threshold"), "msg should contain threshold: {}", msg);
        Ok(())
    }

    #[test]
    fn test_camera_error_parameter_out_of_range() -> TestResult {
        let cam_err = CameraModelError::ParameterOutOfRange {
            param: "alpha".to_string(),
            value: 1.5,
            min: 0.0,
            max: 1.0,
        };
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let msg = apex_err.to_string();
        assert!(msg.contains("alpha"), "msg: {}", msg);
        assert!(msg.contains("1.5"), "msg should preserve value: {}", msg);
        Ok(())
    }

    #[test]
    fn test_camera_error_all_variants_accessible() -> TestResult {
        let errors: Vec<ApexSolverError> = vec![
            CameraModelError::PointBehindCamera { z: -0.5, min_z: 1e-6 }.into(),
            CameraModelError::PointAtCameraCenter.into(),
            CameraModelError::ProjectionOutOfBounds.into(),
            CameraModelError::PointOutsideImage { x: 100.0, y: 200.0 }.into(),
            CameraModelError::DenominatorTooSmall { denom: 1e-15, threshold: 1e-6 }.into(),
            CameraModelError::NumericalError { operation: "unproject".to_string(), details: "convergence failed".to_string() }.into(),
            CameraModelError::FocalLengthNotPositive { fx: -1.0, fy: 500.0 }.into(),
            CameraModelError::FocalLengthNotFinite { fx: f64::INFINITY, fy: 500.0 }.into(),
            CameraModelError::PrincipalPointNotFinite { cx: f64::NAN, cy: 240.0 }.into(),
            CameraModelError::DistortionNotFinite { name: "k1".to_string(), value: f64::NAN }.into(),
            CameraModelError::ParameterOutOfRange { param: "alpha".to_string(), value: 1.5, min: 0.0, max: 1.0 }.into(),
            CameraModelError::InvalidParams("bad".to_string()).into(),
        ];

        for err in &errors {
            assert!(matches!(err, ApexSolverError::Camera(_)));
            assert!(!err.to_string().is_empty(), "Error Display should not be empty");
            assert!(!err.chain_compact().is_empty(), "chain_compact should not be empty");
        }
        Ok(())
    }
}