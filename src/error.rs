//! Error types for the apex-solver library
//!
//! This module provides the main error and result types used throughout the library.
//! All errors use the `thiserror` crate for automatic trait implementations.
//!
//! # Error Hierarchy
//!
//! The library uses a hierarchical error system where:
//! - **`ApexSolverError`** is the top-level error exposed to users via public APIs
//! - **Module errors** (`CoreError`, `OptimizerError`, etc.) are wrapped inside ApexSolverError
//! - **Error sources** are preserved, allowing full error chain inspection
//!
//! Example error chain:
//! ```text
//! ApexSolverError::Core(
//!     CoreError::SymbolicStructure {
//!         message: "Duplicate variable index",
//!         context: "Variable 'x42' at position 15"
//!     }
//! )
//! ```

use crate::{
    core::CoreError, linalg::LinAlgError, observers::ObserverError, optimizer::OptimizerError,
};
use apex_io::IoError;
use apex_manifolds::ManifoldError;
use std::error::Error as StdError;
use thiserror::Error;

/// Main result type used throughout the apex-solver library
pub type ApexSolverResult<T> = Result<T, ApexSolverError>;

/// Main error type for the apex-solver library
///
/// This is the top-level error type exposed by public APIs. It wraps module-specific
/// errors while preserving the full error chain for debugging.
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
    /// Core module errors (problem construction, factors, variables)
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
}

// Module-specific errors are automatically converted via #[from] attributes above
// No manual From implementations needed - thiserror handles it!

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
        // Test automatic conversion via #[from]
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
}
