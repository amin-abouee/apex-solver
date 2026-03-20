//! Core optimization components for the apex-solver library
//!
//! This module contains the fundamental building blocks for nonlinear least squares optimization:
//! - Problem formulation and management
//! - Residual blocks
//! - Variables and manifold handling
//! - Loss functions for robust estimation
//! - Correctors for applying loss functions

pub mod corrector;
pub mod loss_functions;
pub mod problem;
pub mod residual_block;
pub mod variable;

use thiserror::Error;
use tracing::error;

/// Core module error types for optimization problems and factors
#[derive(Debug, Clone, Error)]
pub enum CoreError {
    /// Residual block operation failed
    #[error("Residual block error: {0}")]
    ResidualBlock(String),

    /// Variable initialization or constraint error
    #[error("Variable error: {0}")]
    Variable(String),

    /// Factor linearization failed
    #[error("Factor linearization failed: {0}")]
    FactorLinearization(String),

    /// Symbolic structure construction failed
    #[error("Symbolic structure error: {0}")]
    SymbolicStructure(String),

    /// Parallel computation error (thread/mutex failures)
    #[error("Parallel computation error: {0}")]
    ParallelComputation(String),

    /// Dimension mismatch between residual/Jacobian/variables
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Invalid constraint specification (bounds, fixed indices)
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),

    /// Loss function error
    #[error("Loss function error: {0}")]
    LossFunction(String),

    /// Invalid input parameter or configuration
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl CoreError {
    /// Log the error with tracing::error and return self for chaining
    ///
    /// This method allows for a consistent error logging pattern throughout
    /// the core module, ensuring all errors are properly recorded.
    ///
    /// # Example
    /// ```
    /// # use apex_solver::core::CoreError;
    /// # fn operation() -> Result<(), CoreError> { Ok(()) }
    /// # fn example() -> Result<(), CoreError> {
    /// operation()
    ///     .map_err(|e| e.log())?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn log(self) -> Self {
        error!("{}", self);
        self
    }

    /// Log the error with the original source error from a third-party library
    ///
    /// This method logs both the CoreError and the underlying error
    /// from external libraries. This provides full debugging context when
    /// errors occur in third-party code.
    ///
    /// # Arguments
    /// * `source_error` - The original error from the third-party library (must implement Debug)
    ///
    /// # Example
    /// ```
    /// # use apex_solver::core::CoreError;
    /// # fn matrix_operation() -> Result<(), std::io::Error> { Ok(()) }
    /// # fn example() -> Result<(), CoreError> {
    /// matrix_operation()
    ///     .map_err(|e| {
    ///         CoreError::SymbolicStructure(
    ///             "Failed to build sparse matrix structure".to_string()
    ///         )
    ///         .log_with_source(e)
    ///     })?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        error!("{} | Source: {:?}", self, source_error);
        self
    }
}

/// Result type for core module operations
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // CoreError Display — one test per variant
    // -------------------------------------------------------------------------

    #[test]
    fn test_core_error_residual_block_display() {
        let e = CoreError::ResidualBlock("bad block".into());
        assert!(e.to_string().contains("bad block"));
    }

    #[test]
    fn test_core_error_variable_display() {
        let e = CoreError::Variable("bad var".into());
        assert!(e.to_string().contains("bad var"));
    }

    #[test]
    fn test_core_error_factor_linearization_display() {
        let e = CoreError::FactorLinearization("jacobian fail".into());
        assert!(e.to_string().contains("jacobian fail"));
    }

    #[test]
    fn test_core_error_symbolic_structure_display() {
        let e = CoreError::SymbolicStructure("sparse error".into());
        assert!(e.to_string().contains("sparse error"));
    }

    #[test]
    fn test_core_error_parallel_computation_display() {
        let e = CoreError::ParallelComputation("thread panic".into());
        assert!(e.to_string().contains("thread panic"));
    }

    #[test]
    fn test_core_error_dimension_mismatch_display() {
        let e = CoreError::DimensionMismatch("3 vs 6".into());
        assert!(e.to_string().contains("3 vs 6"));
    }

    #[test]
    fn test_core_error_invalid_constraint_display() {
        let e = CoreError::InvalidConstraint("out of bounds".into());
        assert!(e.to_string().contains("out of bounds"));
    }

    #[test]
    fn test_core_error_loss_function_display() {
        let e = CoreError::LossFunction("huber weight".into());
        assert!(e.to_string().contains("huber weight"));
    }

    #[test]
    fn test_core_error_invalid_input_display() {
        let e = CoreError::InvalidInput("null pointer".into());
        assert!(e.to_string().contains("null pointer"));
    }

    // -------------------------------------------------------------------------
    // log() and log_with_source() return self (variant preserved)
    // -------------------------------------------------------------------------

    #[test]
    fn test_core_error_log_returns_self() {
        let e = CoreError::InvalidInput("test_log".into());
        let returned = e.log();
        assert!(returned.to_string().contains("test_log"));
    }

    #[test]
    fn test_core_error_log_with_source_returns_self() {
        let e = CoreError::DimensionMismatch("mismatch".into());
        let source = std::io::Error::new(std::io::ErrorKind::Other, "source error");
        let returned = e.log_with_source(source);
        assert!(returned.to_string().contains("mismatch"));
    }

    // -------------------------------------------------------------------------
    // CoreResult type alias
    // -------------------------------------------------------------------------

    #[test]
    fn test_core_result_ok() {
        let r: CoreResult<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn test_core_result_err() {
        let r: CoreResult<i32> = Err(CoreError::InvalidInput("oops".into()));
        assert!(r.is_err());
    }
}
