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
}

impl CoreError {
    /// Log the error with tracing::error and return self for chaining
    ///
    /// This method allows for a consistent error logging pattern throughout
    /// the core module, ensuring all errors are properly recorded.
    ///
    /// # Example
    /// ```ignore
    /// operation()
    ///     .map_err(|e| CoreError::from(e).log())?;
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
    /// ```ignore
    /// matrix_operation()
    ///     .map_err(|e| {
    ///         CoreError::SymbolicStructure(
    ///             "Failed to build sparse matrix structure".to_string()
    ///         )
    ///         .log_with_source(e)
    ///     })?;
    /// ```
    #[must_use]
    pub fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        error!("{} | Source: {:?}", self, source_error);
        self
    }
}

/// Result type for core module operations
pub type CoreResult<T> = Result<T, CoreError>;
