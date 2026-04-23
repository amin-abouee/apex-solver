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

/// Result type for core module operations
pub type CoreResult<T> = Result<T, CoreError>;

impl From<crate::linearizer::LinearizerError> for CoreError {
    fn from(err: crate::linearizer::LinearizerError) -> Self {
        match err {
            crate::linearizer::LinearizerError::SymbolicStructure(msg) => {
                CoreError::SymbolicStructure(msg)
            }
            crate::linearizer::LinearizerError::ParallelComputation(msg) => {
                CoreError::ParallelComputation(msg)
            }
            crate::linearizer::LinearizerError::Variable(msg) => CoreError::Variable(msg),
            crate::linearizer::LinearizerError::FactorLinearization(msg) => {
                CoreError::FactorLinearization(msg)
            }
            crate::linearizer::LinearizerError::InvalidInput(msg) => {
                CoreError::InvalidInput(msg)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorLogging;

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
        let source = std::io::Error::other("source error");
        let returned = e.log_with_source(source);
        assert!(returned.to_string().contains("mismatch"));
    }

    // -------------------------------------------------------------------------
    // CoreResult type alias
    // -------------------------------------------------------------------------

    #[test]
    fn test_core_result_ok() {
        let r: CoreResult<i32> = Ok(42);
        assert!(matches!(r, Ok(42)));
    }

    #[test]
    fn test_core_result_err() {
        let r: CoreResult<i32> = Err(CoreError::InvalidInput("oops".into()));
        assert!(r.is_err());
    }
}
