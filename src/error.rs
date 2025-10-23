//! Error types for the apex-solver library
//!
//! This module provides the main error and result types used throughout the library.
//! All errors use the `thiserror` crate for automatic trait implementations.

use crate::{io, linalg, manifold, optimizer};
use thiserror::Error;

/// Main result type used throughout the apex-solver library
pub type ApexResult<T> = Result<T, ApexError>;

/// Main error type for the apex-solver library
#[derive(Debug, Clone, Error)]
pub enum ApexError {
    /// Linear algebra related errors
    #[error("Linear algebra error: {0}")]
    LinearAlgebra(String),

    /// IO related errors (file loading, parsing, etc.)
    #[error("IO error: {0}")]
    Io(String),

    /// Manifold operations errors
    #[error("Manifold error: {0}")]
    Manifold(String),

    /// Solver related errors
    #[error("Solver error: {0}")]
    Solver(String),

    /// General computation errors
    #[error("Computation error: {0}")]
    Computation(String),

    /// Invalid input parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Memory allocation or management errors
    #[error("Memory error: {0}")]
    Memory(String),

    /// Convergence failures
    #[error("Convergence error: {0}")]
    Convergence(String),

    /// Matrix operation errors
    #[error("Matrix operation error: {0}")]
    MatrixOperation(String),

    /// Thread synchronization errors
    #[error("Thread synchronization error: {0}")]
    ThreadError(String),
}

// Conversions from standard library errors

impl From<std::io::Error> for ApexError {
    fn from(err: std::io::Error) -> Self {
        ApexError::Io(err.to_string())
    }
}

impl From<std::num::ParseFloatError> for ApexError {
    fn from(err: std::num::ParseFloatError) -> Self {
        ApexError::InvalidInput(format!("Failed to parse float: {err}"))
    }
}

impl From<std::num::ParseIntError> for ApexError {
    fn from(err: std::num::ParseIntError) -> Self {
        ApexError::InvalidInput(format!("Failed to parse integer: {err}"))
    }
}

// Convert module-specific errors to ApexError

impl From<linalg::LinAlgError> for ApexError {
    fn from(err: linalg::LinAlgError) -> Self {
        ApexError::LinearAlgebra(err.to_string())
    }
}

impl From<optimizer::OptimizerError> for ApexError {
    fn from(err: optimizer::OptimizerError) -> Self {
        ApexError::Solver(err.to_string())
    }
}

impl From<manifold::ManifoldError> for ApexError {
    fn from(err: manifold::ManifoldError) -> Self {
        ApexError::Manifold(err.to_string())
    }
}

impl From<io::ApexSolverIoError> for ApexError {
    fn from(err: io::ApexSolverIoError) -> Self {
        ApexError::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apex_error_display() {
        let error = ApexError::LinearAlgebra("Matrix is singular".to_string());
        assert_eq!(
            error.to_string(),
            "Linear algebra error: Matrix is singular"
        );
    }

    #[test]
    fn test_apex_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let apex_error = ApexError::from(io_error);

        match apex_error {
            ApexError::Io(msg) => assert!(msg.contains("File not found")),
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_apex_result_ok() {
        let result: ApexResult<i32> = Ok(42);
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 42);
        }
    }

    #[test]
    fn test_apex_result_err() {
        let result: ApexResult<i32> = Err(ApexError::Computation("Test error".to_string()));
        assert!(result.is_err());
    }
}
