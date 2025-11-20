//! Error types for the apex-solver library
//!
//! This module provides the main error and result types used throughout the library.
//! All errors use the `thiserror` crate for automatic trait implementations.

use crate::{io::IoError, linalg::LinAlgError, manifold::ManifoldError, optimizer::OptimizerError};
use std::{
    io::Error,
    num::{ParseFloatError, ParseIntError},
};
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

impl From<Error> for ApexError {
    fn from(err: Error) -> Self {
        ApexError::Io(err.to_string())
    }
}

impl From<ParseFloatError> for ApexError {
    fn from(err: ParseFloatError) -> Self {
        ApexError::InvalidInput(format!("Failed to parse float: {err}"))
    }
}

impl From<ParseIntError> for ApexError {
    fn from(err: ParseIntError) -> Self {
        ApexError::InvalidInput(format!("Failed to parse integer: {err}"))
    }
}

// Convert module-specific errors to ApexError

impl From<LinAlgError> for ApexError {
    fn from(err: LinAlgError) -> Self {
        ApexError::LinearAlgebra(err.to_string())
    }
}

impl From<OptimizerError> for ApexError {
    fn from(err: OptimizerError) -> Self {
        ApexError::Solver(err.to_string())
    }
}

impl From<ManifoldError> for ApexError {
    fn from(err: ManifoldError) -> Self {
        ApexError::Manifold(err.to_string())
    }
}

impl From<IoError> for ApexError {
    fn from(err: IoError) -> Self {
        ApexError::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

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
        let io_error = Error::new(ErrorKind::NotFound, "File not found");
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
