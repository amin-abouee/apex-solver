//! Core types and utilities for the apex-solver library
//!
//! This module provides fundamental types used throughout the library,
//! including error handling and result types.

use std::fmt;
pub mod factors;

/// Main result type used throughout the apex-solver library
pub type ApexResult<T> = Result<T, ApexError>;

/// Main error type for the apex-solver library
#[derive(Debug, Clone)]
pub enum ApexError {
    /// Linear algebra related errors
    LinearAlgebra(String),
    /// IO related errors (file loading, parsing, etc.)
    Io(String),
    /// Manifold operations errors
    Manifold(String),
    /// Solver related errors
    Solver(String),
    /// General computation errors
    Computation(String),
    /// Invalid input parameters
    InvalidInput(String),
    /// Memory allocation or management errors
    Memory(String),
    /// Convergence failures
    Convergence(String),
}

impl fmt::Display for ApexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApexError::LinearAlgebra(msg) => write!(f, "Linear algebra error: {msg}"),
            ApexError::Io(msg) => write!(f, "IO error: {msg}"),
            ApexError::Manifold(msg) => write!(f, "Manifold error: {msg}"),
            ApexError::Solver(msg) => write!(f, "Solver error: {msg}"),
            ApexError::Computation(msg) => write!(f, "Computation error: {msg}"),
            ApexError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            ApexError::Memory(msg) => write!(f, "Memory error: {msg}"),
            ApexError::Convergence(msg) => write!(f, "Convergence error: {msg}"),
        }
    }
}

impl std::error::Error for ApexError {
    fn description(&self) -> &str {
        match self {
            ApexError::LinearAlgebra(_) => "Linear algebra operation failed",
            ApexError::Io(_) => "IO operation failed",
            ApexError::Manifold(_) => "Manifold operation failed",
            ApexError::Solver(_) => "Solver operation failed",
            ApexError::Computation(_) => "Computation failed",
            ApexError::InvalidInput(_) => "Invalid input provided",
            ApexError::Memory(_) => "Memory operation failed",
            ApexError::Convergence(_) => "Algorithm did not converge",
        }
    }
}

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

/// Convenience macro for creating linear algebra errors
#[macro_export]
macro_rules! linalg_error {
    ($msg:expr) => {
        $crate::core::ApexError::LinearAlgebra($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::ApexError::LinearAlgebra(format!($fmt, $($arg)*))
    };
}

/// Convenience macro for creating solver errors
#[macro_export]
macro_rules! solver_error {
    ($msg:expr) => {
        $crate::core::ApexError::Solver($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::ApexError::Solver(format!($fmt, $($arg)*))
    };
}

/// Convenience macro for creating computation errors
#[macro_export]
macro_rules! computation_error {
    ($msg:expr) => {
        $crate::core::ApexError::Computation($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::ApexError::Computation(format!($fmt, $($arg)*))
    };
}

/// Trait for problems that can be optimized
pub trait Optimizable {
    /// Parameter type (e.g., Vec<f64>, nalgebra::DVector<f64>)
    type Parameters;
    /// Residuals type (e.g., Vec<f64>, nalgebra::DVector<f64>)
    type Residuals;
    /// Jacobian type (e.g., nalgebra::DMatrix<f64>)
    type Jacobian;

    /// Get the weight matrix for the problem.
    fn weights(&self) -> faer::Mat<f64>;

    /// Evaluate the residuals at the given parameters
    fn evaluate(&self, parameters: &Self::Parameters) -> ApexResult<Self::Residuals>;

    /// Evaluate both residuals and Jacobian at the given parameters
    fn evaluate_with_jacobian(
        &self,
        parameters: &Self::Parameters,
    ) -> ApexResult<(Self::Residuals, Self::Jacobian)>;

    /// Get the number of parameters
    fn parameter_count(&self) -> usize;

    /// Get the number of residuals
    fn residual_count(&self) -> usize;

    /// Compute the cost (sum of squared residuals)
    fn cost(&self, parameters: &Self::Parameters) -> ApexResult<f64>;

    /// Optional: provide analytical Jacobian if available
    fn jacobian(&self, parameters: &Self::Parameters) -> ApexResult<Self::Jacobian> {
        // Default implementation uses evaluate_with_jacobian
        let (_, jacobian) = self.evaluate_with_jacobian(parameters)?;
        Ok(jacobian)
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

    #[test]
    fn test_linalg_error_macro() {
        let error = linalg_error!("Test {}", "message");
        match error {
            ApexError::LinearAlgebra(msg) => assert_eq!(msg, "Test message"),
            _ => panic!("Expected LinearAlgebra error"),
        }
    }

    #[test]
    fn test_solver_error_macro() {
        let error = solver_error!("Convergence failed after {} iterations", 100);
        match error {
            ApexError::Solver(msg) => assert_eq!(msg, "Convergence failed after 100 iterations"),
            _ => panic!("Expected Solver error"),
        }
    }

    #[test]
    fn test_computation_error_macro() {
        let error = computation_error!("Division by zero");
        match error {
            ApexError::Computation(msg) => assert_eq!(msg, "Division by zero"),
            _ => panic!("Expected Computation error"),
        }
    }
}
