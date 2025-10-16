pub mod cholesky;
pub mod qr;

use faer::Mat;
use faer::sparse::SparseColMat;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

impl fmt::Display for LinearSolverType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearSolverType::SparseCholesky => write!(f, "Sparse Cholesky"),
            LinearSolverType::SparseQR => write!(f, "Sparse QR"),
        }
    }
}

/// Linear algebra specific error types for apex-solver
#[derive(Debug, Clone, Error)]
pub enum LinAlgError {
    /// Matrix factorization failed (Cholesky, QR, etc.)
    #[error("Matrix factorization failed: {0}")]
    FactorizationFailed(String),

    /// Singular or near-singular matrix detected
    #[error("Singular matrix detected (matrix is not invertible)")]
    SingularMatrix,

    /// Matrix dimensions are incompatible
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Failed to create sparse matrix from triplets
    #[error("Failed to create sparse matrix: {0}")]
    SparseMatrixCreation(String),

    /// Matrix format conversion failed
    #[error("Matrix conversion failed: {0}")]
    MatrixConversion(String),

    /// Numerical instability detected (NaN, Inf, extreme condition number)
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),

    /// Invalid matrix operation attempted
    #[error("Invalid matrix operation: {0}")]
    InvalidOperation(String),

    /// Linear system solve failed
    #[error("Linear system solve failed: {0}")]
    SolveFailed(String),
}

/// Result type for linear algebra operations
pub type LinAlgResult<T> = Result<T, LinAlgError>;

// /// Contains statistical information about the quality of the optimization solution.
// #[derive(Debug, Clone)]
// pub struct SolverElement {
//     /// The Hessian matrix, computed as `(J^T * W * J)`.
//     ///
//     /// This is `None` if the Hessian could not be computed.
//     pub hessian: Option<SparseColMat<usize, f64>>,

//     /// The gradient vector, computed as `J^T * W * r`.
//     ///
//     /// This is `None` if the gradient could not be computed.
//     pub gradient: Option<Mat<f64>>,

//     /// The parameter covariance matrix, computed as `(J^T * W * J)^-1`.
//     ///
//     /// This is `None` if the Hessian is singular or ill-conditioned.
//     pub covariance_matrix: Option<Mat<f64>>,
//     /// Asymptotic standard errors of the parameters.
//     ///
//     /// This is `None` if the covariance matrix could not be computed.
//     /// Each error is the square root of the corresponding diagonal element
//     /// of the covariance matrix.
//     pub standard_errors: Option<Mat<f64>>,
// }

/// Trait for sparse linear solvers that can solve both normal and augmented equations
pub trait SparseLinearSolver {
    /// Solve the normal equation: (J^T * J) * dx = -J^T * r
    ///
    /// # Errors
    /// Returns `LinAlgError` if:
    /// - Matrix factorization fails
    /// - Matrix is singular or ill-conditioned
    /// - Numerical instability is detected
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>>;

    /// Solve the augmented equation: (J^T * J + λI) * dx = -J^T * r
    ///
    /// # Errors
    /// Returns `LinAlgError` if:
    /// - Matrix factorization fails
    /// - Matrix is singular or ill-conditioned
    /// - Numerical instability is detected
    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>>;

    /// Get the cached Hessian matrix (J^T * J) from the last solve
    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>>;

    /// Get the cached gradient vector (J^T * r) from the last solve
    fn get_gradient(&self) -> Option<&Mat<f64>>;
}

pub use cholesky::SparseCholeskySolver;
pub use qr::SparseQRSolver;
