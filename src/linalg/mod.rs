pub mod cholesky;
pub mod qr;

use crate::core::problem::VariableEnum;
use faer::Mat;
use faer::sparse::SparseColMat;
use std::collections::HashMap;
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

    /// Compute the covariance matrix (H^{-1}) by inverting the cached Hessian
    ///
    /// Returns a reference to the covariance matrix if successful, None otherwise.
    /// The covariance matrix represents parameter uncertainty in the tangent space.
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>>;

    /// Get the cached covariance matrix (H^{-1}) computed from the Hessian
    ///
    /// Returns None if covariance has not been computed yet.
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>>;
}

pub use cholesky::SparseCholeskySolver;
pub use qr::SparseQRSolver;

/// Extract per-variable covariance blocks from the full covariance matrix.
///
/// Given the full covariance matrix H^{-1} (inverse of information matrix),
/// this function extracts the diagonal blocks corresponding to each individual variable.
///
/// # Arguments
/// * `full_covariance` - Full covariance matrix of size n×n (from H^{-1})
/// * `variables` - Map of variable names to their Variable objects
/// * `variable_index_map` - Map from variable names to their starting column index in the full matrix
///
/// # Returns
/// HashMap mapping variable names to their covariance matrices in tangent space.
/// For SE3 variables, this would be 6×6 matrices; for SE2, 3×3; etc.
///
pub fn extract_variable_covariances(
    full_covariance: &Mat<f64>,
    variables: &HashMap<String, VariableEnum>,
    variable_index_map: &HashMap<String, usize>,
) -> HashMap<String, Mat<f64>> {
    let mut result = HashMap::new();

    for (var_name, var) in variables {
        // Get the starting column/row index for this variable
        if let Some(&start_idx) = variable_index_map.get(var_name) {
            // Get the tangent space dimension for this variable
            let dim = var.get_size();

            // Extract the block diagonal covariance for this variable
            // This is the block [start_idx:start_idx+dim, start_idx:start_idx+dim]
            let mut var_cov = Mat::zeros(dim, dim);

            for i in 0..dim {
                for j in 0..dim {
                    var_cov[(i, j)] = full_covariance[(start_idx + i, start_idx + j)];
                }
            }

            result.insert(var_name.clone(), var_cov);
        }
    }

    result
}
