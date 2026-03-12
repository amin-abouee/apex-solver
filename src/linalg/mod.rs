pub mod dense;
pub mod sparse;
pub mod utils;

use crate::core::problem::VariableEnum;
use faer::{Mat, sparse::SparseColMat};
use std::{
    collections::HashMap,
    fmt::{self, Debug, Display, Formatter},
};
use thiserror::Error;
use tracing::error;

// Re-export sparse solver types (backward compatibility)
pub use sparse::{
    IterativeSchurSolver, SchurBlockStructure, SchurOrdering, SchurPreconditioner,
    SchurSolverAdapter, SchurVariant, SparseCholeskySolver, SparseQRSolver,
    SparseSchurComplementSolver,
};

// Re-export dense solver types
pub use dense::DenseCholeskySolver;

// ============================================================================
// Linear solver type selection
// ============================================================================

#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
    SparseSchurComplement,
    DenseCholesky,
}

impl Display for LinearSolverType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LinearSolverType::SparseCholesky => write!(f, "Sparse Cholesky"),
            LinearSolverType::SparseQR => write!(f, "Sparse QR"),
            LinearSolverType::SparseSchurComplement => write!(f, "Sparse Schur Complement"),
            LinearSolverType::DenseCholesky => write!(f, "Dense Cholesky"),
        }
    }
}

// ============================================================================
// Error types
// ============================================================================

/// Linear algebra specific error types for apex-solver
#[derive(Debug, Clone, Error)]
pub enum LinAlgError {
    /// Matrix factorization failed (Cholesky, QR, etc.)
    #[error("Matrix factorization failed: {0}")]
    FactorizationFailed(String),

    /// Singular or near-singular matrix detected
    #[error("Singular matrix detected: {0}")]
    SingularMatrix(String),

    /// Failed to create sparse matrix from triplets
    #[error("Failed to create sparse matrix: {0}")]
    SparseMatrixCreation(String),

    /// Matrix format conversion failed
    #[error("Matrix conversion failed: {0}")]
    MatrixConversion(String),

    /// Invalid input provided to linear solver
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Solver in invalid state (e.g., initialized incorrectly)
    #[error("Invalid solver state: {0}")]
    InvalidState(String),
}

impl LinAlgError {
    /// Log the error with tracing::error and return self for chaining
    #[must_use]
    pub fn log(self) -> Self {
        error!("{}", self);
        self
    }

    /// Log the error with the original source error from a third-party library
    #[must_use]
    pub fn log_with_source<E: Debug>(self, source_error: E) -> Self {
        error!("{} | Source: {:?}", self, source_error);
        self
    }
}

/// Result type for linear algebra operations
pub type LinAlgResult<T> = Result<T, LinAlgError>;

// ============================================================================
// Generic LinearSolver trait (fundamental abstraction)
// ============================================================================

/// A purely generic linear solver boundary.
///
/// Converts a Jacobian and residual into a parameter step, caching the
/// resulting gradient and Hessian in their native formats.
///
/// This trait uses associated types so that each backend can define its own
/// native matrix representations (e.g., `faer::Mat<f64>` for CPU dense,
/// `faer::sparse::SparseColMat` for CPU sparse, `cudarc::CudaSlice` for GPU).
///
/// # Type Parameters
///
/// - `Vector`: The native 1-D array type (e.g., `faer::Mat<f64>`)
/// - `Jacobian`: The native matrix type for the Jacobian J
/// - `Hessian`: The native matrix type for the Hessian approximation J^T · J
/// - `Error`: A generic error type for factorization or memory failures
pub trait LinearSolver {
    /// The native 1-D array type (e.g., `faer::Mat<f64>` or `CudaSlice<f64>`)
    type Vector;

    /// The native matrix type for the Jacobian (J)
    type Jacobian;

    /// The native matrix type for the Hessian approximation (J^T · J)
    type Hessian;

    /// A generic error type for factorization or memory failures
    type Error: Debug;

    // ========================================================================
    // Core solves
    // ========================================================================

    /// Solve the normal equations: (J^T · J) · dx = −J^T · r
    fn solve_normal_equation(
        &mut self,
        residuals: &Self::Vector,
        jacobian: &Self::Jacobian,
    ) -> Result<Self::Vector, Self::Error>;

    /// Solve the augmented equations: (J^T · J + λI) · dx = −J^T · r
    fn solve_augmented_equation(
        &mut self,
        residuals: &Self::Vector,
        jacobian: &Self::Jacobian,
        lambda: f64,
    ) -> Result<Self::Vector, Self::Error>;

    // ========================================================================
    // Cached state access
    // ========================================================================

    /// Retrieve the cached gradient (g = J^T · r) computed during the last solve.
    fn gradient(&self) -> Option<&Self::Vector>;

    /// Retrieve the cached Hessian (H = J^T · J) computed during the last solve.
    fn hessian(&self) -> Option<&Self::Hessian>;
}

// ============================================================================
// SparseLinearSolver trait (optimizer-facing, object-safe bridge)
// ============================================================================

/// Trait for structured sparse linear solvers that require variable information.
///
/// This trait extends the basic sparse solver interface to support solvers that
/// exploit problem structure (e.g., Schur complement for bundle adjustment).
/// These solvers need access to variable information to partition the problem.
pub trait StructuredSparseLinearSolver {
    /// Initialize the solver's block structure from problem variables.
    fn initialize_structure(
        &mut self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> LinAlgResult<()>;

    /// Solve the normal equation: (J^T * J) * dx = -J^T * r
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>>;

    /// Solve the augmented equation: (J^T * J + λI) * dx = -J^T * r
    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>>;

    /// Get the cached Hessian matrix
    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>>;

    /// Get the cached gradient vector
    fn get_gradient(&self) -> Option<&Mat<f64>>;
}

/// Object-safe trait for linear solvers used by the optimizer.
///
/// This trait uses concrete `faer` types (`Mat<f64>` for vectors,
/// `SparseColMat<usize, f64>` for Jacobians/Hessians) so it can be used
/// as `dyn SparseLinearSolver` for runtime solver selection.
///
/// All CPU solvers implement this trait. Dense solvers accept sparse Jacobians
/// and convert internally. GPU solvers will implement via host-side adapters.
pub trait SparseLinearSolver {
    /// Solve the normal equation: (J^T * J) * dx = -J^T * r
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>>;

    /// Solve the augmented equation: (J^T * J + λI) * dx = -J^T * r
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
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>>;

    /// Get the cached covariance matrix (H^{-1}) computed from the Hessian
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>>;
}

// ============================================================================
// Utility functions
// ============================================================================

/// Extract per-variable covariance blocks from the full covariance matrix.
///
/// Given the full covariance matrix H^{-1} (inverse of information matrix),
/// this function extracts the diagonal blocks corresponding to each individual variable.
pub fn extract_variable_covariances(
    full_covariance: &Mat<f64>,
    variables: &HashMap<String, VariableEnum>,
    variable_index_map: &HashMap<String, usize>,
) -> HashMap<String, Mat<f64>> {
    let mut result = HashMap::new();

    for (var_name, var) in variables {
        if let Some(&start_idx) = variable_index_map.get(var_name) {
            let dim = var.get_size();
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
