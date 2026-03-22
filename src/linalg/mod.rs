pub mod dense;
pub mod sparse;
pub mod utils;

use crate::core::problem::VariableEnum;
use faer::Mat;
use std::{
    collections::HashMap,
    fmt::{self, Debug, Display, Formatter},
};
use thiserror::Error;
use tracing::error;

pub use sparse::{
    IterativeSchurSolver, SchurBlockStructure, SchurOrdering, SchurPreconditioner, SchurVariant,
    SparseCholeskySolver, SparseQRSolver, SparseSchurComplementSolver,
};

pub use dense::{DenseCholeskySolver, DenseQRSolver};

pub use crate::linearizer::cpu::{DenseMode, LinearizationMode, SparseMode};

// ============================================================================
// Jacobian mode selection
// ============================================================================

/// Controls which Jacobian assembly strategy the Problem uses.
///
/// Set this when constructing a [`Problem`](crate::core::problem::Problem):
/// - `Problem::new(JacobianMode::Sparse)` — sparse (default, best for large-scale problems)
/// - `Problem::new(JacobianMode::Dense)` — dense (best for small-to-medium problems < ~500 DOF)
/// - `Problem::default()` — equivalent to `JacobianMode::Sparse`
///
/// The optimizer reads this field and dispatches to the appropriate assembly path.
/// `LinearSolverType` selects the specific algorithm within the sparse path.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum JacobianMode {
    /// Sparse Jacobian using symbolic structure and `SparseColMat`. Best for large problems.
    #[default]
    Sparse,
    /// Dense Jacobian using `Mat<f64>`. Best for small-to-medium problems (< ~500 DOF).
    Dense,
}

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
    DenseQR,
}

impl Display for LinearSolverType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LinearSolverType::SparseCholesky => write!(f, "Sparse Cholesky"),
            LinearSolverType::SparseQR => write!(f, "Sparse QR"),
            LinearSolverType::SparseSchurComplement => write!(f, "Sparse Schur Complement"),
            LinearSolverType::DenseCholesky => write!(f, "Dense Cholesky"),
            LinearSolverType::DenseQR => write!(f, "Dense QR"),
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
// StructureAware
// ============================================================================

/// For solvers that need variable structure information before solving.
///
/// Implemented by Schur complement solvers, which must partition variables
/// into camera and landmark blocks before performing any linear solves.
/// Call [`initialize_structure`](StructureAware::initialize_structure) once
/// during solver setup, before passing the solver to an optimizer.
pub trait StructureAware {
    /// Initialize the solver's block structure from problem variables.
    fn initialize_structure(
        &mut self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> LinAlgResult<()>;
}

// ============================================================================
// LinearizationMode — re-exported from linearizer/cpu where it is defined
// ============================================================================

// ============================================================================
// LinearSolver trait (unified solver interface, generic over LinearizationMode)
// ============================================================================

/// Unified linear solver interface parameterized by [`LinearizationMode`].
///
/// This is the single trait implemented by all linear solvers. When `M` is
/// a concrete type (e.g., `SparseMode`), this trait is object-safe and can
/// be used as `dyn LinearSolver<SparseMode>` or `dyn LinearSolver<DenseMode>`.
///
/// - Sparse solvers (`SparseCholeskySolver`, `SparseQRSolver`, `SchurSolverAdapter`)
///   implement `LinearSolver<SparseMode>`.
/// - Dense solvers (`DenseCholeskySolver`, `DenseQRSolver`)
///   implement `LinearSolver<DenseMode>`.
pub trait LinearSolver<M: LinearizationMode> {
    /// Solve the normal equations: (J^T · J) · dx = −J^T · r
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &M::Jacobian,
    ) -> LinAlgResult<Mat<f64>>;

    /// Solve the augmented equations: (J^T · J + λI) · dx = −J^T · r
    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &M::Jacobian,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>>;

    /// Get the cached Hessian matrix (J^T · J) from the last solve
    fn get_hessian(&self) -> Option<&M::Hessian>;

    /// Get the cached gradient vector (J^T · r) from the last solve
    fn get_gradient(&self) -> Option<&Mat<f64>>;

    /// Compute the covariance matrix (H^{-1}) by inverting the cached Hessian.
    ///
    /// Returns `None` for solvers that do not support covariance estimation
    /// (e.g., QR solvers, Schur complement solvers). Only Cholesky-based
    /// solvers provide a real implementation.
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        None
    }

    /// Get the cached covariance matrix (H^{-1}) computed from the Hessian.
    ///
    /// Returns `None` if covariance has not been computed or is not supported.
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        None
    }
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
