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

// Re-export sparse solver types
pub use sparse::{
    IterativeSchurSolver, SchurBlockStructure, SchurOrdering, SchurPreconditioner,
    SchurSolverAdapter, SchurVariant, SparseCholeskySolver, SparseQRSolver,
    SparseSchurComplementSolver,
};

// Re-export dense solver types
pub use dense::{DenseCholeskySolver, DenseQRSolver};

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
// StructuredSparseLinearSolver
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

// ============================================================================
// AssemblyMode trait (static dispatch for sparse vs dense paths)
// ============================================================================

/// Marker trait that defines the matrix types for a linear algebra path.
///
/// This trait enables zero-cost static dispatch between sparse and dense
/// linear algebra backends. Optimizers are generic over `AssemblyMode`,
/// so the compiler generates specialized code for each path.
///
/// Two implementations are provided:
/// - [`SparseMode`]: Jacobian and Hessian are `SparseColMat<usize, f64>`
/// - [`DenseMode`]: Jacobian and Hessian are `Mat<f64>`
pub trait AssemblyMode: 'static {
    /// The Jacobian matrix type (SparseColMat or Mat)
    type Jacobian: Send + Sync;
    /// The Hessian matrix type (SparseColMat or Mat)
    type Hessian: Send + Sync;
}

/// Sparse linear algebra mode.
///
/// Uses `SparseColMat<usize, f64>` for Jacobians and Hessians.
/// Optimal for large-scale problems with sparse structure (e.g., pose graphs).
pub struct SparseMode;

impl AssemblyMode for SparseMode {
    type Jacobian = SparseColMat<usize, f64>;
    type Hessian = SparseColMat<usize, f64>;
}

/// Dense linear algebra mode.
///
/// Uses `Mat<f64>` for Jacobians and Hessians.
/// Optimal for small-to-medium problems (< 500 DOF) or dense Jacobians
/// (e.g., bundle adjustment with few cameras).
pub struct DenseMode;

impl AssemblyMode for DenseMode {
    type Jacobian = Mat<f64>;
    type Hessian = Mat<f64>;
}

// ============================================================================
// LinearSolver trait (unified solver interface, generic over AssemblyMode)
// ============================================================================

/// Unified linear solver interface parameterized by [`AssemblyMode`].
///
/// This is the single trait implemented by all linear solvers. When `M` is
/// a concrete type (e.g., `SparseMode`), this trait is object-safe and can
/// be used as `dyn LinearSolver<SparseMode>` or `dyn LinearSolver<DenseMode>`.
///
/// - Sparse solvers (`SparseCholeskySolver`, `SparseQRSolver`, `SchurSolverAdapter`)
///   implement `LinearSolver<SparseMode>`.
/// - Dense solvers (`DenseCholeskySolver`, `DenseQRSolver`)
///   implement `LinearSolver<DenseMode>`.
pub trait LinearSolver<M: AssemblyMode> {
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
