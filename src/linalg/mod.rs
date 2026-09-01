pub mod covariance;
pub mod dense;
pub mod sparse;
pub mod utils;

use crate::core::{VarKey, variable::ManifoldVariable};
use crate::error::ErrorLogging;
use faer::Mat;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashSet;
use std::fmt::{self, Debug, Display, Formatter};
use thiserror::Error;

#[allow(deprecated)] // re-export kept so existing imports of the old name resolve
pub use sparse::{
    IterativeSchurSolver, SchurBlockStructure, SchurOrdering, SchurPreconditioner, SchurVariant,
    SparseCholeskySolver, SparseQRSolver, SparseSchurComplementSolver,
};
pub use sparse::{
    BlockSpan, ChunkLayout, ChunkedSchurEliminator, ColSlot, EliminatedBlocks, ReducedSystem,
    SchurPartition,
};

pub use dense::{DenseCholeskySolver, DenseQRSolver};

pub use covariance::{Covariance, CovarianceAlgorithm, CovarianceError, CovarianceOptions};

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
// Damping
// ============================================================================

/// Damping applied to the normal equations by a trust-region optimizer.
///
/// The augmented system solved by [`LinearSolver::solve_augmented_equation`] is
///
/// ```text
/// (JᵀJ + λ·D) · dx = −Jᵀr,     D_jj = clamp(JᵀJ_jj, min_diagonal, max_diagonal)
/// ```
///
/// This is Ceres' `LevenbergMarquardtStrategy`: damping each column in
/// proportion to that column's own curvature makes the damped step invariant to
/// a rescaling of the parameters, which uniform `λI` damping is not. `D` is
/// formed from the Hessian the solver actually receives — so when Jacobi column
/// scaling is enabled upstream, `D` is the diagonal of the *scaled* `J̃ᵀJ̃`,
/// matching Ceres.
///
/// [`Damping::identity`] sets `min_diagonal == max_diagonal == 1.0`, giving
/// `D = I` and recovering plain `λI` damping exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Damping {
    /// The damping parameter λ. Must be finite and non-negative.
    pub lambda: f64,
    /// Lower clamp on the damping diagonal (Ceres' `min_lm_diagonal`, 1e-6).
    ///
    /// Bounds the damping away from zero for columns with little curvature —
    /// without it, an unconstrained direction would receive no damping at all.
    pub min_diagonal: f64,
    /// Upper clamp on the damping diagonal (Ceres' `max_lm_diagonal`, 1e32).
    ///
    /// Keeps a single very stiff column from dominating the damped system.
    pub max_diagonal: f64,
}

impl Damping {
    /// Marquardt damping with an explicit clamp range.
    ///
    /// # Errors
    ///
    /// Returns [`LinAlgError::InvalidInput`] if `lambda` is negative or not
    /// finite, if either bound is not finite or non-positive, or if
    /// `min_diagonal > max_diagonal` — `f64::clamp` panics on an inverted range,
    /// so the check has to happen at construction.
    pub fn new(lambda: f64, min_diagonal: f64, max_diagonal: f64) -> LinAlgResult<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(LinAlgError::InvalidInput(format!(
                "damping lambda must be finite and non-negative, got {lambda}"
            ))
            .log());
        }
        if !min_diagonal.is_finite() || min_diagonal <= 0.0 {
            return Err(LinAlgError::InvalidInput(format!(
                "min_diagonal must be finite and positive, got {min_diagonal}"
            ))
            .log());
        }
        if max_diagonal.is_nan() || max_diagonal <= 0.0 {
            return Err(LinAlgError::InvalidInput(format!(
                "max_diagonal must be positive, got {max_diagonal}"
            ))
            .log());
        }
        if min_diagonal > max_diagonal {
            return Err(LinAlgError::InvalidInput(format!(
                "min_diagonal ({min_diagonal}) must not exceed max_diagonal ({max_diagonal})"
            ))
            .log());
        }
        Ok(Self {
            lambda,
            min_diagonal,
            max_diagonal,
        })
    }

    /// Uniform `λI` damping — the classic Levenberg form.
    ///
    /// Used where λ is a numerical stabiliser rather than a trust region: Dog
    /// Leg's μ regularisation of the Gauss-Newton step, and Gauss-Newton's own
    /// `min_diagonal` guard against an exactly singular `JᵀJ`.
    pub fn identity(lambda: f64) -> Self {
        Self {
            lambda,
            min_diagonal: 1.0,
            max_diagonal: 1.0,
        }
    }

    /// The damping to add to diagonal entry `(j, j)`, given `JᵀJ_jj`.
    ///
    /// `JᵀJ_jj = ‖J e_j‖² ≥ 0`, so the clamp is well defined for every input a
    /// Gauss-Newton Hessian can produce.
    #[inline]
    pub fn diagonal_term(&self, hessian_diagonal: f64) -> f64 {
        self.lambda * hessian_diagonal.clamp(self.min_diagonal, self.max_diagonal)
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
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        schur_landmark_keys: &HashSet<VarKey>,
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

    /// Solve the augmented equations: `(JᵀJ + λ·D) · dx = −Jᵀr`.
    ///
    /// See [`Damping`] for the definition of `D`. Pass
    /// [`Damping::identity`] for classic uniform `λI` damping.
    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &M::Jacobian,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>>;

    /// `H·v` for the **un-damped** `H = JᵀJ` of the last solve.
    ///
    /// The optimizers evaluate the true quadratic model through this — Dog
    /// Leg's Cauchy point and every predicted cost reduction — so it must never
    /// carry the damping term `λ·D`, which would corrupt the step-quality
    /// ratio ρ.
    ///
    /// This is the *action* of the Hessian rather than the matrix, because that
    /// is all any consumer needs. A backend holding `JᵀJ` multiplies it; a
    /// backend that never forms it can evaluate `Jᵀ(J·v)` from the Jacobian
    /// alone, which is what makes chunk-wise Schur elimination expressible.
    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>>;

    /// The **un-damped** `JᵀJ`, when this backend happens to hold one.
    ///
    /// Diagnostics and visualisation only. `None` is a valid answer — a backend
    /// that eliminates straight from `J` never materializes `JᵀJ` — so callers
    /// must degrade rather than fail.
    fn get_hessian(&self) -> Option<&M::Hessian> {
        None
    }

    /// The gradient `+Jᵀr` from the last solve.
    ///
    /// Note the sign: this is `Jᵀr`, **not** the right-hand side `−Jᵀr`.
    /// Predicted-reduction formulas depend on it, so a backend that publishes
    /// the negated vector silently inverts every ρ computed from it.
    fn get_gradient(&self) -> Option<&Mat<f64>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorLogging;

    // -------------------------------------------------------------------------
    // JacobianMode
    // -------------------------------------------------------------------------

    #[test]
    fn test_jacobian_mode_default_is_sparse() {
        assert_eq!(JacobianMode::default(), JacobianMode::Sparse);
    }

    #[test]
    fn test_jacobian_mode_equality() {
        assert_eq!(JacobianMode::Sparse, JacobianMode::Sparse);
        assert_eq!(JacobianMode::Dense, JacobianMode::Dense);
        assert_ne!(JacobianMode::Sparse, JacobianMode::Dense);
    }

    // -------------------------------------------------------------------------
    // LinearSolverType Display + Default
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_solver_type_default_is_cholesky() {
        assert_eq!(
            LinearSolverType::default(),
            LinearSolverType::SparseCholesky
        );
    }

    #[test]
    fn test_linear_solver_type_display_all_variants() {
        assert_eq!(
            format!("{}", LinearSolverType::SparseCholesky),
            "Sparse Cholesky"
        );
        assert_eq!(format!("{}", LinearSolverType::SparseQR), "Sparse QR");
        assert_eq!(
            format!("{}", LinearSolverType::SparseSchurComplement),
            "Sparse Schur Complement"
        );
        assert_eq!(
            format!("{}", LinearSolverType::DenseCholesky),
            "Dense Cholesky"
        );
        assert_eq!(format!("{}", LinearSolverType::DenseQR), "Dense QR");
    }

    // -------------------------------------------------------------------------
    // LinAlgError Display — one per variant
    // -------------------------------------------------------------------------

    #[test]
    fn test_lin_alg_error_factorization_failed_display() {
        let e = LinAlgError::FactorizationFailed("non-positive definite".into());
        assert!(e.to_string().contains("non-positive definite"));
    }

    #[test]
    fn test_lin_alg_error_singular_matrix_display() {
        let e = LinAlgError::SingularMatrix("rank deficient".into());
        assert!(e.to_string().contains("rank deficient"));
    }

    #[test]
    fn test_lin_alg_error_sparse_matrix_creation_display() {
        let e = LinAlgError::SparseMatrixCreation("bad triplets".into());
        assert!(e.to_string().contains("bad triplets"));
    }

    #[test]
    fn test_lin_alg_error_matrix_conversion_display() {
        let e = LinAlgError::MatrixConversion("size mismatch".into());
        assert!(e.to_string().contains("size mismatch"));
    }

    #[test]
    fn test_lin_alg_error_invalid_input_display() {
        let e = LinAlgError::InvalidInput("null jacobian".into());
        assert!(e.to_string().contains("null jacobian"));
    }

    #[test]
    fn test_lin_alg_error_invalid_state_display() {
        let e = LinAlgError::InvalidState("not initialized".into());
        assert!(e.to_string().contains("not initialized"));
    }

    // -------------------------------------------------------------------------
    // log() / log_with_source() return self
    // -------------------------------------------------------------------------

    #[test]
    fn test_lin_alg_error_log_returns_self() {
        let e = LinAlgError::InvalidInput("log_test".into());
        let returned = e.log();
        assert!(returned.to_string().contains("log_test"));
    }

    #[test]
    fn test_lin_alg_error_log_with_source_returns_self() {
        let e = LinAlgError::SingularMatrix("source_test".into());
        let source = std::io::Error::other("src");
        let returned = e.log_with_source(source);
        assert!(returned.to_string().contains("source_test"));
    }

    // -------------------------------------------------------------------------
    // LinAlgResult type alias
    // -------------------------------------------------------------------------

    #[test]
    fn test_lin_alg_result_ok() {
        let r: LinAlgResult<i32> = Ok(7);
        assert!(matches!(r, Ok(7)));
    }

    #[test]
    fn test_lin_alg_result_err() {
        let r: LinAlgResult<i32> = Err(LinAlgError::InvalidInput("oops".into()));
        assert!(r.is_err());
    }
}
