pub mod covariance;
pub mod dense;
pub(crate) mod regularization;
pub mod schur;
pub mod sparse;

use crate::core::{VarKey, variable::ManifoldVariable};
use crate::error::ErrorLogging;
use faer::Mat;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashSet;
use std::fmt::{self, Debug, Display, Formatter};
use thiserror::Error;

pub use schur::{
    BlockSpan, ColSlot, EliminatedBlocks, SchurOrdering, SchurPartition, SchurPreconditioner,
};
pub use sparse::{
    ChunkLayout, ChunkedSchurEliminator, ExplicitSchurVariant, ExplicitSparseSchur,
    ImplicitSparseSchur, PatternFingerprint, ReducedSystem, SparseCholeskySolver, SparseQRSolver,
};

pub use dense::{DenseCholeskySolver, DenseQRSolver, ExplicitDenseSchur};

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
    /// Explicit Schur complement over a sparse Hessian: forms `S` and
    /// factorizes (or PCG-solves) it. Equivalent to Ceres's `SPARSE_SCHUR`.
    /// Sub-configured by [`ExplicitSchurVariant`] and, for its `Iterative`
    /// sub-variant, [`SchurPreconditioner`].
    ExplicitSparseSchur,
    /// Implicit (matrix-free) Schur complement, solved with PCG. Neither `S`
    /// nor `JᵀJ` is ever materialized: the operator reads `J` directly, so its
    /// cost scales with `nnz(J)`. Equivalent to Ceres's `ITERATIVE_SCHUR`.
    /// Sub-configured by [`SchurPreconditioner`].
    ImplicitSparseSchur,
    DenseCholesky,
    DenseQR,
    /// Explicit Schur complement over a dense Hessian: forms `S` densely and
    /// factorizes it. Equivalent to Ceres's `DENSE_SCHUR`.
    ExplicitDenseSchur,
}

impl Display for LinearSolverType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LinearSolverType::SparseCholesky => write!(f, "Sparse Cholesky"),
            LinearSolverType::SparseQR => write!(f, "Sparse QR"),
            LinearSolverType::ExplicitSparseSchur => write!(f, "Explicit Sparse Schur Complement"),
            LinearSolverType::ImplicitSparseSchur => write!(f, "Implicit Sparse Schur Complement"),
            LinearSolverType::DenseCholesky => write!(f, "Dense Cholesky"),
            LinearSolverType::DenseQR => write!(f, "Dense QR"),
            LinearSolverType::ExplicitDenseSchur => write!(f, "Explicit Dense Schur Complement"),
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
    /// Returns [`LinAlgError::InvalidDamping`] if `lambda` is negative or not
    /// finite, if either bound is not finite or non-positive, or if
    /// [`LinAlgError::DampingRange`] if `min_diagonal > max_diagonal` — `f64::clamp` panics on an inverted range,
    /// so the check has to happen at construction.
    pub fn new(lambda: f64, min_diagonal: f64, max_diagonal: f64) -> LinAlgResult<Self> {
        let invalid = |param, requirement, value| {
            LinAlgError::InvalidDamping {
                param,
                requirement,
                value,
            }
            .log()
        };
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(invalid("lambda", "finite and non-negative", lambda));
        }
        if !min_diagonal.is_finite() || min_diagonal <= 0.0 {
            return Err(invalid("min_diagonal", "finite and positive", min_diagonal));
        }
        if max_diagonal.is_nan() || max_diagonal <= 0.0 {
            return Err(invalid("max_diagonal", "positive", max_diagonal));
        }
        if min_diagonal > max_diagonal {
            return Err(LinAlgError::DampingRange {
                min: min_diagonal,
                max: max_diagonal,
            }
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
// Rate-limited diagnostics
// ============================================================================

/// Rate limiter for warnings raised per block inside a solve.
///
/// Every call is counted, but [`WarnOnce::hit`] returns the running count
/// only for the first call and then for each power of ten, so a fallback hit
/// on every iteration logs a handful of lines instead of one per block.
pub(crate) struct WarnOnce {
    hits: std::sync::atomic::AtomicUsize,
}

impl WarnOnce {
    pub(crate) const fn new() -> Self {
        Self {
            hits: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Counts one occurrence; `Some(count)` when it is time to log.
    pub(crate) fn hit(&self) -> Option<usize> {
        let count = self
            .hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(1);
        is_power_of_ten(count).then_some(count)
    }
}

fn is_power_of_ten(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(10) {
        n /= 10;
    }
    n == 1
}

/// Counts preconditioner blocks that would not invert and were replaced by
/// the identity (both Schur PCG solvers).
pub(crate) static SINGULAR_PRECONDITIONER_BLOCKS: WarnOnce = WarnOnce::new();

/// The identity standing in for a preconditioner block that would not
/// invert. A preconditioner only speeds PCG up, so the solve stays correct;
/// the warning says why it may converge more slowly.
pub(crate) fn identity_preconditioner_block(dof: usize) -> nalgebra::DMatrix<f64> {
    if let Some(count) = SINGULAR_PRECONDITIONER_BLOCKS.hit() {
        tracing::warn!(
            count,
            "a {dof}×{dof} preconditioner block is singular and was replaced by the identity \
             ({count} so far); PCG may converge more slowly"
        );
    }
    nalgebra::DMatrix::identity(dof, dof)
}

// ============================================================================
// Error types
// ============================================================================

/// A failure reported by faer, kept as the source of a [`LinAlgError`].
#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
pub enum FaerFailure {
    #[error(transparent)]
    Sparse(#[from] faer::sparse::FaerError),

    #[error(transparent)]
    SparseLlt(#[from] faer::sparse::linalg::LltError),

    #[error(transparent)]
    DenseLlt(#[from] faer::linalg::cholesky::llt::factor::LltError),

    #[error(transparent)]
    Creation(#[from] faer::sparse::CreationError),
}

impl FaerFailure {
    /// A non-positive pivot: the values are not positive definite. Every
    /// other failure (allocation, index type) is about resources, not values.
    fn is_numerical(&self) -> bool {
        matches!(
            self,
            Self::DenseLlt(_) | Self::SparseLlt(faer::sparse::linalg::LltError::Numeric(_))
        )
    }
}

/// Errors of the linear solvers and factorizations.
///
/// Logged once where they are built, and the optimizer and covariance code
/// that call the solvers propagate them with a bare `?` — except the
/// [numerical](LinAlgError::is_numerical) ones. A system that will not
/// factorize at one damping is an expected outcome that a damped optimizer
/// retries with a larger one, so it is not logged where it is built; the
/// caller that gives up on it logs it once, with its chain.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum LinAlgError {
    /// A symbolic analysis (factorization pattern or sparse product) failed.
    #[error("symbolic analysis of {context} failed")]
    Symbolic {
        context: &'static str,
        #[source]
        source: FaerFailure,
    },

    /// A numeric factorization failed: the matrix is singular or not
    /// positive definite.
    #[error("factorization of {context} failed, the matrix is singular or not positive definite")]
    Factorization {
        context: &'static str,
        #[source]
        source: FaerFailure,
    },

    /// A system that stayed singular through every diagonal regularization.
    #[error(
        "{context} is singular after {attempts} regularization attempts (largest shift {max_shift:.2e})"
    )]
    SingularAfterRegularization {
        context: &'static str,
        attempts: usize,
        max_shift: f64,
    },

    /// An eliminated variable whose diagonal block cannot be inverted.
    #[error(
        "H_ee block for variable {key:?} ({dof}×{dof}) is singular even after regularization; \
         the variable is unobserved or its observations are degenerate"
    )]
    SingularBlock { key: VarKey, dof: usize },

    /// Triplets that do not form a sparse matrix of the given shape.
    #[error("cannot build the {rows}×{cols} sparse {context}")]
    SparseMatrix {
        context: &'static str,
        rows: usize,
        cols: usize,
        #[source]
        source: FaerFailure,
    },

    /// A solver used before its structure or system was set up.
    #[error("{component} used before it was initialized")]
    NotInitialized { component: &'static str },

    /// A variable key that is not in the index map.
    #[error("variable {key:?} is not in the variable index map")]
    UnknownVariable { key: VarKey },

    /// Operands whose sizes disagree.
    #[error("{what} has {actual} entries, expected {expected}")]
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },

    /// An index past the end of what it indexes.
    #[error("{what} {index} is out of range for {len}")]
    IndexOutOfRange {
        what: &'static str,
        index: usize,
        len: usize,
    },

    /// A count that does not fit the solver's `u32` indices.
    #[error("{what} {value} exceeds the u32 index range")]
    IndexOverflow { what: &'static str, value: usize },

    /// Two variables marked for elimination that share a factor, so `H_ee`
    /// is not block-diagonal.
    #[error(
        "variables {first:?} and {second:?} are both marked for elimination but are connected \
         by a factor, so H_ee is not block-diagonal and Schur elimination would give a wrong \
         step; eliminate only mutually unconnected variables"
    )]
    ConnectedEliminated { first: VarKey, second: VarKey },

    /// A variable marked for elimination that no residual observes.
    #[error("variable {key:?} is marked for elimination but appears in no residual")]
    UnobservedEliminated { key: VarKey },

    /// A Schur partition with no variable on one side.
    #[error("Schur elimination needs at least one {side} variable")]
    EmptyPartition { side: &'static str },

    /// A column of the system that no variable owns.
    #[error("column {column} of the {total}-column system belongs to no variable")]
    UncoveredColumn { column: usize, total: usize },

    /// A column claimed by two variables.
    #[error("column {column} is claimed by more than one variable (at {key:?})")]
    OverlappingColumns { column: usize, key: VarKey },

    /// Chunks whose row ranges are not contiguous and increasing.
    #[error(
        "chunk {chunk} starts at row {start}, but the previous chunk ended at {previous_end}; \
         chunk-wise elimination needs each eliminated variable's rows to form one contiguous, \
         increasing range"
    )]
    ChunkRows {
        chunk: usize,
        start: usize,
        previous_end: usize,
    },

    /// A damping parameter outside its admissible range.
    #[error("damping {param} must be {requirement}, got {value}")]
    InvalidDamping {
        param: &'static str,
        requirement: &'static str,
        value: f64,
    },

    /// Damping bounds in the wrong order.
    #[error("min_diagonal ({min}) must not exceed max_diagonal ({max})")]
    DampingRange { min: f64, max: f64 },
}

impl LinAlgError {
    /// Whether the system's values, not its structure or inputs, failed: a
    /// non-positive pivot or a singular block. A larger damping can cure it,
    /// so it is not logged where it is built; the caller that gives up on it
    /// logs it (see [`ErrorLogging::log_chain`]).
    pub fn is_numerical(&self) -> bool {
        match self {
            Self::Factorization { source, .. } => source.is_numerical(),
            Self::SingularAfterRegularization { .. } | Self::SingularBlock { .. } => true,
            _ => false,
        }
    }

    /// A failed numeric factorization of `context`, logged here unless it is
    /// [numerical](Self::is_numerical).
    #[track_caller]
    pub(crate) fn factorization(context: &'static str, source: FaerFailure) -> Self {
        let error = Self::Factorization { context, source };
        if error.is_numerical() {
            error
        } else {
            error.log()
        }
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
    ///
    /// Freshness contract: these accessors always describe the last
    /// *successful* solve. A failed solve must leave the previously published
    /// values untouched, so an optimizer retrying after an error never reads a
    /// half-published linearization mixed across two iterations.
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
            format!("{}", LinearSolverType::ExplicitSparseSchur),
            "Explicit Sparse Schur Complement"
        );
        assert_eq!(
            format!("{}", LinearSolverType::ImplicitSparseSchur),
            "Implicit Sparse Schur Complement"
        );
        assert_eq!(
            format!("{}", LinearSolverType::DenseCholesky),
            "Dense Cholesky"
        );
        assert_eq!(format!("{}", LinearSolverType::DenseQR), "Dense QR");
        assert_eq!(
            format!("{}", LinearSolverType::ExplicitDenseSchur),
            "Explicit Dense Schur Complement"
        );
    }

    // -------------------------------------------------------------------------
    // LinAlgError
    // -------------------------------------------------------------------------

    #[test]
    fn a_faer_failure_is_the_source_of_a_factorization_error() {
        use crate::error::ErrorChainExt;
        let error = LinAlgError::Factorization {
            context: "H",
            source: FaerFailure::from(faer::sparse::FaerError::OutOfMemory),
        };
        assert_eq!(
            error.chain_compact(),
            "factorization of H failed, the matrix is singular or not positive definite → \
             OutOfMemory"
        );
    }

    #[test]
    fn warn_once_fires_on_powers_of_ten() {
        let limiter = WarnOnce::new();
        let fired: Vec<usize> = (0..120).filter_map(|_| limiter.hit()).collect();
        assert_eq!(fired, [1, 10, 100]);
    }

    #[test]
    fn display_strings_name_their_values() {
        let range = LinAlgError::DampingRange { min: 2.0, max: 1.0 };
        assert_eq!(
            range.to_string(),
            "min_diagonal (2) must not exceed max_diagonal (1)"
        );
        let missing = LinAlgError::NotInitialized {
            component: "explicit Schur solver",
        };
        assert_eq!(
            missing.to_string(),
            "explicit Schur solver used before it was initialized"
        );
    }
}
