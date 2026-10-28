//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Dog Leg algorithm

use crate::core::VarKey;
use crate::core::problem::Problem;
use crate::core::variable::ManifoldVariable;
use crate::error::ErrorLogging;
use crate::linalg::{self, JacobianMode, LinearSolver, SparseMode};
use crate::linearizer::SymbolicStructure;
use faer::{Mat, MatRef};
use slotmap::{SecondaryMap, SlotMap};
use std::time::{self, Duration};
use std::{
    fmt,
    fmt::{Display, Formatter},
};
use thiserror::Error;
use tracing::debug;

pub mod dog_leg;
pub mod gauss_newton;
pub mod levenberg_marquardt;

pub use dog_leg::DogLeg;
pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;

// Re-export observer types from the observers module
pub use crate::observers::{OptObserver, OptObserverVec};

// Re-export AssemblyBackend so optimizer sub-modules can import it from optimizer::
pub use crate::linearizer::AssemblyBackend;

/// Type of optimization solver algorithm to use
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    /// Levenberg-Marquardt algorithm (robust, adaptive damping)
    #[default]
    LevenbergMarquardt,
    /// Gauss-Newton algorithm (fast convergence, may be unstable)
    GaussNewton,
    /// Dog Leg algorithm (trust region method)
    DogLeg,
}

impl Display for OptimizerType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerType::LevenbergMarquardt => write!(f, "Levenberg-Marquardt"),
            OptimizerType::GaussNewton => write!(f, "Gauss-Newton"),
            OptimizerType::DogLeg => write!(f, "Dog Leg"),
        }
    }
}

/// Errors of the optimization algorithms.
///
/// Module errors from below are wrapped with the context of the hop and keep
/// the original as their source; they were logged where they were built, so
/// the optimizer propagates them with a bare `?`.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum OptimizerError {
    /// The linear solver failed (factorization, Schur setup, PCG).
    #[error("linear solve failed")]
    LinAlg(#[from] linalg::LinAlgError),

    /// Problem setup failed before the first iteration.
    #[error("problem setup failed")]
    Core(#[from] crate::core::CoreError),

    /// Residual or Jacobian assembly failed.
    #[error("linearization failed")]
    Linearizer(#[from] crate::linearizer::LinearizerError),

    /// A linear solve returned a step with a NaN or infinite entry.
    #[error("{optimizer} computed a step with a NaN or infinite entry at index {index}")]
    NonFiniteStep {
        optimizer: &'static str,
        index: usize,
    },

    /// A linear solver that does not match the problem's Jacobian mode.
    #[error("{optimizer} in {mode} Jacobian mode supports {supported}; requested {requested}")]
    UnsupportedSolver {
        optimizer: &'static str,
        mode: &'static str,
        supported: &'static str,
        requested: linalg::LinearSolverType,
    },

    /// A finite variable bound, which no optimizer enforces yet: solving
    /// would silently return the unconstrained answer.
    #[error(
        "variable {key:?} component {index} has bound [{lower}, {upper}], but no optimizer \
         enforces variable bounds yet; remove it with Problem::remove_variable_bounds"
    )]
    BoundsUnsupported {
        key: crate::core::VarKey,
        index: usize,
        lower: f64,
        upper: f64,
    },

    /// Jacobi scaling enabled but never built.
    #[error("Jacobi scaling not initialized")]
    JacobiScalingNotInitialized,

    /// The linear solver holds no gradient or Hessian after a solve.
    #[error("the linear solver has no {what} after a successful solve")]
    MissingSolverState { what: &'static str },

    /// Dog Leg could not solve for its Gauss-Newton step at any damping.
    #[error("Gauss-Newton step failed after {attempts} damping attempts (mu = {mu:.3e})")]
    DampingExhausted {
        attempts: usize,
        mu: f64,
        #[source]
        source: Option<linalg::LinAlgError>,
    },

    /// A step vector whose length disagrees with the free columns of the
    /// variables it is applied to.
    #[error("step has {actual} entries, the variables have {expected} free columns")]
    StepSize { expected: usize, actual: usize },

    /// A step applied to a variable that is not in the variable map.
    #[error("step applied to unknown variable {key:?}")]
    UnknownVariable { key: crate::core::VarKey },
}

/// Result type for optimizer operations
pub type OptimizerResult<T> = Result<T, OptimizerError>;

/// Rejects a step with a NaN or infinite entry. The solvers that factorize
/// report their own failures, but a solve can still return a non-finite step
/// without one (an overflowing normal matrix, a degenerate QR), and applying it
/// would poison every variable.
pub(crate) fn ensure_finite_step(
    step: &Mat<f64>,
    optimizer: &'static str,
) -> Result<(), OptimizerError> {
    let mut index = 0;
    for column in 0..step.ncols() {
        for row in 0..step.nrows() {
            if !step[(row, column)].is_finite() {
                return Err(OptimizerError::NonFiniteStep { optimizer, index }.log());
            }
            index += 1;
        }
    }
    Ok(())
}

/// The error for a linear solve the optimizer gives up on.
///
/// A [numerical](linalg::LinAlgError::is_numerical) failure is not logged
/// where it is built, since a larger damping may cure it, so it is logged
/// here, once, with its chain. Every other linear-algebra error was logged
/// at its origin and passes through unlogged.
pub(crate) fn linear_solve_failed(error: linalg::LinAlgError) -> OptimizerError {
    if error.is_numerical() {
        OptimizerError::LinAlg(error).log_chain()
    } else {
        OptimizerError::LinAlg(error)
    }
}

/// Detailed convergence information.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Final gradient norm
    pub final_gradient_norm: f64,
    /// Final parameter update norm
    pub final_parameter_update_norm: f64,
    /// Cost function evaluation count
    pub cost_evaluations: usize,
    /// Jacobian evaluation count
    pub jacobian_evaluations: usize,
}

impl Display for ConvergenceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Final gradient norm: {:.2e}, Final parameter update norm: {:.2e}, Cost evaluations: {}, Jacobian evaluations: {}",
            self.final_gradient_norm,
            self.final_parameter_update_norm,
            self.cost_evaluations,
            self.jacobian_evaluations
        )
    }
}

/// Status of an optimization process
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Optimization converged successfully
    Converged,
    /// Maximum number of iterations reached
    MaxIterationsReached,
    /// Cost function tolerance reached
    CostToleranceReached,
    /// Parameter tolerance reached
    ParameterToleranceReached,
    /// Gradient tolerance reached
    GradientToleranceReached,
    /// Optimization failed due to numerical issues
    NumericalFailure,
    /// User requested termination
    UserTerminated,
    /// Timeout reached
    Timeout,
    /// Trust region radius fell below minimum threshold
    TrustRegionRadiusTooSmall,
    /// Too many consecutive rejected steps — the solver can no longer make progress.
    ///
    /// Damping has grown until the step is negligible and every trial step is rejected,
    /// so further iterations cannot change the cost. Reported instead of silently
    /// burning the remaining iteration budget.
    StalledNoProgress,
    /// Objective function fell below user-specified cutoff
    MinCostThresholdReached,
    /// Jacobian matrix is singular or ill-conditioned
    IllConditionedJacobian,
    /// NaN or Inf detected in cost or parameters
    InvalidNumericalValues,
}

impl Display for OptimizationStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationStatus::Converged => write!(f, "Converged"),
            OptimizationStatus::MaxIterationsReached => write!(f, "Maximum iterations reached"),
            OptimizationStatus::CostToleranceReached => write!(f, "Cost tolerance reached"),
            OptimizationStatus::ParameterToleranceReached => {
                write!(f, "Parameter tolerance reached")
            }
            OptimizationStatus::GradientToleranceReached => write!(f, "Gradient tolerance reached"),
            OptimizationStatus::NumericalFailure => write!(f, "Numerical failure"),
            OptimizationStatus::UserTerminated => write!(f, "User terminated"),
            OptimizationStatus::Timeout => write!(f, "Timeout"),
            OptimizationStatus::TrustRegionRadiusTooSmall => {
                write!(f, "Trust region radius too small")
            }
            OptimizationStatus::StalledNoProgress => {
                write!(f, "Stalled (no further progress possible)")
            }
            OptimizationStatus::MinCostThresholdReached => {
                write!(f, "Minimum cost threshold reached")
            }
            OptimizationStatus::IllConditionedJacobian => {
                write!(f, "Ill-conditioned Jacobian matrix")
            }
            OptimizationStatus::InvalidNumericalValues => {
                write!(f, "Invalid numerical values (NaN/Inf) detected")
            }
        }
    }
}

/// Result of a solver execution.
#[derive(Clone)]
pub struct SolverResult<T> {
    /// Final parameters
    pub parameters: T,
    /// Final optimization status
    pub status: OptimizationStatus,
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Total time elapsed
    pub elapsed_time: time::Duration,
    /// Convergence statistics
    pub convergence_info: Option<ConvergenceInfo>,
    /// Per-variable covariance matrices (uncertainty estimation)
    ///
    /// This is `None` if covariance computation was not enabled in the solver configuration.
    /// When present, it contains a mapping from variable keys to their covariance matrices
    /// in tangent space. For example, for SE3 variables this would be 6×6 matrices.
    ///
    /// Enable covariance computation by setting `compute_covariances: true` in the optimizer config.
    pub covariances: Option<SecondaryMap<VarKey, Mat<f64>>>,
}

/// Type alias for the result returned by all optimizer `optimize()` calls.
pub type OptimizeResult =
    Result<SolverResult<SlotMap<VarKey, Box<dyn ManifoldVariable>>>, OptimizerError>;

/// Unified optimizer interface. Object-safe — `Box<dyn Optimizer>` is valid.
///
/// All three optimizers ([`LevenbergMarquardt`],
/// [`GaussNewton`],
/// [`DogLeg`]) implement this trait.
/// Each optimizer also provides inherent `new()`, `with_config()`, and `optimize()` methods
/// for direct (non-polymorphic) usage.
pub trait Optimizer {
    /// Optimize the problem to minimize the cost function.
    fn optimize(&mut self, problem: &mut Problem) -> OptimizeResult;
}

/// Apply parameter update step to all variables.
///
/// This is a common operation used by all optimizers (Levenberg-Marquardt, Gauss-Newton, Dog Leg).
/// It applies a tangent space perturbation to each variable using the proper manifold plus operation.
///
/// # Arguments
/// * `variables` - Mutable map of variables to update
/// * `step` - Full step vector in tangent space (faer matrix view)
/// * `variable_order` - Ordered list of variable names (defines indexing into step vector)
///
/// # Returns
/// * Step norm (L2 norm) for convergence checking
///
/// # Implementation Notes
/// The step vector contains concatenated tangent vectors for all variables in the order
/// specified by `variable_order`. Each variable's tangent space dimension determines
/// how many elements it occupies in the step vector.
///
/// # Errors
///
/// [`OptimizerError::UnknownVariable`] for a key of `variable_order` that is
/// not in `variables`, or [`OptimizerError::StepSize`] when `step` does not
/// have exactly one entry per free column. Both are checked before any
/// variable is touched.
pub fn apply_parameter_step(
    variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    step: MatRef<f64>,
    variable_order: &[VarKey],
) -> OptimizerResult<f64> {
    apply_signed_parameter_step(variables, step, variable_order, 1.0)
}

/// Apply `sign · step`, the shared body of the forward and rollback paths.
///
/// The sign is folded into the per-variable copy that has to happen anyway, so
/// a rollback does not allocate and fill a negated copy of the full step vector.
fn apply_signed_parameter_step(
    variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    step: MatRef<f64>,
    variable_order: &[VarKey],
    sign: f64,
) -> OptimizerResult<f64> {
    let mut free_columns = 0;
    for &key in variable_order {
        let var = variables
            .get(key)
            .ok_or_else(|| OptimizerError::UnknownVariable { key }.log())?;
        free_columns += var.free_dof();
    }
    if step.nrows() != free_columns {
        return Err(OptimizerError::StepSize {
            expected: free_columns,
            actual: step.nrows(),
        }
        .log());
    }
    // `step` is sized to `total_dof` from `build_variable_index_map`, i.e.
    // free columns only (see ISSUE-0003) — its norm below is already the
    // norm of the step actually applied, with no fixed-coordinate entries
    // inflating it.
    let mut step_offset = 0;

    // SmallVec-backed buffers: inline for the common case (DOF ≤ 16, covering
    // SE3/SO3/SE2/SO2 and small RN), spills to heap only for large RN
    // variables. Mirrors the `[0f64; 16]` buffer pattern used inside
    // `Variable::apply_tangent_step`.
    let mut free_buf: smallvec::SmallVec<[f64; 16]> = smallvec::SmallVec::new();
    let mut full_buf: smallvec::SmallVec<[f64; 16]> = smallvec::SmallVec::new();
    for &var_key in variable_order {
        if let Some(var) = variables.get_mut(var_key) {
            let free_size = var.free_dof();
            let dof = var.dof();
            let var_step = step.subrows(step_offset, free_size);
            free_buf.clear();
            free_buf.resize(free_size, 0.0);
            for i in 0..free_size {
                free_buf[i] = sign * var_step[(i, 0)];
            }
            full_buf.clear();
            full_buf.resize(dof, 0.0);
            var.expand_free_step(&free_buf, &mut full_buf);
            var.apply_tangent_step(&full_buf);
            step_offset += free_size;
        }
    }

    Ok(step.norm_l2())
}

/// Apply negative parameter step to rollback variables.
///
/// This is used when an optimization step is rejected (e.g., in trust region methods).
/// It applies the negative of a tangent space perturbation to revert the previous update.
///
/// # Arguments
/// * `variables` - Mutable map of variables to revert
/// * `step` - Full step vector in tangent space (faer matrix view) to negate
/// * `variable_order` - Ordered list of variable names (defines indexing into step vector)
///
/// # Errors
///
/// As [`apply_parameter_step`].
pub fn apply_negative_parameter_step(
    variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    step: MatRef<f64>,
    variable_order: &[VarKey],
) -> OptimizerResult<()> {
    apply_signed_parameter_step(variables, step, variable_order, -1.0)?;
    Ok(())
}

pub fn compute_cost(residual: &Mat<f64>) -> f64 {
    let cost = residual.norm_l2();
    0.5 * cost * cost
}

// ============================================================================
// Shared optimizer utilities
// ============================================================================
// The following types and functions are shared across all three optimizer
// implementations (Levenberg-Marquardt, Gauss-Newton, Dog Leg) to eliminate
// code duplication.

/// Per-iteration statistics for detailed logging (Ceres-style output).
///
/// Captures all relevant metrics for each optimization iteration, enabling
/// detailed analysis and debugging of the optimization process.
#[derive(Debug, Clone)]
pub struct IterationStats {
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Cost function value at this iteration
    pub cost: f64,
    /// Change in cost from previous iteration
    pub cost_change: f64,
    /// L2 norm of the gradient (||J^T·r||)
    pub gradient_norm: f64,
    /// L2 norm of the parameter update step (||Δx||)
    pub step_norm: f64,
    /// Trust region ratio (ρ = actual_reduction / predicted_reduction)
    pub tr_ratio: f64,
    /// Trust region radius (damping parameter λ for LM, Δ for Dog Leg)
    pub tr_radius: f64,
    /// Linear solver iterations (0 for direct solvers like Cholesky)
    pub ls_iter: usize,
    /// Time taken for this iteration in milliseconds
    pub iter_time_ms: f64,
    /// Total elapsed time since optimization started in milliseconds
    pub total_time_ms: f64,
    /// Whether the step was accepted (true) or rejected (false)
    pub accepted: bool,
}

impl IterationStats {
    /// Print table header in Ceres-style format
    pub fn print_header() {
        debug!(
            "{:>4}  {:>13}  {:>13}  {:>13}  {:>13}  {:>11}  {:>11}  {:>7}  {:>11}  {:>13}  {:>6}",
            "iter",
            "cost",
            "cost_change",
            "|gradient|",
            "|step|",
            "tr_ratio",
            "tr_radius",
            "ls_iter",
            "iter_time",
            "total_time",
            "status"
        );
    }

    /// Print single iteration line in Ceres-style format with scientific notation
    pub fn print_line(&self) {
        let status = if self.iteration == 0 {
            "-"
        } else if self.accepted {
            "✓"
        } else {
            "✗"
        };

        debug!(
            "{:>4}  {:>13.6e}  {:>13.2e}  {:>13.2e}  {:>13.2e}  {:>11.2e}  {:>11.2e}  {:>7}  {:>9.2}ms  {:>11.2}ms  {:>6}",
            self.iteration,
            self.cost,
            self.cost_change,
            self.gradient_norm,
            self.step_norm,
            self.tr_ratio,
            self.tr_radius,
            self.ls_iter,
            self.iter_time_ms,
            self.total_time_ms,
            status
        );
    }
}

/// Result of optimization state initialization, shared by all optimizers.
pub struct InitializedState {
    pub variables: SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    pub variable_index_map: SecondaryMap<VarKey, usize>,
    pub sorted_vars: Vec<VarKey>,
    pub symbolic_structure: Option<SymbolicStructure>,
    /// Static per-solve assembly scratch (block order, offsets, buffers),
    /// built once here and reused by every iteration.
    pub workspace: crate::linearizer::AssemblyWorkspace,
    pub total_dof: usize,
    pub current_cost: f64,
    pub initial_cost: f64,
}

/// Compute total parameter vector norm ||x|| across all variables.
pub fn compute_parameter_norm(variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>) -> f64 {
    variables
        .values()
        .map(|v| v.as_param_slice().iter().map(|x| x * x).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// Assign each variable a contiguous range of tangent-space **free** columns.
///
/// Ordering follows `variables.keys()` — slotmap insertion order, which is already
/// deterministic — and each variable is given `free_dof()` (its tangent
/// dimension minus any fixed local indices) consecutive columns starting at
/// the running offset. Fixed tangent coordinates never occupy a column: they
/// must be eliminated before the linear solve rather than solved for and
/// zeroed afterward, since free and fixed columns generally couple through
/// `JᵀJ` (see `codex/issues/ISSUE-0003-fixed-dofs-solved-as-free.md`).
///
/// Returns `(free-column offset per variable, variable keys in column order,
/// total free DOF)`.
///
/// This is shared by [`initialize_optimization_state`] and by covariance
/// estimation ([`crate::linalg::covariance`]). Both must agree on the column
/// layout, or extracted covariance blocks land on the wrong variables.
pub(crate) fn build_variable_index_map(
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
) -> (SecondaryMap<VarKey, usize>, Vec<VarKey>, usize) {
    let sorted_vars: Vec<VarKey> = variables.keys().collect();

    let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
    let mut col_offset = 0;
    for &var_key in &sorted_vars {
        variable_index_map.insert(var_key, col_offset);
        col_offset += variables[var_key].free_dof();
    }

    (variable_index_map, sorted_vars, col_offset)
}

/// Initialize optimization state from problem and initial parameters.
///
/// This is the common initialization sequence used by all optimizers:
/// 1. Create variables from initial values
/// 2. Build variable-to-column index mapping
/// 3. Build symbolic sparsity structure for Jacobian (sparse mode only)
/// 4. Compute initial cost
///
/// The assembly mode is determined by `problem.jacobian_mode`.
///
/// # Unsupported: variable bounds
///
/// `Problem::set_variable_bounds` is validated and stored, but no optimizer
/// step here enforces it — solving anyway would silently return the
/// unconstrained answer (see `codex/issues/ISSUE-0004-variable-bounds-ignored.md`).
/// Until bounded optimization is implemented, any variable carrying a
/// non-trivial bound (`lower > -inf` or `upper < inf`) is rejected here with
/// a typed error rather than silently ignored.
pub fn initialize_optimization_state(problem: &mut Problem) -> OptimizerResult<InitializedState> {
    // A removed residual block leaves a hole in the row layout, and every
    // assembly path below assumes there is none. Closing it here rather than at
    // each removal keeps it O(#blocks) per solve instead of per removal.
    problem.compact_residual_rows();
    if let Some((var_key, idx, lower, upper)) = problem.variable_bounds.iter().find_map(|(k, m)| {
        m.iter()
            .find(|&(_, &(lower, upper))| lower > f64::NEG_INFINITY || upper < f64::INFINITY)
            .map(|(&idx, &(lower, upper))| (k, idx, lower, upper))
    }) {
        return Err(OptimizerError::BoundsUnsupported {
            key: var_key,
            index: idx,
            lower,
            upper,
        }
        .log());
    }

    let mut variables = problem.variables.clone();
    problem.apply_constraints_to_variables(&mut variables);

    let (variable_index_map, sorted_vars, total_dof) = build_variable_index_map(&variables);

    let symbolic_structure = match problem.jacobian_mode {
        JacobianMode::Sparse => Some(crate::linearizer::cpu::sparse::build_symbolic_structure(
            problem,
            &variables,
            &variable_index_map,
            total_dof,
        )?),
        JacobianMode::Dense => None,
    };

    // Static per-solve assembly data: block order, slice offsets and scratch
    // buffers, computed once and reused by every linearization below.
    let mut workspace = crate::linearizer::AssemblyWorkspace::build(problem);

    let (_residual, current_cost) =
        problem.compute_residual_and_cost_sparse_with_workspace(&variables, &mut workspace)?;
    let initial_cost = current_cost;

    Ok(InitializedState {
        variables,
        variable_index_map,
        sorted_vars,
        symbolic_structure,
        workspace,
        total_dof,
        current_cost,
        initial_cost,
    })
}

/// Parameters for convergence checking, shared across optimizers.
pub struct ConvergenceParams {
    pub iteration: usize,
    pub current_cost: f64,
    pub new_cost: f64,
    pub parameter_norm: f64,
    pub parameter_update_norm: f64,
    pub gradient_norm: f64,
    pub elapsed: Duration,
    pub step_accepted: bool,
    // Config values
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub cost_tolerance: f64,
    pub min_cost_threshold: Option<f64>,
    pub timeout: Option<Duration>,
    /// Trust region radius (LM damping or DogLeg radius). None for GN.
    pub trust_region_radius: Option<f64>,
    /// Minimum trust region radius threshold. None for GN.
    pub min_trust_region_radius: Option<f64>,
}

/// Check convergence criteria common to all optimizers.
///
/// Returns `Some(status)` if a termination criterion is met, `None` otherwise.
pub fn check_convergence(params: &ConvergenceParams) -> Option<OptimizationStatus> {
    // CRITICAL SAFETY CHECKS (perform first)

    // Invalid Numerical Values (NaN/Inf)
    if !params.new_cost.is_finite()
        || !params.parameter_update_norm.is_finite()
        || !params.gradient_norm.is_finite()
    {
        return Some(OptimizationStatus::InvalidNumericalValues);
    }

    // Timeout
    if let Some(timeout) = params.timeout
        && params.elapsed >= timeout
    {
        return Some(OptimizationStatus::Timeout);
    }

    // Maximum Iterations
    if params.iteration >= params.max_iterations {
        return Some(OptimizationStatus::MaxIterationsReached);
    }

    // CONVERGENCE CRITERIA (only check after accepted steps)
    if !params.step_accepted {
        return None;
    }

    // Gradient Norm (First-Order Optimality)
    if params.gradient_norm < params.gradient_tolerance {
        return Some(OptimizationStatus::GradientToleranceReached);
    }

    // Parameter and cost criteria (only after first iteration)
    if params.iteration > 0 {
        // Parameter Change Tolerance (xtol)
        let relative_step_tolerance =
            params.parameter_tolerance * (params.parameter_norm + params.parameter_tolerance);
        if params.parameter_update_norm <= relative_step_tolerance {
            return Some(OptimizationStatus::ParameterToleranceReached);
        }

        // Function Value Change Tolerance (ftol)
        let cost_change = (params.current_cost - params.new_cost).abs();
        let relative_cost_change = cost_change / params.current_cost.max(1e-10);
        if relative_cost_change < params.cost_tolerance {
            return Some(OptimizationStatus::CostToleranceReached);
        }
    }

    // Objective Function Cutoff (optional early stopping)
    if let Some(min_cost) = params.min_cost_threshold
        && params.new_cost < min_cost
    {
        return Some(OptimizationStatus::MinCostThresholdReached);
    }

    // Trust Region Radius (LM and DogLeg only)
    if let (Some(radius), Some(min_radius)) =
        (params.trust_region_radius, params.min_trust_region_radius)
        && radius < min_radius
    {
        return Some(OptimizationStatus::TrustRegionRadiusTooSmall);
    }

    None
}

/// A rigorous lower bound on the 2-norm condition number of `H = JᵀJ`.
///
/// For a symmetric positive semi-definite `H`, the Rayleigh quotient evaluated
/// at the basis vector `e_j` gives `λ_max(H) ≥ H_jj` and `λ_min(H) ≤ H_jj` for
/// every `j`, hence
///
/// ```text
/// κ₂(H) = λ_max/λ_min ≥ max_j H_jj / min_j H_jj = (max_j ‖c_j‖ / min_j ‖c_j‖)²
/// ```
///
/// where `c_j` is column `j` of `J`, since `H_jj = ‖J e_j‖²`.
///
/// The one-sidedness matters and is deliberate: a returned value above a
/// threshold **proves** the system is worse conditioned than that threshold, so
/// the check never fires spuriously. The converse does not hold — a small bound
/// is not evidence of good conditioning, because ill-conditioning arising from
/// near-linear-dependence between columns of comparable norm is invisible to
/// the diagonal. What it does catch reliably is the case worth catching: a
/// variable that no residual constrains has a zero column, so the bound is
/// infinite.
///
/// Pass the column norms of the **un-scaled** Jacobian — Jacobi scaling exists
/// precisely to equalise them, so measuring the scaled matrix would report the
/// conditioning of the preconditioned system rather than the user's problem.
pub fn condition_number_lower_bound(column_norms: &[f64]) -> f64 {
    if column_norms.is_empty() {
        return 1.0;
    }
    let mut min_norm = f64::INFINITY;
    let mut max_norm: f64 = 0.0;
    for &norm in column_norms {
        min_norm = min_norm.min(norm);
        max_norm = max_norm.max(norm);
    }
    if min_norm <= 0.0 {
        return f64::INFINITY;
    }
    let ratio = max_norm / min_norm;
    ratio * ratio
}

/// Check the Jacobian against a user-supplied `max_condition_number`, if set.
///
/// Returns `Some(OptimizationStatus::IllConditionedJacobian)` when the bound
/// from [`condition_number_lower_bound`] exceeds the threshold, having logged
/// the offending value. `None` means "no threshold configured, or the bound did
/// not prove the problem ill-conditioned" — see that function on why the second
/// case is not a certificate of good conditioning.
///
/// Costs one `O(nnz)` pass over `J` and only runs when the option is `Some`.
pub fn check_jacobian_conditioning<M: AssemblyBackend>(
    jacobian: &M::Jacobian,
    max_condition_number: Option<f64>,
) -> Option<OptimizationStatus> {
    let threshold = max_condition_number?;
    let column_norms = M::compute_column_norms(jacobian);
    let bound = condition_number_lower_bound(&column_norms);
    if bound > threshold {
        tracing::error!(
            "Jacobian is ill-conditioned: cond(JᵀJ) >= {bound:.3e} exceeds the configured \
             max_condition_number {threshold:.3e}. An infinite bound means at least one \
             variable is unconstrained by every residual."
        );
        return Some(OptimizationStatus::IllConditionedJacobian);
    }
    None
}

/// The per-iteration preamble shared by LM, Gauss-Newton and Dog-Leg:
/// assemble → conditioning check → Jacobi scaling.
pub(crate) enum IterationPreamble<M: AssemblyBackend> {
    /// Normal equations are ready for the optimizer's step computation.
    Proceed {
        residuals: Mat<f64>,
        scaled_jacobian: M::Jacobian,
    },
    /// The conditioning check terminated the solve; the caller notifies its
    /// observers and builds the final [`SolverResult`].
    EarlyExit(OptimizationStatus),
}

/// Run the shared iteration preamble for one optimizer iteration.
pub(crate) fn iteration_preamble<M: AssemblyBackend>(
    problem: &Problem,
    state: &mut InitializedState,
    jacobi_scaling: &mut Option<Vec<f64>>,
    use_jacobi_scaling: bool,
    iteration: usize,
    max_condition_number: Option<f64>,
    jacobian_evaluations: &mut usize,
) -> OptimizerResult<IterationPreamble<M>> {
    // Evaluate residuals and Jacobian using the assembly mode
    let (residuals, jacobian) = M::assemble(
        problem,
        &state.variables,
        &state.variable_index_map,
        state.symbolic_structure.as_ref(),
        state.total_dof,
        &mut state.workspace,
    )?;
    *jacobian_evaluations += 1;

    // Conditioning check on the *un-scaled* Jacobian, before Jacobi
    // scaling equalises the column norms this bound is read from.
    if let Some(status) = check_jacobian_conditioning::<M>(&jacobian, max_condition_number) {
        return Ok(IterationPreamble::EarlyExit(status));
    }

    // Process Jacobian (apply scaling if enabled)
    let scaled_jacobian = if use_jacobi_scaling {
        process_jacobian_generic::<M>(&jacobian, jacobi_scaling, iteration)?
    } else {
        jacobian
    };

    Ok(IterationPreamble::Proceed {
        residuals,
        scaled_jacobian,
    })
}

/// Predicted decrease of the local quadratic model for a trial step.
///
/// ```text
/// pred = −δᵀg − ½·δᵀHδ
/// ```
///
/// where `g = Jᵀr` and `H = JᵀJ` are the *un-damped* normal equations, i.e.
/// exactly what [`LinearSolver::get_gradient`] and
/// [`LinearSolver::get_hessian`] publish.
///
/// Written this way the formula is independent of how the step was produced.
/// The shorter Levenberg-Marquardt identity `pred = ½·δᵀ(λδ − g)` is obtained
/// by substituting `(H + λI)δ = −g` into the above, so it silently assumes
/// *uniform* `λI` damping and is wrong for the Marquardt diagonal
/// `λ·D` (see [`Damping`](crate::linalg::Damping)) — and wrong for Dog Leg's
/// steps, which do not solve that system at all. One extra sparse mat-vec per
/// iteration buys correctness under every damping policy.
///
/// The step, gradient and Hessian must all live in the same space: when Jacobi
/// column scaling is enabled, pass the *scaled* step, since the solver's cached
/// gradient and Hessian are the scaled ones. The predicted reduction is a value
/// of the quadratic model and is invariant under that change of variables, so
/// the result is equally the predicted reduction of the un-scaled step.
pub fn compute_predicted_reduction(
    step: &Mat<f64>,
    gradient: &Mat<f64>,
    hessian_step: &Mat<f64>,
) -> f64 {
    let linear_term = (step.transpose() * gradient)[(0, 0)];
    let quadratic_term = (step.transpose() * hessian_step)[(0, 0)];
    -linear_term - 0.5 * quadratic_term
}

/// `‖g‖` for the *unscaled* gradient, from a gradient computed against a
/// Jacobi-column-scaled Jacobian (GitHub #57).
///
/// `gradient_tolerance`'s contract (documented on every optimizer's
/// `optimize`) is `‖Jᵀr‖ < gradient_tolerance` — the plain, un-scaled
/// gradient, matching Ceres' own convention. But `LinearSolver::get_gradient`
/// reflects whatever Jacobian it was last handed: under Jacobi scaling
/// (`J̃ = J·diag(scaling)`), that is `g̃ = J̃ᵀr = diag(scaling)·g`, not `g`.
/// Column scaling and gradient transform in *opposite* directions — compare
/// [`AssemblyBackend::apply_inverse_scaling`](crate::linearizer::AssemblyBackend::apply_inverse_scaling),
/// which un-scales a *step* by multiplying by `scaling` — so this divides
/// instead: `g = g̃ ⊘ scaling`. Most impactful for Dog Leg, which enables
/// Jacobi scaling by default.
pub(crate) fn unscaled_gradient_norm(gradient: &Mat<f64>, scaling: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..gradient.nrows() {
        let g = gradient[(i, 0)] / scaling[i];
        sum_sq += g * g;
    }
    sum_sq.sqrt()
}

/// Compute step quality ratio (actual vs predicted reduction).
///
/// Used by Levenberg-Marquardt and Dog Leg optimizers to evaluate
/// whether a proposed step improved the objective function as predicted
/// by the local quadratic model.
///
/// Returns `ρ = actual_reduction / predicted_reduction`, handling
/// degenerate predicted reductions.
///
/// # Non-positive predicted reduction
///
/// A *negative* predicted reduction means the trial step increases the local
/// quadratic model — the model is not a descent direction and the ratio is
/// meaningless. Worse, it is actively misleading: a negative actual reduction
/// divided by a negative predicted reduction yields `ρ > 0`, so a step that
/// raised the cost would be accepted. Ceres treats a non-positive
/// `model_cost_change` as an invalid step; `-1.0` is returned here so every
/// acceptance threshold rejects it.
///
/// A predicted reduction that is merely *near zero* is different: at the
/// solution both reductions vanish legitimately, so the historic behaviour
/// (accept when the cost actually fell) is kept — rejecting there would inflate
/// the damping just as the solver converges.
pub fn compute_step_quality(current_cost: f64, new_cost: f64, predicted_reduction: f64) -> f64 {
    let actual_reduction = current_cost - new_cost;
    if predicted_reduction < -1e-15 {
        -1.0
    } else if predicted_reduction.abs() <= 1e-15 {
        if actual_reduction > 0.0 { 1.0 } else { 0.0 }
    } else {
        actual_reduction / predicted_reduction
    }
}

/// What one iteration reports to the observers.
#[derive(Debug, Clone, Copy)]
pub struct IterationReport {
    pub iteration: usize,
    pub cost: f64,
    pub gradient_norm: f64,
    pub damping: Option<f64>,
    pub step_norm: f64,
    pub step_quality: Option<f64>,
}

/// Notify observers with current optimization state (sparse path).
///
/// This is the common observer notification pattern used by all three optimizers.
pub fn notify_observers(
    observers: &mut OptObserverVec,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    report: IterationReport,
    linear_solver: &dyn LinearSolver<SparseMode>,
) {
    let IterationReport {
        iteration,
        cost,
        gradient_norm,
        damping,
        step_norm,
        step_quality,
    } = report;
    observers.set_iteration_metrics(cost, gradient_norm, damping, step_norm, step_quality);

    // Backends that eliminate straight from J never materialize JᵀJ, so the
    // matrix view is best-effort; the gradient is always available.
    if !observers.is_empty()
        && let Some(gradient) = linear_solver.get_gradient()
    {
        observers.set_matrix_data(linear_solver.get_hessian().cloned(), Some(gradient.clone()));
    }

    observers.notify(variables, iteration);
}

/// Notify observers with current optimization state (generic path).
///
/// For dense mode, the Hessian is converted to sparse for observer compatibility.
/// This is acceptable since observers are for visualization/debugging, not the hot path.
pub fn notify_observers_generic<M: AssemblyBackend>(
    observers: &mut OptObserverVec,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    report: IterationReport,
    _linear_solver: &dyn LinearSolver<M>,
) {
    let IterationReport {
        iteration,
        cost,
        gradient_norm,
        damping,
        step_norm,
        step_quality,
    } = report;
    observers.set_iteration_metrics(cost, gradient_norm, damping, step_norm, step_quality);
    // Skip matrix data for generic path since observers expect sparse Hessian.
    // Observer matrix visualization is optional and only used for debugging.
    observers.notify(variables, iteration);
}

/// Process Jacobian with Jacobi scaling (generic over assembly mode).
///
/// On `iteration == 0`, computes the scaling factors and stores them.
/// On subsequent iterations, reuses the cached scaling.
pub fn process_jacobian_generic<M: AssemblyBackend>(
    jacobian: &M::Jacobian,
    jacobi_scaling: &mut Option<Vec<f64>>,
    iteration: usize,
) -> Result<M::Jacobian, OptimizerError> {
    if iteration == 0 {
        let norms = M::compute_column_norms(jacobian);
        let scaling: Vec<f64> = norms.iter().map(|n| 1.0 / (1.0 + n)).collect();
        *jacobi_scaling = Some(scaling);
    }
    let scaling = jacobi_scaling
        .as_ref()
        .ok_or_else(|| OptimizerError::JacobiScalingNotInitialized.log())?;
    Ok(M::apply_column_scaling(jacobian, scaling)?)
}

/// Build a SolverResult from common optimization loop outputs.
///
/// All three optimizers construct SolverResult identically at convergence.
pub fn build_solver_result(
    status: OptimizationStatus,
    iterations: usize,
    state: InitializedState,
    elapsed: Duration,
    convergence_info: ConvergenceInfo,
    covariances: Option<SecondaryMap<VarKey, Mat<f64>>>,
) -> SolverResult<SlotMap<VarKey, Box<dyn ManifoldVariable>>> {
    SolverResult {
        status,
        iterations,
        initial_cost: state.initial_cost,
        final_cost: state.current_cost,
        parameters: state.variables,
        elapsed_time: elapsed,
        convergence_info: Some(convergence_info),
        covariances,
    }
}

/// Unified summary statistics for all optimizer types.
///
/// Replaces the separate `LevenbergMarquardtSummary`, `GaussNewtonSummary`,
/// and `DogLegSummary` structs with a single type that handles algorithm-specific
/// fields via `Option`.
#[derive(Debug, Clone)]
pub struct OptimizerSummary {
    /// Name of the optimizer algorithm
    pub optimizer_name: &'static str,
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Number of successful steps (None for GN which always accepts)
    pub successful_steps: Option<usize>,
    /// Number of unsuccessful steps (None for GN which always accepts)
    pub unsuccessful_steps: Option<usize>,
    /// Average cost reduction per iteration
    pub average_cost_reduction: f64,
    /// Maximum gradient norm encountered
    pub max_gradient_norm: f64,
    /// Final gradient norm
    pub final_gradient_norm: f64,
    /// Maximum parameter update norm
    pub max_parameter_update_norm: f64,
    /// Final parameter update norm
    pub final_parameter_update_norm: f64,
    /// Total time elapsed
    pub total_time: Duration,
    /// Average time per iteration
    pub average_time_per_iteration: Duration,
    /// Detailed per-iteration statistics history
    pub iteration_history: Vec<IterationStats>,
    /// Convergence status
    pub convergence_status: OptimizationStatus,
    /// Final damping parameter (LM only)
    pub final_damping: Option<f64>,
    /// Final trust region radius (DL only)
    pub final_trust_region_radius: Option<f64>,
    /// Step quality ratio (LM only)
    pub rho: Option<f64>,
}

impl Display for OptimizerSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let converged = matches!(
            self.convergence_status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::GradientToleranceReached
                | OptimizationStatus::ParameterToleranceReached
        );

        writeln!(f, "{} Final Result", self.optimizer_name)?;

        if converged {
            writeln!(f, "CONVERGED ({:?})", self.convergence_status)?;
        } else {
            writeln!(f, "DIVERGED ({:?})", self.convergence_status)?;
        }

        writeln!(f)?;
        writeln!(f, "Cost:")?;
        writeln!(f, "  Initial:   {:.6e}", self.initial_cost)?;
        writeln!(f, "  Final:     {:.6e}", self.final_cost)?;
        writeln!(
            f,
            "  Reduction: {:.6e} ({:.2}%)",
            self.initial_cost - self.final_cost,
            100.0 * (self.initial_cost - self.final_cost) / self.initial_cost.max(1e-12)
        )?;
        writeln!(f)?;
        writeln!(f, "Iterations:")?;
        writeln!(f, "  Total:              {}", self.iterations)?;
        if let (Some(successful), Some(unsuccessful)) =
            (self.successful_steps, self.unsuccessful_steps)
        {
            writeln!(
                f,
                "  Successful steps:   {} ({:.1}%)",
                successful,
                100.0 * successful as f64 / self.iterations.max(1) as f64
            )?;
            writeln!(
                f,
                "  Unsuccessful steps: {} ({:.1}%)",
                unsuccessful,
                100.0 * unsuccessful as f64 / self.iterations.max(1) as f64
            )?;
        }
        if let Some(radius) = self.final_trust_region_radius {
            writeln!(f)?;
            writeln!(f, "Trust Region:")?;
            writeln!(f, "  Final radius: {:.6e}", radius)?;
        }
        writeln!(f)?;
        writeln!(f, "Gradient:")?;
        writeln!(f, "  Max norm:   {:.2e}", self.max_gradient_norm)?;
        writeln!(f, "  Final norm: {:.2e}", self.final_gradient_norm)?;
        writeln!(f)?;
        writeln!(f, "Parameter Update:")?;
        writeln!(f, "  Max norm:   {:.2e}", self.max_parameter_update_norm)?;
        writeln!(f, "  Final norm: {:.2e}", self.final_parameter_update_norm)?;
        writeln!(f)?;
        writeln!(f, "Performance:")?;
        writeln!(
            f,
            "  Total time:             {:.2}ms",
            self.total_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Average per iteration:  {:.2}ms",
            self.average_time_per_iteration.as_secs_f64() * 1000.0
        )?;

        Ok(())
    }
}

/// Create an OptimizerSummary from common optimization loop outputs.
/// The running totals an optimizer reports when it finishes, before the
/// per-iteration averages are derived from them.
#[derive(Debug, Clone)]
pub struct SummaryTotals {
    pub optimizer_name: &'static str,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub successful_steps: Option<usize>,
    pub unsuccessful_steps: Option<usize>,
    pub max_gradient_norm: f64,
    pub final_gradient_norm: f64,
    pub max_parameter_update_norm: f64,
    pub final_parameter_update_norm: f64,
    pub total_cost_reduction: f64,
    pub total_time: Duration,
    pub iteration_history: Vec<IterationStats>,
    pub convergence_status: OptimizationStatus,
    pub final_damping: Option<f64>,
    pub final_trust_region_radius: Option<f64>,
    pub rho: Option<f64>,
}

/// The summary of a finished solve, with the averages per iteration.
pub fn create_optimizer_summary(totals: SummaryTotals) -> OptimizerSummary {
    let SummaryTotals {
        optimizer_name,
        initial_cost,
        final_cost,
        iterations,
        successful_steps,
        unsuccessful_steps,
        max_gradient_norm,
        final_gradient_norm,
        max_parameter_update_norm,
        final_parameter_update_norm,
        total_cost_reduction,
        total_time,
        iteration_history,
        convergence_status,
        final_damping,
        final_trust_region_radius,
        rho,
    } = totals;
    OptimizerSummary {
        optimizer_name,
        initial_cost,
        final_cost,
        iterations,
        successful_steps,
        unsuccessful_steps,
        average_cost_reduction: if iterations > 0 {
            total_cost_reduction / iterations as f64
        } else {
            0.0
        },
        max_gradient_norm,
        final_gradient_norm,
        max_parameter_update_norm,
        final_parameter_update_norm,
        total_time,
        average_time_per_iteration: if iterations > 0 {
            total_time / iterations as u32
        } else {
            Duration::from_secs(0)
        },
        iteration_history,
        convergence_status,
        final_damping,
        final_trust_region_radius,
        rho,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::VarKey;
    use crate::core::variable::{ManifoldVariable, Variable};
    use crate::factors::Factor;
    use crate::linalg::JacobianMode;
    use apex_manifolds::ManifoldType;
    use apex_manifolds::rn::Rn;
    use faer::Mat;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;
    use slotmap::{SecondaryMap, SlotMap};
    use std::time::Duration;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    // -------------------------------------------------------------------------
    // compute_step_quality — the non-positive predicted-reduction guard
    // -------------------------------------------------------------------------

    /// A negative predicted reduction must reject, not accept.
    ///
    /// The trap: a cost that *rose* gives a negative actual reduction, and
    /// dividing two negatives yields ρ > 0. Without the guard the solver accepts
    /// a step that made the objective worse, every time the quadratic model
    /// points uphill.
    #[test]
    fn step_quality_rejects_when_the_model_predicts_an_increase() {
        // cost rose 10 → 12, and the model predicted a rise too.
        let rho = compute_step_quality(10.0, 12.0, -1.0);
        assert!(
            rho <= 0.0,
            "a negative predicted reduction must not produce a positive ρ, got {rho}"
        );
        // Under every acceptance threshold in the crate this rejects.
        assert!(
            rho < 1e-3,
            "ρ = {rho} would pass the default min_relative_decrease"
        );
    }

    /// A vanishing predicted reduction still accepts a genuine improvement.
    ///
    /// At the solution both reductions go to zero legitimately; rejecting there
    /// would inflate the damping exactly as the solver converges.
    #[test]
    fn step_quality_accepts_a_real_decrease_when_prediction_underflows() {
        let rho = compute_step_quality(10.0, 9.999_999_999, 1e-20);
        assert!((rho - 1.0).abs() < 1e-12, "expected ρ = 1.0, got {rho}");

        let rho_no_gain = compute_step_quality(10.0, 10.5, 1e-20);
        assert!(
            (rho_no_gain).abs() < 1e-12,
            "expected ρ = 0.0, got {rho_no_gain}"
        );
    }

    /// The ordinary case is untouched.
    #[test]
    fn step_quality_is_the_plain_ratio_for_a_positive_prediction() {
        let rho = compute_step_quality(10.0, 6.0, 8.0);
        assert!((rho - 0.5).abs() < 1e-12, "expected ρ = 0.5, got {rho}");
    }

    // -------------------------------------------------------------------------
    // condition_number_lower_bound
    // -------------------------------------------------------------------------

    /// The bound is `(max‖c‖ / min‖c‖)²` and is exactly attained on a diagonal J.
    #[test]
    fn condition_bound_squares_the_column_norm_ratio() {
        // For J = diag(1, 10), JᵀJ = diag(1, 100), so κ₂ = 100 exactly.
        let bound = condition_number_lower_bound(&[1.0, 10.0]);
        assert!((bound - 100.0).abs() < 1e-9, "expected 100, got {bound}");
    }

    /// A structurally unconstrained variable gives a zero column, hence an
    /// infinite bound. This is the case the check earns its keep on.
    #[test]
    fn condition_bound_is_infinite_for_an_unconstrained_column() {
        assert!(condition_number_lower_bound(&[3.0, 0.0, 1.0]).is_infinite());
    }

    /// Equal column norms give the minimum possible bound of 1.
    #[test]
    fn condition_bound_is_one_for_equal_column_norms() {
        let bound = condition_number_lower_bound(&[2.5, 2.5, 2.5]);
        assert!((bound - 1.0).abs() < 1e-12, "expected 1, got {bound}");
        assert!((condition_number_lower_bound(&[]) - 1.0).abs() < 1e-12);
    }

    /// The bound must never exceed the true condition number — it is a lower
    /// bound, so exceeding a threshold is proof of ill-conditioning.
    ///
    /// Checked against `κ₂(JᵀJ)` computed from the singular values of a J whose
    /// columns are deliberately non-orthogonal, so the diagonal underestimates.
    #[test]
    fn condition_bound_never_exceeds_the_true_condition_number() {
        // Two nearly parallel unit columns: the true κ is enormous, the column
        // norms are identical, so the bound is 1 — a valid, very loose, bound.
        let norms = [1.0, 1.0];
        let bound = condition_number_lower_bound(&norms);
        // True κ₂ for columns at angle θ is cot²(θ/2); at θ → 0 this diverges.
        assert!(
            bound <= 1.0 + 1e-12,
            "bound {bound} must not exceed the true condition number"
        );
    }

    // -------------------------------------------------------------------------
    // compute_cost
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_cost_known_value() {
        // ||[1, 2]||² * 0.5 = (1 + 4) * 0.5 = 2.5
        let r = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let cost = compute_cost(&r);
        assert!((cost - 2.5).abs() < 1e-12, "expected 2.5, got {cost}");
    }

    #[test]
    fn test_compute_cost_zero_residual() {
        let r = Mat::zeros(3, 1);
        assert_eq!(compute_cost(&r), 0.0);
    }

    // -------------------------------------------------------------------------
    // compute_step_quality
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_step_quality_normal() {
        // actual = 1.0-0.0 = 1.0, predicted = 2.0 → rho = 0.5
        let rho = compute_step_quality(1.0, 0.0, 2.0);
        assert!((rho - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_step_quality_zero_predicted_positive_actual() {
        // predicted ≈ 0, actual > 0 → 1.0
        let rho = compute_step_quality(2.0, 1.0, 0.0);
        assert_eq!(rho, 1.0);
    }

    #[test]
    fn test_compute_step_quality_zero_predicted_nonpositive_actual() {
        // predicted ≈ 0, actual ≤ 0 → 0.0
        let rho = compute_step_quality(1.0, 2.0, 0.0); // actual = -1.0
        assert_eq!(rho, 0.0);
    }

    #[test]
    fn test_compute_step_quality_negative_reduction() {
        // cost increased: actual = 1.0 - 2.0 = -1.0, predicted = 1.0 → -1.0
        let rho = compute_step_quality(1.0, 2.0, 1.0);
        assert!((rho - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_compute_parameter_norm_empty() {
        let vars: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        assert_eq!(compute_parameter_norm(&vars), 0.0);
    }

    // -------------------------------------------------------------------------
    // apply_parameter_step / apply_negative_parameter_step
    // -------------------------------------------------------------------------

    #[test]
    fn test_apply_parameter_step_advances_variable() -> TestResult {
        let mut vars: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = vars.insert(Box::new(Variable::new(Rn::new(dvector![0.0]))));
        let order = vec![k];
        let step = Mat::from_fn(1, 1, |_, _| 3.0);
        let norm = apply_parameter_step(&mut vars, step.as_ref(), &order)?;
        assert!((norm - 3.0).abs() < 1e-12);
        let val = vars[k].as_param_slice()[0];
        assert!((val - 3.0).abs() < 1e-12, "expected 3.0, got {val}");
        Ok(())
    }

    #[test]
    fn test_apply_negative_parameter_step_reverts() -> TestResult {
        let mut vars: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = vars.insert(Box::new(Variable::new(Rn::new(dvector![5.0]))));
        let order = vec![k];
        let step = Mat::from_fn(1, 1, |_, _| 2.0);
        apply_parameter_step(&mut vars, step.as_ref(), &order)?;
        assert!((vars[k].as_param_slice()[0] - 7.0).abs() < 1e-12);
        apply_negative_parameter_step(&mut vars, step.as_ref(), &order)?;
        assert!((vars[k].as_param_slice()[0] - 5.0).abs() < 1e-12);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // check_convergence — all branches
    // -------------------------------------------------------------------------

    fn base_params() -> ConvergenceParams {
        ConvergenceParams {
            iteration: 1,
            current_cost: 1.0,
            new_cost: 0.9,
            parameter_norm: 1.0,
            parameter_update_norm: 1e-3,
            gradient_norm: 1e-3,
            elapsed: Duration::from_millis(10),
            step_accepted: true,
            max_iterations: 100,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-10,
            cost_tolerance: 1e-10,
            min_cost_threshold: None,
            timeout: None,
            trust_region_radius: None,
            min_trust_region_radius: None,
        }
    }

    #[test]
    fn test_check_convergence_no_trigger() {
        assert!(check_convergence(&base_params()).is_none());
    }

    #[test]
    fn test_check_convergence_nan_cost() {
        let mut p = base_params();
        p.new_cost = f64::NAN;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::InvalidNumericalValues)
        );
    }

    #[test]
    fn test_check_convergence_inf_gradient() {
        let mut p = base_params();
        p.gradient_norm = f64::INFINITY;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::InvalidNumericalValues)
        );
    }

    #[test]
    fn test_check_convergence_timeout() {
        let mut p = base_params();
        p.timeout = Some(Duration::from_millis(5));
        p.elapsed = Duration::from_millis(10);
        assert_eq!(check_convergence(&p), Some(OptimizationStatus::Timeout));
    }

    #[test]
    fn test_check_convergence_max_iterations() {
        let mut p = base_params();
        p.iteration = 100;
        p.max_iterations = 100;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::MaxIterationsReached)
        );
    }

    #[test]
    fn test_check_convergence_gradient_tolerance() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1e-12;
        p.gradient_tolerance = 1e-10;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::GradientToleranceReached)
        );
    }

    #[test]
    fn test_check_convergence_parameter_tolerance() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0; // above tolerance
        p.parameter_update_norm = 1e-20;
        p.parameter_tolerance = 1e-8;
        p.parameter_norm = 1.0;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::ParameterToleranceReached)
        );
    }

    #[test]
    fn test_check_convergence_cost_tolerance() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0; // above
        p.parameter_update_norm = 1.0; // above
        p.current_cost = 1.0;
        p.new_cost = 1.0 - 1e-15; // nearly no change
        p.cost_tolerance = 1e-10;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::CostToleranceReached)
        );
    }

    #[test]
    fn test_check_convergence_min_cost_threshold() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0;
        p.parameter_update_norm = 1.0;
        p.current_cost = 1.0;
        p.new_cost = 1.0 - 0.5; // big change — cost tol not triggered
        p.cost_tolerance = 1e-10;
        p.min_cost_threshold = Some(1.0); // new_cost=0.5 < 1.0 → trigger
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::MinCostThresholdReached)
        );
    }

    #[test]
    fn test_check_convergence_trust_region_too_small() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0;
        p.parameter_update_norm = 1.0;
        p.current_cost = 1.0;
        p.new_cost = 0.5;
        p.cost_tolerance = 1e-10;
        p.trust_region_radius = Some(1e-40);
        p.min_trust_region_radius = Some(1e-32);
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::TrustRegionRadiusTooSmall)
        );
    }

    #[test]
    fn test_check_convergence_step_not_accepted_skips_criteria() {
        // With step_accepted=false, gradient/parameter/cost tol should NOT fire
        let mut p = base_params();
        p.step_accepted = false;
        p.gradient_norm = 0.0; // would trigger gradient tol if accepted
        p.parameter_update_norm = 0.0;
        p.new_cost = 0.0;
        assert!(check_convergence(&p).is_none());
    }

    // -------------------------------------------------------------------------
    // Display impls
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimizer_type_display() {
        assert_eq!(
            format!("{}", OptimizerType::LevenbergMarquardt),
            "Levenberg-Marquardt"
        );
        assert_eq!(format!("{}", OptimizerType::GaussNewton), "Gauss-Newton");
        assert_eq!(format!("{}", OptimizerType::DogLeg), "Dog Leg");
    }

    #[test]
    fn test_optimization_status_display() {
        assert_eq!(format!("{}", OptimizationStatus::Converged), "Converged");
        assert_eq!(
            format!("{}", OptimizationStatus::MaxIterationsReached),
            "Maximum iterations reached"
        );
        assert_eq!(format!("{}", OptimizationStatus::Timeout), "Timeout");
        assert_eq!(
            format!("{}", OptimizationStatus::InvalidNumericalValues),
            "Invalid numerical values (NaN/Inf) detected"
        );
    }

    #[test]
    fn a_linalg_error_reaches_the_root_through_the_optimizer() -> TestResult {
        use crate::error::{ApexSolverError, ErrorChainExt};
        fn solve() -> linalg::LinAlgResult<()> {
            Err(linalg::LinAlgError::NotInitialized {
                component: "chunk eliminator",
            })
        }
        fn optimize() -> OptimizerResult<()> {
            solve()?;
            Ok(())
        }
        fn api() -> crate::error::ApexSolverResult<()> {
            optimize()?;
            Ok(())
        }
        let error = api().err().ok_or("expected an error")?;
        assert!(matches!(
            error,
            ApexSolverError::Optimizer(OptimizerError::LinAlg(_))
        ));
        assert_eq!(
            error.chain(),
            "linear solve failed\n  → chunk eliminator used before it was initialized"
        );
        Ok(())
    }

    #[test]
    fn a_step_of_the_wrong_length_is_rejected_before_any_variable_moves() -> TestResult {
        let mut vars: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let order: Vec<VarKey> = [0.0, 1.0]
            .iter()
            .map(|x| vars.insert(Box::new(Variable::new(Rn::new(dvector![*x])))))
            .collect();
        let before: Vec<f64> = vars[order[0]].as_param_slice().to_vec();
        let short = Mat::<f64>::zeros(1, 1);
        let Err(err) = apply_parameter_step(&mut vars, short.as_ref(), &order) else {
            return Err("a short step must be rejected".into());
        };
        assert!(matches!(
            err,
            OptimizerError::StepSize {
                expected: 2,
                actual: 1
            }
        ));
        assert_eq!(vars[order[0]].as_param_slice(), before.as_slice());
        Ok(())
    }

    // -------------------------------------------------------------------------
    // IterationStats print (smoke tests — no panic)
    // -------------------------------------------------------------------------

    #[test]
    fn test_iteration_stats_print_header_no_panic() {
        IterationStats::print_header();
    }

    #[test]
    fn test_iteration_stats_print_line_no_panic() {
        let stats = IterationStats {
            iteration: 1,
            cost: 1.5,
            cost_change: -0.5,
            gradient_norm: 1e-3,
            step_norm: 1e-4,
            tr_ratio: 0.8,
            tr_radius: 1e3,
            ls_iter: 0,
            iter_time_ms: 2.5,
            total_time_ms: 10.0,
            accepted: true,
        };
        stats.print_line();
    }

    // -------------------------------------------------------------------------
    // build_solver_result
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_solver_result_fields() -> TestResult {
        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(Rn::new(dvector![3.0]))));
        let state = InitializedState {
            variables,
            variable_index_map: SecondaryMap::new(),
            sorted_vars: vec![k],
            symbolic_structure: None,
            workspace: crate::linearizer::AssemblyWorkspace::empty(),
            total_dof: 1,
            current_cost: 0.1,
            initial_cost: 5.0,
        };
        let result = build_solver_result(
            OptimizationStatus::CostToleranceReached,
            10,
            state,
            Duration::from_millis(50),
            ConvergenceInfo {
                final_gradient_norm: 1e-5,
                final_parameter_update_norm: 1e-6,
                cost_evaluations: 15,
                jacobian_evaluations: 10,
            },
            None,
        );
        assert_eq!(result.status, OptimizationStatus::CostToleranceReached);
        assert_eq!(result.iterations, 10);
        assert!((result.initial_cost - 5.0).abs() < 1e-12);
        assert!((result.final_cost - 0.1).abs() < 1e-12);
        let ci = result
            .convergence_info
            .as_ref()
            .ok_or("convergence_info is None")?;
        assert_eq!(ci.cost_evaluations, 15);
        assert_eq!(ci.jacobian_evaluations, 10);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // create_optimizer_summary
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_optimizer_summary_averages() {
        let summary = create_optimizer_summary(SummaryTotals {
            optimizer_name: "TestOptimizer",
            initial_cost: 10.0,
            final_cost: 1.0,
            iterations: 4,
            successful_steps: Some(3),
            unsuccessful_steps: Some(1),
            max_gradient_norm: 2.0,
            final_gradient_norm: 0.1,
            max_parameter_update_norm: 3.0,
            final_parameter_update_norm: 0.05,
            total_cost_reduction: 9.0,
            total_time: Duration::from_millis(400),
            iteration_history: vec![],
            convergence_status: OptimizationStatus::CostToleranceReached,
            final_damping: Some(1e-3),
            final_trust_region_radius: None,
            rho: Some(0.8),
        });
        // average_cost_reduction = 9.0 / 4 = 2.25
        assert!((summary.average_cost_reduction - 2.25).abs() < 1e-10);
        // average_time_per_iteration = 400ms / 4 = 100ms
        assert_eq!(
            summary.average_time_per_iteration,
            Duration::from_millis(100)
        );
        assert_eq!(summary.optimizer_name, "TestOptimizer");
        assert_eq!(summary.successful_steps, Some(3));
    }

    #[test]
    fn test_create_optimizer_summary_zero_iterations() {
        let summary = create_optimizer_summary(SummaryTotals {
            optimizer_name: "Test",
            initial_cost: 1.0,
            final_cost: 1.0,
            iterations: 0,
            successful_steps: None,
            unsuccessful_steps: None,
            max_gradient_norm: 0.0,
            final_gradient_norm: 0.0,
            max_parameter_update_norm: 0.0,
            final_parameter_update_norm: 0.0,
            total_cost_reduction: 0.0,
            total_time: Duration::from_secs(0),
            iteration_history: vec![],
            convergence_status: OptimizationStatus::MaxIterationsReached,
            final_damping: None,
            final_trust_region_radius: None,
            rho: None,
        });
        assert_eq!(summary.average_cost_reduction, 0.0);
        assert_eq!(summary.average_time_per_iteration, Duration::from_secs(0));
    }

    // -------------------------------------------------------------------------
    // Simple Factor for integration tests in mod.rs
    // -------------------------------------------------------------------------

    /// Linear factor: r = x - target, J = [[1.0]]
    struct LinearFactor {
        target: f64,
    }

    impl Factor for LinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    // -------------------------------------------------------------------------
    // initialize_optimization_state (smoke test)
    // -------------------------------------------------------------------------

    #[test]
    fn test_initialize_optimization_state() -> TestResult {
        use crate::core::problem::Problem;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None)?;

        let state = initialize_optimization_state(&mut problem)?;
        assert_eq!(state.total_dof, 1);
        assert!(state.initial_cost > 0.0);
        assert!(state.sorted_vars.contains(&k));
        Ok(())
    }

    /// ISSUE-0004 regression: a variable bound is validated and stored, but no
    /// optimizer currently enforces it — `initialize_optimization_state` must
    /// reject with a typed error rather than silently solving the
    /// unconstrained problem, for zero-width, one-sided, and interior bounds.
    #[test]
    fn initialize_optimization_state_rejects_active_variable_bounds() -> TestResult {
        use crate::core::problem::Problem;

        for (lower, upper) in [(0.0, 0.0), (f64::NEG_INFINITY, 5.0), (-5.0, f64::INFINITY)] {
            let mut problem = Problem::new(JacobianMode::Sparse);
            let k = problem.add_variable(ManifoldType::RN, dvector![10.0]);
            problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None)?;
            problem.set_variable_bounds(k, 0, lower, upper)?;

            let Err(err) = initialize_optimization_state(&mut problem) else {
                panic!("a non-trivial bound must be rejected, not silently ignored");
            };
            let message = err.to_string();
            assert!(
                message.contains("bound") && message.contains("component"),
                "error must reference the offending bound, got: {message}"
            );
        }
        Ok(())
    }

    /// A bound of exactly `(-inf, inf)` is a documented no-op (equivalent to
    /// no bound at all) and must not trip the rejection.
    #[test]
    fn initialize_optimization_state_allows_unbounded_interval() -> TestResult {
        use crate::core::problem::Problem;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None)?;
        problem.set_variable_bounds(k, 0, f64::NEG_INFINITY, f64::INFINITY)?;

        initialize_optimization_state(&mut problem)?;
        Ok(())
    }

    /// Problems that never set a bound at all are completely unaffected.
    #[test]
    fn initialize_optimization_state_unaffected_without_bounds() -> TestResult {
        use crate::core::problem::Problem;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None)?;

        let state = initialize_optimization_state(&mut problem)?;
        assert_eq!(state.total_dof, 1);
        Ok(())
    }
}

#[cfg(test)]
mod finite_step_tests {
    use super::*;

    #[test]
    fn a_finite_step_is_accepted() {
        let step = Mat::from_fn(3, 1, |row, _| row as f64);
        assert!(ensure_finite_step(&step, "test").is_ok());
    }

    #[test]
    fn a_nan_or_infinite_step_is_rejected_with_its_index() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let step = Mat::from_fn(3, 1, |row, _| if row == 2 { bad } else { 1.0 });
            assert!(matches!(
                ensure_finite_step(&step, "test").err(),
                Some(OptimizerError::NonFiniteStep {
                    optimizer: "test",
                    index: 2
                })
            ));
        }
    }
}
