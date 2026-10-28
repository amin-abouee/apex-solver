//! How many `ERROR` events a solve emits when its linear system will not
//! factorize.
//!
//! A numerical factorization failure that a damped optimizer cures by raising
//! its damping is an expected per-iteration outcome: it must log nothing. A
//! failure the optimizer gives up on is logged exactly once, by the optimizer,
//! with the whole cause chain.
//!
//! The events are counted by a process-wide subscriber, so events raised on
//! rayon workers are counted too; the tests serialize on a lock and clear the
//! record before each solve.

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use apex_solver::core::problem::Problem;
use apex_solver::factors::Factor;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::{
    DogLeg, ErrorChainExt, GaussNewton, JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig,
    LinearSolverType, ManifoldType,
};
use faer::prelude::ReborrowMut;
use nalgebra::dvector;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Messages of every `ERROR` event since the last [`ErrorEvents::take`].
static ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Serializes the tests of this binary around the shared record.
static SERIAL: Mutex<()> = Mutex::new(());

static INSTALLED: OnceLock<bool> = OnceLock::new();

struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

/// Records the message of every `ERROR` event.
struct ErrorRecorder;

impl<S: Subscriber> Layer<S> for ErrorRecorder {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() == Level::ERROR {
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            ERRORS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(visitor.0);
        }
    }
}

/// Exclusive access to the error record for one test.
struct ErrorEvents {
    _serial: MutexGuard<'static, ()>,
}

impl ErrorEvents {
    fn start() -> Result<Self, Box<dyn std::error::Error>> {
        let serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let installed = *INSTALLED.get_or_init(|| {
            let subscriber = tracing_subscriber::registry().with(ErrorRecorder);
            tracing::subscriber::set_global_default(subscriber).is_ok()
        });
        if !installed {
            return Err("could not install the error-counting subscriber".into());
        }
        ERRORS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        Ok(Self { _serial: serial })
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *ERRORS.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// `r = x₀ + x₁ − 1` with `J = [1, 1]`: `JᵀJ` is singular (rank one), so
/// its Cholesky factorization needs damping that survives `1 + μ` in
/// floating point.
struct RankOneSum;

impl Factor for RankOneSum {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        residual[0] = params[0][0] + params[0][1] - 1.0;
        if let Some(mut jac) = jacobian {
            *jac.rb_mut().get_mut(0, 0) = 1.0;
            *jac.rb_mut().get_mut(0, 1) = 1.0;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 2)
    }
}

/// `r = x − 1` with Jacobian `slope`. A NaN slope keeps the residual finite,
/// so assembly succeeds, but makes `JᵀJ + λ·D` NaN for every λ: the system
/// never factorizes.
struct Scalar {
    slope: f64,
}

impl Factor for Scalar {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        residual[0] = params[0][0] - 1.0;
        if let Some(mut jac) = jacobian {
            *jac.rb_mut().get_mut(0, 0) = self.slope;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 1)
    }
}

fn rank_one_problem(mode: JacobianMode) -> Result<Problem, Box<dyn std::error::Error>> {
    let mut problem = Problem::new(mode);
    let x = problem.add_variable(ManifoldType::RN, dvector![0.0, 0.0]);
    problem.add_residual_block(&[x], Box::new(RankOneSum), None)?;
    Ok(problem)
}

fn nan_problem(mode: JacobianMode) -> Result<Problem, Box<dyn std::error::Error>> {
    let mut problem = Problem::new(mode);
    let x = problem.add_variable(ManifoldType::RN, dvector![0.0]);
    problem.add_residual_block(&[x], Box::new(Scalar { slope: f64::NAN }), None)?;
    Ok(problem)
}

/// Which side of a Schur partition carries the NaN Jacobian.
#[derive(Clone, Copy)]
enum NanSide {
    /// `H_ee` will not invert.
    Eliminated,
    /// The reduced system `S` will not factorize.
    Retained,
}

/// One retained and one eliminated variable, each with its own residual,
/// one of them with a NaN Jacobian.
fn nan_schur_problem(
    mode: JacobianMode,
    side: NanSide,
) -> Result<Problem, Box<dyn std::error::Error>> {
    let (kept_slope, eliminated_slope) = match side {
        NanSide::Eliminated => (1.0, f64::NAN),
        NanSide::Retained => (f64::NAN, 1.0),
    };
    let mut problem = Problem::new(mode);
    let kept = problem.add_variable(ManifoldType::RN, dvector![0.0]);
    let eliminated = problem.add_variable(ManifoldType::RN, dvector![0.0]);
    problem.add_residual_block(&[kept], Box::new(Scalar { slope: kept_slope }), None)?;
    problem.add_residual_block(
        &[eliminated],
        Box::new(Scalar {
            slope: eliminated_slope,
        }),
        None,
    )?;
    problem.mark_for_elimination(eliminated);
    Ok(problem)
}

/// Dog Leg starting at μ = 1e-18, where `1 + μ == 1` and the rank-one
/// system is exactly singular; one ×1e4 retry reaches μ = 1e-14, which
/// factorizes, and `min_mu` keeps μ there for the rest of the solve.
fn dog_leg_one_retry(solver: LinearSolverType) -> DogLeg {
    DogLeg::with_config(
        DogLegConfig::new()
            .with_linear_solver_type(solver)
            .with_jacobi_scaling(false)
            .with_mu_params(1e-18, 1e-14, 1.0, 1e4),
    )
}

/// One `ERROR` event whose message is the returned error's whole chain.
fn assert_logged_once_with_chain(
    events: &[String],
    error: &dyn std::error::Error,
    leaf: &str,
) -> TestResult {
    let chain = error.chain_compact();
    if events.len() != 1 {
        return Err(format!("expected one ERROR event, got {}: {events:?}", events.len()).into());
    }
    if events[0] != chain {
        return Err(format!("logged {:?}, the chain is {chain:?}", events[0]).into());
    }
    if !chain.contains(" → ") || !chain.contains(leaf) {
        return Err(format!("the chain {chain:?} does not reach {leaf:?}").into());
    }
    Ok(())
}

fn assert_recovered(events: &[String], status: &OptimizationStatus, final_cost: f64) -> TestResult {
    if !events.is_empty() {
        return Err(format!("a recovered failure logged {events:?}").into());
    }
    let converged = matches!(
        status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached
            | OptimizationStatus::MinCostThresholdReached
    );
    if !converged || final_cost > 1e-12 {
        return Err(format!("did not converge: {status:?}, final cost {final_cost:e}").into());
    }
    Ok(())
}

#[test]
fn dog_leg_recovers_a_sparse_factorization_failure_silently() -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = rank_one_problem(JacobianMode::Sparse)?;
    let result = dog_leg_one_retry(LinearSolverType::SparseCholesky).optimize(&mut problem)?;
    assert_recovered(&events.take(), &result.status, result.final_cost)
}

#[test]
fn dog_leg_recovers_a_dense_factorization_failure_silently() -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = rank_one_problem(JacobianMode::Dense)?;
    let result = dog_leg_one_retry(LinearSolverType::DenseCholesky).optimize(&mut problem)?;
    assert_recovered(&events.take(), &result.status, result.final_cost)
}

#[test]
fn levenberg_marquardt_logs_an_unfactorizable_sparse_system_once() -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = nan_problem(JacobianMode::Sparse)?;
    let Err(error) =
        LevenbergMarquardt::with_config(LevenbergMarquardtConfig::new()).optimize(&mut problem)
    else {
        return Err("a NaN system must not solve".into());
    };
    assert_logged_once_with_chain(&events.take(), &error, "factorization of")
}

#[test]
fn levenberg_marquardt_logs_an_unfactorizable_dense_system_once() -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = nan_problem(JacobianMode::Dense)?;
    let config =
        LevenbergMarquardtConfig::new().with_linear_solver_type(LinearSolverType::DenseCholesky);
    let Err(error) = LevenbergMarquardt::with_config(config).optimize(&mut problem) else {
        return Err("a NaN system must not solve".into());
    };
    assert_logged_once_with_chain(&events.take(), &error, "factorization of")
}

#[test]
fn dog_leg_logs_exhausted_damping_once() -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = nan_problem(JacobianMode::Sparse)?;
    let Err(error) = DogLeg::with_config(DogLegConfig::new()).optimize(&mut problem) else {
        return Err("a NaN system must not solve".into());
    };
    assert_logged_once_with_chain(&events.take(), &error, "factorization of")
}

#[test]
fn gauss_newton_logs_an_unfactorizable_system_once() -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = nan_problem(JacobianMode::Sparse)?;
    let Err(error) = GaussNewton::with_config(GaussNewtonConfig::new()).optimize(&mut problem)
    else {
        return Err("a NaN system must not solve".into());
    };
    assert_logged_once_with_chain(&events.take(), &error, "factorization of")
}

fn levenberg_marquardt_schur_case(
    mode: JacobianMode,
    solver: LinearSolverType,
    side: NanSide,
) -> TestResult {
    let events = ErrorEvents::start()?;
    let mut problem = nan_schur_problem(mode, side)?;
    let config = LevenbergMarquardtConfig::new().with_linear_solver_type(solver);
    let Err(error) = LevenbergMarquardt::with_config(config).optimize(&mut problem) else {
        return Err(format!("a NaN system must not solve with {solver}").into());
    };
    assert_logged_once_with_chain(&events.take(), &error, "singular")
}

#[test]
fn levenberg_marquardt_logs_a_singular_eliminated_block_once() -> TestResult {
    use LinearSolverType::{ExplicitDenseSchur, ExplicitSparseSchur, ImplicitSparseSchur};
    levenberg_marquardt_schur_case(
        JacobianMode::Sparse,
        ExplicitSparseSchur,
        NanSide::Eliminated,
    )?;
    levenberg_marquardt_schur_case(
        JacobianMode::Sparse,
        ImplicitSparseSchur,
        NanSide::Eliminated,
    )?;
    levenberg_marquardt_schur_case(JacobianMode::Dense, ExplicitDenseSchur, NanSide::Eliminated)
}

#[test]
fn levenberg_marquardt_logs_an_unfactorizable_dense_reduced_system_once() -> TestResult {
    levenberg_marquardt_schur_case(
        JacobianMode::Dense,
        LinearSolverType::ExplicitDenseSchur,
        NanSide::Retained,
    )
}
