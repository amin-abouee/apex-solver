//! A solve whose linear system or step is not finite must fail with an error,
//! never return `Ok` with a status that hides it.

use apex_solver::core::problem::Problem;
use apex_solver::factors::Factor;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::{
    DogLeg, GaussNewton, JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig,
    LinearSolverType, ManifoldType,
};
use faer::prelude::ReborrowMut;
use nalgebra::dvector;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// `r = x − 1` with Jacobian `slope`: a non-finite slope keeps the residual
/// finite, so assembly succeeds and the Hessian is what turns non-finite.
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

/// A retained and an eliminated variable, the retained one with `kept_slope`
/// and the eliminated one with `eliminated_slope`.
fn schur_problem(
    mode: JacobianMode,
    kept_slope: f64,
    eliminated_slope: f64,
) -> Result<Problem, Box<dyn std::error::Error>> {
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

fn mode_of(solver: LinearSolverType) -> JacobianMode {
    use LinearSolverType::{DenseCholesky, DenseQR, ExplicitDenseSchur};
    if matches!(solver, DenseCholesky | DenseQR | ExplicitDenseSchur) {
        JacobianMode::Dense
    } else {
        JacobianMode::Sparse
    }
}

const SOLVERS: [LinearSolverType; 7] = [
    LinearSolverType::SparseCholesky,
    LinearSolverType::SparseQR,
    LinearSolverType::DenseCholesky,
    LinearSolverType::DenseQR,
    LinearSolverType::ExplicitSparseSchur,
    LinearSolverType::ImplicitSparseSchur,
    LinearSolverType::ExplicitDenseSchur,
];

const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

/// Finite slopes whose square, the entry `JᵀJ` is built from, overflows.
const OVERFLOWING: [f64; 2] = [1e200, -1e200];

/// Every optimizer × linear solver × non-finite Jacobian on either side of
/// the partition. Collects the combinations that came back `Ok`.
fn combinations_that_return_ok(slopes: &[f64]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut accepted = Vec::new();
    for solver in SOLVERS {
        for &bad in slopes {
            for (kept, eliminated, side) in [(bad, 1.0, "retained"), (1.0, bad, "eliminated")] {
                let lm = LevenbergMarquardtConfig::new().with_linear_solver_type(solver);
                let gn = GaussNewtonConfig::new().with_linear_solver_type(solver);
                let dl = DogLegConfig::new().with_linear_solver_type(solver);
                let mut problem = schur_problem(mode_of(solver), kept, eliminated)?;
                if LevenbergMarquardt::with_config(lm)
                    .optimize(&mut problem)
                    .is_ok()
                {
                    accepted.push(format!("LM {solver} {bad} {side}"));
                }
                let mut problem = schur_problem(mode_of(solver), kept, eliminated)?;
                if GaussNewton::with_config(gn).optimize(&mut problem).is_ok() {
                    accepted.push(format!("GN {solver} {bad} {side}"));
                }
                let mut problem = schur_problem(mode_of(solver), kept, eliminated)?;
                if DogLeg::with_config(dl).optimize(&mut problem).is_ok() {
                    accepted.push(format!("DL {solver} {bad} {side}"));
                }
            }
        }
    }
    Ok(accepted)
}

fn no_combination_returns_ok(slopes: &[f64]) -> TestResult {
    let accepted = combinations_that_return_ok(slopes)?;
    if accepted.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} combinations returned Ok:\n{}",
            accepted.len(),
            accepted.join("\n")
        )
        .into())
    }
}

#[test]
fn a_non_finite_system_is_an_error_for_every_optimizer_and_linear_solver() -> TestResult {
    no_combination_returns_ok(&NON_FINITE)
}

#[test]
fn a_system_that_overflows_never_returns_a_non_finite_result() -> TestResult {
    let mut non_finite = Vec::new();
    for solver in SOLVERS {
        for slope in OVERFLOWING {
            let lm = LevenbergMarquardtConfig::new().with_linear_solver_type(solver);
            let gn = GaussNewtonConfig::new().with_linear_solver_type(solver);
            let dl = DogLegConfig::new().with_linear_solver_type(solver);
            let mut problem = schur_problem(mode_of(solver), slope, 1.0)?;
            let lm = LevenbergMarquardt::with_config(lm).optimize(&mut problem);
            let mut problem = schur_problem(mode_of(solver), slope, 1.0)?;
            let gn = GaussNewton::with_config(gn).optimize(&mut problem);
            let mut problem = schur_problem(mode_of(solver), slope, 1.0)?;
            let dl = DogLeg::with_config(dl).optimize(&mut problem);
            for (name, outcome) in [("LM", lm), ("GN", gn), ("DL", dl)] {
                if let Ok(result) = outcome {
                    let finite = result.final_cost.is_finite()
                        && result
                            .parameters
                            .values()
                            .all(|v| v.as_param_slice().iter().all(|x| x.is_finite()));
                    if !finite {
                        non_finite.push(format!("{name} {solver} {slope:e}"));
                    }
                }
            }
        }
    }
    if non_finite.is_empty() {
        Ok(())
    } else {
        Err(format!("non-finite results: {non_finite:?}").into())
    }
}
