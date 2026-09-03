//! Gauss-Newton and Dog Leg must solve Schur-backed problems the same way
//! sparse Cholesky does.
//!
//! The solver-level oracle (`schur_generalization.rs`) proves the Schur step
//! equals the Cholesky step for one linearization. These tests prove the
//! optimizer wiring around it: row grouping, block-structure initialization
//! and variant dispatch inside `GaussNewton::optimize` and `DogLeg::optimize`.
//! The fixture is a 5-pose SE2 chain eliminating alternating poses — mutually
//! unconnected, so `H_ee` is block-diagonal and elimination is valid.

use apex_manifolds::se2::SE2;
use apex_solver::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::linalg::{JacobianMode, LinearSolverType, SchurVariant};
use apex_solver::optimizer::dog_leg::{DogLeg, DogLegConfig};
use apex_solver::optimizer::gauss_newton::{GaussNewton, GaussNewtonConfig};
use nalgebra::dvector;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Five-pose odometry chain with a prior anchoring the first pose.
///
/// Poses 1 and 3 are marked for elimination: each touches only kept
/// neighbours, never each other. Initial guesses are perturbed so the
/// optimizers actually iterate.
fn chain_problem() -> (Problem, Vec<VarKey>) {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys = Vec::new();
    for i in 0..5 {
        // Perturbed start: truth is (i, 0, 0).
        let key = problem.add_variable(ManifoldType::SE2, dvector![i as f64 + 0.1, 0.05, 0.02]);
        keys.push(key);
    }
    for w in keys.windows(2) {
        problem.add_residual_block(
            &[w[0], w[1]],
            Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0))),
            None,
        );
    }
    problem.add_residual_block(
        &[keys[0]],
        Box::new(PriorFactor::new(SE2::from_xy_angle(0.0, 0.0, 0.0))),
        None,
    );
    problem.mark_for_elimination(keys[1]);
    problem.mark_for_elimination(keys[3]);
    (problem, keys)
}

fn assert_costs_agree(cholesky: f64, schur: f64, initial: f64, what: &str) {
    assert!(
        schur < initial,
        "{what}: Schur did not improve the cost ({initial:.6e} -> {schur:.6e})"
    );
    let scale = cholesky.abs().max(1.0);
    assert!(
        (cholesky - schur).abs() / scale < 1e-6,
        "{what}: cholesky cost {cholesky:.6e} vs Schur cost {schur:.6e}"
    );
}

/// Gauss-Newton with an explicit Schur solve must land where Cholesky lands.
#[test]
fn gauss_newton_schur_matches_cholesky() -> TestResult {
    let (mut reference_problem, _) = chain_problem();
    let mut reference = GaussNewton::with_config(GaussNewtonConfig::new().with_max_iterations(50));
    let reference = reference.optimize(&mut reference_problem)?;

    let (mut schur_problem, _) = chain_problem();
    let mut schur = GaussNewton::with_config(
        GaussNewtonConfig::new()
            .with_max_iterations(50)
            .with_linear_solver_type(LinearSolverType::SparseSchurComplement),
    );
    let schur = schur.optimize(&mut schur_problem)?;

    assert!(reference.iterations > 0 && schur.iterations > 0);
    assert_costs_agree(
        reference.final_cost,
        schur.final_cost,
        reference.initial_cost,
        "Gauss-Newton Sparse-Schur",
    );
    Ok(())
}

/// Same, through the chunked J-direct variant (exercises row grouping).
#[test]
fn gauss_newton_chunked_schur_matches_cholesky() -> TestResult {
    let (mut reference_problem, _) = chain_problem();
    let mut reference = GaussNewton::with_config(GaussNewtonConfig::new().with_max_iterations(50));
    let reference = reference.optimize(&mut reference_problem)?;

    let (mut schur_problem, _) = chain_problem();
    let mut schur = GaussNewton::with_config(
        GaussNewtonConfig::new()
            .with_max_iterations(50)
            .with_linear_solver_type(LinearSolverType::SparseSchurComplement)
            .with_schur_variant(SchurVariant::ChunkedSparse),
    );
    let schur = schur.optimize(&mut schur_problem)?;

    assert_costs_agree(
        reference.final_cost,
        schur.final_cost,
        reference.initial_cost,
        "Gauss-Newton ChunkedSparse-Schur",
    );
    Ok(())
}

/// Same, on the pure normal equations (`min_diagonal = 0`), covering
/// `solve_normal_equation` through the optimizer rather than only the
/// default augmented path.
#[test]
fn gauss_newton_schur_matches_cholesky_undiagonalized() -> TestResult {
    let base = || {
        GaussNewtonConfig::new()
            .with_max_iterations(50)
            .with_min_diagonal(0.0)
    };
    let (mut reference_problem, _) = chain_problem();
    let mut reference = GaussNewton::with_config(base());
    let reference = reference.optimize(&mut reference_problem)?;

    let (mut schur_problem, _) = chain_problem();
    let mut schur = GaussNewton::with_config(
        base().with_linear_solver_type(LinearSolverType::SparseSchurComplement),
    );
    let schur = schur.optimize(&mut schur_problem)?;

    assert_costs_agree(
        reference.final_cost,
        schur.final_cost,
        reference.initial_cost,
        "Gauss-Newton Sparse-Schur (undamped)",
    );
    Ok(())
}

/// Dog Leg with an explicit Schur solve must land where Cholesky lands.
///
/// This exercises the Schur path through the Cauchy-point machinery, which
/// reads the gradient and Hessian-vector products off the Schur solver.
#[test]
fn dog_leg_schur_matches_cholesky() -> TestResult {
    let (mut reference_problem, _) = chain_problem();
    let mut reference = DogLeg::with_config(DogLegConfig::new().with_max_iterations(50));
    let reference = reference.optimize(&mut reference_problem)?;

    let (mut schur_problem, _) = chain_problem();
    let mut schur = DogLeg::with_config(
        DogLegConfig::new()
            .with_max_iterations(50)
            .with_linear_solver_type(LinearSolverType::SparseSchurComplement),
    );
    let schur = schur.optimize(&mut schur_problem)?;

    assert!(reference.iterations > 0 && schur.iterations > 0);
    assert_costs_agree(
        reference.final_cost,
        schur.final_cost,
        reference.initial_cost,
        "DogLeg Sparse-Schur",
    );
    Ok(())
}
