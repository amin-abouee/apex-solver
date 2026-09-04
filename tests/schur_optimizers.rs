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

/// Star graph: center pose observed by four leaf poses.
///
/// All leaves occupy one contiguous column range and share no factor with
/// each other, which is exactly what the matrix-free (`Iterative`) legacy
/// path requires (contiguous sides, 3-DOF blocks).
fn star_problem() -> (Problem, Vec<VarKey>) {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys = Vec::new();
    for i in 0..5 {
        let key = problem.add_variable(
            ManifoldType::SE2,
            dvector![i as f64 * 0.5 + 0.1, 0.05, 0.02],
        );
        keys.push(key);
    }
    for leaf in 1..5 {
        problem.add_residual_block(
            &[keys[0], keys[leaf]],
            Box::new(BetweenFactor::new(SE2::from_xy_angle(
                0.5 * leaf as f64,
                0.0,
                0.0,
            ))),
            None,
        );
        problem.mark_for_elimination(keys[leaf]);
    }
    problem.add_residual_block(
        &[keys[0]],
        Box::new(PriorFactor::new(SE2::from_xy_angle(0.0, 0.0, 0.0))),
        None,
    );
    (problem, keys)
}

/// Gauss-Newton through the matrix-free (`Iterative`) dispatch arm.
///
/// PCG solves approximately, so this compares at 1e-4 relative with a larger
/// iteration budget — the point is exercising the `IterativeSchurSolver`
/// construction path inside the optimizer, not PCG accuracy itself.
#[test]
fn gauss_newton_iterative_schur_matches_cholesky() -> TestResult {
    let (mut reference_problem, _) = star_problem();
    let mut reference = GaussNewton::with_config(
        GaussNewtonConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-9),
    );
    let reference = reference.optimize(&mut reference_problem)?;

    let (mut schur_problem, _) = star_problem();
    let mut schur = GaussNewton::with_config(
        GaussNewtonConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-9)
            .with_linear_solver_type(LinearSolverType::SparseSchurComplement)
            .with_schur_variant(SchurVariant::Iterative)
            .with_schur_cg_params(500, 1e-9),
    );
    let schur = schur.optimize(&mut schur_problem)?;

    assert!(
        schur.final_cost < schur.initial_cost,
        "Iterative Schur did not improve the cost"
    );
    let scale = reference.final_cost.abs().max(1.0);
    assert!(
        (reference.final_cost - schur.final_cost).abs() / scale < 1e-4,
        "GN Iterative-Schur cost {:.6e} vs Cholesky {:.6e}",
        schur.final_cost,
        reference.final_cost
    );
    Ok(())
}

/// All three configs carry the same Schur knobs with the same defaults and
/// builders — uniformity the dispatch arms rely on.
#[test]
fn schur_config_knobs_are_uniform_across_optimizers() -> TestResult {
    use apex_solver::linalg::SchurPreconditioner;
    use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;

    let lm = LevenbergMarquardtConfig::new()
        .with_schur_variant(SchurVariant::ChunkedSparse)
        .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal)
        .with_schur_cg_params(111, 1e-7);
    let gn = GaussNewtonConfig::new()
        .with_schur_variant(SchurVariant::ChunkedSparse)
        .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal)
        .with_schur_cg_params(111, 1e-7);
    let dl = DogLegConfig::new()
        .with_schur_variant(SchurVariant::ChunkedSparse)
        .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal)
        .with_schur_cg_params(111, 1e-7);
    for (label, variant, precond, iters, tol) in [
        (
            "LM",
            lm.schur_variant,
            lm.schur_preconditioner,
            lm.schur_cg_max_iterations,
            lm.schur_cg_tolerance,
        ),
        (
            "GN",
            gn.schur_variant,
            gn.schur_preconditioner,
            gn.schur_cg_max_iterations,
            gn.schur_cg_tolerance,
        ),
        (
            "DL",
            dl.schur_variant,
            dl.schur_preconditioner,
            dl.schur_cg_max_iterations,
            dl.schur_cg_tolerance,
        ),
    ] {
        assert!(
            matches!(variant, SchurVariant::ChunkedSparse),
            "{label}: variant builder not applied"
        );
        assert!(
            matches!(precond, SchurPreconditioner::BlockDiagonal),
            "{label}: preconditioner builder not applied"
        );
        assert_eq!(iters, 111, "{label}: cg iterations builder not applied");
        assert!(
            (tol - 1e-7).abs() < 1e-18,
            "{label}: cg tolerance builder not applied"
        );
    }

    // Defaults agree as well.
    let (lm_d, gn_d, dl_d) = (
        LevenbergMarquardtConfig::new(),
        GaussNewtonConfig::new(),
        DogLegConfig::new(),
    );
    assert!(matches!(lm_d.schur_variant, SchurVariant::Sparse));
    assert!(matches!(gn_d.schur_variant, SchurVariant::Sparse));
    assert!(matches!(dl_d.schur_variant, SchurVariant::Sparse));
    assert_eq!(lm_d.schur_cg_max_iterations, 200);
    assert_eq!(gn_d.schur_cg_max_iterations, 200);
    assert_eq!(dl_d.schur_cg_max_iterations, 200);
    Ok(())
}
