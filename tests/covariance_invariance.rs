//! Covariance must not depend on solver internals.
//!
//! Regression tests for <https://github.com/amin-abouee/apex-solver/issues/38>.
//!
//! Levenberg-Marquardt damping `λ` and Jacobi column scaling `D` exist to make the
//! *step* well-behaved. Neither carries information about the estimate's
//! uncertainty, so neither may appear in the returned covariance. Before the fix,
//! the solvers returned `(D·JᵀJ·D + λI)⁻¹`, which changed by orders of magnitude
//! when either knob moved.
//!
//! See `cov_issues/01-covariance-absorbs-damping.md` and
//! `cov_issues/02-covariance-absorbs-jacobi-scaling.md`.

use apex_solver::JacobianMode;
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::apex_manifolds::se2::SE2;
use apex_solver::core::VarKey;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::linalg::LinearSolverType;
use apex_solver::linalg::covariance::{Covariance, CovarianceAlgorithm, CovarianceOptions};
use apex_solver::optimizer::dog_leg::{DogLeg, DogLegConfig};
use apex_solver::optimizer::gauss_newton::{GaussNewton, GaussNewtonConfig};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use faer::Mat;
use nalgebra::dvector;
use slotmap::SecondaryMap;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A gauge-fixed 4-pose SE2 chain with a loop closure.
///
/// The prior on `x0` is what makes `H = JᵀJ` invertible. Without it the graph has
/// a 3-dimensional gauge freedom and the covariance is genuinely undefined — the
/// old code only appeared to work there because damping made `H + λI` invertible.
fn pose_graph() -> (Problem, Vec<VarKey>) {
    let mut problem = Problem::new(JacobianMode::Sparse);

    let x0 = problem.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
    let x1 = problem.add_variable(ManifoldType::SE2, dvector![0.95, 0.05, 0.02]);
    let x2 = problem.add_variable(ManifoldType::SE2, dvector![1.9, 0.1, 0.12]);
    let x3 = problem.add_variable(ManifoldType::SE2, dvector![2.85, 0.15, 0.05]);

    for (a, b, m) in [
        (x0, x1, SE2::from_xy_angle(1.0, 0.0, 0.0)),
        (x1, x2, SE2::from_xy_angle(1.0, 0.0, 0.1)),
        (x2, x3, SE2::from_xy_angle(1.0, 0.0, -0.1)),
        (x0, x3, SE2::from_xy_angle(3.0, 0.0, 0.0)),
    ] {
        problem.add_residual_block(&[a, b], Box::new(BetweenFactor::new(m)), None);
    }

    problem.add_residual_block(
        &[x0],
        Box::new(PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        }),
        None,
    );

    (problem, vec![x0, x1, x2, x3])
}

fn assert_covariances_match(
    a: &SecondaryMap<VarKey, Mat<f64>>,
    b: &SecondaryMap<VarKey, Mat<f64>>,
    keys: &[VarKey],
    tolerance: f64,
    context: &str,
) {
    for (idx, &key) in keys.iter().enumerate() {
        let (ca, cb) = (&a[key], &b[key]);
        assert_eq!(
            ca.nrows(),
            cb.nrows(),
            "{context}: block {idx} shape differs"
        );
        for i in 0..ca.nrows() {
            for j in 0..ca.ncols() {
                let (va, vb) = (ca[(i, j)], cb[(i, j)]);
                let scale = va.abs().max(vb.abs()).max(1.0);
                assert!(
                    (va - vb).abs() <= tolerance * scale,
                    "{context}: variable {idx} block[{i},{j}] differs: {va:.12e} vs {vb:.12e}",
                );
            }
        }
    }
}

fn covariances_from_lm(
    use_jacobi_scaling: bool,
    damping: f64,
) -> Result<SecondaryMap<VarKey, Mat<f64>>, Box<dyn std::error::Error>> {
    let (mut problem, _) = pose_graph();
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(200)
        .with_cost_tolerance(1e-14)
        .with_parameter_tolerance(1e-14)
        .with_gradient_tolerance(1e-14)
        .with_damping(damping)
        .with_jacobi_scaling(use_jacobi_scaling)
        .with_compute_covariances(true);

    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
    result
        .covariances
        .ok_or_else(|| "compute_covariances(true) must populate covariances".into())
}

/// Issue #38, part 1: Jacobi scaling must not change the covariance.
///
/// Against the old code this failed dramatically — the covariance came back in
/// scaled coordinates, `(D·H·D)⁻¹` instead of `H⁻¹`.
#[test]
fn covariance_is_invariant_to_jacobi_scaling() -> TestResult {
    let (_, keys) = pose_graph();
    let without = covariances_from_lm(false, 1e-3)?;
    let with = covariances_from_lm(true, 1e-3)?;
    assert_covariances_match(&without, &with, &keys, 1e-6, "jacobi scaling on vs off");
    Ok(())
}

/// Issue #38, part 2: LM damping must not change the covariance.
///
/// Against the old code the three runs differed by roughly the ratio of the
/// damping values, since the returned matrix was `(H + λI)⁻¹`.
#[test]
fn covariance_is_invariant_to_damping() -> TestResult {
    let (_, keys) = pose_graph();
    let reference = covariances_from_lm(false, 1e-8)?;
    for damping in [1e-3, 1.0] {
        let other = covariances_from_lm(false, damping)?;
        assert_covariances_match(
            &reference,
            &other,
            &keys,
            1e-6,
            &format!("damping 1e-8 vs {damping:e}"),
        );
    }
    Ok(())
}

/// All three optimizers must agree: they solve the same problem, so the
/// covariance at the solution is the same regardless of how they got there.
///
/// Dog Leg is the sharpest case — it defaults to `use_jacobi_scaling: true`, so
/// under the old code it was wrong on both counts, out of the box.
#[test]
fn covariance_agrees_across_optimizers() -> TestResult {
    let (_, keys) = pose_graph();
    let reference = covariances_from_lm(false, 1e-3)?;

    let (mut problem, _) = pose_graph();
    let gn = GaussNewton::with_config(
        GaussNewtonConfig::new()
            .with_max_iterations(200)
            .with_cost_tolerance(1e-14)
            .with_parameter_tolerance(1e-14)
            .with_compute_covariances(true),
    )
    .optimize(&mut problem)?
    .covariances
    .ok_or("Gauss-Newton must populate covariances")?;
    assert_covariances_match(&reference, &gn, &keys, 1e-6, "LM vs Gauss-Newton");

    let (mut problem, _) = pose_graph();
    let dl = DogLeg::with_config(
        DogLegConfig::new()
            .with_max_iterations(200)
            .with_cost_tolerance(1e-14)
            .with_parameter_tolerance(1e-14)
            .with_compute_covariances(true),
    )
    .optimize(&mut problem)?
    .covariances
    .ok_or("Dog Leg must populate covariances")?;
    assert_covariances_match(&reference, &dl, &keys, 1e-6, "LM vs Dog Leg");
    Ok(())
}

/// The in-solver flag and the explicit API must produce the same numbers.
///
/// This also pins the fix for the "covariance evaluated at the previous iterate"
/// bug: the flag path now re-linearizes at the returned solution, so recomputing
/// post-hoc from `result.parameters` must agree exactly.
#[test]
fn flag_path_matches_explicit_api() -> TestResult {
    let (mut problem, keys) = pose_graph();
    let result = LevenbergMarquardt::with_config(
        LevenbergMarquardtConfig::new()
            .with_max_iterations(200)
            .with_cost_tolerance(1e-14)
            .with_parameter_tolerance(1e-14)
            .with_compute_covariances(true),
    )
    .optimize(&mut problem)?;

    let from_flag = result.covariances.clone().ok_or("covariances populated")?;
    let from_api = Covariance::compute(CovarianceOptions::default(), &problem, &result.parameters)?
        .per_variable();

    assert_covariances_match(&from_flag, &from_api, &keys, 1e-12, "flag vs explicit API");
    Ok(())
}

/// Cholesky and the SVD pseudo-inverse must agree when `H` is full rank — the
/// pseudo-inverse reduces to the ordinary inverse there.
#[test]
fn dense_svd_matches_cholesky_when_full_rank() -> TestResult {
    let (mut problem, keys) = pose_graph();
    let result = LevenbergMarquardt::with_config(
        LevenbergMarquardtConfig::new()
            .with_max_iterations(200)
            .with_cost_tolerance(1e-14)
            .with_parameter_tolerance(1e-14),
    )
    .optimize(&mut problem)?;

    let cholesky = Covariance::compute(CovarianceOptions::default(), &problem, &result.parameters)?
        .per_variable();

    let svd = Covariance::compute(
        CovarianceOptions::new().with_algorithm(CovarianceAlgorithm::DenseSvd),
        &problem,
        &result.parameters,
    )?
    .per_variable();

    assert_covariances_match(&cholesky, &svd, &keys, 1e-8, "Cholesky vs DenseSvd");
    Ok(())
}

/// Covariance is now solver-independent, so it must be identical across every
/// `LinearSolverType`, sparse and dense alike.
///
/// Previously the QR solvers inverted their own (augmented) factorization and the
/// Schur solvers returned `None` with no error at all, because covariance was
/// derived from whatever the solver happened to cache. It no longer is —
/// `Covariance::compute` does its own assembly and factorization and never
/// consults the linear solver.
///
/// `SparseSchurComplement` is omitted here only because it requires a
/// bipartite camera/landmark structure and rejects a plain pose graph before any
/// covariance question arises; bundle adjustment coverage lives in
/// `tests/bundle_adjustment_integration.rs`.
///
/// See `cov_issues/03-covariance-stale-and-silent.md`.
#[test]
fn covariance_available_for_every_linear_solver() -> TestResult {
    let (_, keys) = pose_graph();
    let reference = covariances_from_lm(false, 1e-3)?;

    for solver_type in [
        LinearSolverType::SparseCholesky,
        LinearSolverType::SparseQR,
        LinearSolverType::DenseCholesky,
        LinearSolverType::DenseQR,
    ] {
        let (mut problem, _) = pose_graph();
        let result = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_linear_solver_type(solver_type)
                .with_max_iterations(200)
                .with_cost_tolerance(1e-14)
                .with_parameter_tolerance(1e-14)
                .with_compute_covariances(true),
        )
        .optimize(&mut problem)?;

        let covariances = result.covariances.ok_or_else(|| {
            format!("{solver_type:?} must populate covariances, not silently return None")
        })?;

        assert_covariances_match(
            &reference,
            &covariances,
            &keys,
            1e-6,
            &format!("reference vs {solver_type:?}"),
        );
    }
    Ok(())
}

/// Removing the gauge anchor makes `H` singular. That must be reported, not
/// papered over by damping the way the old code did.
#[test]
fn gauge_free_problem_is_reported_rather_than_masked() -> TestResult {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let x0 = problem.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
    let x1 = problem.add_variable(ManifoldType::SE2, dvector![1.0, 0.0, 0.0]);
    problem.add_residual_block(
        &[x0, x1],
        Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0))),
        None,
    );

    // The solve itself still succeeds — LM damping makes `H + λI` invertible even
    // though `H` is not. That is exactly what used to hide the problem.
    let result =
        LevenbergMarquardt::with_config(LevenbergMarquardtConfig::new().with_max_iterations(50))
            .optimize(&mut problem)?;

    match Covariance::compute(CovarianceOptions::default(), &problem, &result.parameters) {
        Ok(_) => panic!("a gauge-free problem must not silently produce a covariance"),
        Err(err) => {
            let message = err.to_string();
            assert!(
                message.contains("PriorFactor") && message.contains("DenseSvd"),
                "the error must explain how to proceed, got: {message}"
            );
        }
    }

    // The same problem is computable via the pseudo-inverse, for callers who
    // genuinely want the gauge-free answer.
    Covariance::compute(
        CovarianceOptions::new().with_algorithm(CovarianceAlgorithm::DenseSvd),
        &problem,
        &result.parameters,
    )?;
    Ok(())
}
