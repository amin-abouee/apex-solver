//! Golden-value regression tests: final cost is deterministic for a fixed
//! input and algorithm, so the exact values below are pinned. A drift in these
//! numbers means an algorithmic or numerics change — silently or otherwise.
//!
//! The configs replicate the odometry benchmark's production settings
//! (LM + default SparseCholesky, damping 1e-4, 150/100-iteration caps), so a
//! golden failure also flags any benchmark-trajectory change.
//!
//! Datasets are downloaded on first use (see `apex_io`), matching the other
//! integration tests.
//!
//! Tolerances are RELATIVE (1e-6): faer's SIMD reductions sum in
//! architecture-dependent order, so the last ulps of the final cost differ
//! between x86-64 and aarch64 CI runners. The bound is still orders of
//! magnitude tighter than any algorithmic drift the thresholds catch.

use apex_io::{G2oLoader, GraphLoader};
use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::L2Loss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::linalg::JacobianMode;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use std::collections::HashMap;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// (dataset, is_3d, golden final cost, tolerance)
const GOLDENS: &[(&str, bool, f64, f64)] = &[
    ("ring", false, 2.217_900_322_072e-2, 1e-6),
    ("M3500", false, 1.510_940_460_434e0, 1e-6),
    ("parking-garage", true, 6.245_107_165_929e-1, 1e-6),
    ("sphere2500", true, 2.131_994_494_757e1, 1e-6),
];

fn solve(dataset: &str, is_3d: bool) -> Result<f64, Box<dyn std::error::Error>> {
    apex_io::ensure_odometry_dataset(dataset)?;
    let path = if is_3d {
        format!("{}/{}.g2o", apex_io::ODOMETRY_DATA_DIR_3D, dataset)
    } else {
        format!("{}/{}.g2o", apex_io::ODOMETRY_DATA_DIR_2D, dataset)
    };
    let graph = G2oLoader::load(path.as_str())?;

    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys: HashMap<usize, apex_solver::core::VarKey> = HashMap::new();

    if is_3d {
        let mut ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
        ids.sort();
        for &id in &ids {
            let v = &graph.vertices_se3[&id];
            let q = v.pose.rotation_quaternion();
            let t = v.pose.translation();
            let key = problem.add_variable(
                ManifoldType::SE3,
                nalgebra::dvector![t.x, t.y, t.z, q.w, q.i, q.j, q.k],
            );
            var_keys.insert(id, key);
        }
        for edge in &graph.edges_se3 {
            if let (Some(&kf), Some(&kt)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
                problem.add_residual_block(
                    &[kf, kt],
                    Box::new(BetweenFactor::new(edge.measurement.clone())),
                    Some(Box::new(L2Loss)),
                );
            }
        }
    } else {
        let mut ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
        ids.sort();
        for &id in &ids {
            let v = &graph.vertices_se2[&id];
            let key = problem.add_variable(
                ManifoldType::SE2,
                nalgebra::dvector![v.x(), v.y(), v.theta()],
            );
            var_keys.insert(id, key);
        }
        for edge in &graph.edges_se2 {
            if let (Some(&kf), Some(&kt)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
                problem.add_residual_block(
                    &[kf, kt],
                    Box::new(BetweenFactor::new(edge.measurement.clone())),
                    Some(Box::new(L2Loss)),
                );
            }
        }
    }

    // Both SE2 and SE3 bench paths run LM with the default SparseCholesky
    // solver; the configs differ only in iteration cap and gradient tolerance.
    let (max_iterations, gradient_tol) = if is_3d { (100, 1e-12) } else { (150, 1e-10) };

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(gradient_tol)
        .with_damping(1e-4);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;
    Ok(result.final_cost)
}

#[test]
fn golden_final_costs_are_stable() -> TestResult {
    for (dataset, is_3d, golden, tolerance) in GOLDENS {
        let cost = solve(dataset, *is_3d)?;
        let rel = ((cost - golden) / golden.abs().max(1.0)).abs();
        assert!(
            rel <= *tolerance,
            "{dataset}: final cost {cost:.12e} deviates from pinned golden {golden:.12e} \
             (relative {rel:.3e})"
        );
    }
    Ok(())
}
