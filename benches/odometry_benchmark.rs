//! Criterion benchmark for pose-graph odometry on standard g2o SLAM datasets.
//!
//! Measures full wall-clock solve time of [`LevenbergMarquardt`] over complete
//! factor graphs, from the identical initial state every run (the problem is
//! rebuilt in the untimed setup of each iteration).
//!
//! ## Datasets
//!
//! Downloaded on first use via [`apex_io::ensure_odometry_dataset`]:
//!
//! - `M3500` (2D, 3,500 vertices) and `intel` (2D, 1,228 vertices / 1,483 edges)
//! - `sphere2500` (3D, 2,500 vertices) and `parking-garage` (3D, 1,661 vertices / 6,275 edges)
//! - `torus3D` (3D)
//!
//! ## Accuracy guard
//!
//! Every timed run asserts its final cost against [`GOLDEN_FINAL_COSTS`] at a
//! relative tolerance of [`GOLDEN_REL_TOLERANCE`] — the same values and bound
//! `tests/golden_values.rs` pins. A "faster" solve that converges to a worse
//! optimum (loosened tolerances, truncated sweeps, fewer iterations) fails the
//! benchmark instead of winning it. The configs below must stay exactly the
//! ones the goldens were pinned with.
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench odometry_benchmark                       # all datasets
//! cargo bench --bench odometry_benchmark -- sphere2500         # one dataset
//! cargo bench --bench odometry_benchmark -- --save-baseline true_baseline
//! cargo bench --bench odometry_benchmark -- --baseline true_baseline
//! ```
//!
//! Benchmarks must run strictly sequentially — never launch two bench
//! processes concurrently, they compete for cores and invalidate measurements.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use apex_io::{G2oLoader, GraphLoader};
use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::L2Loss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::BetweenFactor;
use apex_solver::init_logger;
use apex_solver::linalg::JacobianMode;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use tracing::debug;

/// Pinned golden final costs, relative tolerance
/// [`GOLDEN_REL_TOLERANCE`].
///
/// `M3500`, `parking-garage` and `sphere2500` are the exact values
/// `tests/golden_values.rs` pins (same solver configs). `intel` and `torus3D`
/// must be pinned from this benchmark's own validation run before a baseline
/// is recorded — until then they run unguarded.
const GOLDEN_FINAL_COSTS: &[(&str, f64)] = &[
    ("M3500", 1.510_940_460_434e0),
    ("parking-garage", 6.245_093_871_665e-1),
    ("sphere2500", 2.129_064_817_909e1),
    ("intel", 3.893_032_054_396e-1),
    ("torus3D", 1.201_016_810_957e2),
];

/// Maximum relative deviation of a timed run's final cost from its golden.
///
/// faer's SIMD reductions sum in architecture-dependent order, so the last
/// ulps of the final cost differ between hosts — the same bound
/// `tests/golden_values.rs` uses. Any algorithmic drift moves the cost far
/// beyond it.
const GOLDEN_REL_TOLERANCE: f64 = 1e-6;

/// One benched dataset: its loaded graph plus its pinned golden, if any.
struct OdometryCase {
    name: &'static str,
    is_3d: bool,
    graph: apex_io::Graph,
    golden: Option<f64>,
}

/// `(dataset, is_3d)` for every benched graph. The goldens above must cover
/// every dataset listed here before a baseline is recorded.
const DATASETS: &[(&str, bool)] = &[
    ("M3500", false),
    ("intel", false),
    ("sphere2500", true),
    ("parking-garage", true),
    ("torus3D", true),
];

/// Load every dataset, failing hard if one is missing: a benchmark that
/// silently skips a dataset would silently skip its accuracy guard too.
fn load_cases() -> Vec<OdometryCase> {
    DATASETS
        .iter()
        .map(|&(name, is_3d)| {
            let path = apex_io::ensure_odometry_dataset(name)
                .unwrap_or_else(|e| panic!("failed to provision odometry dataset {name}: {e}"));
            let graph =
                G2oLoader::load(&path).unwrap_or_else(|e| panic!("failed to load {name}: {e}"));
            let golden = GOLDEN_FINAL_COSTS
                .iter()
                .find(|(g, _)| *g == name)
                .map(|&(_, cost)| cost);
            OdometryCase {
                name,
                is_3d,
                graph,
                golden,
            }
        })
        .collect()
}

/// Build the pose-graph problem: one SE2/SE3 variable per vertex (in sorted id
/// order, for determinism) and one [`BetweenFactor`] per edge with L2 loss.
///
/// Mirrors `tests/golden_values.rs` exactly — the goldens are only valid for
/// this problem construction.
fn build_problem(case: &OdometryCase) -> Problem {
    let graph = &case.graph;
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys: HashMap<usize, apex_solver::core::VarKey> = HashMap::new();

    if case.is_3d {
        let mut ids: Vec<_> = graph.vertices_se3.keys().copied().collect();
        ids.sort_unstable();
        for &id in &ids {
            let v = &graph.vertices_se3[&id];
            let q = v.pose.rotation_quaternion();
            let t = v.pose.translation();
            let key = problem.add_variable(
                ManifoldType::SE3,
                nalgebra::DVector::from_vec(vec![t.x, t.y, t.z, q.w, q.i, q.j, q.k]),
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
        let mut ids: Vec<_> = graph.vertices_se2.keys().copied().collect();
        ids.sort_unstable();
        for &id in &ids {
            let v = &graph.vertices_se2[&id];
            let key = problem.add_variable(
                ManifoldType::SE2,
                nalgebra::DVector::from_vec(vec![v.x(), v.y(), v.theta()]),
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
    problem
}

/// The exact configs the goldens were pinned with: default SparseCholesky
/// linear solver, damping 1e-4, cost/parameter tolerance 1e-4, iteration cap
/// and gradient tolerance differing between SE2 and SE3. Changing any of
/// these changes the converged cost and invalidates [`GOLDEN_FINAL_COSTS`].
fn config_for(is_3d: bool) -> LevenbergMarquardtConfig {
    let (max_iterations, gradient_tol) = if is_3d { (100, 1e-12) } else { (150, 1e-10) };
    LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(gradient_tol)
        .with_damping(1e-4)
}

/// Assert a timed run converged to the pinned optimum.
fn assert_golden(name: &str, cost: f64, initial: f64, golden: Option<f64>) {
    debug!("{name}: initial cost {initial:.12e}, final cost {cost:.12e}");
    assert!(
        cost.is_finite() && cost < initial,
        "{name}: solve did not make progress (initial {initial:.12e}, final {cost:.12e})"
    );
    if let Some(golden) = golden {
        let rel = ((cost - golden) / golden.abs().max(1.0)).abs();
        assert!(
            rel <= GOLDEN_REL_TOLERANCE,
            "{name}: final cost {cost:.12e} deviates from pinned golden {golden:.12e} \
             (relative {rel:.3e} > {GOLDEN_REL_TOLERANCE})"
        );
    }
}

fn bench_odometry(c: &mut Criterion) {
    init_logger();
    let cases = load_cases();

    let mut group = c.benchmark_group("odometry");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("solve", case.name), case, |b, case| {
            b.iter_batched(
                || build_problem(case),
                |mut problem| {
                    let mut solver = LevenbergMarquardt::with_config(config_for(case.is_3d));
                    let result = solver
                        .optimize(&mut problem)
                        .unwrap_or_else(|e| panic!("solve failed on {}: {e}", case.name));
                    assert_golden(
                        case.name,
                        result.final_cost,
                        result.initial_cost,
                        case.golden,
                    );
                    black_box(result.final_cost)
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_odometry);
criterion_main!(benches);
