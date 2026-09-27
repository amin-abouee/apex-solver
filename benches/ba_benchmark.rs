//! Criterion benchmark for bundle adjustment on BAL datasets across the
//! Schur-complement solver family.
//!
//! Measures full wall-clock solve time of [`LevenbergMarquardt`] with the
//! [`LevenbergMarquardtConfig::for_bundle_adjustment`] preset, from the
//! identical initial state every run (the problem is rebuilt in the untimed
//! setup of each iteration).
//!
//! ## Datasets
//!
//! Downloaded on first use via `apex_io::utils::ensure_ba_dataset`:
//!
//! - `trafalgar-21` (21 cameras, 11,315 landmarks, 36,455 observations)
//! - `trafalgar-257` (257 cameras, 65,132 landmarks, 225,911 observations)
//! - `dubrovnik-135` (135 cameras, 90,642 landmarks, 553,336 observations)
//! - `venice-52` (52 cameras, 64,053 landmarks, 347,173 observations)
//!
//! This spans a ~7x observation range — enough to see Schur-solver scaling —
//! while keeping a full run to well under an hour on a mid-range laptop. The
//! largest BAL problems (dubrovnik-356, ladybug-1723, venice-1778) are
//! minutes to tens of minutes per solve; exercise them occasionally through
//! `bundle_adjustment_benchmark`, not in every Criterion run.
//!
//! ## Solver variants
//!
//! - `schur_explicit_sparse` — [`ExplicitSchurVariant::Sparse`]: forms the
//!   reduced system `S`, sparse-Cholesky factorizes it
//! - `schur_explicit_iterative` — [`ExplicitSchurVariant::Iterative`]: PCG on
//!   the formed `S`
//!
//! These two are the optimization targets (explicit Schur is the VIO path);
//! the implicit and chunked variants were dropped from the routine matrix to
//! keep each benchmark iteration fast — they live in git history and in
//! `bundle_adjustment_benchmark` for occasional deep checks.
//!
//! ## Accuracy guard
//!
//! Every timed run asserts its final cost against [`GOLDEN_FINAL_COSTS`]:
//! cost at most [`COST_TOLERANCE_RATIO`]x the golden, and reprojection RMSE
//! (`sqrt(2·cost/observations)`, the convention `bundle_adjustment_benchmark.rs`
//! uses) at most [`RMSE_TOLERANCE_PX`] px above it. A "faster" solve that
//! converges to a worse optimum fails the benchmark instead of winning it.
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench ba_benchmark                          # all datasets x variants
//! cargo bench --bench ba_benchmark -- schur_implicit        # one variant everywhere
//! cargo bench --bench ba_benchmark -- trafalgar-257         # one dataset
//! cargo bench --bench ba_benchmark -- --save-baseline true_baseline
//! cargo bench --bench ba_benchmark -- --baseline true_baseline
//! ```
//!
//! A full run takes tens of minutes (the large datasets alone are ~15 s per
//! solve, ten samples each); filter while iterating, run everything before
//! accepting a change. Benchmarks must run strictly sequentially — never
//! launch two bench processes concurrently.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_io::utils::ensure_ba_dataset;
use apex_io::{BalDataset, BalLoader};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::so3::SO3;
use apex_solver::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::SelfCalibration;
use apex_solver::factors::visual::ProjectionFactor;
use apex_solver::init_logger;
use apex_solver::linalg::{ExplicitSchurVariant, JacobianMode, LinearSolverType};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};
use tracing::debug;

/// Pinned golden final costs per `(dataset, variant)`, bound by
/// [`COST_TOLERANCE_RATIO`] and [`RMSE_TOLERANCE_PX`].
///
/// Pinned from this benchmark's validation runs on the configs below (see
/// `benchmarking_results/baseline_criterion.md` for the matching timings).
/// The iterative variant may legitimately converge to a slightly different
/// optimum than the sparse direct solve — each variant is guarded against its
/// own golden, never across variants.
const GOLDEN_FINAL_COSTS: &[(&str, &str, f64)] = &[
    ("trafalgar-21", "schur_explicit_sparse", 1.370_013_056_594e4),
    (
        "trafalgar-21",
        "schur_explicit_iterative",
        1.370_020_699_431e4,
    ),
    (
        "trafalgar-257",
        "schur_explicit_sparse",
        6.863_327_074_594e4,
    ),
    (
        "trafalgar-257",
        "schur_explicit_iterative",
        6.799_201_774_758e4,
    ),
    (
        "dubrovnik-135",
        "schur_explicit_sparse",
        1.888_767_976_834e5,
    ),
    (
        "dubrovnik-135",
        "schur_explicit_iterative",
        1.905_530_913_190e5,
    ),
    ("venice-52", "schur_explicit_sparse", 9.716_652_695_390e4),
    ("venice-52", "schur_explicit_iterative", 9.195_236_768_263e4),
];

/// Maximum ratio of a timed run's final cost to its golden.
const COST_TOLERANCE_RATIO: f64 = 1.0005;

/// Maximum reprojection-RMSE increase (px) of a timed run over its golden.
const RMSE_TOLERANCE_PX: f64 = 0.01;

/// Which BAL problem to run: registry key for download, label for benchmark
/// IDs and goldens.
#[derive(Debug, Clone, Copy)]
struct BaDataset {
    registry: &'static str,
    label: &'static str,
    cameras: u32,
    points: u32,
}

/// Every dataset runs the two explicit variants; trafalgar-21 doubles as the
/// fast per-hypothesis probe via Criterion filters.
const DATASETS: &[BaDataset] = &[
    BaDataset {
        registry: "trafalgar",
        label: "trafalgar-21",
        cameras: 21,
        points: 11315,
    },
    BaDataset {
        registry: "trafalgar",
        label: "trafalgar-257",
        cameras: 257,
        points: 65132,
    },
    BaDataset {
        registry: "dubrovnik",
        label: "dubrovnik-135",
        cameras: 135,
        points: 90642,
    },
    BaDataset {
        registry: "venice",
        label: "venice-52",
        cameras: 52,
        points: 64053,
    },
];

/// One Schur-solver row of the benchmark matrix.
#[derive(Debug, Clone, Copy)]
struct SolverVariant {
    label: &'static str,
    linear_solver_type: LinearSolverType,
    /// Only meaningful when `linear_solver_type` is `ExplicitSparseSchur`.
    schur_variant: ExplicitSchurVariant,
}

/// The variant matrix: the two explicit Schur paths on every dataset.
fn variants_for() -> Vec<SolverVariant> {
    vec![
        SolverVariant {
            label: "schur_explicit_sparse",
            linear_solver_type: LinearSolverType::ExplicitSparseSchur,
            schur_variant: ExplicitSchurVariant::Sparse,
        },
        SolverVariant {
            label: "schur_explicit_iterative",
            linear_solver_type: LinearSolverType::ExplicitSparseSchur,
            schur_variant: ExplicitSchurVariant::Iterative,
        },
    ]
}

/// The preset with the variant's Schur solver selected; everything else
/// (damping, tolerances, iteration cap) stays whatever the library tunes.
fn config_for(variant: SolverVariant) -> LevenbergMarquardtConfig {
    LevenbergMarquardtConfig::for_bundle_adjustment()
        .with_linear_solver_type(variant.linear_solver_type)
        .with_schur_variant(variant.schur_variant)
}

/// Assert a timed run made progress and matched its pinned golden.
fn assert_accuracy(label: &str, variant: &str, cost: f64, initial_cost: f64, num_obs: usize) {
    debug!("{label}/{variant}: initial cost {initial_cost:.12e}, final cost {cost:.12e}");
    assert!(
        cost.is_finite() && cost < initial_cost,
        "{label}/{variant}: solve did not make progress \
         (initial {initial_cost:.12e}, final {cost:.12e})"
    );
    if let Some(&(_, _, golden)) = GOLDEN_FINAL_COSTS
        .iter()
        .find(|(d, v, _)| *d == label && *v == variant)
    {
        assert!(
            cost <= golden * COST_TOLERANCE_RATIO,
            "{label}/{variant}: final cost {cost:.6e} exceeds pinned golden \
             {golden:.6e} beyond {COST_TOLERANCE_RATIO}x"
        );
        // Solver cost = 0.5 * sum ||r||², so MSE = mean ||r||² = 2·cost/n.
        let rmse = (2.0 * cost / num_obs as f64).sqrt();
        let golden_rmse = (2.0 * golden / num_obs as f64).sqrt();
        assert!(
            rmse - golden_rmse <= RMSE_TOLERANCE_PX,
            "{label}/{variant}: reprojection RMSE {rmse:.6}px exceeds pinned \
             golden {golden_rmse:.6}px beyond {RMSE_TOLERANCE_PX}px"
        );
    }
}

/// Convert a BAL axis-angle rotation to `SO3`.
fn axis_angle_to_so3(axis_angle: &Vector3<f64>) -> SO3 {
    let angle = axis_angle.norm();
    if angle < 1e-10 {
        SO3::identity()
    } else {
        let axis = axis_angle / angle;
        SO3::from_axis_angle(&axis, angle)
    }
}

/// Build the SelfCalibration BA problem: an SE3 pose and 3-parameter
/// intrinsics per camera, an eliminated `Rn` landmark per observed point, one
/// Huber(1 px) projection factor per observation, camera 0 fixed for gauge
/// freedom. Mirrors `bundle_adjustment_benchmark.rs::build_ba_problem`.
fn build_ba_problem(dataset: &BalDataset) -> Result<Problem, String> {
    let observations = &dataset.observations;
    let mut problem = Problem::new(JacobianMode::Sparse);

    // Add cameras as SE3 poses plus a 3-parameter intrinsics block each.
    let mut pose_keys: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    let mut intr_keys: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    for cam in &dataset.cameras {
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let pose = SE3::from_translation_so3(translation, axis_angle_to_so3(&axis_angle));

        pose_keys.push(problem.add_variable(
            ManifoldType::SE3,
            DVector::from_column_slice(pose.as_param_slice()),
        ));
        intr_keys.push(problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![cam.focal_length, cam.k1, cam.k2]),
        ));
    }

    // Only points the observations reference become variables: an unobserved
    // landmark contributes no residual and would leave its `H_ee` block
    // singular for every Schur solver to trip over.
    let mut point_indices: Vec<usize> = observations.iter().map(|o| o.point_index).collect();
    point_indices.sort_unstable();
    point_indices.dedup();

    let mut pt_keys: HashMap<usize, VarKey> = HashMap::with_capacity(point_indices.len());
    for index in point_indices {
        let position = &dataset.points[index].position;
        let pt_key = problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![position.x, position.y, position.z]),
        );
        problem.mark_for_elimination(pt_key);
        pt_keys.insert(index, pt_key);
    }

    for obs in observations {
        let cam = &dataset.cameras[obs.camera_index];
        let camera = BALPinholeCameraStrict::new(
            PinholeParams {
                fx: cam.focal_length,
                fy: cam.focal_length,
                cx: 0.0,
                cy: 0.0,
            },
            DistortionModel::Radial {
                k1: cam.k1,
                k2: cam.k2,
            },
        )
        .map_err(|e| format!("Invalid camera parameters: {e}"))?;

        let measurements = Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, SelfCalibration> =
            ProjectionFactor::new(measurements, camera);

        let pt_key = pt_keys
            .get(&obs.point_index)
            .ok_or_else(|| format!("no variable for point {}", obs.point_index))?;

        let loss = HuberLoss::new(1.0).map_err(|e| format!("Invalid Huber loss: {e}"))?;
        problem.add_residual_block(
            &[
                pose_keys[obs.camera_index],
                *pt_key,
                intr_keys[obs.camera_index],
            ],
            Box::new(factor),
            Some(Box::new(loss)),
        );
    }

    // Fix first camera pose (gauge freedom) - all 6 DOF.
    let first_pose = *pose_keys.first().ok_or("dataset has no cameras")?;
    for dof in 0..6 {
        problem.fix_variable(first_pose, dof);
    }

    Ok(problem)
}

fn bench_bundle_adjustment(c: &mut Criterion) {
    init_logger();
    let mut group = c.benchmark_group("bundle_adjustment");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for spec in DATASETS {
        let path = ensure_ba_dataset(spec.registry, spec.cameras, spec.points)
            .unwrap_or_else(|e| panic!("failed to provision BAL dataset {}: {e}", spec.label));
        let dataset =
            BalLoader::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", spec.label));

        for variant in variants_for() {
            group.bench_with_input(
                BenchmarkId::new("solve", format!("{}/{}", spec.label, variant.label)),
                &dataset,
                |b, dataset| {
                    let num_obs = dataset.observations.len();
                    b.iter_batched(
                        || {
                            build_ba_problem(dataset).unwrap_or_else(|e| {
                                panic!(
                                    "failed to build BA problem for {}/{}: {e}",
                                    spec.label, variant.label
                                )
                            })
                        },
                        |mut problem| {
                            let mut solver = LevenbergMarquardt::with_config(config_for(variant));
                            let result = solver.optimize(&mut problem).unwrap_or_else(|e| {
                                panic!("solve failed for {}/{}: {e}", spec.label, variant.label)
                            });
                            assert_accuracy(
                                spec.label,
                                variant.label,
                                result.final_cost,
                                result.initial_cost,
                                num_obs,
                            );
                            black_box(result.final_cost)
                        },
                        BatchSize::PerIteration,
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_bundle_adjustment);
criterion_main!(benches);
