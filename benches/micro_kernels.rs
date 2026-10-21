//! Criterion micro-benchmarks for the per-iteration hot kernels:
//!
//! - [`normal_eq`]: parallel formation of `H = JᵀJ` and `g = Jᵀr`
//! - [`cholesky`]/[`qr`]: damped sparse solve `(JᵀJ + λD)·dx = −Jᵀr`
//! - `pcg` group: `ImplicitSparseSchur` (matrix-free PCG), `ExplicitSparseSchur`
//!   with `Iterative` (PCG on the formed `S`) and default (Cholesky-on-`S`)
//!   variants, `ExplicitDenseSchur`, and the shared `schur::pcg` primitive in
//!   isolation — all at matched problem size, so the numbers compare directly
//! - `schur_window` group: `ExplicitSparseSchur` on a visual-inertial window
//!   whose coupling rows include stored zeros
//! - [`assemble`]: residual/Jacobian assembly for a synthetic factor graph
//! - [`fingerprint`]: structural pattern hashing guarding the sparse caches
//!
//! ```bash
//! cargo bench --bench micro_kernels
//! ```

use apex_manifolds::rn::Rn;
use apex_manifolds::{LieGroup, se3::SE3};
use apex_solver::ManifoldType;
use apex_solver::core::problem::Problem;
use apex_solver::core::variable::Variable;
use apex_solver::factors::Factor;
use apex_solver::linalg::LinearSolverType;
use apex_solver::linalg::dense::schur::ExplicitDenseSchur;
use apex_solver::linalg::schur::{PcgParams, solve_pcg};
use apex_solver::linalg::sparse::normal_eq::NormalEquationsCache;
use apex_solver::linalg::sparse::pattern::PatternFingerprint;
use apex_solver::linalg::sparse::qr::SparseQRSolver;
use apex_solver::linalg::sparse::{
    ExplicitSchurVariant, ExplicitSparseSchur, ImplicitSparseSchur, SparseCholeskySolver,
};
use apex_solver::linalg::{Damping, DenseMode, LinearSolver, StructureAware};
use apex_solver::linearizer::cpu::sparse::assemble_sparse;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use faer::Mat;
use faer::prelude::ReborrowMut;
use faer::sparse::{SparseColMat, Triplet};
use slotmap::{SecondaryMap, SlotMap};
use std::hint::black_box;

type VarMap =
    SlotMap<apex_solver::core::VarKey, Box<dyn apex_solver::core::variable::ManifoldVariable>>;

type BaSystem = (
    SparseColMat<usize, f64>,
    Mat<f64>,
    VarMap,
    SecondaryMap<apex_solver::core::VarKey, usize>,
    std::collections::HashSet<apex_solver::core::VarKey>,
);

/// Deterministic pseudo-random sparse matrix: `rows × cols`, `nnz_per_col`
/// entries per column.
fn sample_jacobian(rows: usize, cols: usize, nnz_per_col: usize) -> SparseColMat<usize, f64> {
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut state: u64 = 0x9E3779B97F4A7C15;
    for col in 0..cols {
        for _k in 0..nnz_per_col {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let row = (state as usize) % rows;
            let value = ((state >> 33) % 1000) as f64 / 250.0 - 2.0;
            triplets.push(Triplet::new(row, col, value));
        }
    }
    SparseColMat::try_new_from_triplets(rows, cols, &triplets)
        .unwrap_or_else(|e| panic!("valid triplets required: {e:?}"))
}

/// Synthetic BA-like Jacobian: `groups` blocks of 2 cameras + 3 landmarks,
/// each camera observing each landmark of its group with a 6-row factor.
fn ba_like_system(groups: usize) -> BaSystem {
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let cam_dof = 6;
    let lm_dof = 3;
    // Layout: all camera columns first, then all landmark columns — the Schur
    // extraction routines assume these are two contiguous column ranges.
    let cam_block = 2 * groups * cam_dof;
    let group_rows = 6 * 2 * 3; // 6 observations of 6 rows
    let total_rows = groups * group_rows;
    let total_cols = cam_block + 3 * groups * lm_dof;

    let mut variables: SlotMap<
        apex_solver::core::VarKey,
        Box<dyn apex_solver::core::variable::ManifoldVariable>,
    > = SlotMap::with_key();
    let mut index_map: SecondaryMap<apex_solver::core::VarKey, usize> = SecondaryMap::new();
    let mut landmark_keys = std::collections::HashSet::new();

    let cam_se3 = nalgebra::DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let pt_zero = nalgebra::DVector::from_vec(vec![0.0, 0.0, 0.0]);

    for g in 0..groups {
        let row_base = g * group_rows;

        let mut cam_cols = Vec::new();
        for c in 0..2 {
            let key = variables.insert(Box::new(Variable::new(SE3::from_param_slice(
                cam_se3.as_slice(),
            ))));
            let col = (2 * g + c) * cam_dof;
            index_map.insert(key, col);
            cam_cols.push(col);
        }
        let mut lm_cols = Vec::new();
        for l in 0..3 {
            let key = variables.insert(Box::new(Variable::new(Rn::new(pt_zero.clone()))));
            let col = cam_block + (3 * g + l) * lm_dof;
            index_map.insert(key, col);
            landmark_keys.insert(key);
            lm_cols.push(col);
        }

        for (ci, &cc) in cam_cols.iter().enumerate() {
            for (li, &lc) in lm_cols.iter().enumerate() {
                let rb = row_base + (ci * 3 + li) * 6;
                for k in 0..6 {
                    triplets.push(Triplet::new(rb + k, cc + k, 1.0));
                    triplets.push(Triplet::new(rb + k, lc + (k % 3), 1.0));
                }
            }
        }
    }

    let jacobian = SparseColMat::try_new_from_triplets(total_rows, total_cols, &triplets)
        .unwrap_or_else(|e| panic!("valid triplets required: {e:?}"));
    let residuals = Mat::from_fn(total_rows, 1, |i, _| ((i * 13) % 17) as f64 * 0.1);
    (jacobian, residuals, variables, index_map, landmark_keys)
}

/// Synthetic sliding-window system shaped like a visual-inertial back end:
/// `frames` retained 9-DOF navigation states and one 6-DOF extrinsic, with
/// 1-DOF inverse-range landmarks eliminated. Each landmark is seen from a host
/// frame and three observers, two rows per observation, and every row stores
/// its state's last three columns (velocity) as explicit zeros — a
/// reprojection residual does not depend on velocity, but the factor over
/// `SE_2(3)` still emits the full 9-column block.
fn window_like_system(frames: usize, landmarks: usize) -> BaSystem {
    const STATE_DOF: usize = 9;
    const POSE_DOF: usize = 6;
    const EXTRINSIC_DOF: usize = 6;
    let extrinsic_col = frames * STATE_DOF;
    let landmark_base = extrinsic_col + EXTRINSIC_DOF;
    let total_cols = landmark_base + landmarks;
    let total_rows = landmarks * 6;

    let mut variables: VarMap = SlotMap::with_key();
    let mut index_map: SecondaryMap<apex_solver::core::VarKey, usize> = SecondaryMap::new();
    let mut landmark_keys = std::collections::HashSet::new();
    for f in 0..frames {
        let key = variables.insert(Box::new(Variable::new(Rn::new(nalgebra::DVector::zeros(
            STATE_DOF,
        )))));
        index_map.insert(key, f * STATE_DOF);
    }
    let key = variables.insert(Box::new(Variable::new(Rn::new(nalgebra::DVector::zeros(
        EXTRINSIC_DOF,
    )))));
    index_map.insert(key, extrinsic_col);
    for l in 0..landmarks {
        let key = variables.insert(Box::new(Variable::new(Rn::new(nalgebra::DVector::zeros(
            1,
        )))));
        index_map.insert(key, landmark_base + l);
        landmark_keys.insert(key);
    }

    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut value = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((state >> 33) % 1000) as f64 + 1.0) / 250.0 - 2.0
    };
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for l in 0..landmarks {
        let host = l % frames;
        let stride = 1 + l % 3;
        for j in 0..3 {
            let observer = (host + 1 + j * stride) % frames;
            for r in 0..2 {
                let row = l * 6 + j * 2 + r;
                for frame in [host, observer] {
                    for k in 0..STATE_DOF {
                        let v = if k < POSE_DOF { value() } else { 0.0 };
                        triplets.push(Triplet::new(row, frame * STATE_DOF + k, v));
                    }
                }
                for k in 0..EXTRINSIC_DOF {
                    triplets.push(Triplet::new(row, extrinsic_col + k, value()));
                }
                triplets.push(Triplet::new(row, landmark_base + l, value()));
            }
        }
    }

    let jacobian = SparseColMat::try_new_from_triplets(total_rows, total_cols, &triplets)
        .unwrap_or_else(|e| panic!("valid triplets required: {e:?}"));
    let residuals = Mat::from_fn(total_rows, 1, |i, _| ((i * 13) % 17) as f64 * 0.1);
    (jacobian, residuals, variables, index_map, landmark_keys)
}

/// A synthetic visual-inertial observation: two 9-column states (the last
/// three, velocity, have zero weight and so never move the residual), a
/// 6-column extrinsic and a one-column landmark, two residual rows.
struct WindowObservation {
    weights: [[f64; 25]; 2],
    target: [f64; 2],
}

impl Factor for WindowObservation {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let mut jacobian = jacobian;
        for (r, weights) in self.weights.iter().enumerate() {
            let x = params.iter().flat_map(|block| block.iter().copied());
            let s: f64 = x.zip(weights).map(|(x, w)| x * w).sum();
            residual[r] = s.sin() - self.target[r];
            if let Some(jac) = jacobian.as_mut() {
                let c = s.cos();
                for (col, w) in weights.iter().enumerate() {
                    jac[(r, col)] = c * w;
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 25)
    }
}

/// [`window_like_system`] as a solvable `Problem`: the same layout, with a
/// smooth nonlinear residual so Levenberg-Marquardt needs several iterations.
fn window_problem(frames: usize, landmarks: usize) -> Problem {
    let mut problem = Problem::new(apex_solver::JacobianMode::Sparse);
    let states: Vec<_> = (0..frames)
        .map(|_| problem.add_variable(ManifoldType::RN, nalgebra::DVector::zeros(9)))
        .collect();
    let extrinsic = problem.add_variable(ManifoldType::RN, nalgebra::DVector::zeros(6));

    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % 1000) as f64 / 1000.0 - 0.5
    };
    // The solve starts at zero and targets a hidden true state, so the
    // residuals are consistent and the iterations converge.
    let truth_states: Vec<[f64; 9]> = (0..frames)
        .map(|_| std::array::from_fn(|k| if k < 6 { next() } else { 0.0 }))
        .collect();
    let truth_extrinsic: [f64; 6] = std::array::from_fn(|_| next());
    for l in 0..landmarks {
        let landmark = problem.add_variable(ManifoldType::RN, nalgebra::DVector::zeros(1));
        problem.mark_for_elimination(landmark);
        let truth_landmark = next();
        let host = l % frames;
        let stride = 1 + l % 3;
        for j in 0..3 {
            let observer = (host + 1 + j * stride) % frames;
            let mut weights = [[0.0; 25]; 2];
            let mut target = [0.0; 2];
            for (row, t) in weights.iter_mut().zip(&mut target) {
                for (col, w) in row.iter_mut().enumerate() {
                    let velocity = (6..9).contains(&col) || (15..18).contains(&col);
                    if !velocity {
                        *w = next();
                    }
                }
                let truth = truth_states[host]
                    .iter()
                    .chain(&truth_states[observer])
                    .chain(&truth_extrinsic)
                    .chain(std::iter::once(&truth_landmark));
                *t = truth.zip(row.iter()).map(|(x, w)| x * w).sum::<f64>().sin();
            }
            problem.add_residual_block(
                &[states[host], states[observer], extrinsic, landmark],
                Box::new(WindowObservation { weights, target }),
                None,
            );
        }
    }
    problem
}

type BaSystemDense = (
    Mat<f64>,
    Mat<f64>,
    VarMap,
    SecondaryMap<apex_solver::core::VarKey, usize>,
    std::collections::HashSet<apex_solver::core::VarKey>,
);

/// Dense counterpart of [`ba_like_system`], for `ExplicitDenseSchur` (targets
/// < ~500 DOF, so `groups` should stay small — a handful, not hundreds).
fn ba_like_system_dense(groups: usize) -> BaSystemDense {
    let cam_dof = 6;
    let lm_dof = 3;
    let cam_block = 2 * groups * cam_dof;
    let group_rows = 6 * 2 * 3;
    let total_rows = groups * group_rows;
    let total_cols = cam_block + 3 * groups * lm_dof;

    let mut variables: SlotMap<
        apex_solver::core::VarKey,
        Box<dyn apex_solver::core::variable::ManifoldVariable>,
    > = SlotMap::with_key();
    let mut index_map: SecondaryMap<apex_solver::core::VarKey, usize> = SecondaryMap::new();
    let mut landmark_keys = std::collections::HashSet::new();

    let cam_se3 = nalgebra::DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let pt_zero = nalgebra::DVector::from_vec(vec![0.0, 0.0, 0.0]);

    let mut jacobian = Mat::<f64>::zeros(total_rows, total_cols);
    for g in 0..groups {
        let row_base = g * group_rows;

        let mut cam_cols = Vec::new();
        for c in 0..2 {
            let key = variables.insert(Box::new(Variable::new(SE3::from_param_slice(
                cam_se3.as_slice(),
            ))));
            let col = (2 * g + c) * cam_dof;
            index_map.insert(key, col);
            cam_cols.push(col);
        }
        let mut lm_cols = Vec::new();
        for l in 0..3 {
            let key = variables.insert(Box::new(Variable::new(Rn::new(pt_zero.clone()))));
            let col = cam_block + (3 * g + l) * lm_dof;
            index_map.insert(key, col);
            landmark_keys.insert(key);
            lm_cols.push(col);
        }

        for (ci, &cc) in cam_cols.iter().enumerate() {
            for (li, &lc) in lm_cols.iter().enumerate() {
                let rb = row_base + (ci * 3 + li) * 6;
                for k in 0..6 {
                    let cam_val = 1.0 + 0.1 * (ci as f64 + 1.0);
                    let lm_val = 0.3 + 0.05 * ((k + li + 2 * ci) % 5) as f64;
                    jacobian[(rb + k, cc + k)] = cam_val;
                    jacobian[(rb + k, lc + (k % 3))] = lm_val;
                }
            }
        }
    }
    let residuals = Mat::from_fn(total_rows, 1, |i, _| ((i * 13) % 17) as f64 * 0.1);
    (jacobian, residuals, variables, index_map, landmark_keys)
}

fn bench_normal_equations(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_eq");
    let jacobian = sample_jacobian(20_000, 5_000, 4);
    let residuals = Mat::from_fn(20_000, 1, |i, _| (i % 7) as f64 - 3.0);

    let mut cache =
        NormalEquationsCache::try_new(&jacobian).unwrap_or_else(|e| panic!("cache builds: {e:?}"));
    group.bench_function("hessian_5k_cols", |b| {
        b.iter(|| {
            let ne = cache
                .compute(black_box(&residuals), black_box(&jacobian))
                .unwrap_or_else(|e| panic!("compute failed: {e:?}"));
            black_box(ne.hessian.nrows())
        })
    });
    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    let jacobian = sample_jacobian(20_000, 5_000, 4);
    let residuals = Mat::from_fn(20_000, 1, |i, _| (i % 7) as f64 - 3.0);
    let damping = Damping::new(1e-4, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = SparseCholeskySolver::new();
    group.bench_function("damped_solve_5k_cols", |b| {
        b.iter(|| {
            let dx = solver
                .solve_augmented_equation(
                    black_box(&residuals),
                    black_box(&jacobian),
                    black_box(&damping),
                )
                .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr");
    let jacobian = sample_jacobian(20_000, 5_000, 4);
    let residuals = Mat::from_fn(20_000, 1, |i, _| (i % 7) as f64 - 3.0);
    let damping = Damping::new(1e-4, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = SparseQRSolver::new();
    group.bench_function("damped_solve_5k_cols", |b| {
        b.iter(|| {
            let dx = solver
                .solve_augmented_equation(
                    black_box(&residuals),
                    black_box(&jacobian),
                    black_box(&damping),
                )
                .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.finish();
}

fn bench_pcg_iterative_schur(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");
    let (jacobian, residuals, variables, index_map, landmark_keys) = ba_like_system(250);
    let damping = Damping::new(1e-3, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = ImplicitSparseSchur::new();
    solver
        .initialize_structure(&variables, &index_map, &landmark_keys)
        .unwrap_or_else(|e| panic!("structure init: {e:?}"));

    group.bench_function("iterative_schur_500_cameras", |b| {
        b.iter(|| {
            let dx = solver
                .solve_augmented_equation(
                    black_box(&residuals),
                    black_box(&jacobian),
                    black_box(&damping),
                )
                .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.finish();
}

/// PCG on the *explicit* `S` (`ExplicitSchurVariant::Iterative`), same
/// fixture as [`bench_pcg_iterative_schur`] — isolates the cost of PCG on a
/// formed `S` vs. the matrix-free operator, at matched problem size.
fn bench_pcg_explicit_schur(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");
    let (jacobian, residuals, variables, index_map, landmark_keys) = ba_like_system(250);
    let damping = Damping::new(1e-3, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = ExplicitSparseSchur::new()
        .with_variant(ExplicitSchurVariant::Iterative)
        .with_cg_params(200, 1e-6);
    solver
        .initialize_structure(&variables, &index_map, &landmark_keys)
        .unwrap_or_else(|e| panic!("structure init: {e:?}"));

    group.bench_function("explicit_iterative_schur_500_cameras", |b| {
        b.iter(|| {
            let dx = solver
                .solve_augmented_equation(
                    black_box(&residuals),
                    black_box(&jacobian),
                    black_box(&damping),
                )
                .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.finish();
}

/// Direct Cholesky on the explicit `S` — the baseline every iterative variant
/// above is measured against, same fixture and size.
fn bench_explicit_sparse_schur(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");
    let (jacobian, residuals, variables, index_map, landmark_keys) = ba_like_system(250);
    let damping = Damping::new(1e-3, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = ExplicitSparseSchur::new();
    solver
        .initialize_structure(&variables, &index_map, &landmark_keys)
        .unwrap_or_else(|e| panic!("structure init: {e:?}"));

    group.bench_function("explicit_sparse_schur_500_cameras", |b| {
        b.iter(|| {
            let dx = solver
                .solve_augmented_equation(
                    black_box(&residuals),
                    black_box(&jacobian),
                    black_box(&damping),
                )
                .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.finish();
}

/// `ExplicitSparseSchur` on a visual-inertial window: 11 states of nine
/// columns whose velocity columns are stored zeros, and 560 one-column
/// landmarks. The BA fixtures above have no stored zeros, so this is the one
/// that sees coupling rows that contribute nothing.
fn bench_explicit_sparse_schur_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("schur_window");
    let (jacobian, residuals, variables, index_map, landmark_keys) = window_like_system(11, 560);
    let damping = Damping::new(1e-3, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = ExplicitSparseSchur::new();
    solver
        .initialize_structure(&variables, &index_map, &landmark_keys)
        .unwrap_or_else(|e| panic!("structure init: {e:?}"));

    group.bench_function("explicit_sparse_schur_11_states_560_landmarks", |b| {
        b.iter(|| {
            let dx = solver
                .solve_augmented_equation(
                    black_box(&residuals),
                    black_box(&jacobian),
                    black_box(&damping),
                )
                .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.bench_function("lm_8_iterations_11_states_560_landmarks", |b| {
        b.iter_batched(
            || window_problem(11, 560),
            |mut problem| {
                let config = LevenbergMarquardtConfig::new()
                    .with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
                    .with_max_iterations(8)
                    .with_cost_tolerance(0.0)
                    .with_parameter_tolerance(0.0)
                    .with_gradient_tolerance(0.0);
                let result = LevenbergMarquardt::with_config(config).optimize(&mut problem);
                black_box(result.map(|r| r.final_cost))
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

/// `ExplicitDenseSchur` on a small BA-like problem — dense mode's target is
/// < ~500 DOF, so this uses far fewer groups than the sparse PCG benches.
fn bench_explicit_dense_schur(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");
    let (jacobian, residuals, variables, index_map, landmark_keys) = ba_like_system_dense(15);
    let damping = Damping::new(1e-3, 1e-6, 1e32).unwrap_or_else(|e| panic!("valid bounds: {e:?}"));

    let mut solver = ExplicitDenseSchur::new();
    solver
        .initialize_structure(&variables, &index_map, &landmark_keys)
        .unwrap_or_else(|e| panic!("structure init: {e:?}"));

    group.bench_function("explicit_dense_schur_15_groups", |b| {
        b.iter(|| {
            let dx = LinearSolver::<DenseMode>::solve_augmented_equation(
                &mut solver,
                black_box(&residuals),
                black_box(&jacobian),
                black_box(&damping),
            )
            .unwrap_or_else(|e| panic!("solve failed: {e:?}"));
            black_box(dx.nrows())
        })
    });
    group.finish();
}

/// The shared PCG primitive on its own, against a plain dense SPD system —
/// decoupled from any Schur-specific operator or preconditioner, so a
/// regression here can only be the loop itself, not the caller's setup.
fn bench_pcg_primitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");
    let n = 200;
    // Diagonally dominant SPD system.
    let a = Mat::from_fn(n, n, |i, j| {
        if i == j {
            10.0
        } else {
            ((i + j) % 5) as f64 * 0.05
        }
    });
    let b = Mat::from_fn(n, 1, |i, _| (i % 7) as f64 * 0.1 + 1.0);
    let inv_diag: Vec<f64> = (0..n).map(|i| 1.0 / a[(i, i)]).collect();

    group.bench_function("shared_pcg_dense_spd_200", |b_| {
        b_.iter(|| {
            let result = solve_pcg(
                black_box(&b),
                &PcgParams::new(200, 1e-9),
                |p, ap| {
                    for i in 0..n {
                        let mut acc = 0.0;
                        for j in 0..n {
                            acc += a[(i, j)] * p[(j, 0)];
                        }
                        ap[(i, 0)] += acc;
                    }
                },
                |r, z| {
                    for i in 0..n {
                        z[(i, 0)] = inv_diag[i] * r[(i, 0)];
                    }
                },
            );
            black_box(result.iterations)
        })
    });
    group.finish();
}

/// A 2-variable, 6-row synthetic factor: residual reads both parameters,
/// Jacobian is dense ones — enough work to exercise the write path.
struct PairFactor;

impl apex_solver::Factor for PairFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        for (i, r) in residual.iter_mut().enumerate() {
            *r = params[i / 3][i % 3] * 0.5;
        }
        if let Some(mut jac) = jacobian {
            for col in 0..6 {
                for row in 0..6 {
                    *jac.rb_mut().get_mut(row, col) = 1.0;
                }
            }
        }
    }
    fn residual_dim(&self) -> usize {
        6
    }
    fn jacobian_shape(&self) -> (usize, usize) {
        (6, 6)
    }
}

fn bench_assemble(c: &mut Criterion) {
    let mut group = c.benchmark_group("assemble");

    // Chain graph: 5 001 variables, 5 000 factors, 6-row residuals.
    let n_vars = 5_001;
    let mut problem = Problem::new(apex_solver::linalg::JacobianMode::Sparse);
    let mut keys: Vec<apex_solver::core::VarKey> = Vec::with_capacity(n_vars);
    for i in 0..n_vars {
        keys.push(problem.add_variable(ManifoldType::RN, nalgebra::dvector![i as f64, 0.5, -0.25]));
    }
    for i in 0..n_vars - 1 {
        problem.add_residual_block(&[keys[i], keys[i + 1]], Box::new(PairFactor), None);
    }

    // Production initialization path: variables, index map, symbolic
    // structure and the assembly workspace come back ready for iteration.
    let mut state = apex_solver::optimizer::initialize_optimization_state(&mut problem)
        .unwrap_or_else(|e| panic!("optimization state: {e:?}"));
    let symbolic = match state.symbolic_structure.as_ref() {
        Some(symbolic) => symbolic,
        None => panic!("sparse mode must produce a symbolic structure"),
    };

    group.bench_function("chain_5k_factors", |b| {
        b.iter(|| {
            let (res, jac) = assemble_sparse(
                black_box(&problem),
                black_box(&state.variables),
                &state.variable_index_map,
                symbolic,
                &mut state.workspace,
            )
            .unwrap_or_else(|e| panic!("assemble failed: {e:?}"));
            black_box((res.nrows(), jac.compute_nnz()))
        })
    });
    group.finish();
}

/// Pattern fingerprinting: the O(nnz) integer hash behind every sparse-cache
/// `matches()` check (normal equations, chunk layout, diagonality verdict,
/// visibility index). It runs once per check against O(nnz) floating-point
/// assembly plus factorization, so it must stay in the noise — this bench
/// pins that. The window case is the size a keyframe solve checks on every
/// iteration.
fn bench_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint");

    // BA-like Jacobian: 40 groups × (2 cams × 3 landmarks × 6 rows).
    let (jacobian, _, _, _, _) = ba_like_system(40);
    let nnz = jacobian.compute_nnz();
    group.bench_function(format!("ba_40_groups_{nnz}_nnz"), |b| {
        b.iter(|| black_box(PatternFingerprint::of(black_box(&jacobian))))
    });

    // Chain Jacobian from the assemble bench (5 001 variables).
    let mut problem = Problem::new(apex_solver::linalg::JacobianMode::Sparse);
    let mut keys: Vec<apex_solver::core::VarKey> = Vec::new();
    for i in 0..5_001 {
        keys.push(problem.add_variable(ManifoldType::RN, nalgebra::dvector![i as f64, 0.5, -0.25]));
    }
    for i in 0..5_000 {
        problem.add_residual_block(&[keys[i], keys[i + 1]], Box::new(PairFactor), None);
    }
    let mut state = apex_solver::optimizer::initialize_optimization_state(&mut problem)
        .unwrap_or_else(|e| panic!("optimization state: {e:?}"));
    let symbolic = match state.symbolic_structure.as_ref() {
        Some(symbolic) => symbolic,
        None => panic!("sparse mode must produce a symbolic structure"),
    };
    let (_, chain_jacobian) = assemble_sparse(
        &problem,
        &state.variables,
        &state.variable_index_map,
        symbolic,
        &mut state.workspace,
    )
    .unwrap_or_else(|e| panic!("assemble failed: {e:?}"));
    let chain_nnz = chain_jacobian.compute_nnz();
    group.bench_function(format!("chain_5k_{chain_nnz}_nnz"), |b| {
        b.iter(|| black_box(PatternFingerprint::of(black_box(&chain_jacobian))))
    });

    // Visual-inertial window Jacobian: the one checked on every iteration of a
    // keyframe solve.
    let (window_jacobian, _, _, _, _) = window_like_system(11, 560);
    let window_nnz = window_jacobian.compute_nnz();
    group.bench_function(
        format!("window_11_states_560_landmarks_{window_nnz}_nnz"),
        |b| b.iter(|| black_box(PatternFingerprint::of(black_box(&window_jacobian)))),
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_normal_equations,
    bench_cholesky,
    bench_qr,
    bench_pcg_iterative_schur,
    bench_pcg_explicit_schur,
    bench_explicit_sparse_schur,
    bench_explicit_sparse_schur_window,
    bench_explicit_dense_schur,
    bench_pcg_primitive,
    bench_assemble,
    bench_fingerprint
);
criterion_main!(benches);
