//! Criterion micro-benchmarks for the per-iteration hot kernels:
//!
//! - [`normal_eq`]: parallel formation of `H = JᵀJ` and `g = Jᵀr`
//! - [`cholesky`]/[`qr`]: damped sparse solve `(JᵀJ + λD)·dx = −Jᵀr`
//! - [`pcg`]: iterative Schur (matrix-free PCG with Schur–Jacobi preconditioner)
//! - [`assemble`]: residual/Jacobian assembly for a synthetic factor graph
//!
//! ```bash
//! cargo bench --bench micro_kernels
//! ```

use apex_manifolds::rn::Rn;
use apex_manifolds::{LieGroup, se3::SE3};
use apex_solver::ManifoldType;
use apex_solver::core::problem::Problem;
use apex_solver::core::variable::Variable;
use apex_solver::linalg::sparse::normal_eq::NormalEquationsCache;
use apex_solver::linalg::sparse::qr::SparseQRSolver;
use apex_solver::linalg::sparse::{IterativeSchurSolver, SparseCholeskySolver};
use apex_solver::linalg::{Damping, LinearSolver, StructureAware};
use apex_solver::linearizer::cpu::sparse::assemble_sparse;
use criterion::{Criterion, criterion_group, criterion_main};
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

    let mut solver = IterativeSchurSolver::new();
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

criterion_group!(
    benches,
    bench_normal_equations,
    bench_cholesky,
    bench_qr,
    bench_pcg_iterative_schur,
    bench_assemble
);
criterion_main!(benches);
