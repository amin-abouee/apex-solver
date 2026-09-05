//! Cross-cutting mathematical properties every Schur complement solver must
//! share, checked across all three: `ExplicitSparseSchur`, `ExplicitDenseSchur`
//! and `ImplicitSparseSchur`.
//!
//! `schur_generalization.rs` already proves each solver's step matches plain
//! Cholesky individually. This file adds two properties that only show up
//! when the three are compared *side by side* on the same system:
//!
//! - **`S` is SPD**: a successful Cholesky factorization of the reduced
//!   system is exactly the statement that `S` is symmetric positive
//!   definite, so a solver that returns `Ok` has already proven it.
//! - **Explicit-sparse, explicit-dense and implicit-sparse agree on the same
//!   step**, including on non-contiguous, mixed-DOF partitions — the exact
//!   generalization this consolidation added.

use apex_manifolds::rn::Rn;
use apex_solver::core::VarKey;
use apex_solver::core::variable::{ManifoldVariable, Variable};
use apex_solver::linalg::{
    ExplicitDenseSchur, ExplicitSparseSchur, ImplicitSparseSchur, LinearSolver, StructureAware,
};
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::DVector;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashSet;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A synthetic system built once as a dense Jacobian and once as the
/// structurally-identical sparse Jacobian, so every `LinearizationMode` can
/// solve the "same" problem.
struct System {
    variables: SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    index_map: SecondaryMap<VarKey, usize>,
    landmark_keys: HashSet<VarKey>,
    dense_jacobian: Mat<f64>,
    sparse_jacobian: SparseColMat<usize, f64>,
    residuals: Mat<f64>,
    total_dof: usize,
}

/// Build a system with the given kept/eliminated DOF layout (in column
/// order) and full bipartite coupling between every kept and every
/// eliminated block — deliberately non-contiguous when `dofs` interleaves
/// `is_eliminated` values, and mixed-DOF when the eliminated sizes differ.
///
/// Coefficients vary by index so the coupling structure is not degenerate
/// (see `ExplicitDenseSchur`'s own test fixture for why a uniform
/// coefficient would make plain Cholesky fail on a rank-deficient `S`).
fn build_system(dofs: &[(usize, bool)]) -> Result<System, Box<dyn std::error::Error>> {
    let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
    let mut index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
    let mut landmark_keys = HashSet::new();
    let mut col_starts = Vec::new();
    let mut col = 0usize;
    for &(dof, _) in dofs {
        col_starts.push(col);
        col += dof;
    }
    let total_dof = col;

    let kept_idx: Vec<usize> = dofs
        .iter()
        .enumerate()
        .filter(|&(_, &(_, elim))| !elim)
        .map(|(i, _)| i)
        .collect();
    let elim_idx: Vec<usize> = dofs
        .iter()
        .enumerate()
        .filter(|&(_, &(_, elim))| elim)
        .map(|(i, _)| i)
        .collect();

    for (i, &(dof, is_elim)) in dofs.iter().enumerate() {
        let key = variables.insert(Box::new(Variable::new(Rn::new(DVector::zeros(dof)))));
        index_map.insert(key, col_starts[i]);
        if is_elim {
            landmark_keys.insert(key);
        }
    }

    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();

    // Full bipartite coupling: one row block per (kept, eliminated) pair, so
    // every eliminated block is observed by every kept block.
    let pairs: Vec<(usize, usize)> = kept_idx
        .iter()
        .flat_map(|&k| elim_idx.iter().map(move |&e| (k, e)))
        .collect();
    for (pair_idx, &(ki, ei)) in pairs.iter().enumerate() {
        let (k_dof, _) = dofs[ki];
        let (e_dof, _) = dofs[ei];
        let rows_here = k_dof.max(e_dof).max(1);
        for r in 0..rows_here {
            let mut row = vec![0.0f64; total_dof];
            for c in 0..k_dof {
                let v = 0.8 + ((pair_idx + r + c) % 5) as f64 * 0.31;
                row[col_starts[ki] + c] = v;
            }
            for c in 0..e_dof {
                let v = 0.4 + ((pair_idx * 3 + r + c) % 4) as f64 * 0.27;
                row[col_starts[ei] + c] = v;
            }
            rows.push(row);
        }
    }
    // Prior rows so every column has independent curvature (SPD guarantee).
    for c in 0..total_dof {
        let mut row = vec![0.0f64; total_dof];
        row[c] = 1.1 + (c % 3) as f64 * 0.2;
        rows.push(row);
    }

    let dense_jacobian = Mat::from_fn(rows.len(), total_dof, |r, c| rows[r][c]);
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                triplets.push(Triplet::new(r, c, v));
            }
        }
    }
    let sparse_jacobian = SparseColMat::try_new_from_triplets(rows.len(), total_dof, &triplets)?;
    let residuals = Mat::from_fn(rows.len(), 1, |i, _| 0.15 + (i % 7) as f64 * 0.09);

    Ok(System {
        variables,
        index_map,
        landmark_keys,
        dense_jacobian,
        sparse_jacobian,
        residuals,
        total_dof,
    })
}

/// `ExplicitSparseSchur`'s default (`Sparse`) variant returns `Ok` only if
/// its Cholesky factorization of `S` succeeded — i.e. only if `S` is SPD.
fn assert_explicit_sparse_solves(system: &System) -> TestResult {
    let mut solver = ExplicitSparseSchur::new();
    solver.initialize_structure(&system.variables, &system.index_map, &system.landmark_keys)?;
    apex_solver::linalg::LinearSolver::<apex_solver::linalg::SparseMode>::solve_normal_equation(
        &mut solver,
        &system.residuals,
        &system.sparse_jacobian,
    )?;
    Ok(())
}

fn assert_explicit_dense_solves(system: &System) -> TestResult {
    let mut solver = ExplicitDenseSchur::new();
    solver.initialize_structure(&system.variables, &system.index_map, &system.landmark_keys)?;
    apex_solver::linalg::LinearSolver::<apex_solver::linalg::DenseMode>::solve_normal_equation(
        &mut solver,
        &system.residuals,
        &system.dense_jacobian,
    )?;
    Ok(())
}

/// `S` is SPD on a classic contiguous BA-shaped partition, proven by every
/// explicit solver's Cholesky succeeding.
#[test]
fn s_is_spd_for_classic_contiguous_partition() -> TestResult {
    let system = build_system(&[(6, false), (6, false), (3, true), (3, true), (3, true)])?;
    assert_explicit_sparse_solves(&system)?;
    assert_explicit_dense_solves(&system)
}

/// `S` is SPD even when the eliminated blocks interleave with the retained
/// ones and carry mixed DOF (1-DOF inverse depth + 3-DOF point) — the
/// generalization this consolidation added.
#[test]
fn s_is_spd_for_non_contiguous_mixed_dof_partition() -> TestResult {
    // kept(6) elim(1) kept(6) elim(3) elim(1)
    let system = build_system(&[(6, false), (1, true), (6, false), (3, true), (1, true)])?;
    assert_explicit_sparse_solves(&system)?;
    assert_explicit_dense_solves(&system)
}

/// All three Schur solvers must land on the same step for the classic
/// contiguous shape.
#[test]
fn explicit_sparse_dense_and_implicit_agree_on_classic_shape() -> TestResult {
    let system = build_system(&[(6, false), (6, false), (3, true), (3, true), (3, true)])?;
    assert_three_way_agreement(&system, 1e-6)
}

/// All three Schur solvers must land on the same step for a non-contiguous,
/// mixed-DOF partition — `ImplicitSparseSchur` could not even express this
/// before it moved onto `SchurPartition`.
#[test]
fn explicit_sparse_dense_and_implicit_agree_on_non_contiguous_mixed_dof_partition() -> TestResult {
    let system = build_system(&[(6, false), (1, true), (6, false), (3, true), (1, true)])?;
    // PCG is iterative, so the tolerance is looser than the direct-solver
    // comparisons — matching `schur_generalization.rs`'s convention for any
    // comparison involving `ImplicitSparseSchur`/`ExplicitSchurVariant::Iterative`.
    assert_three_way_agreement(&system, 1e-4)
}

fn assert_three_way_agreement(system: &System, tolerance: f64) -> TestResult {
    use apex_solver::linalg::{DenseMode, SparseMode};

    let mut explicit_sparse = ExplicitSparseSchur::new();
    explicit_sparse.initialize_structure(
        &system.variables,
        &system.index_map,
        &system.landmark_keys,
    )?;
    let sparse_step = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut explicit_sparse,
        &system.residuals,
        &system.sparse_jacobian,
    )?;

    let mut explicit_dense = ExplicitDenseSchur::new();
    explicit_dense.initialize_structure(
        &system.variables,
        &system.index_map,
        &system.landmark_keys,
    )?;
    let dense_step = LinearSolver::<DenseMode>::solve_normal_equation(
        &mut explicit_dense,
        &system.residuals,
        &system.dense_jacobian,
    )?;

    // This file's premise is that all three solvers agree on the *same* step,
    // which needs the linear system solved exactly. The default forcing
    // sequence deliberately truncates it, so it is disabled here.
    let mut implicit = ImplicitSparseSchur::with_cg_params(1000, 1e-12).with_cg_q_tolerance(0.0);
    implicit.initialize_structure(&system.variables, &system.index_map, &system.landmark_keys)?;
    let implicit_step = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut implicit,
        &system.residuals,
        &system.sparse_jacobian,
    )?;

    assert_eq!(sparse_step.nrows(), system.total_dof);
    assert_eq!(dense_step.nrows(), system.total_dof);
    assert_eq!(implicit_step.nrows(), system.total_dof);

    let scale = sparse_step.norm_l2().max(1.0);
    for i in 0..system.total_dof {
        let d_diff = (sparse_step[(i, 0)] - dense_step[(i, 0)]).abs() / scale;
        assert!(
            d_diff < 1e-9,
            "explicit sparse vs dense diverge at {i}: {} vs {} (rel {d_diff:.3e})",
            sparse_step[(i, 0)],
            dense_step[(i, 0)]
        );
        let i_diff = (sparse_step[(i, 0)] - implicit_step[(i, 0)]).abs() / scale;
        assert!(
            i_diff < tolerance,
            "explicit sparse vs implicit diverge at {i}: {} vs {} (rel {i_diff:.3e})",
            sparse_step[(i, 0)],
            implicit_step[(i, 0)]
        );
    }
    Ok(())
}
