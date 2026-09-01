//! Schur elimination must be correct for *any* variable partition, not just
//! "6-DOF cameras kept, 3-DOF points eliminated".
//!
//! The check used throughout is the strongest one available: eliminating a
//! variable is an algebraic reordering of the *same* linear system, so the
//! Schur solver and a plain sparse Cholesky on the full system must return the
//! same step. If elimination were mis-indexing blocks — the failure mode when
//! sizes or layouts are assumed — the two would diverge.
//!
//! Covered here, none of which the pre-generalization solver could express:
//!
//! - 1-DOF eliminated blocks (inverse-depth parameterization)
//! - 6-DOF eliminated blocks (sliding-window pose marginalization)
//! - mixed sizes eliminated in one solve (depths and points together)
//! - eliminated variables interleaved between retained ones, so neither side
//!   occupies a contiguous column range
//! - eliminating variables coupled to each other, which is *invalid* and must
//!   be reported rather than silently producing a wrong step

use apex_solver::core::VarKey;
use apex_solver::core::variable::{ManifoldVariable, Variable};
use apex_solver::linalg::{
    LinearSolver, SparseCholeskySolver, SparseMode, SparseSchurComplementSolver, StructureAware,
};
use apex_manifolds::rn::Rn;
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::DVector;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashSet;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A synthetic least-squares problem with a chosen variable layout.
struct System {
    variables: SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    index_map: SecondaryMap<VarKey, usize>,
    jacobian: SparseColMat<usize, f64>,
    residuals: Mat<f64>,
    keys: Vec<VarKey>,
}

/// Build a problem whose variables have the given DOFs, in the given order.
///
/// `couplings` lists `(variable_a, variable_b)` pairs that share a factor. Each
/// coupling contributes a dense row block over both variables' columns, and
/// every column additionally gets a prior row so `JᵀJ` is positive definite and
/// the comparison is well posed.
///
/// Values are deterministic and irrational-ish so that an indexing mistake
/// cannot coincidentally cancel.
fn build_system(dofs: &[usize], couplings: &[(usize, usize)]) -> Result<System, Box<dyn std::error::Error>> {
    let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
    let mut index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
    let mut keys = Vec::new();
    let mut col_starts = Vec::new();

    let mut col = 0usize;
    for &dof in dofs {
        let key = variables.insert(Box::new(Variable::new(Rn::new(DVector::zeros(dof)))));
        index_map.insert(key, col);
        keys.push(key);
        col_starts.push(col);
        col += dof;
    }
    let total_cols = col;

    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut row = 0usize;

    // Coupling rows: each pair shares one dense block of rows.
    for (pair_idx, &(a, b)) in couplings.iter().enumerate() {
        let rows_here = dofs[a].max(dofs[b]);
        for r in 0..rows_here {
            for k in 0..dofs[a] {
                let v = 1.0 + ((pair_idx + r + k) % 7) as f64 * 0.37;
                triplets.push(Triplet::new(row + r, col_starts[a] + k, v));
            }
            for k in 0..dofs[b] {
                let v = 0.5 + ((pair_idx * 3 + r + k) % 5) as f64 * 0.29;
                triplets.push(Triplet::new(row + r, col_starts[b] + k, v));
            }
        }
        row += rows_here;
    }

    // Prior rows keep JᵀJ non-singular without coupling anything.
    for c in 0..total_cols {
        triplets.push(Triplet::new(row + c, c, 0.9 + (c % 3) as f64 * 0.15));
    }
    row += total_cols;

    let jacobian = SparseColMat::try_new_from_triplets(row, total_cols, &triplets)?;
    let residuals = Mat::from_fn(row, 1, |i, _| 0.1 + (i % 11) as f64 * 0.07);

    Ok(System {
        variables,
        index_map,
        jacobian,
        residuals,
        keys,
    })
}

/// Solve with sparse Cholesky on the full system, and with Schur eliminating
/// `eliminate`. Returns both steps for comparison.
fn solve_both(
    system: &System,
    eliminate: &HashSet<VarKey>,
) -> Result<(Mat<f64>, Mat<f64>), Box<dyn std::error::Error>> {
    let mut cholesky = SparseCholeskySolver::new();
    let dx_reference = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut cholesky,
        &system.residuals,
        &system.jacobian,
    )?;

    let mut schur = SparseSchurComplementSolver::new();
    schur.initialize_structure(&system.variables, &system.index_map, eliminate)?;
    let dx_schur = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut schur,
        &system.residuals,
        &system.jacobian,
    )?;

    Ok((dx_reference, dx_schur))
}

fn assert_steps_agree(reference: &Mat<f64>, schur: &Mat<f64>, what: &str) {
    assert_eq!(
        reference.nrows(),
        schur.nrows(),
        "{what}: step length differs"
    );
    let scale = reference.norm_l2().max(1.0);
    for i in 0..reference.nrows() {
        let diff = (reference[(i, 0)] - schur[(i, 0)]).abs();
        assert!(
            diff / scale < 1e-9,
            "{what}: component {i} differs — cholesky {}, schur {} (rel {:.3e})",
            reference[(i, 0)],
            schur[(i, 0)],
            diff / scale
        );
    }
}

/// Control: the classic bundle-adjustment shape the solver was written for.
#[test]
fn schur_matches_cholesky_for_classic_bundle_adjustment_shape() -> TestResult {
    // two 6-DOF poses kept, three 3-DOF points eliminated
    let system = build_system(
        &[6, 6, 3, 3, 3],
        &[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
    )?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3], system.keys[4]]
        .into_iter()
        .collect();

    let (reference, schur) = solve_both(&system, &eliminate)?;
    assert_steps_agree(&reference, &schur, "classic BA (3-DOF points)");
    Ok(())
}

/// Inverse-depth parameterization: each landmark is a single scalar.
#[test]
fn schur_matches_cholesky_for_one_dof_inverse_depth() -> TestResult {
    let system = build_system(
        &[6, 6, 1, 1, 1, 1],
        &[
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
        ],
    )?;
    let eliminate: HashSet<VarKey> = system.keys[2..].iter().copied().collect();

    let (reference, schur) = solve_both(&system, &eliminate)?;
    assert_steps_agree(&reference, &schur, "1-DOF inverse depth");
    Ok(())
}

/// Sliding-window marginalization eliminates whole 6-DOF poses.
#[test]
fn schur_matches_cholesky_for_six_dof_marginalization() -> TestResult {
    // Keep two poses, marginalize two others. The eliminated poses touch only
    // retained ones, never each other.
    let system = build_system(&[6, 6, 6, 6], &[(0, 2), (1, 2), (0, 3), (1, 3)])?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3]].into_iter().collect();

    let (reference, schur) = solve_both(&system, &eliminate)?;
    assert_steps_agree(&reference, &schur, "6-DOF marginalization");
    Ok(())
}

/// Depths, points and a larger feature eliminated together in one solve.
#[test]
fn schur_matches_cholesky_for_mixed_eliminated_sizes() -> TestResult {
    // kept: 6-DOF pose, 3-DOF intrinsics.  eliminated: 1, 3 and 4 DOF.
    let system = build_system(
        &[6, 3, 1, 3, 4],
        &[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
    )?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3], system.keys[4]]
        .into_iter()
        .collect();

    let (reference, schur) = solve_both(&system, &eliminate)?;
    assert_steps_agree(&reference, &schur, "mixed 1/3/4-DOF elimination");
    Ok(())
}

/// The eliminated variables sit *between* retained ones in column order, so
/// neither side is a contiguous range. This is the layout marginalization
/// produces, and the pre-generalization solver could not represent it.
#[test]
fn schur_matches_cholesky_for_non_contiguous_partition() -> TestResult {
    // columns: kept(6) | elim(3) | kept(6) | elim(3) | kept(6)
    let system = build_system(
        &[6, 3, 6, 3, 6],
        &[(0, 1), (2, 1), (2, 3), (4, 3), (0, 3), (4, 1)],
    )?;
    let eliminate: HashSet<VarKey> = [system.keys[1], system.keys[3]].into_iter().collect();

    let (reference, schur) = solve_both(&system, &eliminate)?;
    assert_steps_agree(&reference, &schur, "non-contiguous partition");
    Ok(())
}

/// A single retained variable with many eliminated ones — the degenerate end of
/// the range, where the reduced system is as small as it gets.
#[test]
fn schur_matches_cholesky_for_single_retained_variable() -> TestResult {
    let system = build_system(&[6, 1, 1, 3], &[(0, 1), (0, 2), (0, 3)])?;
    let eliminate: HashSet<VarKey> = system.keys[1..].iter().copied().collect();

    let (reference, schur) = solve_both(&system, &eliminate)?;
    assert_steps_agree(&reference, &schur, "single retained variable");
    Ok(())
}

/// Eliminating two variables that share a factor makes `H_ee` non
/// block-diagonal. Inverting it blockwise would then be wrong, so this must be
/// an error rather than a plausible-looking answer.
#[test]
fn coupled_eliminated_variables_are_rejected() -> TestResult {
    // keys[1] and keys[2] are coupled directly to each other.
    let system = build_system(&[6, 3, 3], &[(0, 1), (0, 2), (1, 2)])?;
    let eliminate: HashSet<VarKey> = [system.keys[1], system.keys[2]].into_iter().collect();

    let mut schur = SparseSchurComplementSolver::new();
    schur.initialize_structure(&system.variables, &system.index_map, &eliminate)?;
    let result = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut schur,
        &system.residuals,
        &system.jacobian,
    );

    let Err(err) = result else {
        panic!("eliminating two coupled variables must be rejected, not silently solved");
    };
    let message = err.to_string();
    assert!(
        message.contains("block-diagonal"),
        "error should explain the precondition, got: {message}"
    );
    Ok(())
}

/// The same problem, eliminated two different ways, must give the same answer.
///
/// This is the sharpest statement of what generality means: the partition is a
/// solver strategy, not part of the problem definition.
#[test]
fn different_elimination_choices_agree() -> TestResult {
    let system = build_system(&[6, 3, 3, 1], &[(0, 1), (0, 2), (0, 3), (1, 0), (2, 0)])?;

    let eliminate_a: HashSet<VarKey> = [system.keys[1], system.keys[2]].into_iter().collect();
    let eliminate_b: HashSet<VarKey> = [system.keys[3]].into_iter().collect();
    let eliminate_c: HashSet<VarKey> = [system.keys[1], system.keys[2], system.keys[3]]
        .into_iter()
        .collect();

    let (reference, schur_a) = solve_both(&system, &eliminate_a)?;
    assert_steps_agree(&reference, &schur_a, "eliminate {points}");

    let (_, schur_b) = solve_both(&system, &eliminate_b)?;
    assert_steps_agree(&reference, &schur_b, "eliminate {depth}");

    let (_, schur_c) = solve_both(&system, &eliminate_c)?;
    assert_steps_agree(&reference, &schur_c, "eliminate {points, depth}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Matrix-free (implicit) Schur
// ---------------------------------------------------------------------------

/// The matrix-free solver never forms `S`; it applies the Schur operator
/// through `H_ke`/`H_ee⁻¹` products inside PCG. It must still land on the same
/// step as a direct factorization of the full system.
///
/// PCG is iterative, so the tolerance here is looser than the direct
/// comparisons above — but far tighter than any indexing mistake would survive.
#[test]
fn matrix_free_schur_matches_cholesky_on_bundle_adjustment_shape() -> TestResult {
    use apex_solver::linalg::IterativeSchurSolver;

    // Two 6-DOF poses kept, three 3-DOF points eliminated: the shape the
    // matrix-free path was written for.
    let system = build_system(
        &[6, 6, 3, 3, 3],
        &[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
    )?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3], system.keys[4]]
        .into_iter()
        .collect();

    let mut cholesky = SparseCholeskySolver::new();
    let reference = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut cholesky,
        &system.residuals,
        &system.jacobian,
    )?;

    let mut implicit = IterativeSchurSolver::with_cg_params(500, 1e-12);
    implicit.initialize_structure(&system.variables, &system.index_map, &eliminate)?;
    let step = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut implicit,
        &system.residuals,
        &system.jacobian,
    )?;

    assert_eq!(reference.nrows(), step.nrows());
    let scale = reference.norm_l2().max(1.0);
    for i in 0..reference.nrows() {
        let diff = (reference[(i, 0)] - step[(i, 0)]).abs();
        assert!(
            diff / scale < 1e-6,
            "matrix-free component {i}: cholesky {}, implicit {} (rel {:.3e})",
            reference[(i, 0)],
            step[(i, 0)],
            diff / scale
        );
    }
    Ok(())
}

/// Build a system whose rows are already grouped by eliminated variable, the
/// layout `Problem::group_rows_for_elimination` produces.
///
/// Rows are emitted as: every prior on a *retained* column first, then, per
/// eliminated variable, all of its coupling rows followed by its own prior
/// rows. That keeps each eliminated variable's rows in one contiguous range,
/// which chunk-wise elimination requires.
fn build_grouped_system(
    dofs: &[usize],
    eliminated: &[usize],
    couplings: &[(usize, usize)],
) -> Result<System, Box<dyn std::error::Error>> {
    let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
    let mut index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
    let mut keys = Vec::new();
    let mut col_starts = Vec::new();

    let mut col = 0usize;
    for &dof in dofs {
        let key = variables.insert(Box::new(Variable::new(Rn::new(DVector::zeros(dof)))));
        index_map.insert(key, col);
        keys.push(key);
        col_starts.push(col);
        col += dof;
    }

    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut row = 0usize;

    let emit_coupling = |row: &mut usize,
                             triplets: &mut Vec<Triplet<usize, usize, f64>>,
                             pair_idx: usize,
                             a: usize,
                             b: usize| {
        let rows_here = dofs[a].max(dofs[b]);
        for r in 0..rows_here {
            for k in 0..dofs[a] {
                let v = 1.0 + ((pair_idx + r + k) % 7) as f64 * 0.37;
                triplets.push(Triplet::new(*row + r, col_starts[a] + k, v));
            }
            for k in 0..dofs[b] {
                let v = 0.5 + ((pair_idx * 3 + r + k) % 5) as f64 * 0.29;
                triplets.push(Triplet::new(*row + r, col_starts[b] + k, v));
            }
        }
        *row += rows_here;
    };

    // Priors on retained columns, ahead of every chunk.
    for (v, &dof) in dofs.iter().enumerate() {
        if eliminated.contains(&v) {
            continue;
        }
        for k in 0..dof {
            triplets.push(Triplet::new(row, col_starts[v] + k, 0.9 + (k % 3) as f64 * 0.15));
            row += 1;
        }
    }

    // Per eliminated variable: its couplings, then its own priors — contiguous.
    for &e in eliminated {
        for (pair_idx, &(a, b)) in couplings.iter().enumerate() {
            if a == e || b == e {
                emit_coupling(&mut row, &mut triplets, pair_idx, a, b);
            }
        }
        for k in 0..dofs[e] {
            triplets.push(Triplet::new(row, col_starts[e] + k, 0.9 + (k % 3) as f64 * 0.15));
            row += 1;
        }
    }

    let jacobian = SparseColMat::try_new_from_triplets(row, col, &triplets)?;
    let residuals = Mat::from_fn(row, 1, |i, _| 0.1 + (i % 11) as f64 * 0.07);

    Ok(System {
        variables,
        index_map,
        jacobian,
        residuals,
        keys,
    })
}

/// Chunk-wise elimination must give the same step as a direct factorization.
///
/// It forms the reduced system straight from `J`, never materializing `JᵀJ`, so
/// this is the check that the reordered accumulation is still the same algebra.
///
/// The fixture's rows are already grouped by eliminated variable, which is what
/// `Problem::group_rows_for_elimination` arranges for real problems.
#[test]
fn chunked_schur_matches_cholesky() -> TestResult {
    use apex_solver::linalg::SchurVariant;

    // rows are emitted coupling-by-coupling, so listing each eliminated
    // variable's couplings together makes its rows contiguous.
    let system = build_grouped_system(&[6, 6, 3, 3], &[2, 3], &[(0, 2), (1, 2), (0, 3), (1, 3)])?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3]].into_iter().collect();

    let mut cholesky = SparseCholeskySolver::new();
    let reference = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut cholesky,
        &system.residuals,
        &system.jacobian,
    )?;

    let mut chunked = SparseSchurComplementSolver::new().with_variant(SchurVariant::ChunkedSparse);
    chunked.initialize_structure(&system.variables, &system.index_map, &eliminate)?;
    let step = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut chunked,
        &system.residuals,
        &system.jacobian,
    )?;

    assert_steps_agree(&reference, &step, "chunked elimination");
    Ok(())
}

/// The chunked path must serve the quadratic model without ever holding `JᵀJ`.
#[test]
fn chunked_schur_serves_hessian_action_without_the_matrix() -> TestResult {
    use apex_solver::linalg::SchurVariant;

    let system = build_grouped_system(&[6, 6, 3, 3], &[2, 3], &[(0, 2), (1, 2), (0, 3), (1, 3)])?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3]].into_iter().collect();

    let mut chunked = SparseSchurComplementSolver::new().with_variant(SchurVariant::ChunkedSparse);
    chunked.initialize_structure(&system.variables, &system.index_map, &eliminate)?;
    let step = LinearSolver::<SparseMode>::solve_normal_equation(
        &mut chunked,
        &system.residuals,
        &system.jacobian,
    )?;

    // No matrix...
    assert!(
        LinearSolver::<SparseMode>::get_hessian(&chunked).is_none(),
        "the chunked path must not materialize JtJ"
    );

    // ...but the action must still be right. Compare against the solver that
    // does hold JtJ.
    let mut cholesky = SparseCholeskySolver::new();
    LinearSolver::<SparseMode>::solve_normal_equation(
        &mut cholesky,
        &system.residuals,
        &system.jacobian,
    )?;
    let want = LinearSolver::<SparseMode>::hessian_vec_product(&cholesky, &step)
        .ok_or("cholesky must provide H*v")?;
    let got = LinearSolver::<SparseMode>::hessian_vec_product(&chunked, &step)
        .ok_or("chunked must provide H*v")?;

    let scale = want.norm_l2().max(1.0);
    for i in 0..want.nrows() {
        assert!(
            (want[(i, 0)] - got[(i, 0)]).abs() / scale < 1e-10,
            "H*v[{i}]: cholesky {}, chunked {}",
            want[(i, 0)],
            got[(i, 0)]
        );
    }
    Ok(())
}

/// The *damped* chunked solve must match the damped direct solve.
///
/// Regression: damping was first applied to `S` rather than to `H_kk`. Since
/// `D_jj = clamp(H_jj, …)` reads the diagonal it is applied to, damping after
/// elimination clamps against the wrong matrix. The undamped tests above could
/// not see it — only a solve with `λ > 0` can.
#[test]
fn chunked_schur_matches_cholesky_when_damped() -> TestResult {
    use apex_solver::linalg::{Damping, SchurVariant};

    let system = build_grouped_system(&[6, 6, 3, 3], &[2, 3], &[(0, 2), (1, 2), (0, 3), (1, 3)])?;
    let eliminate: HashSet<VarKey> = [system.keys[2], system.keys[3]].into_iter().collect();
    let damping = Damping::new(1e-2, 1e-6, 1e32)?;

    let mut cholesky = SparseCholeskySolver::new();
    let reference = LinearSolver::<SparseMode>::solve_augmented_equation(
        &mut cholesky,
        &system.residuals,
        &system.jacobian,
        &damping,
    )?;

    let mut chunked = SparseSchurComplementSolver::new().with_variant(SchurVariant::ChunkedSparse);
    chunked.initialize_structure(&system.variables, &system.index_map, &eliminate)?;
    let step = LinearSolver::<SparseMode>::solve_augmented_equation(
        &mut chunked,
        &system.residuals,
        &system.jacobian,
        &damping,
    )?;

    assert_steps_agree(&reference, &step, "chunked elimination, damped");
    Ok(())
}
