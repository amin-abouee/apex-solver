//! Contract tests for the [`LinearSolver`] trait, checked across every backend.
//!
//! `get_gradient` and `get_hessian` are documented to publish `+Jᵀr` and the
//! **un-damped** `JᵀJ`. Both are consumed by the optimizers to build the local
//! quadratic model, so a backend that quietly publishes `−Jᵀr` (or the augmented
//! Hessian) inverts every step-quality ratio ρ computed from it — the solve
//! itself still looks fine, which is exactly what makes the bug survive
//! per-backend testing. These tests pin the contract for all six backends at
//! once, so a new backend cannot get it wrong silently.

use apex_solver::linalg::{
    Damping, DenseCholeskySolver, DenseMode, DenseQRSolver, LinearSolver, SparseCholeskySolver,
    SparseMode, SparseQRSolver,
};
use faer::Mat;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type SparseSystem = Result<(SparseColMat<usize, f64>, Mat<f64>), Box<dyn std::error::Error>>;

/// A small overdetermined least-squares problem: 4 residuals, 3 parameters.
///
/// Deliberately asymmetric and full rank so that `Jᵀr` has no zero entries and a
/// sign flip cannot hide behind a symmetry.
fn dense_system() -> (Mat<f64>, Mat<f64>) {
    let j = Mat::from_fn(4, 3, |i, k| match (i, k) {
        (0, 0) => 1.0,
        (0, 1) => 2.0,
        (0, 2) => 0.0,
        (1, 0) => 0.0,
        (1, 1) => 1.0,
        (1, 2) => 3.0,
        (2, 0) => 4.0,
        (2, 1) => 0.0,
        (2, 2) => 1.0,
        (3, 0) => 1.0,
        (3, 1) => 1.0,
        (3, 2) => 1.0,
        _ => 0.0,
    });
    let r = Mat::from_fn(4, 1, |i, _| [0.5, -1.5, 2.0, -0.25][i]);
    (j, r)
}

fn sparse_system() -> SparseSystem {
    let (dense_j, r) = dense_system();
    let mut triplets = Vec::new();
    for i in 0..dense_j.nrows() {
        for k in 0..dense_j.ncols() {
            if dense_j[(i, k)] != 0.0 {
                triplets.push(Triplet::new(i, k, dense_j[(i, k)]));
            }
        }
    }
    Ok((
        SparseColMat::try_new_from_triplets(dense_j.nrows(), dense_j.ncols(), &triplets)?,
        r,
    ))
}

/// Reference values computed directly from the dense `J` and `r`.
fn expected_gradient_and_hessian() -> (Mat<f64>, Mat<f64>) {
    let (j, r) = dense_system();
    (j.transpose() * &r, j.transpose() * &j)
}

fn assert_publishes_contract(
    name: &str,
    gradient: &Mat<f64>,
    hessian_diagonal: &[f64],
) -> TestResult {
    let (expected_g, expected_h) = expected_gradient_and_hessian();

    for i in 0..expected_g.nrows() {
        let got = gradient[(i, 0)];
        let want = expected_g[(i, 0)];
        assert!(
            (got - want).abs() < 1e-9,
            "{name}: get_gradient() must publish +Jᵀr. Entry {i} is {got}, expected {want}\
             {}",
            if (got + want).abs() < 1e-9 {
                " — this is exactly -Jᵀr, i.e. the sign convention is inverted"
            } else {
                ""
            }
        );
    }

    for i in 0..expected_h.nrows() {
        let got = hessian_diagonal[i];
        let want = expected_h[(i, i)];
        assert!(
            (got - want).abs() < 1e-9,
            "{name}: get_hessian() must publish the un-damped JᵀJ. Diagonal entry \
             {i} is {got}, expected {want} — a larger value means the damping term \
             leaked into the published Hessian"
        );
    }
    Ok(())
}

fn sparse_diagonal(h: &SparseColMat<usize, f64>) -> Vec<f64> {
    (0..h.ncols())
        .map(|col| {
            let rows = h.symbolic().row_idx_of_col_raw(col);
            let vals = h.val_of_col(col);
            rows.iter()
                .position(|&r| r == col)
                .map(|k| vals[k])
                .unwrap_or(0.0)
        })
        .collect()
}

/// Damping strong enough that a leaked damping term cannot hide in the tolerance.
fn probe_damping() -> Damping {
    Damping::identity(10.0)
}

/// The same contract holds for the un-damped entry point, which Gauss-Newton
/// uses when `min_diagonal` is zero.
#[test]
fn sparse_backends_publish_the_contract_from_solve_normal_equation() -> TestResult {
    let (j, r) = sparse_system()?;

    let mut cholesky = SparseCholeskySolver::new();
    LinearSolver::<SparseMode>::solve_normal_equation(&mut cholesky, &r, &j)?;
    assert_publishes_contract(
        "SparseCholeskySolver::solve_normal_equation",
        LinearSolver::<SparseMode>::get_gradient(&cholesky).ok_or("no gradient")?,
        &sparse_diagonal(LinearSolver::<SparseMode>::get_hessian(&cholesky).ok_or("no hessian")?),
    )?;

    let mut qr = SparseQRSolver::new();
    LinearSolver::<SparseMode>::solve_normal_equation(&mut qr, &r, &j)?;
    assert_publishes_contract(
        "SparseQRSolver::solve_normal_equation",
        LinearSolver::<SparseMode>::get_gradient(&qr).ok_or("no gradient")?,
        &sparse_diagonal(LinearSolver::<SparseMode>::get_hessian(&qr).ok_or("no hessian")?),
    )
}

#[test]
fn dense_backends_publish_the_contract_from_solve_normal_equation() -> TestResult {
    let (j, r) = dense_system();

    let mut cholesky = DenseCholeskySolver::new();
    LinearSolver::<DenseMode>::solve_normal_equation(&mut cholesky, &r, &j)?;
    let h = LinearSolver::<DenseMode>::get_hessian(&cholesky).ok_or("no hessian")?;
    let diag: Vec<f64> = (0..h.ncols()).map(|i| h[(i, i)]).collect();
    assert_publishes_contract(
        "DenseCholeskySolver::solve_normal_equation",
        LinearSolver::<DenseMode>::get_gradient(&cholesky).ok_or("no gradient")?,
        &diag,
    )?;

    let mut qr = DenseQRSolver::new();
    LinearSolver::<DenseMode>::solve_normal_equation(&mut qr, &r, &j)?;
    let h = LinearSolver::<DenseMode>::get_hessian(&qr).ok_or("no hessian")?;
    let diag: Vec<f64> = (0..h.ncols()).map(|i| h[(i, i)]).collect();
    assert_publishes_contract(
        "DenseQRSolver::solve_normal_equation",
        LinearSolver::<DenseMode>::get_gradient(&qr).ok_or("no gradient")?,
        &diag,
    )
}

#[test]
fn sparse_cholesky_publishes_positive_gradient_and_undamped_hessian() -> TestResult {
    let (j, r) = sparse_system()?;
    let mut solver = SparseCholeskySolver::new();
    LinearSolver::<SparseMode>::solve_augmented_equation(&mut solver, &r, &j, &probe_damping())?;
    let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("no gradient")?;
    let h = LinearSolver::<SparseMode>::get_hessian(&solver).ok_or("no hessian")?;
    assert_publishes_contract("SparseCholeskySolver", g, &sparse_diagonal(h))
}

#[test]
fn sparse_qr_publishes_positive_gradient_and_undamped_hessian() -> TestResult {
    let (j, r) = sparse_system()?;
    let mut solver = SparseQRSolver::new();
    LinearSolver::<SparseMode>::solve_augmented_equation(&mut solver, &r, &j, &probe_damping())?;
    let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("no gradient")?;
    let h = LinearSolver::<SparseMode>::get_hessian(&solver).ok_or("no hessian")?;
    assert_publishes_contract("SparseQRSolver", g, &sparse_diagonal(h))
}

#[test]
fn dense_cholesky_publishes_positive_gradient_and_undamped_hessian() -> TestResult {
    let (j, r) = dense_system();
    let mut solver = DenseCholeskySolver::new();
    LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, &probe_damping())?;
    let g = LinearSolver::<DenseMode>::get_gradient(&solver).ok_or("no gradient")?;
    let h = LinearSolver::<DenseMode>::get_hessian(&solver).ok_or("no hessian")?;
    let diag: Vec<f64> = (0..h.ncols()).map(|i| h[(i, i)]).collect();
    assert_publishes_contract("DenseCholeskySolver", g, &diag)
}

#[test]
fn dense_qr_publishes_positive_gradient_and_undamped_hessian() -> TestResult {
    let (j, r) = dense_system();
    let mut solver = DenseQRSolver::new();
    LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, &probe_damping())?;
    let g = LinearSolver::<DenseMode>::get_gradient(&solver).ok_or("no gradient")?;
    let h = LinearSolver::<DenseMode>::get_hessian(&solver).ok_or("no hessian")?;
    let diag: Vec<f64> = (0..h.ncols()).map(|i| h[(i, i)]).collect();
    assert_publishes_contract("DenseQRSolver", g, &diag)
}

/// The augmented solve must satisfy `(JᵀJ + λ·D)·dx = −Jᵀr`.
///
/// This is what ties the two published quantities together: it fails if the
/// right-hand side uses `+Jᵀr`, and it fails if the damping applied inside
/// differs from what `Damping` describes.
#[test]
fn augmented_solve_satisfies_its_own_normal_equations() -> TestResult {
    let (dense_j, r) = dense_system();
    let (j, _) = sparse_system()?;
    let damping = Damping::new(0.25, 1e-6, 1e32)?;

    let mut solver = SparseCholeskySolver::new();
    let dx = LinearSolver::<SparseMode>::solve_augmented_equation(&mut solver, &r, &j, &damping)?;

    let h = dense_j.transpose() * &dense_j;
    let g = dense_j.transpose() * &r;
    for i in 0..h.nrows() {
        let mut lhs = damping.diagonal_term(h[(i, i)]) * dx[(i, 0)];
        for k in 0..h.ncols() {
            lhs += h[(i, k)] * dx[(k, 0)];
        }
        assert!(
            (lhs + g[(i, 0)]).abs() < 1e-8,
            "row {i}: (JᵀJ + λ·D)·dx should equal −Jᵀr, got {lhs} vs {}",
            -g[(i, 0)]
        );
    }
    Ok(())
}

/// `Damping::identity` must reproduce classic uniform `λI` damping exactly.
///
/// This is the escape hatch documented on [`Damping`]: it lets a caller opt out
/// of Marquardt scaling entirely, so it has to be bit-for-bit the old behaviour
/// rather than merely close.
#[test]
fn identity_damping_adds_lambda_to_every_diagonal() -> TestResult {
    let (dense_j, r) = dense_system();
    let (j, _) = sparse_system()?;
    let lambda = 0.75;

    let mut solver = SparseCholeskySolver::new();
    let dx = LinearSolver::<SparseMode>::solve_augmented_equation(
        &mut solver,
        &r,
        &j,
        &Damping::identity(lambda),
    )?;

    let mut h = dense_j.transpose() * &dense_j;
    for i in 0..h.nrows() {
        h[(i, i)] += lambda;
    }
    let g = dense_j.transpose() * &r;
    let expected = h.as_ref().llt(faer::Side::Lower)?.solve(-&g);

    for i in 0..expected.nrows() {
        assert!(
            (dx[(i, 0)] - expected[(i, 0)]).abs() < 1e-10,
            "entry {i}: identity damping should match H + λI exactly, got {} vs {}",
            dx[(i, 0)],
            expected[(i, 0)]
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Schur solvers: same contract, plus StructureAware setup and graceful
// degradation where `JᵀJ` is never formed.
// -----------------------------------------------------------------------------
//
// Every Schur solver publishes the *full* system's `+Jᵀr`/`JᵀJ` through
// `get_gradient`/`get_hessian` — not the reduced camera system — so the same
// `assert_publishes_contract` reference applies unchanged. The fixture below
// reuses `dense_system`/`sparse_system`'s exact 4x3 shape (so
// `expected_gradient_and_hessian` is still the ground truth) and marks its
// last column as the eliminated ("group 0") variable.

mod schur_contract {
    use super::*;
    use apex_manifolds::rn::Rn;
    use apex_solver::core::VarKey;
    use apex_solver::core::variable::{ManifoldVariable, Variable};
    use apex_solver::linalg::{
        ExplicitDenseSchur, ExplicitSchurVariant, ExplicitSparseSchur, ImplicitSparseSchur,
        StructureAware,
    };
    use slotmap::{SecondaryMap, SlotMap};
    use std::collections::HashSet;

    type Setup = (
        SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        HashSet<VarKey>,
    );

    /// One kept 2-DOF variable (columns 0-1, "group 1") and one eliminated
    /// 1-DOF variable (column 2, "group 0"), matching `dense_system`'s 4x3
    /// shape column-for-column.
    fn schur_variables() -> Setup {
        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let kept = variables.insert(Box::new(Variable::new(Rn::new(nalgebra::DVector::zeros(
            2,
        )))));
        let eliminated = variables.insert(Box::new(Variable::new(Rn::new(
            nalgebra::DVector::zeros(1),
        ))));
        let mut index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        index_map.insert(kept, 0);
        index_map.insert(eliminated, 2);
        let mut landmark_keys = HashSet::new();
        landmark_keys.insert(eliminated);
        (variables, index_map, landmark_keys)
    }

    #[test]
    fn explicit_sparse_schur_publishes_the_contract() -> TestResult {
        let (j, r) = sparse_system()?;
        let (variables, index_map, landmarks) = schur_variables();
        let mut solver = ExplicitSparseSchur::new();
        solver.initialize_structure(&variables, &index_map, &landmarks)?;
        LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &probe_damping(),
        )?;
        let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("no gradient")?;
        let h = LinearSolver::<SparseMode>::get_hessian(&solver).ok_or("no hessian")?;
        assert_publishes_contract("ExplicitSparseSchur", g, &sparse_diagonal(h))
    }

    /// `ImplicitSparseSchur` is matrix-free with respect to `JᵀJ` as well as
    /// `S` — Ceres's `ITERATIVE_SCHUR` semantics — so like the chunked path it
    /// must publish no Hessian while still serving the true, un-damped
    /// quadratic model through `hessian_vec_product`.
    #[test]
    fn implicit_sparse_schur_degrades_hessian_gracefully() -> TestResult {
        let (j, r) = sparse_system()?;
        let (variables, index_map, landmarks) = schur_variables();
        let mut solver = ImplicitSparseSchur::new();
        solver.initialize_structure(&variables, &index_map, &landmarks)?;
        LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &probe_damping(),
        )?;

        assert!(
            LinearSolver::<SparseMode>::get_hessian(&solver).is_none(),
            "the matrix-free path never forms JᵀJ and must not publish one"
        );

        let (expected_g, expected_h) = expected_gradient_and_hessian();
        let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("no gradient")?;
        for i in 0..expected_g.nrows() {
            assert!(
                (g[(i, 0)] - expected_g[(i, 0)]).abs() < 1e-9,
                "ImplicitSparseSchur: get_gradient() must publish +Jᵀr. Entry {i} is {}, \
                 expected {}",
                g[(i, 0)],
                expected_g[(i, 0)]
            );
        }

        // The model must be the *un-damped* JᵀJ even though the solve was
        // damped, or the step-quality ratio would be computed against the
        // wrong quadratic.
        let v = Mat::<f64>::from_fn(3, 1, |i, _| (i + 1) as f64);
        let hv = LinearSolver::<SparseMode>::hessian_vec_product(&solver, &v)
            .ok_or("no hessian_vec_product")?;
        let expected_hv = &expected_h * &v;
        for i in 0..3 {
            assert!(
                (hv[(i, 0)] - expected_hv[(i, 0)]).abs() < 1e-9,
                "hessian_vec_product mismatch at {i}: {} vs {}",
                hv[(i, 0)],
                expected_hv[(i, 0)]
            );
        }
        Ok(())
    }

    #[test]
    fn explicit_dense_schur_publishes_the_contract() -> TestResult {
        let (j, r) = dense_system();
        let (variables, index_map, landmarks) = schur_variables();
        let mut solver = ExplicitDenseSchur::new();
        solver.initialize_structure(&variables, &index_map, &landmarks)?;
        LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, &probe_damping())?;
        let g = LinearSolver::<DenseMode>::get_gradient(&solver).ok_or("no gradient")?;
        let h = LinearSolver::<DenseMode>::get_hessian(&solver).ok_or("no hessian")?;
        let diag: Vec<f64> = (0..h.ncols()).map(|i| h[(i, i)]).collect();
        assert_publishes_contract("ExplicitDenseSchur", g, &diag)
    }

    /// The chunked path never forms `JᵀJ`, so it must degrade gracefully:
    /// `get_hessian` returns `None` while `get_gradient` and
    /// `hessian_vec_product` still serve the true, un-damped quadratic model.
    #[test]
    fn explicit_sparse_schur_chunked_degrades_hessian_gracefully() -> TestResult {
        let (j, r) = sparse_system()?;
        let (variables, index_map, landmarks) = schur_variables();
        let mut solver = ExplicitSparseSchur::new().with_variant(ExplicitSchurVariant::Chunked);
        solver.initialize_structure(&variables, &index_map, &landmarks)?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        assert!(
            LinearSolver::<SparseMode>::get_hessian(&solver).is_none(),
            "the chunked path never forms JᵀJ and must not publish one"
        );

        let (expected_g, expected_h) = expected_gradient_and_hessian();
        let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("no gradient")?;
        for i in 0..expected_g.nrows() {
            assert!(
                (g[(i, 0)] - expected_g[(i, 0)]).abs() < 1e-9,
                "gradient mismatch at {i}: {} vs {}",
                g[(i, 0)],
                expected_g[(i, 0)]
            );
        }

        let v = Mat::<f64>::from_fn(3, 1, |i, _| (i + 1) as f64);
        let hv = LinearSolver::<SparseMode>::hessian_vec_product(&solver, &v)
            .ok_or("no hessian_vec_product")?;
        let expected_hv = &expected_h * &v;
        for i in 0..3 {
            assert!(
                (hv[(i, 0)] - expected_hv[(i, 0)]).abs() < 1e-9,
                "hessian_vec_product mismatch at {i}: {} vs {}",
                hv[(i, 0)],
                expected_hv[(i, 0)]
            );
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Optimizer dispatch: no silent solver substitution
// -----------------------------------------------------------------------------
//
// Requesting a solver an optimizer cannot run used to silently fall back to
// plain Cholesky, making Schur-vs-Cholesky comparisons under GN/DogLeg report
// identical numbers. The dispatch must reject unsupported combinations.

mod dispatch {
    use apex_solver::ManifoldType;
    use apex_solver::core::problem::Problem;
    use apex_solver::linalg::JacobianMode;
    use apex_solver::linalg::LinearSolverType;
    use apex_solver::optimizer::dog_leg::{DogLeg, DogLegConfig};
    use apex_solver::optimizer::gauss_newton::{GaussNewton, GaussNewtonConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn small_problem(mode: JacobianMode) -> Problem {
        let mut problem = Problem::new(mode);
        let a = problem.add_variable(ManifoldType::RN, nalgebra::dvector![1.0]);
        let b = problem.add_variable(ManifoldType::RN, nalgebra::dvector![2.0]);
        problem.add_residual_block(&[a], Box::new(OffsetFactor { target: 0.5 }), None);
        problem.add_residual_block(&[b], Box::new(OffsetFactor { target: 1.5 }), None);
        problem
    }

    struct OffsetFactor {
        target: f64,
    }

    impl apex_solver::Factor for OffsetFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            _jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            // Jacobian left at zero: the optimizer errors out before any solve.
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    fn gn(solver: LinearSolverType) -> GaussNewton {
        GaussNewton::with_config(GaussNewtonConfig::new().with_linear_solver_type(solver))
    }

    fn dl(solver: LinearSolverType) -> DogLeg {
        DogLeg::with_config(DogLegConfig::new().with_linear_solver_type(solver))
    }

    fn assert_rejected(
        mut optimizer: impl apex_solver::Optimizer,
        mode: JacobianMode,
        label: &str,
    ) -> TestResult {
        let mut problem = small_problem(mode);
        let err = optimizer
            .optimize(&mut problem)
            .err()
            .ok_or_else(|| format!("{label} should be rejected"))?;
        assert!(
            err.to_string().contains("supports"),
            "{label}: unexpected error: {err}"
        );
        Ok(())
    }

    #[test]
    fn gn_accepts_schur_under_sparse_mode() -> TestResult {
        // Schur is dispatched like any other sparse solver now; the factor has
        // a zero Jacobian so the solve fails numerically — any error other
        // than the dispatch rejection is fine here.
        let mut optimizer = gn(LinearSolverType::ExplicitSparseSchur);
        let mut problem = small_problem(JacobianMode::Sparse);
        let result = optimizer.optimize(&mut problem);
        if let Err(e) = result {
            assert!(
                !e.to_string().contains("supports"),
                "GN + Schur (sparse): dispatch must accept, got {e}"
            );
        }
        Ok(())
    }

    #[test]
    fn gn_rejects_dense_cholesky_under_sparse_mode() -> TestResult {
        assert_rejected(
            gn(LinearSolverType::DenseCholesky),
            JacobianMode::Sparse,
            "GN + DenseCholesky (sparse)",
        )
    }

    #[test]
    fn gn_rejects_sparse_solvers_under_dense_mode() -> TestResult {
        assert_rejected(
            gn(LinearSolverType::SparseCholesky),
            JacobianMode::Dense,
            "GN + SparseCholesky (dense)",
        )?;
        assert_rejected(
            gn(LinearSolverType::ExplicitSparseSchur),
            JacobianMode::Dense,
            "GN + ExplicitSparseSchur (dense)",
        )?;
        assert_rejected(
            gn(LinearSolverType::ImplicitSparseSchur),
            JacobianMode::Dense,
            "GN + ImplicitSparseSchur (dense)",
        )
    }

    #[test]
    fn gn_accepts_implicit_schur_under_sparse_mode() -> TestResult {
        let mut optimizer = gn(LinearSolverType::ImplicitSparseSchur);
        let mut problem = small_problem(JacobianMode::Sparse);
        let result = optimizer.optimize(&mut problem);
        if let Err(e) = result {
            assert!(
                !e.to_string().contains("supports"),
                "GN + ImplicitSparseSchur (sparse): dispatch must accept, got {e}"
            );
        }
        Ok(())
    }

    #[test]
    fn gn_rejects_explicit_dense_schur_under_sparse_mode() -> TestResult {
        assert_rejected(
            gn(LinearSolverType::ExplicitDenseSchur),
            JacobianMode::Sparse,
            "GN + ExplicitDenseSchur (sparse)",
        )
    }

    #[test]
    fn gn_accepts_explicit_dense_schur_under_dense_mode() -> TestResult {
        let mut optimizer = gn(LinearSolverType::ExplicitDenseSchur);
        let mut problem = small_problem(JacobianMode::Dense);
        let result = optimizer.optimize(&mut problem);
        if let Err(e) = result {
            assert!(
                !e.to_string().contains("supports"),
                "GN + ExplicitDenseSchur (dense): dispatch must accept, got {e}"
            );
        }
        Ok(())
    }

    #[test]
    fn dog_leg_accepts_schur_under_sparse_mode() -> TestResult {
        // See `gn_accepts_schur_under_sparse_mode`: dispatched, not rejected.
        let mut optimizer = dl(LinearSolverType::ExplicitSparseSchur);
        let mut problem = small_problem(JacobianMode::Sparse);
        let result = optimizer.optimize(&mut problem);
        if let Err(e) = result {
            assert!(
                !e.to_string().contains("supports"),
                "DogLeg + Schur (sparse): dispatch must accept, got {e}"
            );
        }
        Ok(())
    }

    #[test]
    fn dog_leg_rejects_sparse_solvers_under_dense_mode() -> TestResult {
        assert_rejected(
            dl(LinearSolverType::SparseQR),
            JacobianMode::Dense,
            "DogLeg + SparseQR (dense)",
        )
    }

    #[test]
    fn gn_accepts_supported_combinations() -> TestResult {
        // Supported combos must dispatch without the rejection error. The factor
        // has a zero Jacobian, so the solve fails numerically — any error other
        // than the dispatch rejection is fine here.
        for (solver, mode, label) in [
            (
                LinearSolverType::SparseCholesky,
                JacobianMode::Sparse,
                "GN sparse+cholesky",
            ),
            (
                LinearSolverType::DenseCholesky,
                JacobianMode::Dense,
                "GN dense+cholesky",
            ),
        ] {
            let mut optimizer = gn(solver);
            let mut problem = small_problem(mode);
            let result = optimizer.optimize(&mut problem);
            if let Err(e) = result {
                assert!(
                    !e.to_string().contains("supports"),
                    "{label}: dispatch must accept this combination, got {e}"
                );
            }
        }
        Ok(())
    }
}
