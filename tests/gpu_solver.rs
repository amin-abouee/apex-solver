//! GPU solver correctness, against the CPU solvers as ground truth.
//!
//! **These tests need an NVIDIA GPU.** They are `#[ignore]`d so that
//! `cargo test --features cuda` stays green on machines without one (including
//! all four CI runners), and each additionally probes `cuda::is_available` and
//! skips with a message rather than failing.
//!
//! Run them on a CUDA machine with:
//!
//! ```bash
//! cargo test --features cuda -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use apex_solver::JacobianMode;
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::apex_manifolds::se2::SE2;
use apex_solver::core::VarKey;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, Factor, PriorFactor};
use apex_solver::cuda::{Reordering, is_available};
use apex_solver::linalg::{
    CudaSparseCholeskySolver, CudaSparseQRSolver, LinearSolver, LinearSolverType,
    SparseCholeskySolver, SparseMode,
};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::dvector;

type TestResult = Result<(), Box<dyn std::error::Error>>;
/// A linear system: the Jacobian and its residual vector.
type LinearSystem = (SparseColMat<usize, f64>, Mat<f64>);

/// Skip (rather than fail) when there is no GPU on this machine.
macro_rules! require_gpu {
    () => {
        if !is_available() {
            eprintln!("skipping: no CUDA device available");
            return Ok(());
        }
    };
}

/// A small overdetermined least-squares system whose `JᵀJ` is well conditioned.
fn well_conditioned_system() -> Result<LinearSystem, Box<dyn std::error::Error>> {
    // 4x3 Jacobian.
    let triplets = vec![
        Triplet::new(0, 0, 2.0),
        Triplet::new(0, 1, 1.0),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 3.0),
        Triplet::new(1, 2, 1.0),
        Triplet::new(2, 1, 1.0),
        Triplet::new(2, 2, 2.0),
        Triplet::new(3, 0, 1.5),
        Triplet::new(3, 2, 0.5),
    ];
    let jacobian = SparseColMat::try_new_from_triplets(4, 3, &triplets)?;
    let residuals = Mat::from_fn(4, 1, |i, _| [1.0, 2.0, 0.5, 1.5][i]);
    Ok((jacobian, residuals))
}

fn assert_steps_match(gpu: &Mat<f64>, cpu: &Mat<f64>, tolerance: f64, context: &str) {
    assert_eq!(gpu.nrows(), cpu.nrows(), "{context}: dimension mismatch");
    for i in 0..cpu.nrows() {
        let (g, c) = (gpu[(i, 0)], cpu[(i, 0)]);
        let scale = g.abs().max(c.abs()).max(1.0);
        assert!(
            (g - c).abs() <= tolerance * scale,
            "{context}: dx[{i}] GPU {g:.15e} vs CPU {c:.15e}"
        );
    }
}

/// The GPU normal-equation solve must reproduce the CPU one.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn gpu_cholesky_matches_cpu_on_normal_equations() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    let mut cpu = SparseCholeskySolver::new();
    let cpu_dx =
        LinearSolver::<SparseMode>::solve_normal_equation(&mut cpu, &residuals, &jacobian)?;

    let mut gpu = CudaSparseCholeskySolver::new()?;
    let gpu_dx =
        LinearSolver::<SparseMode>::solve_normal_equation(&mut gpu, &residuals, &jacobian)?;

    assert_steps_match(&gpu_dx, &cpu_dx, 1e-10, "normal equations");
    Ok(())
}

/// Same, with Levenberg-Marquardt damping applied.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn gpu_cholesky_matches_cpu_on_augmented_equations() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    for lambda in [1e-8, 1e-3, 1.0] {
        let mut cpu = SparseCholeskySolver::new();
        let cpu_dx = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut cpu, &residuals, &jacobian, lambda,
        )?;

        let mut gpu = CudaSparseCholeskySolver::new()?;
        let gpu_dx = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut gpu, &residuals, &jacobian, lambda,
        )?;

        assert_steps_match(
            &gpu_dx,
            &cpu_dx,
            1e-10,
            &format!("augmented lambda={lambda:e}"),
        );
    }
    Ok(())
}

/// GPU QR must agree with GPU Cholesky — they solve the same `JᵀJ` system.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn gpu_qr_matches_gpu_cholesky() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    let mut chol = CudaSparseCholeskySolver::new()?;
    let chol_dx =
        LinearSolver::<SparseMode>::solve_normal_equation(&mut chol, &residuals, &jacobian)?;

    let mut qr = CudaSparseQRSolver::new()?;
    let qr_dx = LinearSolver::<SparseMode>::solve_normal_equation(&mut qr, &residuals, &jacobian)?;

    assert_steps_match(&qr_dx, &chol_dx, 1e-8, "GPU QR vs GPU Cholesky");
    Ok(())
}

/// Every reordering must produce the same answer — it changes fill-in and
/// runtime, never the solution.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn reordering_does_not_change_the_solution() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    let mut reference = CudaSparseCholeskySolver::new()?;
    let reference_dx =
        LinearSolver::<SparseMode>::solve_normal_equation(&mut reference, &residuals, &jacobian)?;

    for reordering in [
        Reordering::None,
        Reordering::SymRcm,
        Reordering::SymAmd,
        Reordering::Metis,
    ] {
        let mut solver = CudaSparseCholeskySolver::new()?.with_reordering(reordering);
        let dx =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_steps_match(&dx, &reference_dx, 1e-9, &format!("{reordering:?}"));
    }
    Ok(())
}

/// A star graph: one hub connected to every other node, plus a prior anchoring
/// the hub.
///
/// `JᵀJ` is an arrow matrix, whose hub has by far the highest degree — so every
/// fill-reducing ordering moves it to the end and the permutation is emphatically
/// **not** the identity. That is what makes it useful for testing permutation
/// handling; the small well-conditioned system cannot, because minimum degree
/// leaves a 3x3 matrix alone.
fn star_system(spokes: usize) -> Result<LinearSystem, Box<dyn std::error::Error>> {
    let mut triplets = Vec::new();
    // Row 0: prior on the hub, so the system is not gauge-free.
    triplets.push(Triplet::new(0, 0, 1.0));
    for spoke in 1..=spokes {
        // One edge row per spoke, coupling the hub (column 0) to column `spoke`.
        triplets.push(Triplet::new(spoke, 0, 1.0));
        triplets.push(Triplet::new(spoke, spoke, -1.0));
    }
    let jacobian = SparseColMat::try_new_from_triplets(spokes + 1, spokes + 1, &triplets)?;
    let residuals = Mat::from_fn(spokes + 1, 1, |i, _| 0.5 + (i as f64) * 0.25);
    Ok((jacobian, residuals))
}

/// Changing the reordering on a solver that has already run must re-analyze from
/// the Hessian's own pattern.
///
/// The reusable path permutes its cached CSR arrays in place, so a naive
/// re-analysis would permute an already-permuted pattern — composing two
/// permutations and corrupting the value map, which shows up as a wrong step
/// rather than an error.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn changing_reordering_after_a_solve_still_solves_correctly() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = star_system(31)?;

    let mut cpu = SparseCholeskySolver::new();
    let expected =
        LinearSolver::<SparseMode>::solve_normal_equation(&mut cpu, &residuals, &jacobian)?;

    let mut solver = CudaSparseCholeskySolver::new()?.with_reordering(Reordering::Metis);
    LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

    // Re-analysis with the same pattern but a different ordering.
    let mut solver = solver.with_reordering(Reordering::SymAmd);
    let dx = LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
    assert_steps_match(&dx, &expected, 1e-9, "reordering changed after first solve");

    // And back again, to catch a permutation that only composes in one direction.
    let mut solver = solver.with_reordering(Reordering::Metis);
    let dx = LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
    assert_steps_match(&dx, &expected, 1e-9, "reordering changed back");
    Ok(())
}

/// The cached quantities must match the CPU contract exactly: `get_hessian`
/// returns the **undamped** `JᵀJ` even after an augmented solve, and
/// `get_gradient` the **positive** `Jᵀr`. Levenberg-Marquardt depends on both.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn augmented_solve_caches_the_undamped_hessian() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    let mut gpu = CudaSparseCholeskySolver::new()?;
    LinearSolver::<SparseMode>::solve_augmented_equation(&mut gpu, &residuals, &jacobian, 5.0)?;

    let mut cpu = SparseCholeskySolver::new();
    LinearSolver::<SparseMode>::solve_augmented_equation(&mut cpu, &residuals, &jacobian, 5.0)?;

    let gpu_h = LinearSolver::<SparseMode>::get_hessian(&gpu).ok_or("GPU hessian missing")?;
    let cpu_h = LinearSolver::<SparseMode>::get_hessian(&cpu).ok_or("CPU hessian missing")?;
    for c in 0..cpu_h.ncols() {
        for r in 0..cpu_h.nrows() {
            let (g, k) = (
                gpu_h.get(r, c).copied().unwrap_or(0.0),
                cpu_h.get(r, c).copied().unwrap_or(0.0),
            );
            assert!((g - k).abs() < 1e-12, "H[{r},{c}]: GPU {g} vs CPU {k}");
        }
    }

    let gpu_g = LinearSolver::<SparseMode>::get_gradient(&gpu).ok_or("GPU gradient missing")?;
    let cpu_g = LinearSolver::<SparseMode>::get_gradient(&cpu).ok_or("CPU gradient missing")?;
    assert_steps_match(gpu_g, cpu_g, 1e-12, "cached gradient sign/value");
    Ok(())
}

/// Rank-deficient `J`: column 1 is twice column 0, so `JᵀJ` is singular.
fn rank_deficient_system() -> Result<LinearSystem, Box<dyn std::error::Error>> {
    let triplets = vec![
        Triplet::new(0, 0, 1.0),
        Triplet::new(0, 1, 2.0),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 2.0),
    ];
    let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
    let residuals = Mat::from_fn(2, 1, |_, _| 1.0);
    Ok((jacobian, residuals))
}

/// `csrlsvqr` reports a singular system with the offending row — strictly better
/// than the CPU path's opaque factorization failure.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn singular_system_is_reported_with_a_row_index_by_qr() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = rank_deficient_system()?;

    let mut gpu = CudaSparseQRSolver::new()?;
    match LinearSolver::<SparseMode>::solve_normal_equation(&mut gpu, &residuals, &jacobian) {
        Ok(_) => panic!("a singular system must not solve successfully"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("singular") || message.contains("positive definite"),
                "error should describe the singularity: {message}"
            );
        }
    }
    Ok(())
}

/// Pins a cuSOLVER limitation rather than a property we want.
///
/// The **device** `cusolverSpDcsrlsvchol` leaves `singularity` at `-1` even for
/// a rank-deficient matrix — only the `...Host` variant and `csrlsvqr` run the
/// zero-pivot check. Verified on CUDA 13.0 / driver 580.178.04.
///
/// On this consistent rank-deficient system it returns a vector that genuinely
/// satisfies `H·dx = −g`, so the outcome is benign here; for an *inconsistent*
/// singular system it would return an unflagged meaningless step. Callers who
/// need singularity detection on the GPU must use [`CudaSparseQRSolver`]. If a
/// future CUDA starts populating the out-parameter this test will fail, which is
/// the point — the limitation is then gone and the docs should follow.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn device_cholesky_does_not_flag_singularity() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = rank_deficient_system()?;

    let mut gpu = CudaSparseCholeskySolver::new()?;
    let dx = LinearSolver::<SparseMode>::solve_normal_equation(&mut gpu, &residuals, &jacobian)?;

    // H = JᵀJ = [[2, 4], [4, 8]], g = Jᵀr = [2, 4]; every solution of H·dx = −g
    // satisfies dx₀ + 2·dx₁ = −1.
    let on_solution_set = dx[(0, 0)] + 2.0 * dx[(1, 0)];
    assert!(
        (on_solution_set + 1.0).abs() < 1e-9,
        "returned step should still satisfy the consistent system: dx = [{:e}, {:e}]",
        dx[(0, 0)],
        dx[(1, 0)]
    );
    Ok(())
}

/// A gauge-fixed 4-pose SE2 chain with a loop closure.
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

    // Anchor the gauge: without a prior, JᵀJ is singular by 3 and Cholesky
    // legitimately fails.
    problem.add_residual_block(
        &[x0],
        Box::new(PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        }),
        None,
    );

    (problem, vec![x0, x1, x2, x3])
}

fn final_cost_with(solver_type: LinearSolverType) -> Result<f64, Box<dyn std::error::Error>> {
    let (mut problem, _) = pose_graph();
    let result = LevenbergMarquardt::with_config(
        LevenbergMarquardtConfig::new()
            .with_linear_solver_type(solver_type)
            .with_max_iterations(100)
            .with_cost_tolerance(1e-12)
            .with_parameter_tolerance(1e-12),
    )
    .optimize(&mut problem)?;
    Ok(result.final_cost)
}

/// End-to-end: Levenberg-Marquardt driven by the GPU solver must reach the same
/// optimum as the CPU one.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn lm_converges_to_the_same_cost_on_gpu() -> TestResult {
    require_gpu!();

    let cpu = final_cost_with(LinearSolverType::SparseCholesky)?;
    for gpu_type in [
        LinearSolverType::GpuSparseCholesky,
        LinearSolverType::GpuSparseQR,
    ] {
        let gpu = final_cost_with(gpu_type)?;
        let scale = cpu.abs().max(gpu.abs()).max(1.0);
        assert!(
            (cpu - gpu).abs() <= 1e-8 * scale,
            "{gpu_type}: final cost {gpu:.12e} vs CPU {cpu:.12e}"
        );
    }
    Ok(())
}

/// Covariance must work on the GPU path rather than silently returning `None`,
/// and must equal the CPU result — Levenberg-Marquardt's damping must not leak
/// into it. See `cov_issues/01-covariance-absorbs-damping.md`.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn gpu_solver_reports_covariance() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    let mut gpu = CudaSparseCholeskySolver::new()?;
    LinearSolver::<SparseMode>::solve_augmented_equation(&mut gpu, &residuals, &jacobian, 0.75)?;

    let covariance = LinearSolver::<SparseMode>::compute_covariance_matrix(&mut gpu)
        .ok_or("GPU solver must not return None for covariance")?
        .clone();

    // Σ · H must be the identity, with H the UNDAMPED Hessian.
    let hessian = LinearSolver::<SparseMode>::get_hessian(&gpu).ok_or("hessian missing")?;
    let n = covariance.nrows();
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += covariance[(i, k)] * hessian.get(k, j).copied().unwrap_or(0.0);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (acc - expected).abs() < 1e-8,
                "(Sigma*H)[{i},{j}] = {acc}, expected {expected} — damping must not leak in"
            );
        }
    }
    Ok(())
}

/// Reusing one solver across two different problems must not return a stale
/// covariance from the first.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn covariance_cache_is_invalidated_between_solves() -> TestResult {
    require_gpu!();
    let (jacobian, residuals) = well_conditioned_system()?;

    let mut gpu = CudaSparseCholeskySolver::new()?;
    LinearSolver::<SparseMode>::solve_normal_equation(&mut gpu, &residuals, &jacobian)?;
    let first = LinearSolver::<SparseMode>::compute_covariance_matrix(&mut gpu)
        .ok_or("first covariance missing")?
        .clone();

    // Same pattern, different values => a different covariance.
    let scaled = SparseColMat::try_new_from_triplets(
        4,
        3,
        &[
            Triplet::new(0, 0, 4.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 6.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 2, 4.0),
            Triplet::new(3, 0, 1.5),
            Triplet::new(3, 2, 0.5),
        ],
    )?;
    LinearSolver::<SparseMode>::solve_normal_equation(&mut gpu, &residuals, &scaled)?;
    let second = LinearSolver::<SparseMode>::compute_covariance_matrix(&mut gpu)
        .ok_or("second covariance missing")?
        .clone();

    let changed = (0..first.nrows()).any(|i| (first[(i, i)] - second[(i, i)]).abs() > 1e-9);
    assert!(
        changed,
        "covariance must be recomputed after a new solve, not served from cache"
    );
    Ok(())
}

/// Selecting a GPU solver without a device must produce a clear error rather
/// than silently running on the CPU — a silent fallback would make a GPU
/// benchmark measure the wrong thing.
#[test]
fn gpu_selection_without_a_device_errors_rather_than_falling_back() {
    if is_available() {
        return; // Meaningless on a machine that has a GPU.
    }
    let (mut problem, _) = pose_graph();
    let result = LevenbergMarquardt::with_config(
        LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::GpuSparseCholesky)
            .with_max_iterations(5),
    )
    .optimize(&mut problem);

    match result {
        Ok(_) => panic!("GPU solver must not silently fall back to the CPU"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("CUDA") || message.contains("cuda"),
                "error should explain the GPU is unavailable: {message}"
            );
        }
    }
}

/// The `Factor` trait import is used by the pose-graph builder; this keeps the
/// import honest if the builder changes.
#[allow(dead_code)]
fn _assert_factor_in_scope(f: &dyn Factor) -> usize {
    f.residual_dim()
}
