//! A dense Cholesky solve of an unchanged shape allocates nothing once warm.
//!
//! The Gauss-Newton loops that call it run thousands of solves a frame, so a
//! per-solve allocation is a per-frame cost. This is a regression guard for
//! exactly that: the solver keeps its normal equations, factor and scratch
//! between calls, and hands the step back from its own buffer.

use apex_solver::linalg::{Damping, DenseCholeskySolver};
use faer::Mat;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn a_warm_solve_of_the_same_shape_allocates_nothing() -> TestResult {
    // The counter sees every thread. Left parallel, faer may hand a tiny matmul
    // to the rayon pool, whose workers allocate their own bookkeeping on first
    // use, which this test would count against the solver.
    faer::set_global_parallelism(faer::Par::Seq);
    let _profiler = dhat::Profiler::builder().testing().build();

    let jacobian = Mat::<f64>::from_fn(52, 3, |i, j| ((i * 7 + j * 3) % 11) as f64 - 4.5);
    let residual = Mat::<f64>::from_fn(52, 1, |i, _| ((i * 5) % 9) as f64 - 3.0);
    let damping = Damping::identity(1e-10);
    let mut solver = DenseCholeskySolver::new();

    // The first solves size every buffer and any lazily initialised state.
    for _ in 0..2 {
        solver.solve_augmented_equation_ref(&residual, &jacobian, &damping)?;
    }

    // Each solve is bracketed on its own, so the harness's bookkeeping
    // between solves is not charged to the solver.
    let mut blocks = 0u64;
    for _ in 0..100 {
        let before = dhat::HeapStats::get().total_blocks;
        solver.solve_augmented_equation_ref(&residual, &jacobian, &damping)?;
        blocks += dhat::HeapStats::get().total_blocks - before;
    }

    assert_eq!(blocks, 0, "100 warm solves must not touch the heap");
    Ok(())
}
