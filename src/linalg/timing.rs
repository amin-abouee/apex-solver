//! A [`LinearSolver`] decorator that measures how much of an optimization is
//! actually spent in the linear solve.
//!
//! An optimizer iteration is roughly *assemble `J`* → *form `H = JᵀJ`, `g = Jᵀr`*
//! → *factorize and solve*. Only the last step differs between the CPU and GPU
//! solvers, so end-to-end wall clock alone cannot say whether a backend swap
//! helped: the shared assembly cost caps the achievable gain (Amdahl). Wrapping
//! a solver in [`TimedSolver`] separates the two.
//!
//! ```no_run
//! use apex_solver::linalg::{SparseCholeskySolver, SparseMode, TimedSolver};
//! use apex_solver::optimizer::LevenbergMarquardt;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut problem = apex_solver::core::problem::Problem::default();
//! let mut solver = TimedSolver::new(SparseCholeskySolver::new());
//! let result = LevenbergMarquardt::new().optimize_with_mode::<SparseMode>(&mut problem, &mut solver)?;
//! println!(
//!     "{} solves took {:?} of {:?} total",
//!     solver.solve_count(),
//!     solver.solve_time(),
//!     result.elapsed_time,
//! );
//! # Ok(()) }
//! ```

use std::time::{Duration, Instant};

use faer::Mat;
use faer::sparse::SparseColMat;

use crate::linalg::{LinAlgResult, LinearSolver, SparseMode};

/// Wraps a sparse linear solver and accumulates the time spent inside it.
///
/// Every trait method delegates; only [`solve_normal_equation`] and
/// [`solve_augmented_equation`] are timed, since those are the calls that
/// factorize. Covariance computation is deliberately *not* counted — it happens
/// once after convergence, not per iteration.
///
/// Generic over the wrapped solver so [`inner`](Self::inner) can hand back the
/// concrete type — which is how a CUDA solver's
/// [`profile`](crate::linalg::CudaSparseCholeskySolver::profile) is read after a
/// run. A solver chosen at runtime still works: `Box<dyn LinearSolver<SparseMode>>`
/// implements the trait itself, so `TimedSolver<Box<dyn _>>` is valid.
///
/// [`solve_normal_equation`]: LinearSolver::solve_normal_equation
/// [`solve_augmented_equation`]: LinearSolver::solve_augmented_equation
pub struct TimedSolver<S> {
    inner: S,
    solve_time: Duration,
    solve_count: usize,
}

impl<S: LinearSolver<SparseMode>> TimedSolver<S> {
    /// Wrap `inner`, starting from a zeroed accumulator.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            solve_time: Duration::ZERO,
            solve_count: 0,
        }
    }

    /// The wrapped solver, for backend-specific results such as a CUDA profile.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Unwrap, discarding the accumulated timings.
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Total time spent inside the wrapped solver's factorize-and-solve calls.
    pub fn solve_time(&self) -> Duration {
        self.solve_time
    }

    /// How many such calls were made.
    ///
    /// Exceeds the iteration count whenever Levenberg-Marquardt rejects a step
    /// and re-solves with a larger `λ`.
    pub fn solve_count(&self) -> usize {
        self.solve_count
    }

    /// Mean time per solve, or zero if none were made.
    pub fn mean_solve_time(&self) -> Duration {
        self.solve_time
            .checked_div(self.solve_count.try_into().unwrap_or(u32::MAX))
            .unwrap_or(Duration::ZERO)
    }
}

impl<S> std::fmt::Debug for TimedSolver<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimedSolver")
            .field("solve_time", &self.solve_time)
            .field("solve_count", &self.solve_count)
            .finish_non_exhaustive()
    }
}

impl<S: LinearSolver<SparseMode>> LinearSolver<SparseMode> for TimedSolver<S> {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let started = Instant::now();
        let result = self.inner.solve_normal_equation(residuals, jacobian);
        self.solve_time += started.elapsed();
        self.solve_count += 1;
        result
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        let started = Instant::now();
        let result = self
            .inner
            .solve_augmented_equation(residuals, jacobian, lambda);
        self.solve_time += started.elapsed();
        self.solve_count += 1;
        result
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.inner.get_hessian()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.inner.get_gradient()
    }

    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        self.inner.compute_covariance_matrix()
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        self.inner.get_covariance_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::SparseCholeskySolver;
    use faer::sparse::Triplet;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn small_system() -> Result<(SparseColMat<usize, f64>, Mat<f64>), faer::sparse::CreationError> {
        let triplets = vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 3.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| [1.0, 2.0][i]);
        Ok((jacobian, residuals))
    }

    #[test]
    fn counts_and_times_every_solve() -> TestResult {
        let (jacobian, residuals) = small_system()?;
        let mut timed = TimedSolver::new(SparseCholeskySolver::new());

        assert_eq!(timed.solve_count(), 0);
        assert_eq!(timed.solve_time(), Duration::ZERO);
        assert_eq!(timed.mean_solve_time(), Duration::ZERO);

        LinearSolver::<SparseMode>::solve_normal_equation(&mut timed, &residuals, &jacobian)?;
        LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut timed, &residuals, &jacobian, 1e-3,
        )?;

        assert_eq!(timed.solve_count(), 2, "both call kinds must be counted");
        assert!(timed.solve_time() > Duration::ZERO);
        Ok(())
    }

    /// The decorator must be transparent: the step it returns and the Hessian it
    /// exposes have to be exactly what the wrapped solver produced.
    #[test]
    fn delegates_results_unchanged() -> TestResult {
        let (jacobian, residuals) = small_system()?;

        let mut plain = SparseCholeskySolver::new();
        let expected = LinearSolver::<SparseMode>::solve_normal_equation(
            &mut plain, &residuals, &jacobian,
        )?;

        let mut timed = TimedSolver::new(SparseCholeskySolver::new());
        let actual =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut timed, &residuals, &jacobian)?;

        assert_eq!(actual.nrows(), expected.nrows());
        for i in 0..expected.nrows() {
            assert!((actual[(i, 0)] - expected[(i, 0)]).abs() < 1e-15);
        }
        assert!(
            LinearSolver::<SparseMode>::get_hessian(&timed).is_some(),
            "cached Hessian must pass through"
        );
        Ok(())
    }
}
