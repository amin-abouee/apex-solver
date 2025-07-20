pub mod cholesky;
pub mod qr;

use faer::Mat;
use faer::sparse::SparseColMat;

/// Trait for sparse linear solvers that can solve both normal and augmented equations
pub trait SparseLinearSolver {
    /// Solve the normal equation: (J^T * W * J) * dx = -J^T * W * r
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        weights: &Mat<f64>,
    ) -> Option<Mat<f64>>;

    /// Solve the augmented equation: (J^T * W * J + λI) * dx = -J^T * W * r
    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        weights: &Mat<f64>,
        lambda: f64,
    ) -> Option<Mat<f64>>;
}

#[derive(Default, Clone)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

pub use cholesky::SparseCholeskySolver;
pub use qr::SparseQRSolver;
