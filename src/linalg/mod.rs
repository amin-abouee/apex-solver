pub mod cholesky;
pub mod qr;

use faer::Mat;
use faer::sparse::SparseColMat;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

// /// Contains statistical information about the quality of the optimization solution.
// #[derive(Debug, Clone)]
// pub struct SolverElement {
//     /// The Hessian matrix, computed as `(J^T * W * J)`.
//     ///
//     /// This is `None` if the Hessian could not be computed.
//     pub hessian: Option<SparseColMat<usize, f64>>,

//     /// The gradient vector, computed as `J^T * W * r`.
//     ///
//     /// This is `None` if the gradient could not be computed.
//     pub gradient: Option<Mat<f64>>,

//     /// The parameter covariance matrix, computed as `(J^T * W * J)^-1`.
//     ///
//     /// This is `None` if the Hessian is singular or ill-conditioned.
//     pub covariance_matrix: Option<Mat<f64>>,
//     /// Asymptotic standard errors of the parameters.
//     ///
//     /// This is `None` if the covariance matrix could not be computed.
//     /// Each error is the square root of the corresponding diagonal element
//     /// of the covariance matrix.
//     pub standard_errors: Option<Mat<f64>>,
// }

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

pub use cholesky::SparseCholeskySolver;
pub use qr::SparseQRSolver;

#[cfg(test)]
mod integration_tests;
