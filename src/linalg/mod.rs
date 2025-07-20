#[derive(Default, Clone)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

pub trait SparseLinearSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>>;

    fn solve_augmented_equation(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
        lambda: f64,
    ) -> Option<faer::Mat<f64>>;
}
