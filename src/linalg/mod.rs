#[derive(Default, Clone)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

pub trait SparseLinearSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;

    fn solve_weighted(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>>;

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;

    fn solve_jtj_weighted(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>>;
}
