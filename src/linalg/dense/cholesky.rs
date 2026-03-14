use faer::{
    Mat, Side,
    linalg::solvers::{Llt, LltError, Solve},
    sparse::SparseColMat,
};

use crate::linalg::{
    DenseMode, LinAlgError, LinAlgResult, LinAlgSolver, LinearSolver, SparseLinearSolver,
    utils::sparse_to_dense,
};

/// Dense Cholesky (LLT) linear solver for CPU.
///
/// Optimal for small-to-medium problems (< 500 DOF) where the Hessian is
/// moderately dense. Avoids sparse data structure overhead and benefits
/// from dense BLAS routines.
///
/// When used through the `SparseLinearSolver` trait (the optimizer bridge),
/// this solver accepts sparse Jacobians and converts them to dense internally.
#[derive(Debug, Clone)]
pub struct DenseCholeskySolver {
    /// Dense Hessian H = J^T · J
    hessian: Option<Mat<f64>>,

    /// Dense gradient g = J^T · r
    gradient: Option<Mat<f64>>,

    /// Cached dense Cholesky factorization for covariance computation
    factorizer: Option<Llt<f64>>,

    /// The parameter covariance matrix (H^{-1}), computed lazily
    covariance_matrix: Option<Mat<f64>>,

    /// Cached sparse Hessian for `get_hessian()` compatibility
    hessian_sparse_cache: Option<SparseColMat<usize, f64>>,
}

impl DenseCholeskySolver {
    pub fn new() -> Self {
        Self {
            hessian: None,
            gradient: None,
            factorizer: None,
            covariance_matrix: None,
            hessian_sparse_cache: None,
        }
    }

    /// Solve with dense Jacobian directly (the core dense implementation).
    fn solve_dense_normal(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // H = J^T · J
        let hessian = jacobian.transpose() * jacobian;
        // g = J^T · r
        let gradient = jacobian.transpose() * residuals;

        // Dense Cholesky factorization
        let llt = hessian
            .as_ref()
            .llt(Side::Lower)
            .map_err(|e| map_llt_error(e, "Dense Cholesky factorization failed"))?;

        // Solve H · dx = -g
        let dx = llt.solve(-&gradient);

        self.factorizer = Some(llt);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        // Invalidate caches
        self.hessian_sparse_cache = None;
        self.covariance_matrix = None;

        Ok(dx)
    }

    /// Solve with dense Jacobian and damping (the core dense augmented implementation).
    fn solve_dense_augmented(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        // H = J^T · J
        let hessian = jacobian.transpose() * jacobian;
        // g = J^T · r
        let gradient = jacobian.transpose() * residuals;

        // H_aug = H + λI
        let n = hessian.nrows();
        let mut augmented = hessian.clone();
        for i in 0..n {
            augmented[(i, i)] += lambda;
        }

        // Dense Cholesky factorization on augmented system
        let llt = augmented
            .as_ref()
            .llt(Side::Lower)
            .map_err(|e| map_llt_error(e, "Augmented dense Cholesky factorization failed"))?;

        // Solve H_aug · dx = -g
        let dx = llt.solve(-&gradient);

        // Cache the un-augmented Hessian (DogLeg/LM need the true quadratic model)
        self.factorizer = Some(llt);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        // Invalidate caches
        self.hessian_sparse_cache = None;
        self.covariance_matrix = None;

        Ok(dx)
    }
}

impl Default for DenseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Generic LinearSolver implementation (dense types)
// ============================================================================

impl LinearSolver for DenseCholeskySolver {
    type Vector = Mat<f64>;
    type Jacobian = Mat<f64>;
    type Hessian = Mat<f64>;
    type Error = LinAlgError;

    fn solve_normal_equation(
        &mut self,
        residuals: &Self::Vector,
        jacobian: &Self::Jacobian,
    ) -> Result<Self::Vector, Self::Error> {
        self.solve_dense_normal(residuals, jacobian)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Self::Vector,
        jacobian: &Self::Jacobian,
        lambda: f64,
    ) -> Result<Self::Vector, Self::Error> {
        self.solve_dense_augmented(residuals, jacobian, lambda)
    }

    fn gradient(&self) -> Option<&Self::Vector> {
        self.gradient.as_ref()
    }

    fn hessian(&self) -> Option<&Self::Hessian> {
        self.hessian.as_ref()
    }
}

// ============================================================================
// LinAlgSolver<DenseMode> (native dense path, no conversions)
// ============================================================================

impl LinAlgSolver<DenseMode> for DenseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_dense_normal(residuals, jacobian)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_dense_augmented(residuals, jacobian, lambda)
    }

    fn get_hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        if self.covariance_matrix.is_none()
            && let Some(hessian) = &self.hessian
            && let Some(factorizer) = &self.factorizer
        {
            let n = hessian.nrows();
            let identity = Mat::identity(n, n);
            let cov = factorizer.solve(&identity);
            self.covariance_matrix = Some(cov);
        }
        self.covariance_matrix.as_ref()
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        self.covariance_matrix.as_ref()
    }
}

// ============================================================================
// SparseLinearSolver bridge (accepts sparse input, converts internally)
// ============================================================================

impl SparseLinearSolver for DenseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let j_dense = sparse_to_dense(jacobians);
        self.solve_dense_normal(residuals, &j_dense)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        let j_dense = sparse_to_dense(jacobians);
        self.solve_dense_augmented(residuals, &j_dense, lambda)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian_sparse_cache.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        if self.covariance_matrix.is_none()
            && let Some(hessian) = &self.hessian
            && let Some(factorizer) = &self.factorizer
        {
            let n = hessian.nrows();
            let identity = Mat::identity(n, n);
            let cov = factorizer.solve(&identity);
            self.covariance_matrix = Some(cov);
        }
        self.covariance_matrix.as_ref()
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        self.covariance_matrix.as_ref()
    }
}

/// Map faer's LLT error to our LinAlgError
fn map_llt_error(e: LltError, context: &str) -> LinAlgError {
    LinAlgError::FactorizationFailed(format!("{context}: {e:?}")).log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};

    const TOLERANCE: f64 = 1e-10;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn create_test_data() -> (Mat<f64>, Mat<f64>) {
        // 4×2 Jacobian (overdetermined)
        let mut j = Mat::zeros(4, 2);
        j[(0, 0)] = 2.0;
        j[(0, 1)] = 1.0;
        j[(1, 0)] = 1.0;
        j[(1, 1)] = 3.0;
        j[(2, 0)] = 1.0;
        j[(2, 1)] = 1.0;
        j[(3, 0)] = 0.5;
        j[(3, 1)] = 2.0;

        let mut r = Mat::zeros(4, 1);
        r[(0, 0)] = 1.0;
        r[(1, 0)] = 2.0;
        r[(2, 0)] = 0.5;
        r[(3, 0)] = 1.5;

        (j, r)
    }

    fn create_sparse_test_data() -> (SparseColMat<usize, f64>, Mat<f64>) {
        let (j_dense, r) = create_test_data();
        let nrows = j_dense.nrows();
        let ncols = j_dense.ncols();

        let mut triplets = Vec::new();
        for col in 0..ncols {
            for row in 0..nrows {
                let val = j_dense[(row, col)];
                if val.abs() > 1e-15 {
                    triplets.push(Triplet::new(row, col, val));
                }
            }
        }
        let j_sparse = SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).unwrap();
        (j_sparse, r)
    }

    #[test]
    fn test_dense_cholesky_solve_normal() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseCholeskySolver::new();

        let dx = LinearSolver::solve_normal_equation(&mut solver, &r, &j)?;

        // Verify: J^T·J·dx ≈ -J^T·r
        let jtj = j.transpose() * &j;
        let jtr = j.transpose() * &r;
        let residual = &jtj * &dx + &jtr;

        for i in 0..dx.nrows() {
            assert!(
                residual[(i, 0)].abs() < TOLERANCE,
                "Residual at index {}: {}",
                i,
                residual[(i, 0)]
            );
        }

        assert!(solver.hessian.is_some());
        assert!(solver.gradient.is_some());

        Ok(())
    }

    #[test]
    fn test_dense_cholesky_solve_augmented() -> TestResult {
        let (j, r) = create_test_data();
        let lambda = 0.1;
        let mut solver = DenseCholeskySolver::new();

        let dx = LinearSolver::solve_augmented_equation(&mut solver, &r, &j, lambda)?;

        // Verify: (J^T·J + λI)·dx ≈ -J^T·r
        let mut jtj = j.transpose() * &j;
        let jtr = j.transpose() * &r;
        for i in 0..jtj.nrows() {
            jtj[(i, i)] += lambda;
        }
        let residual = &jtj * &dx + &jtr;

        for i in 0..dx.nrows() {
            assert!(
                residual[(i, 0)].abs() < TOLERANCE,
                "Residual at index {}: {}",
                i,
                residual[(i, 0)]
            );
        }

        Ok(())
    }

    #[test]
    fn test_sparse_bridge_matches_dense() -> TestResult {
        let (j_dense, r) = create_test_data();
        let (j_sparse, _) = create_sparse_test_data();

        let mut dense_solver = DenseCholeskySolver::new();
        let mut bridge_solver = DenseCholeskySolver::new();

        let dx_dense = LinearSolver::solve_normal_equation(&mut dense_solver, &r, &j_dense)?;
        let dx_bridge =
            SparseLinearSolver::solve_normal_equation(&mut bridge_solver, &r, &j_sparse)?;

        for i in 0..dx_dense.nrows() {
            assert!(
                (dx_dense[(i, 0)] - dx_bridge[(i, 0)]).abs() < TOLERANCE,
                "Dense vs bridge mismatch at {}: {} vs {}",
                i,
                dx_dense[(i, 0)],
                dx_bridge[(i, 0)]
            );
        }

        Ok(())
    }

    #[test]
    fn test_covariance_computation() -> TestResult {
        let (j_sparse, r) = create_sparse_test_data();
        let mut solver = DenseCholeskySolver::new();

        let _ = SparseLinearSolver::solve_normal_equation(&mut solver, &r, &j_sparse)?;

        let cov = SparseLinearSolver::compute_covariance_matrix(&mut solver);
        assert!(cov.is_some(), "Covariance should be computable");

        let cov = cov.unwrap();
        let n = cov.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < TOLERANCE,
                    "Covariance not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        for i in 0..n {
            assert!(cov[(i, i)] > 0.0, "Diagonal entry {} should be positive", i);
        }

        Ok(())
    }
}
