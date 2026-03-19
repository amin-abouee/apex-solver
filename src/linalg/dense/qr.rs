use faer::{Mat, linalg::solvers::Solve};

use crate::linalg::{DenseMode, LinAlgResult, LinearSolver};

/// Dense QR (column-pivoting) linear solver for CPU.
///
/// Optimal for small-to-medium problems (< 500 DOF) where the Hessian may be
/// nearly rank-deficient. More robust than Cholesky for ill-conditioned systems
/// because QR decomposition never fails on singular or near-singular matrices.
#[derive(Debug, Clone)]
pub struct DenseQRSolver {
    /// Cached column-pivoting QR factorization of the Hessian (or augmented Hessian)
    factorizer: Option<faer::linalg::solvers::ColPivQr<f64>>,

    /// Dense Hessian H = J^T · J (un-augmented, for covariance)
    hessian: Option<Mat<f64>>,

    /// Dense gradient g = J^T · r
    gradient: Option<Mat<f64>>,

    /// The parameter covariance matrix (H^{-1}), computed lazily
    covariance_matrix: Option<Mat<f64>>,

    /// Asymptotic standard errors sqrt(diag(H^{-1})), computed lazily
    standard_errors: Option<Mat<f64>>,
}

impl DenseQRSolver {
    pub fn new() -> Self {
        Self {
            factorizer: None,
            hessian: None,
            gradient: None,
            covariance_matrix: None,
            standard_errors: None,
        }
    }

    pub fn hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    /// Compute and cache standard errors as sqrt of covariance diagonal.
    pub fn compute_standard_errors(&mut self) -> Option<&Mat<f64>> {
        if self.covariance_matrix.is_none() {
            LinearSolver::<DenseMode>::compute_covariance_matrix(self);
        }

        let n = self.hessian.as_ref()?.ncols();
        if let Some(cov) = &self.covariance_matrix {
            let mut std_errors = Mat::zeros(n, 1);
            for i in 0..n {
                let diag_val = cov[(i, i)];
                if diag_val >= 0.0 {
                    std_errors[(i, 0)] = diag_val.sqrt();
                } else {
                    return None;
                }
            }
            self.standard_errors = Some(std_errors);
        }
        self.standard_errors.as_ref()
    }

    /// Reset covariance computation state (useful for iterative optimization).
    pub fn reset_covariance(&mut self) {
        self.covariance_matrix = None;
        self.standard_errors = None;
    }

    /// Solve with dense Jacobian directly (the core dense QR implementation).
    fn solve_dense_normal(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // H = J^T · J
        let hessian = jacobian.transpose() * jacobian;
        // g = J^T · r
        let gradient = jacobian.transpose() * residuals;

        // Dense column-pivoting QR factorization (never fails, handles rank-deficient cases)
        let qr = hessian.as_ref().col_piv_qr();

        // Solve H · dx = -g
        let dx = qr.solve(-&gradient);

        self.factorizer = Some(qr);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        self.covariance_matrix = None;
        self.standard_errors = None;

        Ok(dx)
    }

    /// Solve with dense Jacobian and LM damping (the core dense QR augmented implementation).
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

        // QR factorization on augmented system
        let qr = augmented.as_ref().col_piv_qr();

        // Solve H_aug · dx = -g
        let dx = qr.solve(-&gradient);

        // Cache the un-augmented Hessian (DogLeg/LM need the true quadratic model)
        self.factorizer = Some(qr);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        self.covariance_matrix = None;
        self.standard_errors = None;

        Ok(dx)
    }
}

impl Default for DenseQRSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LinearSolver<DenseMode>
// ============================================================================

impl LinearSolver<DenseMode> for DenseQRSolver {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_dense_qr_solver_creation() {
        let solver = DenseQRSolver::new();
        assert!(solver.factorizer.is_none());
        assert!(solver.hessian.is_none());
        assert!(solver.gradient.is_none());

        let default_solver = DenseQRSolver::default();
        assert!(default_solver.factorizer.is_none());
    }

    #[test]
    fn test_dense_qr_solve_normal_equation() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        let dx = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        // Verify: J^T·J·dx ≈ -J^T·r
        let jtj = j.transpose() * &j;
        let jtr = j.transpose() * &r;
        let residual = &jtj * &dx + &jtr;

        for i in 0..dx.nrows() {
            assert!(
                residual[(i, 0)].abs() < TOLERANCE,
                "Residual at index {i}: {}",
                residual[(i, 0)]
            );
        }

        assert!(solver.hessian.is_some());
        assert!(solver.gradient.is_some());
        assert!(solver.factorizer.is_some());

        Ok(())
    }

    #[test]
    fn test_dense_qr_solve_augmented_equation() -> TestResult {
        let (j, r) = create_test_data();
        let lambda = 0.1;
        let mut solver = DenseQRSolver::new();

        let dx = LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, lambda)?;

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
                "Residual at index {i}: {}",
                residual[(i, 0)]
            );
        }

        Ok(())
    }

    #[test]
    fn test_dense_qr_augmented_different_lambdas() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        let dx1 = LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, 0.01)?;
        let dx2 = LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, 1.0)?;

        let mut different = false;
        for i in 0..dx1.nrows() {
            if (dx1[(i, 0)] - dx2[(i, 0)]).abs() > TOLERANCE {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Solutions should differ with different lambda values"
        );

        Ok(())
    }

    #[test]
    fn test_dense_qr_rank_deficient_matrix() -> TestResult {
        let mut solver = DenseQRSolver::new();

        // Rank-deficient Jacobian (3×3, rank 2): second row = 2 × first row
        let mut j = Mat::zeros(3, 3);
        j[(0, 0)] = 1.0;
        j[(0, 1)] = 2.0;
        j[(0, 2)] = 3.0;
        j[(1, 0)] = 2.0;
        j[(1, 1)] = 4.0;
        j[(1, 2)] = 6.0;
        j[(2, 0)] = 0.0;
        j[(2, 1)] = 0.0;
        j[(2, 2)] = 1.0;

        let mut r = Mat::zeros(3, 1);
        r[(0, 0)] = 1.0;
        r[(1, 0)] = 2.0;
        r[(2, 0)] = 3.0;

        let result = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j);
        assert!(result.is_ok(), "QR should handle rank-deficient matrices");

        Ok(())
    }

    #[test]
    fn test_dense_qr_numerical_accuracy() -> TestResult {
        let mut solver = DenseQRSolver::new();

        // Identity system: I * x = -b → solution should be b
        let mut j = Mat::zeros(3, 3);
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j[(2, 2)] = 1.0;

        let mut r = Mat::zeros(3, 1);
        r[(0, 0)] = -1.0;
        r[(1, 0)] = -2.0;
        r[(2, 0)] = -3.0;

        let dx = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        for i in 0..3 {
            let expected = (i + 1) as f64;
            assert!(
                (dx[(i, 0)] - expected).abs() < TOLERANCE,
                "Expected {expected}, got {}",
                dx[(i, 0)]
            );
        }

        Ok(())
    }

    #[test]
    fn test_dense_qr_solver_clone() {
        let solver1 = DenseQRSolver::new();
        let solver2 = solver1.clone();

        assert!(solver1.factorizer.is_none());
        assert!(solver2.factorizer.is_none());
    }

    #[test]
    fn test_dense_qr_zero_lambda_augmented() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        let normal_dx = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;
        let augmented_dx =
            LinearSolver::<DenseMode>::solve_augmented_equation(&mut solver, &r, &j, 0.0)?;

        for i in 0..normal_dx.nrows() {
            assert!(
                (normal_dx[(i, 0)] - augmented_dx[(i, 0)]).abs() < 1e-8,
                "Zero-lambda augmented should match normal equation"
            );
        }

        Ok(())
    }

    #[test]
    fn test_dense_qr_covariance_computation() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        let cov = LinearSolver::<DenseMode>::compute_covariance_matrix(&mut solver)
            .ok_or("covariance should be computable")?;
        let n = cov.nrows();

        // Symmetry
        for i in 0..n {
            for k in 0..n {
                assert!(
                    (cov[(i, k)] - cov[(k, i)]).abs() < TOLERANCE,
                    "Covariance not symmetric at ({i}, {k})"
                );
            }
        }

        // Positive diagonal
        for i in 0..n {
            assert!(cov[(i, i)] > 0.0, "Diagonal entry {i} should be positive");
        }

        Ok(())
    }

    #[test]
    fn test_dense_qr_standard_errors_computation() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        // clone to release the borrow on `solver` so we can access covariance_matrix next
        let errors = solver
            .compute_standard_errors()
            .ok_or("standard errors should be computable")?
            .clone();
        let cov = solver
            .covariance_matrix
            .as_ref()
            .ok_or("covariance matrix not available")?;

        assert_eq!(errors.nrows(), cov.nrows());
        assert_eq!(errors.ncols(), 1);

        for i in 0..errors.nrows() {
            assert!(
                errors[(i, 0)] > 0.0,
                "Standard error at {i} should be positive"
            );
            let expected = cov[(i, i)].sqrt();
            assert!(
                (errors[(i, 0)] - expected).abs() < TOLERANCE,
                "Standard error should equal sqrt of covariance diagonal"
            );
        }

        Ok(())
    }

    #[test]
    fn test_dense_qr_covariance_well_conditioned() -> TestResult {
        let mut solver = DenseQRSolver::new();

        // J = diag(2, 3) → H = diag(4, 9) → H^{-1} = diag(0.25, 1/9)
        let mut j = Mat::zeros(2, 2);
        j[(0, 0)] = 2.0;
        j[(1, 1)] = 3.0;

        let mut r = Mat::zeros(2, 1);
        r[(0, 0)] = 1.0;
        r[(1, 0)] = 2.0;

        LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        let cov = LinearSolver::<DenseMode>::compute_covariance_matrix(&mut solver)
            .ok_or("covariance computation failed")?;
        assert!(
            (cov[(0, 0)] - 0.25).abs() < TOLERANCE,
            "cov[0,0] should be 0.25"
        );
        assert!(
            (cov[(1, 1)] - 1.0 / 9.0).abs() < TOLERANCE,
            "cov[1,1] should be 1/9"
        );
        assert!(cov[(0, 1)].abs() < TOLERANCE);
        assert!(cov[(1, 0)].abs() < TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_dense_qr_covariance_caching() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        LinearSolver::<DenseMode>::compute_covariance_matrix(&mut solver);
        let ptr1 = solver
            .covariance_matrix
            .as_ref()
            .ok_or("covariance not cached after first call")?
            .as_ptr();

        // Second call should return cached result (same pointer)
        LinearSolver::<DenseMode>::compute_covariance_matrix(&mut solver);
        let ptr2 = solver
            .covariance_matrix
            .as_ref()
            .ok_or("covariance not cached after second call")?
            .as_ptr();

        assert_eq!(ptr1, ptr2, "Covariance matrix should be cached");

        Ok(())
    }

    #[test]
    fn test_dense_qr_covariance_singular_system() -> TestResult {
        let mut solver = DenseQRSolver::new();

        // Singular Jacobian: second row = 2 × first row
        let mut j = Mat::zeros(2, 2);
        j[(0, 0)] = 1.0;
        j[(0, 1)] = 2.0;
        j[(1, 0)] = 2.0;
        j[(1, 1)] = 4.0;

        let mut r = Mat::zeros(2, 1);
        r[(0, 0)] = 0.0;
        r[(1, 0)] = 1.0;

        let result = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j);
        if result.is_ok() {
            let cov = LinearSolver::<DenseMode>::compute_covariance_matrix(&mut solver);
            if let Some(cov) = cov {
                assert_eq!(cov.nrows(), 2);
                assert_eq!(cov.ncols(), 2);
            }
        }

        Ok(())
    }

    #[test]
    fn test_dense_qr_reset_covariance() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;
        LinearSolver::<DenseMode>::compute_covariance_matrix(&mut solver);
        assert!(solver.covariance_matrix.is_some());

        solver.reset_covariance();
        assert!(solver.covariance_matrix.is_none());
        assert!(solver.standard_errors.is_none());

        Ok(())
    }
}
