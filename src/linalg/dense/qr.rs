use faer::{Mat, linalg::solvers::Solve};

use crate::linalg::{Damping, DenseMode, LinAlgResult, LinearSolver};

/// Dense QR (column-pivoting) linear solver for CPU.
///
/// Optimal for small-to-medium problems (< 500 DOF) where the Hessian may be
/// nearly rank-deficient. More robust than Cholesky for ill-conditioned systems
/// because QR decomposition never fails on singular or near-singular matrices.
#[derive(Debug, Clone)]
pub struct DenseQRSolver {
    /// Dense Hessian H = J^T · J (un-augmented, for covariance)
    hessian: Option<Mat<f64>>,

    /// Dense gradient g = J^T · r
    gradient: Option<Mat<f64>>,
}

impl DenseQRSolver {
    pub fn new() -> Self {
        Self {
            hessian: None,
            gradient: None,
        }
    }

    pub fn hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
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

        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

        Ok(dx)
    }

    /// Solve with dense Jacobian and LM damping (the core dense QR augmented implementation).
    fn solve_dense_augmented(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        // H = J^T · J
        let hessian = jacobian.transpose() * jacobian;
        // g = J^T · r
        let gradient = jacobian.transpose() * residuals;

        // H_aug = H + λ·D, D_jj = clamp(H_jj, min_diagonal, max_diagonal)
        let n = hessian.nrows();
        let mut augmented = hessian.clone();
        for i in 0..n {
            augmented[(i, i)] += damping.diagonal_term(hessian[(i, i)]);
        }

        // QR factorization on augmented system
        let qr = augmented.as_ref().col_piv_qr();

        // Solve H_aug · dx = -g
        let dx = qr.solve(-&gradient);

        // Cache the un-augmented Hessian (DogLeg/LM need the true quadratic model)
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

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
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_dense_augmented(residuals, jacobian, damping)
    }

    fn get_hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
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
        assert!(solver.hessian.is_none());
        assert!(solver.gradient.is_none());

        let default_solver = DenseQRSolver::default();
        assert!(default_solver.hessian.is_none());
        assert!(default_solver.gradient.is_none());
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

        Ok(())
    }

    #[test]
    fn test_dense_qr_solve_augmented_equation() -> TestResult {
        let (j, r) = create_test_data();
        let lambda = 0.1;
        let mut solver = DenseQRSolver::new();

        let dx = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &Damping::identity(lambda),
        )?;

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

        let dx1 = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &Damping::identity(0.01),
        )?;
        let dx2 = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &Damping::identity(1.0),
        )?;

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
        assert!(solver2.hessian.is_none());
        assert!(solver2.gradient.is_none());
    }

    #[test]
    fn test_dense_qr_zero_lambda_augmented() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseQRSolver::new();

        let normal_dx = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;
        let augmented_dx = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &Damping::identity(0.0),
        )?;

        for i in 0..normal_dx.nrows() {
            assert!(
                (normal_dx[(i, 0)] - augmented_dx[(i, 0)]).abs() < 1e-8,
                "Zero-lambda augmented should match normal equation"
            );
        }

        Ok(())
    }
}
