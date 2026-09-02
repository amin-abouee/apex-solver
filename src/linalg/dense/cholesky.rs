use faer::{
    Mat, Side,
    linalg::solvers::{LltError, Solve},
};

use crate::error::ErrorLogging;
use crate::linalg::{Damping, DenseMode, LinAlgError, LinAlgResult, LinearSolver};

/// Dense Cholesky (LLT) linear solver for CPU.
///
/// Optimal for small-to-medium problems (< 500 DOF) where the Hessian is
/// moderately dense. Avoids sparse data structure overhead and benefits
/// from dense BLAS routines.
#[derive(Debug, Clone)]
pub struct DenseCholeskySolver {
    /// Dense Hessian H = J^T · J
    hessian: Option<Mat<f64>>,

    /// Dense gradient g = J^T · r
    gradient: Option<Mat<f64>>,
}

impl DenseCholeskySolver {
    pub fn new() -> Self {
        Self {
            hessian: None,
            gradient: None,
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

        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

        Ok(dx)
    }

    /// Solve with dense Jacobian and damping (the core dense augmented implementation).
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

        // Dense Cholesky factorization on augmented system
        let llt = augmented
            .as_ref()
            .llt(Side::Lower)
            .map_err(|e| map_llt_error(e, "Augmented dense Cholesky factorization failed"))?;

        // Solve H_aug · dx = -g
        let dx = llt.solve(-&gradient);

        // Cache the un-augmented Hessian (DogLeg/LM need the true quadratic model)
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

        Ok(dx)
    }
}

impl Default for DenseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LinearSolver<DenseMode>
// ============================================================================

impl LinearSolver<DenseMode> for DenseCholeskySolver {
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

/// Map faer's LLT error to our LinAlgError
fn map_llt_error(e: LltError, context: &str) -> LinAlgError {
    LinAlgError::FactorizationFailed(format!("{context}: {e:?}")).log()
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
    fn test_dense_cholesky_solve_normal() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseCholeskySolver::new();

        let dx = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

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
                "Residual at index {}: {}",
                i,
                residual[(i, 0)]
            );
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // New tests for previously uncovered code paths
    // -------------------------------------------------------------------------

    /// get_hessian() and get_gradient() should return None before any solve.
    #[test]
    fn test_accessors_before_solve() {
        let solver = DenseCholeskySolver::new();
        assert!(
            LinearSolver::<DenseMode>::get_hessian(&solver).is_none(),
            "hessian should be None before solve"
        );
        assert!(
            LinearSolver::<DenseMode>::get_gradient(&solver).is_none(),
            "gradient should be None before solve"
        );
    }

    /// DenseCholeskySolver::default() should behave identically to new().
    #[test]
    fn test_default_equals_new() {
        let solver = DenseCholeskySolver::default();
        assert!(solver.hessian.is_none());
        assert!(solver.gradient.is_none());
    }

    /// A rank-deficient (singular) Jacobian should return an Err from solve_normal_equation.
    #[test]
    fn test_singular_jacobian_returns_error() {
        let mut solver = DenseCholeskySolver::new();

        // Jacobian with an all-zero column → H = J^T J is singular
        let mut j = Mat::zeros(3, 2);
        j[(0, 0)] = 1.0;
        j[(1, 0)] = 2.0;
        j[(2, 0)] = 3.0;
        // column 1 is all zeros

        let mut r = Mat::zeros(3, 1);
        r[(0, 0)] = 1.0;

        let result = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j);
        assert!(
            result.is_err(),
            "Singular Jacobian must produce a factorization error"
        );
    }

    /// solve_augmented_equation with lambda=0 should produce the same result as solve_normal_equation.
    #[test]
    fn test_augmented_with_zero_lambda_matches_normal() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver_n = DenseCholeskySolver::new();
        let mut solver_a = DenseCholeskySolver::new();

        let dx_normal = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver_n, &r, &j)?;
        let dx_augmented = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver_a,
            &r,
            &j,
            &Damping::identity(0.0),
        )?;

        for i in 0..dx_normal.nrows() {
            assert!(
                (dx_normal[(i, 0)] - dx_augmented[(i, 0)]).abs() < TOLERANCE,
                "Element {} differs: normal={}, augmented(λ=0)={}",
                i,
                dx_normal[(i, 0)],
                dx_augmented[(i, 0)]
            );
        }
        Ok(())
    }
}
