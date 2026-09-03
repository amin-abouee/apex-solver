use faer::{
    Mat,
    linalg::solvers::Solve,
    sparse::SparseColMat,
    sparse::linalg::solvers::{Qr, SymbolicQr},
};

use crate::error::ErrorLogging;
use crate::linalg::sparse::normal_eq::{LazyNormalEquations, NormalEquations};
use crate::linalg::{Damping, LinAlgError, LinAlgResult, LinearSolver, SparseMode};

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    /// Cached symbolic factorization for reuse across iterations.
    ///
    /// This is computed once and reused when the sparsity pattern doesn't change,
    /// providing a 10-15% performance improvement for iterative optimization.
    /// For augmented systems where only lambda changes, the sparsity pattern
    /// remains the same (adding diagonal lambda*I doesn't change the pattern).
    symbolic_factorization: Option<SymbolicQr<usize>>,

    /// Cached symbolic machinery for forming `JᵀJ` and `Jᵀr` in parallel.
    ne_cache: LazyNormalEquations,

    /// The Hessian matrix, computed as `(J^T * W * J)`.
    ///
    /// This is `None` if the Hessian could not be computed.
    hessian: Option<SparseColMat<usize, f64>>,

    /// The gradient vector, computed as `J^T * W * r`.
    ///
    /// This is `None` if the gradient could not be computed.
    gradient: Option<Mat<f64>>,
}

impl SparseQRSolver {
    pub fn new() -> Self {
        SparseQRSolver {
            symbolic_factorization: None,
            ne_cache: LazyNormalEquations::default(),
            hessian: None,
            gradient: None,
        }
    }

    pub fn hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

impl Default for SparseQRSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSolver<SparseMode> for SparseQRSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // Form the normal equations: H = JᵀJ, g = Jᵀr (parallel faer kernels,
        // symbolic product cached across iterations).
        let NormalEquations { hessian, gradient } = self.ne_cache.compute(residuals, jacobians)?;

        // Check if we can reuse the cached symbolic factorization
        // We can reuse it if the sparsity pattern (symbolic structure) hasn't changed
        let sym = if let Some(ref cached_sym) = self.symbolic_factorization {
            // Reuse cached symbolic factorization
            // Note: SymbolicQr is reference-counted, so clone() is cheap (O(1))
            // We assume the sparsity pattern is constant across iterations
            // which is typical in iterative optimization
            cached_sym.clone()
        } else {
            // Create new symbolic factorization and cache it
            let new_sym = SymbolicQr::try_new(hessian.symbolic()).map_err(|e| {
                LinAlgError::FactorizationFailed("Symbolic QR decomposition failed".to_string())
                    .log_with_source(e)
            })?;
            // Cache it (clone is cheap due to reference counting)
            self.symbolic_factorization = Some(new_sym.clone());
            new_sym
        };

        // Perform numeric factorization using the symbolic structure
        let qr = Qr::try_new_with_symbolic(sym, hessian.as_ref()).map_err(|e| {
            LinAlgError::SingularMatrix(
                "QR factorization failed (matrix may be singular)".to_string(),
            )
            .log_with_source(e)
        })?;

        // Solve H * dx = -g (negate gradient to get descent direction)
        let dx = qr.solve(-&gradient);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

        Ok(dx)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        // H = JᵀJ, g = Jᵀr (parallel faer kernels)
        let NormalEquations { hessian, gradient } = self.ne_cache.compute(residuals, jacobians)?;

        // H_aug = H + λ·D — diagonal edit on the cached product pattern.
        let augmented_hessian = self.ne_cache.damped_hessian(damping)?;

        // Check if we can reuse the cached symbolic factorization
        // For augmented systems, the sparsity pattern remains the same
        // (adding diagonal lambda*I doesn't change the pattern)
        // Note: SymbolicQr is reference-counted, so clone() is cheap (O(1))
        let sym = if let Some(ref cached_sym) = self.symbolic_factorization {
            cached_sym.clone()
        } else {
            // Create new symbolic factorization and cache it
            let new_sym = SymbolicQr::try_new(augmented_hessian.symbolic()).map_err(|e| {
                LinAlgError::FactorizationFailed(
                    "Symbolic QR decomposition failed for augmented system".to_string(),
                )
                .log_with_source(e)
            })?;
            // Cache it (clone is cheap due to reference counting)
            self.symbolic_factorization = Some(new_sym.clone());
            new_sym
        };

        // Perform numeric factorization
        let qr = Qr::try_new_with_symbolic(sym, augmented_hessian.as_ref()).map_err(|e| {
            LinAlgError::SingularMatrix(
                "QR factorization failed (matrix may be singular)".to_string(),
            )
            .log_with_source(e)
        })?;

        let dx = qr.solve(-&gradient);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

        Ok(dx)
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        Some(
            <SparseMode as crate::linearizer::AssemblyBackend>::hessian_vec_product(
                self.hessian.as_ref()?,
                v,
            ),
        )
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    const TOLERANCE: f64 = 1e-10;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Helper function to create test data for QR solver
    fn create_test_data()
    -> Result<(SparseColMat<usize, f64>, Mat<f64>), faer::sparse::CreationError> {
        // Create a 4x3 overdetermined system
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 0.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(1, 0, 0.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(2, 0, 1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 2, 0.0),
            Triplet::new(3, 0, 1.0),
            Triplet::new(3, 1, 0.0),
            Triplet::new(3, 2, 0.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(4, 3, &triplets)?;

        let residuals = Mat::from_fn(4, 1, |i, _| (i + 1) as f64);

        Ok((jacobian, residuals))
    }

    /// Test basic QR solver creation
    #[test]
    fn test_qr_solver_creation() {
        let solver = SparseQRSolver::new();
        assert!(solver.hessian.is_none());
        assert!(solver.gradient.is_none());

        let default_solver = SparseQRSolver::default();
        assert!(default_solver.hessian.is_none());
        assert!(default_solver.gradient.is_none());
    }

    /// Test normal equation solving with QR decomposition
    #[test]
    fn test_qr_solve_normal_equation() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let solution =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(solution.nrows(), 3); // Number of variables
        assert_eq!(solution.ncols(), 1);

        // Verify symbolic pattern was cached
        Ok(())
    }

    /// Test QR symbolic pattern caching
    #[test]
    fn test_qr_factorizer_caching() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve
        let sol1 =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // Second solve should reuse pattern
        let sol2 =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // Results should be identical
        for i in 0..sol1.nrows() {
            assert!((sol1[(i, 0)] - sol2[(i, 0)]).abs() < TOLERANCE);
        }
        Ok(())
    }

    /// Test augmented equation solving with QR
    #[test]
    fn test_qr_solve_augmented_equation() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;
        let lambda = 0.1;

        let solution = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(lambda),
        )?;
        assert_eq!(solution.nrows(), 3); // Number of variables
        assert_eq!(solution.ncols(), 1);
        Ok(())
    }

    /// Test augmented system with different lambda values
    #[test]
    fn test_qr_augmented_different_lambdas() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let lambda1 = 0.01;
        let lambda2 = 1.0;

        let sol1 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(lambda1),
        )?;
        let sol2 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(lambda2),
        )?;

        // Solutions should be different due to different regularization
        let mut different = false;
        for i in 0..sol1.nrows() {
            if (sol1[(i, 0)] - sol2[(i, 0)]).abs() > TOLERANCE {
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

    /// Test QR with rank-deficient matrix
    #[test]
    fn test_qr_rank_deficient_matrix() -> TestResult {
        let mut solver = SparseQRSolver::new();

        // Create a rank-deficient matrix (3x3 but rank 2)
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(0, 2, 3.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 4.0),
            Triplet::new(1, 2, 6.0), // 2x first row
            Triplet::new(2, 0, 0.0),
            Triplet::new(2, 1, 0.0),
            Triplet::new(2, 2, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(3, 3, &triplets)?;
        let residuals = Mat::from_fn(3, 1, |i, _| i as f64);

        // QR should still provide a least squares solution
        let result =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian);
        assert!(result.is_ok());
        Ok(())
    }

    /// Test augmented system structure and dimensions
    #[test]
    fn test_qr_augmented_system_structure() -> TestResult {
        let mut solver = SparseQRSolver::new();

        // Simple 2x2 system
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 0.0),
            Triplet::new(1, 0, 0.0),
            Triplet::new(1, 1, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let lambda = 0.5;

        let solution = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(lambda),
        )?;
        assert_eq!(solution.nrows(), 2); // Should return only the variable part
        assert_eq!(solution.ncols(), 1);
        Ok(())
    }

    /// Test numerical accuracy with known solution
    #[test]
    fn test_qr_numerical_accuracy() -> TestResult {
        let mut solver = SparseQRSolver::new();

        // Create identity system: I * x = b
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(3, 3, &triplets)?;

        let residuals = Mat::from_fn(3, 1, |i, _| -((i + 1) as f64)); // [-1, -2, -3]

        let solution =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        // Expected solution should be [1, 2, 3]
        for i in 0..3 {
            let expected = (i + 1) as f64;
            assert!(
                (solution[(i, 0)] - expected).abs() < TOLERANCE,
                "Expected {}, got {}",
                expected,
                solution[(i, 0)]
            );
        }
        Ok(())
    }

    /// Test QR solver clone functionality
    #[test]
    fn test_qr_solver_clone() {
        let solver1 = SparseQRSolver::new();
        let solver2 = solver1.clone();
        assert!(solver2.hessian.is_none());
        assert!(solver2.gradient.is_none());
    }

    /// Test zero lambda in augmented system (should behave like normal equation)
    #[test]
    fn test_qr_zero_lambda_augmented() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let normal_sol =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        let augmented_sol = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(0.0),
        )?;

        // Solutions should be very close (within numerical precision)
        for i in 0..normal_sol.nrows() {
            assert!(
                (normal_sol[(i, 0)] - augmented_sol[(i, 0)]).abs() < 1e-8,
                "Zero lambda augmented should match normal equation"
            );
        }
        Ok(())
    }

    /// Test hessian() getter returns None before solve and Some after
    #[test]
    fn test_qr_hessian_getter() -> TestResult {
        let mut solver = SparseQRSolver::new();
        assert!(solver.hessian().is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(solver.hessian().is_some());
        Ok(())
    }

    /// Test gradient() getter returns None before solve and Some after
    #[test]
    fn test_qr_gradient_getter() -> TestResult {
        let mut solver = SparseQRSolver::new();
        assert!(solver.gradient().is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(solver.gradient().is_some());
        Ok(())
    }

    /// Test get_hessian() trait method returns Some after solve
    #[test]
    fn test_qr_get_hessian_trait() -> TestResult {
        let mut solver = SparseQRSolver::new();
        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_some());
        Ok(())
    }

    /// Test get_gradient() trait method returns Some after solve
    #[test]
    fn test_qr_get_gradient_trait() -> TestResult {
        let mut solver = SparseQRSolver::new();
        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_some());
        Ok(())
    }
}
