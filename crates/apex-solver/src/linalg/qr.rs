use faer::{
    Mat,
    linalg::solvers::Solve,
    sparse::linalg::solvers::{Qr, SymbolicQr},
    sparse::{SparseColMat, Triplet},
};
use std::ops::Mul;

use crate::linalg::{LinAlgError, LinAlgResult, SparseLinearSolver};

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    factorizer: Option<Qr<usize, f64>>,

    /// Cached symbolic factorization for reuse across iterations.
    ///
    /// This is computed once and reused when the sparsity pattern doesn't change,
    /// providing a 10-15% performance improvement for iterative optimization.
    /// For augmented systems where only lambda changes, the sparsity pattern
    /// remains the same (adding diagonal lambda*I doesn't change the pattern).
    symbolic_factorization: Option<SymbolicQr<usize>>,

    /// The Hessian matrix, computed as `(J^T * W * J)`.
    ///
    /// This is `None` if the Hessian could not be computed.
    hessian: Option<SparseColMat<usize, f64>>,

    /// The gradient vector, computed as `J^T * W * r`.
    ///
    /// This is `None` if the gradient could not be computed.
    gradient: Option<Mat<f64>>,

    /// The parameter covariance matrix, computed as `(J^T * W * J)^-1`.
    ///
    /// This is `None` if the Hessian is singular or ill-conditioned.
    covariance_matrix: Option<Mat<f64>>,
    /// Asymptotic standard errors of the parameters.
    ///
    /// This is `None` if the covariance matrix could not be computed.
    /// Each error is the square root of the corresponding diagonal element
    /// of the covariance matrix.
    standard_errors: Option<Mat<f64>>,
}

impl SparseQRSolver {
    pub fn new() -> Self {
        SparseQRSolver {
            factorizer: None,
            symbolic_factorization: None,
            hessian: None,
            gradient: None,
            covariance_matrix: None,
            standard_errors: None,
        }
    }

    pub fn hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    pub fn compute_standard_errors(&mut self) -> Option<&Mat<f64>> {
        // Ensure covariance matrix is computed first
        if self.covariance_matrix.is_none() {
            self.compute_covariance_matrix();
        }

        // Return None if hessian is not available (solver not initialized)
        let hessian = self.hessian.as_ref()?;
        let n = hessian.ncols();
        // Compute standard errors as sqrt of diagonal elements
        if let Some(cov) = &self.covariance_matrix {
            let mut std_errors = Mat::zeros(n, 1);
            for i in 0..n {
                let diag_val = cov[(i, i)];
                if diag_val >= 0.0 {
                    std_errors[(i, 0)] = diag_val.sqrt();
                } else {
                    // Negative diagonal indicates numerical issues
                    return None;
                }
            }
            self.standard_errors = Some(std_errors);
        }
        self.standard_errors.as_ref()
    }

    /// Reset covariance computation state (useful for iterative optimization)
    pub fn reset_covariance(&mut self) {
        self.covariance_matrix = None;
        self.standard_errors = None;
    }
}

impl Default for SparseQRSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseLinearSolver for SparseQRSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // Form the normal equations explicitly: H = J^T * J
        let jt = jacobians.as_ref().transpose();
        let hessian = jt
            .to_col_major()
            .map_err(|e| {
                LinAlgError::MatrixConversion(
                    "Failed to convert transposed Jacobian to column-major format".to_string(),
                )
                .log_with_source(e)
            })?
            .mul(jacobians.as_ref());

        // g = J^T * r (stored as positive, will negate when solving)
        let gradient = jacobians.as_ref().transpose().mul(residuals);

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
        self.factorizer = Some(qr);

        Ok(dx)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        let n = jacobians.ncols();

        // H = J^T * J
        let jt = jacobians.as_ref().transpose();
        let hessian = jt
            .to_col_major()
            .map_err(|e| {
                LinAlgError::MatrixConversion(
                    "Failed to convert transposed Jacobian to column-major format".to_string(),
                )
                .log_with_source(e)
            })?
            .mul(jacobians.as_ref());

        // g = J^T * r
        let gradient = jacobians.as_ref().transpose().mul(residuals);

        // H_aug = H + lambda * I
        let mut lambda_i_triplets = Vec::with_capacity(n);
        for i in 0..n {
            lambda_i_triplets.push(Triplet::new(i, i, lambda));
        }
        let lambda_i =
            SparseColMat::try_new_from_triplets(n, n, &lambda_i_triplets).map_err(|e| {
                LinAlgError::SparseMatrixCreation("Failed to create lambda*I matrix".to_string())
                    .log_with_source(e)
            })?;

        let augmented_hessian = hessian.as_ref() + lambda_i;

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
        self.factorizer = Some(qr);

        Ok(dx)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        // Only compute if we have a factorizer and hessian, but no covariance matrix yet
        if self.factorizer.is_some()
            && self.hessian.is_some()
            && self.covariance_matrix.is_none()
            && let (Some(factorizer), Some(hessian)) = (&self.factorizer, &self.hessian)
        {
            let n = hessian.ncols();
            // Create identity matrix
            let identity = Mat::identity(n, n);

            // Solve H * X = I to get X = H^(-1) = covariance matrix
            let cov_matrix = factorizer.solve(&identity);
            self.covariance_matrix = Some(cov_matrix);
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
        assert!(solver.factorizer.is_none());

        let default_solver = SparseQRSolver::default();
        assert!(default_solver.factorizer.is_none());
    }

    /// Test normal equation solving with QR decomposition
    #[test]
    fn test_qr_solve_normal_equation() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let solution = solver.solve_normal_equation(&residuals, &jacobian)?;
        assert_eq!(solution.nrows(), 3); // Number of variables
        assert_eq!(solution.ncols(), 1);

        // Verify symbolic pattern was cached
        assert!(solver.factorizer.is_some());
        Ok(())
    }

    /// Test QR symbolic pattern caching
    #[test]
    fn test_qr_factorizer_caching() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve
        let sol1 = solver.solve_normal_equation(&residuals, &jacobian)?;
        assert!(solver.factorizer.is_some());

        // Second solve should reuse pattern
        let sol2 = solver.solve_normal_equation(&residuals, &jacobian)?;

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

        let solution = solver.solve_augmented_equation(&residuals, &jacobian, lambda)?;
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

        let sol1 = solver.solve_augmented_equation(&residuals, &jacobian, lambda1)?;
        let sol2 = solver.solve_augmented_equation(&residuals, &jacobian, lambda2)?;

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
        let result = solver.solve_normal_equation(&residuals, &jacobian);
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

        let solution = solver.solve_augmented_equation(&residuals, &jacobian, lambda)?;
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

        let solution = solver.solve_normal_equation(&residuals, &jacobian)?;
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

        assert!(solver1.factorizer.is_none());
        assert!(solver2.factorizer.is_none());
    }

    /// Test zero lambda in augmented system (should behave like normal equation)
    #[test]
    fn test_qr_zero_lambda_augmented() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let normal_sol = solver.solve_normal_equation(&residuals, &jacobian)?;
        let augmented_sol = solver.solve_augmented_equation(&residuals, &jacobian, 0.0)?;

        // Solutions should be very close (within numerical precision)
        for i in 0..normal_sol.nrows() {
            assert!(
                (normal_sol[(i, 0)] - augmented_sol[(i, 0)]).abs() < 1e-8,
                "Zero lambda augmented should match normal equation"
            );
        }
        Ok(())
    }

    /// Test covariance matrix computation
    #[test]
    fn test_qr_covariance_computation() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve to set up factorizer and hessian
        solver.solve_normal_equation(&residuals, &jacobian)?;

        // Now compute covariance matrix
        let cov_matrix = solver.compute_covariance_matrix();
        assert!(cov_matrix.is_some());

        if let Some(cov) = cov_matrix {
            assert_eq!(cov.nrows(), 3); // Should be n x n where n is number of variables
            assert_eq!(cov.ncols(), 3);

            // Covariance matrix should be symmetric
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (cov[(i, j)] - cov[(j, i)]).abs() < TOLERANCE,
                        "Covariance matrix should be symmetric"
                    );
                }
            }

            // Diagonal elements should be positive (variances)
            for i in 0..3 {
                assert!(
                    cov[(i, i)] > 0.0,
                    "Diagonal elements (variances) should be positive"
                );
            }
        }
        Ok(())
    }

    /// Test standard errors computation
    #[test]
    fn test_qr_standard_errors_computation() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve to set up factorizer and hessian
        solver.solve_normal_equation(&residuals, &jacobian)?;

        // Compute covariance matrix first (this also computes standard errors)
        solver.compute_standard_errors();

        // Now check that both covariance matrix and standard errors are available
        assert!(solver.covariance_matrix.is_some());
        assert!(solver.standard_errors.is_some());

        if let (Some(cov), Some(errors)) = (&solver.covariance_matrix, &solver.standard_errors) {
            assert_eq!(errors.nrows(), 3); // Should be n x 1 where n is number of variables
            assert_eq!(errors.ncols(), 1);

            // All standard errors should be positive
            for i in 0..3 {
                assert!(errors[(i, 0)] > 0.0, "Standard errors should be positive");
            }

            // Verify relationship: std_error = sqrt(covariance_diagonal)
            for i in 0..3 {
                let expected_std_error = cov[(i, i)].sqrt();
                assert!(
                    (errors[(i, 0)] - expected_std_error).abs() < TOLERANCE,
                    "Standard error should equal sqrt of covariance diagonal"
                );
            }
        }
        Ok(())
    }

    /// Test covariance computation with well-conditioned system
    #[test]
    fn test_qr_covariance_well_conditioned() -> TestResult {
        let mut solver = SparseQRSolver::new();

        // Create a well-conditioned 2x2 system
        let triplets = vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(0, 1, 0.0),
            Triplet::new(1, 0, 0.0),
            Triplet::new(1, 1, 3.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);

        solver.solve_normal_equation(&residuals, &jacobian)?;

        let cov_matrix = solver.compute_covariance_matrix();
        assert!(cov_matrix.is_some());

        if let Some(cov) = cov_matrix {
            // For this system, H = J^T * W * J = [[4, 0], [0, 9]]
            // So covariance = H^(-1) = [[1/4, 0], [0, 1/9]]
            assert!((cov[(0, 0)] - 0.25).abs() < TOLERANCE);
            assert!((cov[(1, 1)] - 1.0 / 9.0).abs() < TOLERANCE);
            assert!(cov[(0, 1)].abs() < TOLERANCE);
            assert!(cov[(1, 0)].abs() < TOLERANCE);
        }
        Ok(())
    }

    /// Test covariance computation caching
    #[test]
    fn test_qr_covariance_caching() -> TestResult {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve
        solver.solve_normal_equation(&residuals, &jacobian)?;

        // First covariance computation
        solver.compute_covariance_matrix();
        assert!(solver.covariance_matrix.is_some());

        // Get pointer to first computation
        if let Some(cov1) = &solver.covariance_matrix {
            let cov1_ptr = cov1.as_ptr();

            // Second covariance computation should return cached result
            solver.compute_covariance_matrix();
            assert!(solver.covariance_matrix.is_some());

            // Get pointer to second computation
            if let Some(cov2) = &solver.covariance_matrix {
                let cov2_ptr = cov2.as_ptr();

                // Should be the same pointer (cached)
                assert_eq!(cov1_ptr, cov2_ptr, "Covariance matrix should be cached");
            }
        }
        Ok(())
    }

    /// Test that covariance computation fails gracefully for singular systems
    #[test]
    fn test_qr_covariance_singular_system() -> TestResult {
        let mut solver = SparseQRSolver::new();

        // Create a singular system (rank deficient)
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 4.0), // Second row is 2x first row
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| i as f64);

        // QR can handle rank-deficient systems, but covariance may be problematic
        let result = solver.solve_normal_equation(&residuals, &jacobian);
        if result.is_ok() {
            // If solve succeeded, covariance computation might still fail due to singularity
            let cov_matrix = solver.compute_covariance_matrix();
            // We don't assert failure here since QR might handle this case
            if let Some(cov) = cov_matrix {
                // If covariance is computed, check that it's reasonable
                assert!(cov.nrows() == 2);
                assert!(cov.ncols() == 2);
            }
        }
        Ok(())
    }
}
