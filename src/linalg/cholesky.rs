use std::ops::Mul;

use faer::{
    Mat,
    linalg::solvers::Solve,
    sparse::{SparseColMat, linalg::solvers},
};

use crate::linalg::SparseLinearSolver;

#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    factorizer: Option<solvers::Llt<usize, f64>>,

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

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver {
            factorizer: None,
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

    pub fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        // Only compute if we have a factorizer and hessian, but no covariance matrix yet
        if self.factorizer.is_some() && self.hessian.is_some() && self.covariance_matrix.is_none() {
            if let (Some(factorizer), Some(hessian)) = (&self.factorizer, &self.hessian) {
                let n = hessian.ncols();
                // Create identity matrix
                let identity = Mat::identity(n, n);

                // Solve H * X = I to get X = H^(-1) = covariance matrix
                let cov_matrix = factorizer.solve(&identity);
                self.covariance_matrix = Some(cov_matrix);
            }
        }
        self.covariance_matrix.as_ref()
    }

    pub fn compute_standard_errors(&mut self) -> Option<&Mat<f64>> {
        // // Ensure covariance matrix is computed first
        if self.covariance_matrix.is_none() {
            self.compute_covariance_matrix();
        }
        
        let n = self.hessian.as_ref().unwrap().ncols();
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

impl Default for SparseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseLinearSolver for SparseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        weights: &Mat<f64>,
    ) -> Option<Mat<f64>> {
        let m = jacobians.nrows();

        // Create a sparse diagonal matrix from the weights vector
        let mut w_triplets = Vec::with_capacity(m);
        for i in 0..m {
            w_triplets.push(faer::sparse::Triplet::new(i, i, weights[(i, 0)]));
        }
        let weights_diag =
            faer::sparse::SparseColMat::try_new_from_triplets(m, m, &w_triplets).unwrap();

        // Form the normal equations: H = J^T * W * J
        let hessian = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(weights_diag.as_ref().mul(jacobians.as_ref()));

        // g = J^T * W * -r
        let gradient = jacobians
            .as_ref()
            .transpose()
            .mul(weights_diag.as_ref().mul(-residuals));

        let sym = solvers::SymbolicLlt::try_new(hessian.symbolic(), faer::Side::Lower).unwrap();
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym.clone(), hessian.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(&gradient);
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
            self.factorizer = Some(cholesky);
            Some(dx)
        } else {
            None
        }
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        weights: &Mat<f64>,
        lambda: f64,
    ) -> Option<Mat<f64>> {
        let m = jacobians.nrows();
        let n = jacobians.ncols();

        // Create a sparse diagonal matrix from weights
        let mut w_triplets = Vec::with_capacity(m);
        for i in 0..m {
            w_triplets.push(faer::sparse::Triplet::new(i, i, weights[(i, 0)]));
        }
        let weights_diag =
            faer::sparse::SparseColMat::try_new_from_triplets(m, m, &w_triplets).unwrap();

        // H = J^T * W * J
        let hessian = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(weights_diag.as_ref().mul(jacobians.as_ref()));

        // g = J^T * W * -r
        let gradient = jacobians
            .as_ref()
            .transpose()
            .mul(weights_diag.as_ref().mul(-residuals));

        // H_aug = H + lambda * I
        let mut lambda_i_triplets = Vec::with_capacity(n);
        for i in 0..n {
            lambda_i_triplets.push(faer::sparse::Triplet::new(i, i, lambda));
        }
        let lambda_i =
            faer::sparse::SparseColMat::try_new_from_triplets(n, n, &lambda_i_triplets).unwrap();

        let augmented_hessian = &hessian + lambda_i;

        // Don't cache symbolic pattern for augmented system as sparsity may change
        let sym =
            solvers::SymbolicLlt::try_new(augmented_hessian.symbolic(), faer::Side::Lower).unwrap();
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym, augmented_hessian.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(&gradient);
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
            self.factorizer = Some(cholesky);
            Some(dx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use faer::sparse::SparseColMat;

    const TOLERANCE: f64 = 1e-10;

    /// Helper function to create a simple test matrix and vectors
    fn create_test_data() -> (SparseColMat<usize, f64>, Mat<f64>, Mat<f64>) {
        // Create an overdetermined system (4x3) so that weights have an effect
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 2.0),
            faer::sparse::Triplet::new(0, 1, 1.0),
            faer::sparse::Triplet::new(1, 0, 1.0),
            faer::sparse::Triplet::new(1, 1, 3.0),
            faer::sparse::Triplet::new(1, 2, 1.0),
            faer::sparse::Triplet::new(2, 1, 1.0),
            faer::sparse::Triplet::new(2, 2, 2.0),
            faer::sparse::Triplet::new(3, 0, 1.5), // Add a 4th row for overdetermined system
            faer::sparse::Triplet::new(3, 2, 0.5),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(4, 3, &triplets).unwrap();

        let residuals = Mat::from_fn(4, 1, |i, _| match i {
            0 => 1.0,
            1 => -2.0,
            2 => 0.5,
            3 => 1.2,
            _ => 0.0,
        });

        let weights = Mat::from_fn(4, 1, |_, _| 1.0);

        (jacobian, residuals, weights)
    }

    /// Test basic solver creation and default implementation
    #[test]
    fn test_solver_creation() {
        let solver = SparseCholeskySolver::new();
        assert!(solver.factorizer.is_none());

        let default_solver = SparseCholeskySolver::default();
        assert!(default_solver.factorizer.is_none());
    }

    /// Test normal equation solving with well-conditioned matrix
    #[test]
    fn test_solve_normal_equation_well_conditioned() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        let solution = result.unwrap();
        assert_eq!(solution.nrows(), 3);
        assert_eq!(solution.ncols(), 1);

        // Verify the symbolic pattern was cached
        assert!(solver.factorizer.is_some());
    }

    /// Test that symbolic pattern is reused on subsequent calls
    #[test]
    fn test_symbolic_pattern_caching() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        // First solve
        let result1 = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result1.is_some());
        assert!(solver.factorizer.is_some());

        // Second solve should reuse pattern
        let result2 = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result2.is_some());

        // Results should be identical
        let sol1 = result1.unwrap();
        let sol2 = result2.unwrap();
        for i in 0..sol1.nrows() {
            assert!((sol1[(i, 0)] - sol2[(i, 0)]).abs() < TOLERANCE);
        }
    }

    /// Test augmented equation solving
    #[test]
    fn test_solve_augmented_equation() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();
        let lambda = 0.1;

        let result = solver.solve_augmented_equation(&residuals, &jacobian, &weights, lambda);
        assert!(result.is_some());

        let solution = result.unwrap();
        assert_eq!(solution.nrows(), 3);
        assert_eq!(solution.ncols(), 1);
    }

    /// Test with different lambda values in augmented system
    #[test]
    fn test_augmented_equation_different_lambdas() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        let lambda1 = 0.01;
        let lambda2 = 1.0;

        let result1 = solver.solve_augmented_equation(&residuals, &jacobian, &weights, lambda1);
        let result2 = solver.solve_augmented_equation(&residuals, &jacobian, &weights, lambda2);

        assert!(result1.is_some());
        assert!(result2.is_some());

        // Solutions should be different due to different regularization
        let sol1 = result1.unwrap();
        let sol2 = result2.unwrap();
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
    }

    /// Test with singular matrix (should return None)
    #[test]
    fn test_singular_matrix() {
        let mut solver = SparseCholeskySolver::new();

        // Create a singular matrix
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 1.0),
            faer::sparse::Triplet::new(0, 1, 2.0),
            faer::sparse::Triplet::new(1, 0, 2.0),
            faer::sparse::Triplet::new(1, 1, 4.0), // Second row is 2x first row
        ];
        let singular_jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let residuals = Mat::from_fn(2, 1, |i, _| i as f64);
        let weights = Mat::from_fn(2, 1, |_, _| 1.0);

        let result = solver.solve_normal_equation(&residuals, &singular_jacobian, &weights);
        assert!(result.is_none(), "Singular matrix should return None");
    }

    /// Test with different weight values
    #[test]
    fn test_different_weights() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, _) = create_test_data();

        let weights1 = Mat::from_fn(4, 1, |_, _| 1.0);
        let weights2 = Mat::from_fn(4, 1, |i, _| (i + 1) as f64);

        let result1 = solver.solve_normal_equation(&residuals, &jacobian, &weights1);
        let result2 = solver.solve_normal_equation(&residuals, &jacobian, &weights2);

        assert!(result1.is_some());
        assert!(result2.is_some());

        // Solutions should be different due to different weights
        let sol1 = result1.unwrap();
        let sol2 = result2.unwrap();
        let mut different = false;
        for i in 0..sol1.nrows() {
            if (sol1[(i, 0)] - sol2[(i, 0)]).abs() > TOLERANCE {
                different = true;
                break;
            }
        }
        assert!(different, "Solutions should differ with different weights");
    }

    /// Test with empty matrix (edge case)
    #[test]
    fn test_empty_matrix() {
        let mut solver = SparseCholeskySolver::new();

        let empty_jacobian = SparseColMat::try_new_from_triplets(0, 0, &[]).unwrap();
        let empty_residuals = Mat::zeros(0, 1);
        let empty_weights = Mat::zeros(0, 1);

        let result =
            solver.solve_normal_equation(&empty_residuals, &empty_jacobian, &empty_weights);
        if let Some(solution) = result {
            assert_eq!(solution.nrows(), 0);
        }
    }

    /// Test numerical accuracy with known solution
    #[test]
    fn test_numerical_accuracy() {
        let mut solver = SparseCholeskySolver::new();

        // Create a simple 2x2 system with known solution
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 1.0),
            faer::sparse::Triplet::new(0, 1, 0.0),
            faer::sparse::Triplet::new(1, 0, 0.0),
            faer::sparse::Triplet::new(1, 1, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let residuals = Mat::from_fn(2, 1, |i, _| -((i + 1) as f64)); // [-1, -2]
        let weights = Mat::from_fn(2, 1, |_, _| 1.0);

        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        let solution = result.unwrap();
        // Expected solution should be [1, 2] since J^T * J = I and J^T * (-r) = [1, 2]
        assert!((solution[(0, 0)] - 1.0).abs() < TOLERANCE);
        assert!((solution[(1, 0)] - 2.0).abs() < TOLERANCE);
    }

    /// Test clone functionality
    #[test]
    fn test_solver_clone() {
        let solver1 = SparseCholeskySolver::new();
        let solver2 = solver1.clone();

        assert!(solver1.factorizer.is_none());
        assert!(solver2.factorizer.is_none());
    }

    /// Test covariance matrix computation
    #[test]
    fn test_cholesky_covariance_computation() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        // First solve to set up factorizer and hessian
        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        // Now compute covariance matrix
        let cov_matrix = solver.compute_covariance_matrix();
        assert!(cov_matrix.is_some());

        let cov = cov_matrix.unwrap();
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

    /// Test standard errors computation
    #[test]
    fn test_cholesky_standard_errors_computation() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        // First solve to set up factorizer and hessian
        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        // Compute covariance matrix first (this also computes standard errors)
        solver.compute_standard_errors();

        // Now check that both covariance matrix and standard errors are available
        assert!(solver.covariance_matrix.is_some());
        assert!(solver.standard_errors.is_some());

        let cov = solver.covariance_matrix.as_ref().unwrap();
        let errors = solver.standard_errors.as_ref().unwrap();

        assert_eq!(errors.nrows(), 3); // Should be n x 1 where n is number of variables
        assert_eq!(errors.ncols(), 1);

        // All standard errors should be positive
        for i in 0..3 {
            assert!(
                errors[(i, 0)] > 0.0,
                "Standard errors should be positive"
            );
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

    /// Test covariance computation with well-conditioned positive definite system
    #[test]
    fn test_cholesky_covariance_positive_definite() {
        let mut solver = SparseCholeskySolver::new();

        // Create a well-conditioned positive definite system
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 3.0),
            faer::sparse::Triplet::new(0, 1, 1.0),
            faer::sparse::Triplet::new(1, 0, 1.0),
            faer::sparse::Triplet::new(1, 1, 2.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let weights = Mat::from_fn(2, 1, |_, _| 1.0);

        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        let cov_matrix = solver.compute_covariance_matrix();
        assert!(cov_matrix.is_some());

        let cov = cov_matrix.unwrap();
        // For this system, H = J^T * W * J = [[10, 5], [5, 5]]
        // Covariance = H^(-1) = (1/25) * [[5, -5], [-5, 10]] = [[0.2, -0.2], [-0.2, 0.4]]
        assert!((cov[(0, 0)] - 0.2).abs() < TOLERANCE);
        assert!((cov[(1, 1)] - 0.4).abs() < TOLERANCE);
        assert!((cov[(0, 1)] - (-0.2)).abs() < TOLERANCE);
        assert!((cov[(1, 0)] - (-0.2)).abs() < TOLERANCE);
    }

    /// Test covariance computation caching
    #[test]
    fn test_cholesky_covariance_caching() {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        // First solve
        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        // First covariance computation
        solver.compute_covariance_matrix();
        assert!(solver.covariance_matrix.is_some());

        // Get pointer to first computation
        let cov1_ptr = solver.covariance_matrix.as_ref().unwrap().as_ptr();

        // Second covariance computation should return cached result
        solver.compute_covariance_matrix();
        assert!(solver.covariance_matrix.is_some());

        // Get pointer to second computation
        let cov2_ptr = solver.covariance_matrix.as_ref().unwrap().as_ptr();

        // Should be the same pointer (cached)
        assert_eq!(cov1_ptr, cov2_ptr, "Covariance matrix should be cached");
    }

    /// Test Cholesky decomposition properties
    #[test]
    fn test_cholesky_decomposition_properties() {
        let mut solver = SparseCholeskySolver::new();

        // Create a simple positive definite system
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 2.0),
            faer::sparse::Triplet::new(1, 1, 3.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let weights = Mat::from_fn(2, 1, |_, _| 1.0);

        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        // Verify that we have a factorizer and hessian
        assert!(solver.factorizer.is_some());
        assert!(solver.hessian.is_some());

        // The hessian should be positive definite for Cholesky to work
        let hessian = solver.hessian.as_ref().unwrap();
        assert_eq!(hessian.nrows(), 2);
        assert_eq!(hessian.ncols(), 2);
    }

    /// Test numerical stability with different condition numbers
    #[test]
    fn test_cholesky_numerical_stability() {
        let mut solver = SparseCholeskySolver::new();

        // Create a well-conditioned system
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 1.0),
            faer::sparse::Triplet::new(1, 1, 1.0),
            faer::sparse::Triplet::new(2, 2, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap();
        let residuals = Mat::from_fn(3, 1, |i, _| -((i + 1) as f64)); // [-1, -2, -3]
        let weights = Mat::from_fn(3, 1, |_, _| 1.0);

        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        let solution = result.unwrap();
        // Expected solution should be [1, 2, 3] since H = I and g = [1, 2, 3]
        for i in 0..3 {
            let expected = (i + 1) as f64;
            assert!(
                (solution[(i, 0)] - expected).abs() < TOLERANCE,
                "Expected {}, got {}",
                expected,
                solution[(i, 0)]
            );
        }

        // Covariance should be identity matrix (inverse of identity)
        let cov_matrix = solver.compute_covariance_matrix();
        assert!(cov_matrix.is_some());
        let cov = cov_matrix.unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cov[(i, j)] - expected).abs() < TOLERANCE,
                    "Covariance[{}, {}] expected {}, got {}",
                    i, j, expected, cov[(i, j)]
                );
            }
        }
    }
}
