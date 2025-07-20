use std::ops::Mul;

use super::SparseLinearSolver;
use faer::{
    Mat,
    linalg::solvers::Solve,
    sparse::{SparseColMat, linalg::solvers},
};

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    // Symbolic factorization can be cached if the sparsity pattern of the matrix is constant.
    // For augmented systems where lambda changes, this might not be safe to reuse.
    symbolic_pattern: Option<solvers::SymbolicQr<usize>>,
}

impl SparseQRSolver {
    pub fn new() -> Self {
        SparseQRSolver {
            symbolic_pattern: None,
        }
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
        weights: &Mat<f64>,
    ) -> Option<Mat<f64>> {
        let m = jacobians.nrows();

        // Create a sparse diagonal matrix from the weights vector.
        let mut w_triplets = Vec::with_capacity(m);
        for i in 0..m {
            w_triplets.push(faer::sparse::Triplet::new(i, i, weights[(i, 0)]));
        }
        let weights_diag = SparseColMat::try_new_from_triplets(m, m, &w_triplets).unwrap();

        // Form the normal equations explicitly: H = J^T * W * J
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

        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern = Some(solvers::SymbolicQr::try_new(hessian.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), hessian.as_ref()) {
            let dx = qr.solve(gradient);
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
        let weights_diag = SparseColMat::try_new_from_triplets(m, m, &w_triplets).unwrap();

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
        let lambda_i = SparseColMat::try_new_from_triplets(n, n, &lambda_i_triplets).unwrap();

        let augmented_hessian = hessian + lambda_i;

        // Don't cache symbolic pattern for augmented system as sparsity may change
        let sym = solvers::SymbolicQr::try_new(augmented_hessian.symbolic()).unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym, augmented_hessian.as_ref()) {
            let dx = qr.solve(gradient);
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

    /// Helper function to create test data for QR solver
    fn create_test_data() -> (SparseColMat<usize, f64>, Mat<f64>, Mat<f64>) {
        // Create a 4x3 overdetermined system
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 1.0),
            faer::sparse::Triplet::new(0, 1, 0.0),
            faer::sparse::Triplet::new(0, 2, 1.0),
            faer::sparse::Triplet::new(1, 0, 0.0),
            faer::sparse::Triplet::new(1, 1, 1.0),
            faer::sparse::Triplet::new(1, 2, 1.0),
            faer::sparse::Triplet::new(2, 0, 1.0),
            faer::sparse::Triplet::new(2, 1, 1.0),
            faer::sparse::Triplet::new(2, 2, 0.0),
            faer::sparse::Triplet::new(3, 0, 1.0),
            faer::sparse::Triplet::new(3, 1, 0.0),
            faer::sparse::Triplet::new(3, 2, 0.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(4, 3, &triplets).unwrap();

        let residuals = Mat::from_fn(4, 1, |i, _| (i + 1) as f64);
        let weights = Mat::from_fn(4, 1, |_, _| 1.0);

        (jacobian, residuals, weights)
    }

    /// Test basic QR solver creation
    #[test]
    fn test_qr_solver_creation() {
        let solver = SparseQRSolver::new();
        assert!(solver.symbolic_pattern.is_none());

        let default_solver = SparseQRSolver::default();
        assert!(default_solver.symbolic_pattern.is_none());
    }

    /// Test normal equation solving with QR decomposition
    #[test]
    fn test_qr_solve_normal_equation() {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());

        let solution = result.unwrap();
        assert_eq!(solution.nrows(), 3); // Number of variables
        assert_eq!(solution.ncols(), 1);

        // Verify symbolic pattern was cached
        assert!(solver.symbolic_pattern.is_some());
    }

    /// Test QR symbolic pattern caching
    #[test]
    fn test_qr_symbolic_pattern_caching() {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        // First solve
        let result1 = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result1.is_some());
        assert!(solver.symbolic_pattern.is_some());

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

    /// Test augmented equation solving with QR
    #[test]
    fn test_qr_solve_augmented_equation() {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals, weights) = create_test_data();
        let lambda = 0.1;

        let result = solver.solve_augmented_equation(&residuals, &jacobian, &weights, lambda);
        assert!(result.is_some());

        let solution = result.unwrap();
        assert_eq!(solution.nrows(), 3); // Number of variables
        assert_eq!(solution.ncols(), 1);
    }

    /// Test augmented system with different lambda values
    #[test]
    fn test_qr_augmented_different_lambdas() {
        let mut solver = SparseQRSolver::new();
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

    /// Test QR with rank-deficient matrix
    #[test]
    fn test_qr_rank_deficient_matrix() {
        let mut solver = SparseQRSolver::new();

        // Create a rank-deficient matrix (3x3 but rank 2)
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 1.0),
            faer::sparse::Triplet::new(0, 1, 2.0),
            faer::sparse::Triplet::new(0, 2, 3.0),
            faer::sparse::Triplet::new(1, 0, 2.0),
            faer::sparse::Triplet::new(1, 1, 4.0),
            faer::sparse::Triplet::new(1, 2, 6.0), // 2x first row
            faer::sparse::Triplet::new(2, 0, 0.0),
            faer::sparse::Triplet::new(2, 1, 0.0),
            faer::sparse::Triplet::new(2, 2, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap();
        let residuals = Mat::from_fn(3, 1, |i, _| i as f64);
        let weights = Mat::from_fn(3, 1, |_, _| 1.0);

        // QR should still provide a least squares solution
        let result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        assert!(result.is_some());
    }

    /// Test QR with different weight distributions
    #[test]
    fn test_qr_different_weights() {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals, _) = create_test_data();

        let uniform_weights = Mat::from_fn(4, 1, |_, _| 1.0);
        let varying_weights = Mat::from_fn(4, 1, |i, _| (i + 1) as f64 * 0.5);

        let result1 = solver.solve_normal_equation(&residuals, &jacobian, &uniform_weights);
        let result2 = solver.solve_normal_equation(&residuals, &jacobian, &varying_weights);

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

    /// Test augmented system structure and dimensions
    #[test]
    fn test_qr_augmented_system_structure() {
        let mut solver = SparseQRSolver::new();

        // Simple 2x2 system
        let triplets = vec![
            faer::sparse::Triplet::new(0, 0, 1.0),
            faer::sparse::Triplet::new(0, 1, 0.0),
            faer::sparse::Triplet::new(1, 0, 0.0),
            faer::sparse::Triplet::new(1, 1, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let weights = Mat::from_fn(2, 1, |_, _| 1.0);
        let lambda = 0.5;

        let result = solver.solve_augmented_equation(&residuals, &jacobian, &weights, lambda);
        assert!(result.is_some());

        let solution = result.unwrap();
        assert_eq!(solution.nrows(), 2); // Should return only the variable part
        assert_eq!(solution.ncols(), 1);
    }

    /// Test numerical accuracy with known solution
    #[test]
    fn test_qr_numerical_accuracy() {
        let mut solver = SparseQRSolver::new();

        // Create identity system: I * x = b
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
    }

    /// Test QR solver clone functionality
    #[test]
    fn test_qr_solver_clone() {
        let solver1 = SparseQRSolver::new();
        let solver2 = solver1.clone();

        assert!(solver1.symbolic_pattern.is_none());
        assert!(solver2.symbolic_pattern.is_none());
    }

    /// Test zero lambda in augmented system (should behave like normal equation)
    #[test]
    fn test_qr_zero_lambda_augmented() {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals, weights) = create_test_data();

        let normal_result = solver.solve_normal_equation(&residuals, &jacobian, &weights);
        let augmented_result =
            solver.solve_augmented_equation(&residuals, &jacobian, &weights, 0.0);

        assert!(normal_result.is_some());
        assert!(augmented_result.is_some());

        let normal_sol = normal_result.unwrap();
        let augmented_sol = augmented_result.unwrap();

        // Solutions should be very close (within numerical precision)
        for i in 0..normal_sol.nrows() {
            assert!(
                (normal_sol[(i, 0)] - augmented_sol[(i, 0)]).abs() < 1e-8,
                "Zero lambda augmented should match normal equation"
            );
        }
    }

    /// Test with very small weights (near-zero)
    #[test]
    fn test_qr_small_weights() {
        let mut solver = SparseQRSolver::new();
        let (jacobian, residuals, _) = create_test_data();

        let small_weights = Mat::from_fn(4, 1, |_, _| 1e-12);

        let result = solver.solve_normal_equation(&residuals, &jacobian, &small_weights);
        // Should still work, though solution might be less meaningful
        assert!(result.is_some());
    }
}
