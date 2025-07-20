use std::ops::Mul;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers;

use super::SparseLinearSolver;

#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    symbolic_pattern: Option<solvers::SymbolicLlt<usize>>,
}

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver {
            symbolic_pattern: None,
        }
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
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>> {
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

        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicLlt::try_new(hessian.symbolic(), faer::Side::Lower).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym.clone(), hessian.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(gradient);
            Some(dx)
        } else {
            None
        }
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
        lambda: f64,
    ) -> Option<faer::Mat<f64>> {
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

        let augmented_hessian = hessian + lambda_i;

        // Don't cache symbolic pattern for augmented system as sparsity may change
        let sym =
            solvers::SymbolicLlt::try_new(augmented_hessian.symbolic(), faer::Side::Lower).unwrap();
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym, augmented_hessian.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(gradient);
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
        assert!(solver.symbolic_pattern.is_none());

        let default_solver = SparseCholeskySolver::default();
        assert!(default_solver.symbolic_pattern.is_none());
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
        assert!(solver.symbolic_pattern.is_some());
    }

    /// Test that symbolic pattern is reused on subsequent calls
    #[test]
    fn test_symbolic_pattern_caching() {
        let mut solver = SparseCholeskySolver::new();
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

        assert!(solver1.symbolic_pattern.is_none());
        assert!(solver2.symbolic_pattern.is_none());
    }
}
