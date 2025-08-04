#[cfg(test)]
mod integration_tests {
    use crate::linalg::{SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
    use faer::{Mat, sparse::SparseColMat};

    const TOLERANCE: f64 = 1e-10;

    /// Helper function to create a realistic optimization problem
    fn create_optimization_problem() -> (SparseColMat<usize, f64>, Mat<f64>, Mat<f64>) {
        // Create a 6x4 overdetermined system representing a typical optimization problem
        // This could represent 6 residuals with 4 parameters
        let triplets = vec![
            // First residual: depends on parameters 0 and 1
            faer::sparse::Triplet::new(0, 0, 1.5),
            faer::sparse::Triplet::new(0, 1, 0.8),
            // Second residual: depends on parameters 1 and 2
            faer::sparse::Triplet::new(1, 1, 2.0),
            faer::sparse::Triplet::new(1, 2, 1.2),
            // Third residual: depends on parameters 0 and 3
            faer::sparse::Triplet::new(2, 0, 0.9),
            faer::sparse::Triplet::new(2, 3, 1.8),
            // Fourth residual: depends on parameters 2 and 3
            faer::sparse::Triplet::new(3, 2, 1.1),
            faer::sparse::Triplet::new(3, 3, 0.7),
            // Fifth residual: depends on all parameters
            faer::sparse::Triplet::new(4, 0, 0.5),
            faer::sparse::Triplet::new(4, 1, 0.3),
            faer::sparse::Triplet::new(4, 2, 0.4),
            faer::sparse::Triplet::new(4, 3, 0.6),
            // Sixth residual: depends on parameters 0 and 2
            faer::sparse::Triplet::new(5, 0, 1.3),
            faer::sparse::Triplet::new(5, 2, 0.9),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(6, 4, &triplets).unwrap();

        // Create residuals with some realistic values
        let residuals = Mat::from_fn(6, 1, |i, _| match i {
            0 => 0.5,
            1 => -0.3,
            2 => 0.8,
            3 => -0.2,
            4 => 0.1,
            5 => -0.6,
            _ => 0.0,
        });

        // Create weights (could represent measurement uncertainties)
        let weights = Mat::from_fn(6, 1, |i, _| match i {
            0 => 2.0, // High confidence
            1 => 1.5,
            2 => 1.0, // Medium confidence
            3 => 1.0,
            4 => 0.5, // Low confidence
            5 => 1.8, // High confidence
            _ => 1.0,
        });

        (jacobian, residuals, weights)
    }

    /// Test that QR and Cholesky solvers produce consistent covariance matrices
    #[test]
    fn test_solver_covariance_consistency() {
        let (jacobian, residuals, weights) = create_optimization_problem();

        let mut qr_solver = SparseQRSolver::new();
        let mut cholesky_solver = SparseCholeskySolver::new();

        // Solve with both methods
        let qr_result = qr_solver.solve_normal_equation(&residuals, &jacobian, &weights);
        let cholesky_result =
            cholesky_solver.solve_normal_equation(&residuals, &jacobian, &weights);

        assert!(qr_result.is_some());
        assert!(cholesky_result.is_some());

        // Solutions should be very close
        let qr_sol = qr_result.unwrap();
        let chol_sol = cholesky_result.unwrap();
        for i in 0..4 {
            assert!(
                (qr_sol[(i, 0)] - chol_sol[(i, 0)]).abs() < 1e-8,
                "Solutions should be consistent between QR and Cholesky"
            );
        }

        // Compute covariance matrices
        let qr_cov = qr_solver.compute_covariance_matrix();
        let chol_cov = cholesky_solver.compute_covariance_matrix();

        assert!(qr_cov.is_some());
        assert!(chol_cov.is_some());

        // Covariance matrices should be very close
        let qr_cov_mat = qr_cov.unwrap();
        let chol_cov_mat = chol_cov.unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (qr_cov_mat[(i, j)] - chol_cov_mat[(i, j)]).abs() < 1e-8,
                    "Covariance matrices should be consistent between QR and Cholesky"
                );
            }
        }
    }

    /// Test covariance computation in iterative optimization scenario
    #[test]
    fn test_iterative_covariance_computation() {
        let (jacobian, _residuals, weights) = create_optimization_problem();
        let mut solver = SparseQRSolver::new();

        // Simulate multiple iterations of an optimization algorithm
        for iteration in 0..3 {
            // Modify residuals slightly to simulate optimization progress
            let modified_residuals = Mat::from_fn(6, 1, |i, _| {
                let base_residual = match i {
                    0 => 0.5,
                    1 => -0.3,
                    2 => 0.8,
                    3 => -0.2,
                    4 => 0.1,
                    5 => -0.6,
                    _ => 0.0,
                };
                // Residuals should decrease with iterations
                base_residual * (0.8_f64).powi(iteration as i32)
            });

            let result = solver.solve_normal_equation(&modified_residuals, &jacobian, &weights);
            assert!(result.is_some(), "Iteration {} should succeed", iteration);

            // Reset covariance computation for next iteration
            solver.reset_covariance();
        }

        // Final covariance computation
        let final_cov = solver.compute_covariance_matrix();
        assert!(final_cov.is_some());

        let cov = final_cov.unwrap();
        // Verify covariance matrix properties
        assert_eq!(cov.nrows(), 4);
        assert_eq!(cov.ncols(), 4);

        // Should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < TOLERANCE,
                    "Covariance matrix should be symmetric"
                );
            }
        }

        // Diagonal elements should be positive
        for i in 0..4 {
            assert!(cov[(i, i)] > 0.0, "Diagonal elements should be positive");
        }
    }

    /// Test covariance computation with different weight distributions
    #[test]
    fn test_covariance_with_weight_variations() {
        let (jacobian, residuals, _) = create_optimization_problem();

        // Test with uniform weights
        let uniform_weights = Mat::from_fn(6, 1, |_, _| 1.0);
        let mut solver1 = SparseQRSolver::new();
        solver1.solve_normal_equation(&residuals, &jacobian, &uniform_weights);
        let cov1 = solver1.compute_covariance_matrix().unwrap();

        // Test with varying weights
        let varying_weights = Mat::from_fn(6, 1, |i, _| (i + 1) as f64 * 0.5);
        let mut solver2 = SparseQRSolver::new();
        solver2.solve_normal_equation(&residuals, &jacobian, &varying_weights);
        let cov2 = solver2.compute_covariance_matrix().unwrap();

        // Covariance matrices should be different due to different weights
        let mut different = false;
        for i in 0..4 {
            for j in 0..4 {
                if (cov1[(i, j)] - cov2[(i, j)]).abs() > TOLERANCE {
                    different = true;
                    break;
                }
            }
            if different {
                break;
            }
        }
        assert!(
            different,
            "Different weights should produce different covariance matrices"
        );

        // Both should still be valid covariance matrices
        for cov in [cov1, cov2] {
            // Symmetric
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        (cov[(i, j)] - cov[(j, i)]).abs() < TOLERANCE,
                        "Covariance matrix should be symmetric"
                    );
                }
            }
            // Positive diagonal
            for i in 0..4 {
                assert!(cov[(i, i)] > 0.0, "Diagonal elements should be positive");
            }
        }
    }

    /// Test that covariance computation handles augmented systems correctly
    #[test]
    fn test_covariance_with_augmented_system() {
        let (jacobian, residuals, weights) = create_optimization_problem();
        let mut solver = SparseQRSolver::new();

        // Solve augmented system (Levenberg-Marquardt style)
        let lambda = 0.1;
        let result = solver.solve_augmented_equation(&residuals, &jacobian, &weights, lambda);
        assert!(result.is_some());

        // Compute covariance matrix
        let cov_matrix = solver.compute_covariance_matrix();
        assert!(cov_matrix.is_some());

        let cov = cov_matrix.unwrap();

        // Verify properties
        assert_eq!(cov.nrows(), 4);
        assert_eq!(cov.ncols(), 4);

        // Should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < TOLERANCE,
                    "Covariance matrix should be symmetric"
                );
            }
        }

        // Diagonal elements should be positive but smaller than normal equation
        // (due to regularization)
        for i in 0..4 {
            assert!(cov[(i, i)] > 0.0, "Diagonal elements should be positive");
        }

        // Compare with normal equation solution
        let mut normal_solver = SparseQRSolver::new();
        normal_solver.solve_normal_equation(&residuals, &jacobian, &weights);
        let normal_cov = normal_solver.compute_covariance_matrix().unwrap();

        // Augmented system should generally have smaller variances (more confident)
        for i in 0..4 {
            assert!(
                cov[(i, i)] <= normal_cov[(i, i)] + TOLERANCE,
                "Augmented system should have smaller or equal variances"
            );
        }
    }
}
