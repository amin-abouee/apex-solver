//! Prior factor for unary constraints on variables.

use super::Factor;
use nalgebra::{DMatrix, DVector};

/// Prior factor (unary constraint) on a single variable.
///
/// Represents a direct measurement or prior belief about a variable's value. This is used
/// to anchor variables to known values or to incorporate prior knowledge into the optimization.
///
/// # Mathematical Formulation
///
/// The residual is simply the difference between the current value and the prior:
///
/// ```text
/// r = x - x_prior
/// ```
///
/// The Jacobian is the identity matrix: `J = I`.
///
/// # Use Cases
///
/// - **Anchoring**: Fix the first pose in SLAM to prevent drift
/// - **GPS measurements**: Constrain a pose to a known global position
/// - **Prior knowledge**: Incorporate measurements from other sensors
/// - **Regularization**: Prevent variables from drifting too far from initial values
///
/// # Example
///
/// ```
/// use apex_solver::factors::{Factor, PriorFactor};
/// use nalgebra::DVector;
///
/// // Prior: first pose should be at origin
/// let prior = PriorFactor {
///     data: DVector::from_vec(vec![0.0, 0.0, 0.0]),
/// };
///
/// // Current estimate (slightly off)
/// let current_pose = DVector::from_vec(vec![0.1, 0.05, 0.02]);
///
/// // Compute residual (shows deviation from prior)
/// let (residual, jacobian) = prior.linearize(&[current_pose], true);
/// ```
///
/// # Implementation Note
///
/// This is a simple "Euclidean" prior that works for any vector space. For manifold
/// variables (SE2, SE3, etc.), consider using manifold-aware priors that respect the
/// geometry (not yet implemented).
#[derive(Debug, Clone)]
pub struct PriorFactor {
    /// The prior value (measurement or known value)
    pub data: DVector<f64>,
}

impl Factor for PriorFactor {
    /// Compute residual and Jacobian for prior factor.
    ///
    /// # Arguments
    ///
    /// * `params` - Single variable value
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// - Residual: N×1 vector `(x_current - x_prior)`
    /// - Jacobian: N×N identity matrix
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::factors::{Factor, PriorFactor};
    /// use nalgebra::DVector;
    ///
    /// let prior = PriorFactor {
    ///     data: DVector::from_vec(vec![1.0, 2.0]),
    /// };
    ///
    /// let current = DVector::from_vec(vec![1.5, 2.3]);
    /// let (residual, jacobian) = prior.linearize(&[current], true);
    ///
    /// // Residual shows difference
    /// assert!((residual[0] - 0.5).abs() < 1e-10);
    /// assert!((residual[1] - 0.3).abs() < 1e-10);
    ///
    /// // Jacobian is identity
    /// if let Some(jac) = jacobian {
    ///     assert_eq!(jac[(0, 0)], 1.0);
    ///     assert_eq!(jac[(1, 1)], 1.0);
    /// }
    /// ```
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let residual = &params[0] - &self.data;
        let jacobian = if compute_jacobian {
            Some(DMatrix::<f64>::identity(residual.nrows(), residual.nrows()))
        } else {
            None
        };
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    const TOLERANCE: f64 = 1e-10;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_get_dimension_matches_prior_length() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![1.0, 2.0, 3.0]),
        };
        assert_eq!(prior.get_dimension(), 3);
    }

    #[test]
    fn test_get_dimension_scalar() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![5.0]),
        };
        assert_eq!(prior.get_dimension(), 1);
    }

    // ── Residual computation ──────────────────────────────────────────────────

    #[test]
    fn test_residual_zero_when_at_prior() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![1.0, 2.0, 3.0]),
        };
        let current = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let (residual, _) = prior.linearize(&[current], false);

        assert_eq!(residual.len(), 3);
        assert!(
            residual.norm() < TOLERANCE,
            "Expected zero residual, got norm {}",
            residual.norm()
        );
    }

    #[test]
    fn test_residual_is_current_minus_prior() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![1.0, 2.0]),
        };
        let current = DVector::from_vec(vec![1.5, 2.3]);
        let (residual, _) = prior.linearize(&[current], false);

        assert!((residual[0] - 0.5).abs() < TOLERANCE);
        assert!((residual[1] - 0.3).abs() < TOLERANCE);
    }

    #[test]
    fn test_residual_negative_when_below_prior() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![3.0, 4.0]),
        };
        let current = DVector::from_vec(vec![1.0, 2.0]);
        let (residual, _) = prior.linearize(&[current], false);

        assert!((residual[0] - (-2.0)).abs() < TOLERANCE);
        assert!((residual[1] - (-2.0)).abs() < TOLERANCE);
    }

    #[test]
    fn test_residual_zero_prior() {
        let prior = PriorFactor {
            data: DVector::zeros(4),
        };
        let current = DVector::from_vec(vec![0.1, -0.2, 0.3, -0.4]);
        let (residual, _) = prior.linearize(&[current.clone()], false);

        for i in 0..4 {
            assert!((residual[i] - current[i]).abs() < TOLERANCE);
        }
    }

    // ── Jacobian computation ──────────────────────────────────────────────────

    #[test]
    fn test_jacobian_is_identity_when_requested() {
        let n = 3;
        let prior = PriorFactor {
            data: DVector::zeros(n),
        };
        let current = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let (_, jacobian) = prior.linearize(&[current], true);

        let jac = jacobian.expect("Jacobian should be Some when compute_jacobian=true");
        assert_eq!(jac.nrows(), n);
        assert_eq!(jac.ncols(), n);

        let identity = DMatrix::<f64>::identity(n, n);
        assert!(
            (jac - identity).norm() < TOLERANCE,
            "Jacobian must be identity matrix"
        );
    }

    #[test]
    fn test_jacobian_none_when_not_requested() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![1.0, 2.0]),
        };
        let current = DVector::from_vec(vec![1.5, 2.5]);
        let (_, jacobian) = prior.linearize(&[current], false);
        assert!(jacobian.is_none());
    }

    #[test]
    fn test_jacobian_is_square() {
        for n in [1, 3, 6, 7] {
            let prior = PriorFactor {
                data: DVector::zeros(n),
            };
            let current = DVector::zeros(n);
            let (_, jacobian) = prior.linearize(&[current], true);
            let jac = jacobian.unwrap();
            assert_eq!(jac.nrows(), n);
            assert_eq!(jac.ncols(), n);
        }
    }

    // ── Jacobian numerical verification ──────────────────────────────────────

    #[test]
    fn test_jacobian_matches_finite_differences() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![1.0, -0.5, 2.0]),
        };
        let x0 = DVector::from_vec(vec![1.2, -0.3, 2.5]);
        let (r0, jacobian) = prior.linearize(&[x0.clone()], true);
        let jac = jacobian.unwrap();

        let eps = 1e-6;
        let n = x0.len();
        for j in 0..n {
            let mut x_plus = x0.clone();
            x_plus[j] += eps;
            let (r_plus, _) = prior.linearize(&[x_plus], false);
            let fd_col = (r_plus - &r0) / eps;

            for i in 0..n {
                assert!(
                    (jac[(i, j)] - fd_col[i]).abs() < 1e-5,
                    "J[{i},{j}] analytical={} FD={}",
                    jac[(i, j)],
                    fd_col[i]
                );
            }
        }
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_clone_is_independent() {
        let prior = PriorFactor {
            data: DVector::from_vec(vec![1.0, 2.0]),
        };
        let mut cloned = prior.clone();
        cloned.data[0] = 99.0;
        assert_eq!(
            prior.data[0], 1.0,
            "Original should not be affected by clone mutation"
        );
    }

    #[test]
    fn test_large_dimension() {
        let n = 100;
        let prior = PriorFactor {
            data: DVector::from_element(n, 1.0),
        };
        let current = DVector::from_element(n, 2.0);
        let (residual, jacobian) = prior.linearize(&[current], true);

        assert_eq!(residual.len(), n);
        assert!(residual.iter().all(|&r| (r - 1.0).abs() < TOLERANCE));

        let jac = jacobian.unwrap();
        assert_eq!(jac, DMatrix::<f64>::identity(n, n));
    }
}
