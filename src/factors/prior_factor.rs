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
