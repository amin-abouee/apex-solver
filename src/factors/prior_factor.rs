//! Prior factor for unary constraints on variables.

use super::Factor;
use faer::prelude::ReborrowMut;
use nalgebra::DVector;

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
#[derive(Debug, Clone)]
pub struct PriorFactor {
    /// The prior value (measurement or known value)
    pub data: DVector<f64>,
}

impl Factor for PriorFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let n = self.data.len();
        for i in 0..n {
            residual[i] = params[0][i] - self.data[i];
        }
        if let Some(mut jac) = jacobian {
            for i in 0..n {
                *jac.rb_mut().get_mut(i, i) = 1.0;
            }
        }
    }

    fn residual_dim(&self) -> usize {
        self.data.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        let n = self.data.len();
        (n, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_prior_factor_residual() {
        let prior = PriorFactor {
            data: dvector![1.0, 2.0],
        };
        let current = dvector![1.5f64, 2.3f64];
        let params: Vec<&[f64]> = vec![current.as_slice()];
        let mut residual = vec![0.0f64; 2];
        prior.linearize(&params, &mut residual, None);
        assert!((residual[0] - 0.5).abs() < 1e-10);
        assert!((residual[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_prior_factor_jacobian_identity() {
        let prior = PriorFactor {
            data: dvector![1.0, 2.0],
        };
        let current = dvector![1.5f64, 2.3f64];
        let params: Vec<&[f64]> = vec![current.as_slice()];
        let mut residual = vec![0.0f64; 2];
        let mut jac_buf = vec![0.0f64; 4];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 2, 2);
        prior.linearize(&params, &mut residual, Some(jac_mut));
        // Column-major: jac_buf[col*rows + row]
        assert!((jac_buf[0] - 1.0).abs() < 1e-10); // (0,0)
        assert!((jac_buf[3] - 1.0).abs() < 1e-10); // (1,1)
        assert!(jac_buf[1].abs() < 1e-10); // (1,0)
        assert!(jac_buf[2].abs() < 1e-10); // (0,1)
    }

    #[test]
    fn test_prior_factor_residual_dim() {
        let prior = PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        };
        assert_eq!(prior.residual_dim(), 3);
    }

    #[test]
    fn test_prior_factor_jacobian_shape() {
        let prior = PriorFactor {
            data: dvector![0.0, 0.0],
        };
        assert_eq!(prior.jacobian_shape(), (2, 2));
    }
}
