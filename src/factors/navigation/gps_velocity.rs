//! GNSS velocity factor (GTSAM `GPSVelocityFactor` analogue).
//!
//! 3D velocity prior from GNSS (e.g. Doppler-derived velocity in the ECEF or
//! local tangent frame). The connected variable is a plain R³ velocity.
//! For a *raw* Doppler range-rate along the satellite line of sight see
//! [`DopplerFactor`](super::doppler_factor::DopplerFactor).

use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// GNSS velocity factor over a single R³ velocity variable.
#[derive(Clone)]
pub struct GpsVelocityFactor {
    /// Measured 3D velocity [m/s].
    pub measurement: Vector3<f64>,
}

impl GpsVelocityFactor {
    /// Create the factor from the measured velocity.
    pub fn new(measurement: Vector3<f64>) -> Self {
        Self { measurement }
    }
}

impl Factor for GpsVelocityFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 1, "GpsVelocityFactor expects one R³ block");
        debug_assert_eq!(params[0].len(), 3, "params[0] must be a 3D velocity");

        for i in 0..3 {
            residual[i] = params[0][i] - self.measurement[i];
        }

        let Some(mut jac) = jacobian else { return };
        for row in 0..3 {
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, col) = if row == col { 1.0 } else { 0.0 };
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 3)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 1 || variables[0].as_param_slice().len() != 3 {
            return Err("GpsVelocityFactor expects a single R³ velocity variable".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_and_jacobian_are_identity() {
        let factor = GpsVelocityFactor::new(Vector3::new(1.0, -2.0, 0.5));
        let v = [1.2, -2.0, 0.3];

        let mut residual = vec![0.0; 3];
        let mut jac_buf = vec![0.0; 9];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 3, 3);
        factor.linearize(&[&v], &mut residual, Some(jac_mut));

        assert!((residual[0] - 0.2).abs() < 1e-12);
        assert!(residual[1].abs() < 1e-12);
        assert!((residual[2] + 0.2).abs() < 1e-12);
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((jac_buf[col * 3 + row] - expected).abs() < 1e-12);
            }
        }
    }
}
