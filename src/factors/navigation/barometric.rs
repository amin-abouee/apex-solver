//! Barometric altimeter factor (GTSAM `BarometricFactor` analogue).
//!
//! Constrains an SE(3) pose's altitude against a pressure-derived height,
//! together with a slowly-varying R¹ altimeter bias:
//!
//! ```text
//! r = (z_pose + bias) − z_measured
//! ```

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Barometric altitude factor over `[pose (7), baro bias (1)]`
/// (GTSAM `BarometricFactor` layout).
///
/// Residual: `r = z + bias − h_measured` — the bias absorbs slow drift of
/// the pressure reference so consecutive height measurements stay unbiased.
/// The z-Jacobian follows the world-to-body pose retraction:
/// `∂z/∂δρ = R[2,:]`, `∂z/∂δθ = −(R·p̂_w)[2,:]`.
#[derive(Clone)]
pub struct BarometricFactor {
    /// Measured altitude [m].
    pub measured_height: f64,
}

impl BarometricFactor {
    /// Create the factor from the pressure-derived altitude.
    pub fn new(measured_height: f64) -> Self {
        Self { measured_height }
    }
}

impl Factor for BarometricFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "BarometricFactor expects [pose, bias]");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 1, "params[1] must be a scalar bias");

        let pose = SE3::from_param_slice(params[0]);
        let p_world = pose.translation();
        let bias = params[1][0];
        residual[0] = p_world.z + bias - self.measured_height;

        let Some(mut jac) = jacobian else { return };
        // Retraction t' = t + R·δρ: ∂z/∂δρ = R[2,:]; rotation does not move
        // the origin; ∂z/∂bias = 1.
        let rotation = pose.rotation_so3().rotation_matrix();
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = rotation[(2, col)];
            *jac.rb_mut().get_mut(0, 3 + col) = 0.0;
        }
        *jac.rb_mut().get_mut(0, 6) = 1.0;
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 7)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != SE3::REP_SIZE
            || variables[1].as_param_slice().len() != 1
        {
            return Err("BarometricFactor expects [SE3 pose, R¹ baro bias]".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion as UQ;

    #[test]
    fn barometric_residual_and_jacobian() {
        use apex_manifolds::Tangent;
        use apex_manifolds::se3::SE3Tangent;

        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 2.0, 48.5),
            UQ::from_euler_angles(0.05, -0.03, 0.02),
        ));
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();
        let bias = vec![1.0];
        let factor = BarometricFactor::new(50.0);

        let mut residual = vec![0.0; 1];
        let mut jac_buf = vec![0.0; 7];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 1, 7);
        factor.linearize(&[&pose_v, &bias], &mut residual, Some(jac_mut));

        assert!((residual[0] - (-0.5)).abs() < 1e-12);
        assert!((jac_buf[6] - 1.0).abs() < 1e-12); // ∂r/∂bias

        // FD check over the pose tangent.
        const EPS: f64 = 1e-6;
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let perturbed: Vec<f64> = pose
                .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                .as_param_slice()
                .to_vec();
            let mut r_pert = vec![0.0; 1];
            factor.linearize(&[&perturbed, &bias], &mut r_pert, None);
            let fd = (r_pert[0] - residual[0]) / EPS;
            let ana = jac_buf[col];
            assert!(
                (fd - ana).abs() < 1e-4,
                "baro J[{col}]: analytical={ana:.6} fd={fd:.6}"
            );
        }
    }
}
