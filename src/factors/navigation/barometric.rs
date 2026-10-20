//! Barometric altimeter factor (GTSAM `BarometricFactor` analogue).
//!
//! Constrains an SE(3) pose's altitude against a pressure-derived height,
//! together with a slowly-varying R¹ altimeter bias:
//!
//! ```text
//! r = (z_pose + bias) − z_measured
//! ```

use std::marker::PhantomData;

use apex_manifolds::capability::ActsOnPoints3;
use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::group_point::act_point;

/// Barometric altitude factor over `[pose (7), baro bias (1)]`
/// (GTSAM `BarometricFactor` layout).
///
/// Residual: `r = z + bias − h_measured` — the bias absorbs slow drift of
/// the pressure reference so consecutive height measurements stay unbiased.
/// The z-Jacobian follows the world-to-body pose retraction:
/// `∂z/∂δρ = R[2,:]`, `∂z/∂δθ = −(R·p̂_w)[2,:]`.
#[derive(Clone)]
pub struct BarometricFactor<G: ActsOnPoints3> {
    /// Measured altitude [m].
    pub measured_height: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> BarometricFactor<G> {
    /// Create the factor from the pressure-derived altitude.
    pub fn new(measured_height: f64) -> Self {
        Self {
            measured_height,
            group: PhantomData,
        }
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for BarometricFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let origin = act_point(&G::from_param_slice(params[0]), &Vector3::zeros());
        let bias = params[1][0];
        residual[0] = origin.world.z + bias - self.measured_height;

        let Some(mut jac) = jacobian else { return };
        // ∂z/∂state is the z row of the origin's chain; ∂z/∂bias = 1.
        for col in 0..G::DOF {
            *jac.rb_mut().get_mut(0, col) = origin.d_state[(2, col)];
        }
        *jac.rb_mut().get_mut(0, G::DOF) = 1.0;
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, G::DOF + 1)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        match variables {
            [state, bias]
                if state.manifold_type_name() == G::NAME && bias.as_param_slice().len() == 1 =>
            {
                Ok(())
            }
            _ => Err(format!(
                "BarometricFactor expects [{} state, R¹ baro bias]",
                G::NAME
            )),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::se3::SE3;
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
        let factor = BarometricFactor::<SE3>::new(50.0);

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
