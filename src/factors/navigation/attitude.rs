//! Attitude factor from a reference direction (GTSAM `AttitudeFactor` /
//! `MagFactor` analogue).
//!
//! Constrains an SE(3) pose's rotation so a known world-frame direction
//! (gravity, or the magnetic field) maps onto the direction the sensor
//! measures in the body frame. One such factor pins two of the three rotation
//! degrees of freedom; add a gravity factor and a magnetometer factor for a
//! full AHRS.

use std::marker::PhantomData;

use apex_manifolds::capability::ActsOnPoints3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::core::{CoreError, CoreResult};
use crate::factors::Factor;
use crate::factors::common::group_point::act_point;

/// Attitude constraint from a measured reference direction (gravity via
/// accelerometer, or magnetic field via magnetometer).
///
/// The world-frame direction `d_world` (e.g. `[0, 0, −g]` for gravity) is
/// known; the instrument measures the same direction in the body frame,
/// `d_body_meas`. The connected SE3 pose follows the crate's world-to-body
/// convention (`p_body = R·p_world + t`), so a world direction expressed in
/// the body frame is `R·d_world`:
///
/// ```text
/// r = R·d_world − d_body_meas         (3D, effectively rank-2)
/// ```
///
/// Zero residual when the attitude is consistent with the measurement. Add
/// two of these factors (gravity + magnetometer) to fully constrain yaw —
/// the AHRS composition.
#[derive(Clone)]
pub struct AttitudeFactor<G: ActsOnPoints3> {
    /// Known reference direction in the world frame (normalized internally).
    pub direction_world: Vector3<f64>,
    /// Measured direction in the body frame (normalized internally).
    pub direction_body: Vector3<f64>,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> AttitudeFactor<G> {
    /// Create the factor; both directions are normalized, and near-zero
    /// directions are rejected.
    pub fn new(direction_world: Vector3<f64>, direction_body: Vector3<f64>) -> CoreResult<Self> {
        let nw = direction_world.norm();
        let nb = direction_body.norm();
        if !(nw.is_finite() && nw > 1e-12 && nb.is_finite() && nb > 1e-12) {
            return Err(CoreError::InvalidInput(
                "AttitudeFactor requires non-zero, finite directions".into(),
            ));
        }
        Ok(Self {
            direction_world: direction_world / nw,
            direction_body: direction_body / nb,
            group: PhantomData,
        })
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for AttitudeFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = G::from_param_slice(params[0]);

        // The rotation part of the action on a direction: the difference of
        // the action on `d` and on the origin cancels the translation, and the
        // normalization removes any scale (`Sim3`).
        let at_direction = act_point(&state, &self.direction_world);
        let at_origin = act_point(&state, &Vector3::zeros());
        let rotated = at_direction.world - at_origin.world;
        let norm = rotated.norm();
        let predicted_body = rotated / norm;
        let r = predicted_body - self.direction_body;
        residual.copy_from_slice(r.as_slice());

        let Some(mut jac) = jacobian else { return };

        // ∂n/∂v = (I − nnᵀ)/‖v‖, with v the rotated direction.
        let projector = (Matrix3::identity() - predicted_body * predicted_body.transpose()) / norm;
        let d_rotated = at_direction.d_state - at_origin.d_state;
        let d_r = projector * d_rotated;
        for row in 0..3 {
            for col in 0..G::DOF {
                *jac.rb_mut().get_mut(row, col) = d_r[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, G::DOF)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        match variables {
            [state] if state.manifold_type_name() == G::NAME => Ok(()),
            _ => Err(format!(
                "AttitudeFactor expects a single {} variable",
                G::NAME
            )),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se3::SE3Tangent;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;
    #[test]
    fn attitude_zero_residual_at_consistent_attitude() -> TestResult<()> {
        // Gravity measured in body frame for a known attitude.
        let g_world = Vector3::new(0.0, 0.0, -9.81);
        let attitude = nalgebra::UnitQuaternion::from_euler_angles(0.05, -0.1, 0.3);
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 2.0, 3.0),
            attitude,
        ));
        let g_body = attitude.to_rotation_matrix() * g_world;
        let factor = AttitudeFactor::<SE3>::new(g_world, g_body)?;

        let mut residual = vec![0.0; 3];
        factor.linearize(&[pose.as_param_slice()], &mut residual, None);
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-12, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn attitude_fd_jacobians() -> TestResult<()> {
        let g_world = Vector3::new(0.0, 0.0, -1.0);
        let attitude = nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.2, 0.15);
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, -0.2, 0.3),
            attitude,
        ));
        let g_body = attitude.to_rotation_matrix() * g_world;
        let factor = AttitudeFactor::<SE3>::new(g_world, g_body)?;
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();

        let mut r0 = vec![0.0; 3];
        let mut jac_buf = vec![0.0; 18];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 3, 6);
        factor.linearize(&[&pose_vec], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let perturbed: Vec<f64> = pose
                .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                .as_param_slice()
                .to_vec();
            let mut r_pert = vec![0.0; 3];
            factor.linearize(&[&perturbed], &mut r_pert, None);
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let ana = jac_buf[col * 3 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "J[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn attitude_rejects_zero_directions() {
        assert!(
            AttitudeFactor::<SE3>::new(Vector3::zeros(), Vector3::new(0.0, 0.0, -1.0)).is_err()
        );
        assert!(
            AttitudeFactor::<SE3>::new(Vector3::new(0.0, 0.0, -1.0), Vector3::zeros()).is_err()
        );
    }
}
