//! Range and bearing-range factors (GTSAM `RangeFactor` / `BearingRangeFactor`
//! analogues) — used for UWB, radar, and landmark-based positioning.
//!
//! Generic over every group that moves 3-D points ([`ActsOnPoints3`]): the
//! frame origin is the group's action on the zero point, so range rows pick up
//! exactly the translation columns of the group's own `act` Jacobian.

use std::marker::PhantomData;

use apex_manifolds::capability::ActsOnPoints3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

use crate::core::CoreError;
use crate::core::CoreResult;
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::group_point::{
    act_point, expect_state_and_point, expect_two_states, into_anchor,
};

/// 1D range constraint between the origins (translations) of two states.
///
/// Residual: `‖p_i − p_j‖ − d_measured`. Jacobians act on the origin's
/// derivative in each state's tangent (rotation does not move the origin).
#[derive(Clone)]
pub struct PosePoseRangeFactor<G: ActsOnPoints3> {
    /// Measured distance between the two frame origins [m].
    pub measured_range: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> PosePoseRangeFactor<G> {
    /// Create the factor from the measured range.
    pub fn new(measured_range: f64) -> Self {
        Self {
            measured_range,
            group: PhantomData,
        }
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for PosePoseRangeFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let origin = Vector3::zeros();
        let ti = act_point(&G::from_param_slice(params[0]), &origin);
        let tj = act_point(&G::from_param_slice(params[1]), &origin);
        let delta = ti.world - tj.world;
        let dist = delta.norm();

        if dist < 1e-12 {
            residual[0] = -self.measured_range;
        } else {
            residual[0] = dist - self.measured_range;
        }

        let Some(mut jac) = jacobian else { return };

        // ∂dist/∂p_i = (p_i−p_j)ᵀ/d, carried through each state's origin chain.
        if dist < 1e-12 {
            for c in 0..2 * G::DOF {
                *jac.rb_mut().get_mut(0, c) = 0.0;
            }
            return;
        }
        for col in 0..G::DOF {
            *jac.rb_mut().get_mut(0, col) = delta.dot(&ti.d_state.column(col)) / dist;
            *jac.rb_mut().get_mut(0, G::DOF + col) = -delta.dot(&tj.d_state.column(col)) / dist;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 2 * G::DOF)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_two_states::<G>(variables, "PosePoseRangeFactor")
    }
}

/// 1D range constraint between a state origin and a 3D point.
#[derive(Clone)]
pub struct PosePointRangeFactor<G: ActsOnPoints3> {
    /// Measured distance between the state origin and the point [m].
    pub measured_range: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> PosePointRangeFactor<G> {
    /// Create the factor from the measured range.
    pub fn new(measured_range: f64) -> Self {
        Self {
            measured_range,
            group: PhantomData,
        }
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for PosePointRangeFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let origin = act_point(&G::from_param_slice(params[0]), &Vector3::zeros());
        let point = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let delta = origin.world - point;
        let dist = delta.norm();

        if dist < 1e-12 {
            residual[0] = -self.measured_range;
        } else {
            residual[0] = dist - self.measured_range;
        }

        let Some(mut jac) = jacobian else { return };
        if dist < 1e-12 {
            for c in 0..G::DOF + 3 {
                *jac.rb_mut().get_mut(0, c) = 0.0;
            }
            return;
        }
        for col in 0..G::DOF {
            *jac.rb_mut().get_mut(0, col) = delta.dot(&origin.d_state.column(col)) / dist;
        }
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, G::DOF + col) = -delta[col] / dist;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, G::DOF + 3)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_state_and_point::<G>(variables, "PosePointRangeFactor")
    }
}

/// Bearing + range constraint between a state and a 3D landmark.
///
/// Residual (4D = 3 bearing rows + 1 range row): the unit bearing from the
/// state to the point, expressed in the body frame, compared against the
/// measured unit direction, plus the scalar range error:
///
/// ```text
/// b_pred = u/‖u‖,  u = X⁻¹ ⊙ p_world          (body-frame direction)
/// r      = [ b_pred − b_measured ; ‖p_world − origin(X)‖ − d_measured ]
/// ```
#[derive(Clone)]
pub struct BearingRangeFactor<G: ActsOnPoints3> {
    /// Measured unit bearing in the body frame.
    pub measured_bearing: Vector3<f64>,
    /// Measured range [m].
    pub measured_range: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> BearingRangeFactor<G> {
    /// Create the factor; the bearing is normalized internally.
    pub fn new(measured_bearing: Vector3<f64>, measured_range: f64) -> CoreResult<Self> {
        let n = measured_bearing.norm();
        if !(n.is_finite() && n > 1e-12) {
            return Err(CoreError::InvalidInput(
                "measured bearing must be non-zero and finite".into(),
            ));
        }
        Ok(Self {
            measured_bearing: measured_bearing / n,
            measured_range,
            group: PhantomData,
        })
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for BearingRangeFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = G::from_param_slice(params[0]);
        let point = Vector3::new(params[1][0], params[1][1], params[1][2]);

        let origin = act_point(&state, &Vector3::zeros());
        let delta_w = point - origin.world;
        let dist = delta_w.norm();

        if dist < 1e-12 {
            // Landmark exactly at the pose origin: bearing undefined.
            residual.copy_from_slice(self.measured_bearing.as_slice());
            if let Some(mut jac) = jacobian {
                for row in 0..4 {
                    for c in 0..G::DOF + 3 {
                        *jac.rb_mut().get_mut(row, c) = 0.0;
                    }
                }
            }
            return;
        }

        // Body-frame direction to the landmark. The state is body-in-world, so
        // the landmark is carried *into* the body by the group's inverse.
        let chain = into_anchor(&state, &point);
        let p_body = chain.point;
        let body_norm = p_body.norm();
        let b_pred = p_body / body_norm;
        residual[0..3].copy_from_slice((b_pred - self.measured_bearing).as_slice());
        residual[3] = dist - self.measured_range;

        let Some(mut jac) = jacobian else { return };

        // ∂b/∂u = (I − bbᵀ)/‖u‖, then through the group's own chain.
        let proj = (Matrix3::identity() - b_pred * b_pred.transpose()) / body_norm;
        let d_b_d_point = proj * chain.d_world;
        for (col, axis) in chain.d_anchor.iter().enumerate() {
            let column = proj * Vector3::from_column_slice(axis);
            for row in 0..3 {
                *jac.rb_mut().get_mut(row, col) = column[row];
            }
        }
        for row in 0..3 {
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, G::DOF + col) = d_b_d_point[(row, col)];
            }
        }
        // Range row: d = ‖p − origin(X)‖. Only the origin moves with the state.
        let q_hat = delta_w / dist;
        for col in 0..G::DOF {
            *jac.rb_mut().get_mut(3, col) = -q_hat.dot(&origin.d_state.column(col));
        }
        for col in 0..3 {
            *jac.rb_mut().get_mut(3, G::DOF + col) = q_hat[col];
        }
    }

    fn residual_dim(&self) -> usize {
        4
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (4, G::DOF + 3)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_state_and_point::<G>(variables, "BearingRangeFactor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::{LieGroup, Tangent};

    fn fd_check_1d(
        factor: &dyn Factor,
        pose: &SE3,
        second: Vec<f64>,
        second_is_pose: bool,
    ) -> Result<(), CoreError> {
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();
        let cols = factor.jacobian_shape().1;
        let mut r0 = vec![0.0; 1];
        let mut jac_buf = vec![0.0; cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 1, cols);
        factor.linearize(&[&pose_vec, &second], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let perturbed: Vec<f64> = pose
                .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                .as_param_slice()
                .to_vec();
            let mut r_pert = vec![0.0; 1];
            factor.linearize(&[&perturbed, &second], &mut r_pert, None);
            let fd = (r_pert[0] - r0[0]) / EPS;
            let ana = jac_buf[col];
            if (fd - ana).abs() > TOL {
                return Err(CoreError::InvalidInput(format!(
                    "pose[{col}]: analytical={ana:.6} fd={fd:.6}"
                )));
            }
        }
        let second_len = if second_is_pose { 6 } else { second.len() };
        for col in 0..second_len {
            let (plus, minus) = if second_is_pose {
                // SE3 block: perturb in the manifold tangent.
                let mut tan = [0.0f64; 6];
                tan[col] = EPS;
                let se3 = SE3::from_param_slice(&second);
                (
                    se3.right_plus(&SE3Tangent::from_slice(&tan), None, None)
                        .as_param_slice()
                        .to_vec(),
                    {
                        let mut tan_m = tan;
                        tan_m[col] = -EPS;
                        se3.right_plus(&SE3Tangent::from_slice(&tan_m), None, None)
                            .as_param_slice()
                            .to_vec()
                    },
                )
            } else {
                let mut plus = second.clone();
                let mut minus = second.clone();
                plus[col] += EPS;
                minus[col] -= EPS;
                (plus, minus)
            };
            let mut r_plus = vec![0.0; 1];
            let mut r_minus = vec![0.0; 1];
            factor.linearize(&[&pose_vec, &plus], &mut r_plus, None);
            factor.linearize(&[&pose_vec, &minus], &mut r_minus, None);
            let fd = (r_plus[0] - r_minus[0]) / (2.0 * EPS);
            let ana = jac_buf[6 + col];
            if (fd - ana).abs() > TOL {
                return Err(CoreError::InvalidInput(format!(
                    "second[{col}]: analytical={ana:.6} fd={fd:.6}"
                )));
            }
        }
        Ok(())
    }

    #[test]
    fn pose_pose_range_zero_residual_and_fd() -> Result<(), CoreError> {
        let ti = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.2, 0.4, -0.1),
            nalgebra::UnitQuaternion::from_euler_angles(0.02, 0.03, -0.01),
        ));
        let tj = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(-0.5, 0.1, 1.2),
            nalgebra::UnitQuaternion::from_euler_angles(-0.04, 0.01, 0.05),
        ));
        let d = (ti.translation() - tj.translation()).norm();
        let factor = PosePoseRangeFactor::<SE3>::new(d);
        let ti_v: Vec<f64> = ti.as_param_slice().to_vec();
        let tj_v: Vec<f64> = tj.as_param_slice().to_vec();

        let mut residual = vec![0.0; 1];
        factor.linearize(&[&ti_v, &tj_v], &mut residual, None);
        assert!(residual[0].abs() < 1e-12);

        fd_check_1d(&factor, &ti, tj_v.clone(), true)?;
        Ok(())
    }

    #[test]
    fn pose_point_range_zero_residual_and_fd() -> Result<(), CoreError> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.3, -0.2, 0.6),
            nalgebra::UnitQuaternion::from_euler_angles(0.05, -0.02, 0.04),
        ));
        let point = Vector3::new(2.0, 1.5, -0.5);
        let d = (pose.translation() - point).norm();
        let factor = PosePointRangeFactor::<SE3>::new(d);
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();
        let point_v = vec![point.x, point.y, point.z];

        let mut residual = vec![0.0; 1];
        factor.linearize(&[&pose_v, &point_v], &mut residual, None);
        assert!(residual[0].abs() < 1e-12);

        fd_check_1d(&factor, &pose, point_v, false)?;
        Ok(())
    }

    #[test]
    fn bearing_range_zero_residual_and_fd() -> Result<(), CoreError> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, 0.3, -0.2),
            nalgebra::UnitQuaternion::from_euler_angles(0.03, 0.02, -0.05),
        ));
        let point = Vector3::new(1.5, -0.5, 2.5);
        let delta = point - pose.translation();
        let bearing = pose.rotation_so3().rotation_matrix().transpose() * (delta / delta.norm());
        let factor = BearingRangeFactor::<SE3>::new(bearing, delta.norm())?;
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();
        let point_v = vec![point.x, point.y, point.z];

        let mut residual = vec![0.0; 4];
        factor.linearize(&[&pose_v, &point_v], &mut residual, None);
        for r in &residual {
            assert!(r.abs() < 1e-12);
        }

        // FD on all 3 rows, pose block and point block.
        let mut r0 = vec![0.0; 4];
        let mut jac_buf = vec![0.0; 36];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 4, 9);
        factor.linearize(&[&pose_v, &point_v], &mut r0, Some(jac_mut));
        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let perturbed: Vec<f64> = pose
                .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                .as_param_slice()
                .to_vec();
            let mut r_pert = vec![0.0; 4];
            factor.linearize(&[&perturbed, &point_v], &mut r_pert, None);
            for row in 0..4 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let ana = jac_buf[col * 4 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "pose[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        for col in 0..3 {
            let mut plus = point_v.clone();
            let mut minus = point_v.clone();
            plus[col] += EPS;
            minus[col] -= EPS;
            let mut r_plus = vec![0.0; 4];
            let mut r_minus = vec![0.0; 4];
            factor.linearize(&[&pose_v, &plus], &mut r_plus, None);
            factor.linearize(&[&pose_v, &minus], &mut r_minus, None);
            for row in 0..4 {
                let fd = (r_plus[row] - r_minus[row]) / (2.0 * EPS);
                let ana = jac_buf[(6 + col) * 4 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "point[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }

    /// A hand-computed bearing, which is what actually pins the SE(3)
    /// convention. Finite differences cannot: they only check the Jacobian
    /// against this factor's own residual, so a residual that rotates the wrong
    /// way stays self-consistent and passes. That is exactly how
    /// `BearingRangeFactor` shipped using `R` where it needed `Rᵀ`.
    #[test]
    fn bearing_range_golden_value_pins_the_convention() -> Result<(), CoreError> {
        // Body yawed +90° about z at the origin. A landmark 2 m along world +x
        // therefore sits along the body's −y axis.
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        ));
        let point = Vector3::new(2.0, 0.0, 0.0);
        let expected = Vector3::new(0.0, -1.0, 0.0);

        let factor = BearingRangeFactor::<SE3>::new(expected, 2.0)?;
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();
        let point_v = vec![point.x, point.y, point.z];
        let mut residual = vec![0.0; 4];
        factor.linearize(&[&pose_v, &point_v], &mut residual, None);

        for (i, r) in residual.iter().enumerate() {
            assert!(
                r.abs() < 1e-12,
                "expected zero residual for the hand-computed bearing, row {i} = {r:.3e}"
            );
        }
        Ok(())
    }

    /// `BearingFactor` and `BearingRangeFactor` must agree on which way the
    /// bearing points; they disagreed before this convention was fixed.
    #[test]
    fn bearing_range_agrees_with_bearing_factor() -> Result<(), CoreError> {
        use crate::factors::ranging::bearing::BearingFactor;

        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, 0.3, -0.2),
            nalgebra::UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5),
        ));
        let point = Vector3::new(1.5, -0.5, 2.5);
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();
        let point_v = vec![point.x, point.y, point.z];

        // A bearing that BearingFactor reports as exact must also be exact for
        // BearingRangeFactor's first three rows.
        let delta = point - pose.translation();
        let truth = pose.rotation_so3().rotation_matrix().transpose() * (delta / delta.norm());

        let bearing_only = BearingFactor::<SE3>::new_isotropic(truth, 1.0);
        let mut r_bearing = vec![0.0; 2];
        bearing_only.linearize(&[&pose_v, &point_v], &mut r_bearing, None);
        for (i, r) in r_bearing.iter().enumerate() {
            assert!(r.abs() < 1e-12, "BearingFactor row {i} = {r:.3e}");
        }

        let combined = BearingRangeFactor::<SE3>::new(truth, delta.norm())?;
        let mut r_combined = vec![0.0; 4];
        combined.linearize(&[&pose_v, &point_v], &mut r_combined, None);
        for (i, r) in r_combined.iter().enumerate() {
            assert!(r.abs() < 1e-12, "BearingRangeFactor row {i} = {r:.3e}");
        }
        Ok(())
    }

    #[test]
    fn bearing_rejects_zero_direction() {
        assert!(BearingRangeFactor::<SE3>::new(Vector3::zeros(), 1.0).is_err());
    }
}
