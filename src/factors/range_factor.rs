//! Range and bearing-range factors (GTSAM `RangeFactor` / `BearingRangeFactor`
//! analogues) — used for UWB, radar, and landmark-based positioning.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// 1D range constraint between the origins (translations) of two poses.
///
/// Residual: `‖p_i − p_j‖ − d_measured`. Jacobians act on the translation
/// tangent components only (rotation does not move the origins).
#[derive(Clone)]
pub struct PosePoseRangeFactor {
    /// Measured distance between the two frame origins [m].
    pub measured_range: f64,
}

impl PosePoseRangeFactor {
    /// Create the factor from the measured range.
    pub fn new(measured_range: f64) -> Self {
        Self { measured_range }
    }
}

impl Factor for PosePoseRangeFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            2,
            "PosePoseRangeFactor expects two SE3 blocks"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 7, "params[1] must be SE3 (7D)");

        let ti = SE3::from_param_slice(params[0]);
        let tj = SE3::from_param_slice(params[1]);
        let delta = ti.translation() - tj.translation();
        let dist = delta.norm();

        if dist < 1e-12 {
            residual[0] = -self.measured_range;
        } else {
            residual[0] = dist - self.measured_range;
        }

        let Some(mut jac) = jacobian else { return };

        // ∂dist/∂p_i = (p_i−p_j)ᵀ/d;  p_i' = p_i + R_i·δρ_i → ∂p_i/∂δρ_i = R_i.
        if dist < 1e-12 {
            for c in 0..12 {
                *jac.rb_mut().get_mut(0, c) = 0.0;
            }
            return;
        }
        let ri = ti.rotation_so3().rotation_matrix();
        let rj = tj.rotation_so3().rotation_matrix();
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = delta.dot(&ri.column(col)) / dist;
            *jac.rb_mut().get_mut(0, 6 + col) = -delta.dot(&rj.column(col)) / dist;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 12)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != SE3::REP_SIZE
            || variables[1].as_param_slice().len() != SE3::REP_SIZE
        {
            return Err("PosePoseRangeFactor expects two SE3 pose variables".into());
        }
        Ok(())
    }
}

/// 1D range constraint between a pose origin and a 3D point.
#[derive(Clone)]
pub struct PosePointRangeFactor {
    /// Measured distance between the pose origin and the point [m].
    pub measured_range: f64,
}

impl PosePointRangeFactor {
    /// Create the factor from the measured range.
    pub fn new(measured_range: f64) -> Self {
        Self { measured_range }
    }
}

impl Factor for PosePointRangeFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            2,
            "PosePointRangeFactor expects [pose, point]"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be a 3D point");

        let pose = SE3::from_param_slice(params[0]);
        let point = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let delta = pose.translation() - point;
        let dist = delta.norm();

        if dist < 1e-12 {
            residual[0] = -self.measured_range;
        } else {
            residual[0] = dist - self.measured_range;
        }

        let Some(mut jac) = jacobian else { return };
        if dist < 1e-12 {
            for c in 0..9 {
                *jac.rb_mut().get_mut(0, c) = 0.0;
            }
            return;
        }
        let rotation = pose.rotation_so3().rotation_matrix();
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = delta.dot(&rotation.column(col)) / dist;
            *jac.rb_mut().get_mut(0, 6 + col) = -delta[col] / dist;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != SE3::REP_SIZE
            || variables[1].as_param_slice().len() != 3
        {
            return Err("PosePointRangeFactor expects [SE3 pose, 3D point]".into());
        }
        Ok(())
    }
}

/// Bearing + range constraint between a pose and a 3D landmark.
///
/// Residual (4D = 3 bearing rows + 1 range row): the unit bearing from the
/// pose to the point, expressed in the body frame, compared against the
/// measured unit direction, plus the scalar range error:
///
/// ```text
/// b_pred = R·(p_world − t)/‖p_world − t‖     (world-to-body pose convention)
/// r      = [ b_pred − b_measured ; ‖p_world − t‖ − d_measured ]
/// ```
#[derive(Clone)]
pub struct BearingRangeFactor {
    /// Measured unit bearing in the body frame.
    pub measured_bearing: Vector3<f64>,
    /// Measured range [m].
    pub measured_range: f64,
}

impl BearingRangeFactor {
    /// Create the factor; the bearing is normalized internally.
    pub fn new(measured_bearing: Vector3<f64>, measured_range: f64) -> Result<Self, String> {
        let n = measured_bearing.norm();
        if !(n.is_finite() && n > 1e-12) {
            return Err("measured bearing must be non-zero and finite".into());
        }
        Ok(Self {
            measured_bearing: measured_bearing / n,
            measured_range,
        })
    }
}

impl Factor for BearingRangeFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "BearingRangeFactor expects [pose, point]");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be a 3D point");

        let pose = SE3::from_param_slice(params[0]);
        let point = Vector3::new(params[1][0], params[1][1], params[1][2]);

        let rotation = pose.rotation_so3().rotation_matrix();
        let delta_w = point - pose.translation();
        let dist = delta_w.norm();

        if dist < 1e-12 {
            // Landmark exactly at the pose origin: bearing undefined.
            residual.copy_from_slice(self.measured_bearing.as_slice());
            if let Some(mut jac) = jacobian {
                for c in 0..9 {
                    *jac.rb_mut().get_mut(0, c) = 0.0;
                }
            }
            return;
        }

        let b_pred = rotation * (delta_w / dist);
        residual[0..3].copy_from_slice((b_pred - self.measured_bearing).as_slice());
        residual[3] = dist - self.measured_range;

        let Some(mut jac) = jacobian else { return };

        // δp_body = R(δθ × q) − R·R·δρ with q = p_world − t (the body-tangent
        // translation step moves the origin by R·δρ in the world frame);
        // b = p_body/d:
        // ∂b/∂(δρ, δθ) = (I − bbᵀ)/d · [−R² | −R·q̂]; ∂b/∂p_world = (I − bbᵀ)R/d.
        let q_x = Matrix3::new(
            0.0, -delta_w.z, delta_w.y, delta_w.z, 0.0, -delta_w.x, -delta_w.y, delta_w.x, 0.0,
        );
        let proj = Matrix3::identity() - b_pred * b_pred.transpose();
        let scale = 1.0 / dist;
        let d_b_d_rho = (proj * (-rotation * rotation)) * scale; // 3×3
        let d_b_d_theta = (proj * (-rotation * q_x)) * scale; // 3×3
        let d_b_d_point = (proj * rotation) * scale; // 3×3

        let mut j_pose = SMatrix::<f64, 3, 6>::zeros();
        j_pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&d_b_d_rho);
        j_pose.fixed_view_mut::<3, 3>(0, 3).copy_from(&d_b_d_theta);
        for row in 0..3 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = j_pose[(row, col)];
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, 6 + col) = d_b_d_point[(row, col)];
            }
        }
        // Range row: d = ‖q‖ with q = p_world − t. The body-tangent
        // translation step moves the origin by R·δρ in the world, so
        // ∂d/∂δρ = −q̂ᵀR; rotation does not change the origin; point block
        // is q̂ᵀ.
        let q_hat = delta_w / dist;
        for col in 0..3 {
            *jac.rb_mut().get_mut(3, col) = -q_hat.dot(&rotation.column(col));
            *jac.rb_mut().get_mut(3, 3 + col) = 0.0;
            *jac.rb_mut().get_mut(3, 6 + col) = q_hat[col];
        }
    }

    fn residual_dim(&self) -> usize {
        4
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (4, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != SE3::REP_SIZE
            || variables[1].as_param_slice().len() != 3
        {
            return Err("BearingRangeFactor expects [SE3 pose, 3D point]".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;

    fn fd_check_1d(
        factor: &dyn Factor,
        pose: &SE3,
        second: Vec<f64>,
        second_is_pose: bool,
    ) -> Result<(), String> {
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
                return Err(format!("pose[{col}]: analytical={ana:.6} fd={fd:.6}"));
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
                return Err(format!("second[{col}]: analytical={ana:.6} fd={fd:.6}"));
            }
        }
        Ok(())
    }

    #[test]
    fn pose_pose_range_zero_residual_and_fd() -> Result<(), String> {
        let ti = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.2, 0.4, -0.1),
            nalgebra::UnitQuaternion::from_euler_angles(0.02, 0.03, -0.01),
        ));
        let tj = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(-0.5, 0.1, 1.2),
            nalgebra::UnitQuaternion::from_euler_angles(-0.04, 0.01, 0.05),
        ));
        let d = (ti.translation() - tj.translation()).norm();
        let factor = PosePoseRangeFactor::new(d);
        let ti_v: Vec<f64> = ti.as_param_slice().to_vec();
        let tj_v: Vec<f64> = tj.as_param_slice().to_vec();

        let mut residual = vec![0.0; 1];
        factor.linearize(&[&ti_v, &tj_v], &mut residual, None);
        assert!(residual[0].abs() < 1e-12);

        fd_check_1d(&factor, &ti, tj_v.clone(), true)?;
        Ok(())
    }

    #[test]
    fn pose_point_range_zero_residual_and_fd() -> Result<(), String> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.3, -0.2, 0.6),
            nalgebra::UnitQuaternion::from_euler_angles(0.05, -0.02, 0.04),
        ));
        let point = Vector3::new(2.0, 1.5, -0.5);
        let d = (pose.translation() - point).norm();
        let factor = PosePointRangeFactor::new(d);
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();
        let point_v = vec![point.x, point.y, point.z];

        let mut residual = vec![0.0; 1];
        factor.linearize(&[&pose_v, &point_v], &mut residual, None);
        assert!(residual[0].abs() < 1e-12);

        fd_check_1d(&factor, &pose, point_v, false)?;
        Ok(())
    }

    #[test]
    fn bearing_range_zero_residual_and_fd() -> Result<(), String> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, 0.3, -0.2),
            nalgebra::UnitQuaternion::from_euler_angles(0.03, 0.02, -0.05),
        ));
        let point = Vector3::new(1.5, -0.5, 2.5);
        let delta = point - pose.translation();
        let bearing = pose.rotation_so3().rotation_matrix() * (delta / delta.norm());
        let factor = BearingRangeFactor::new(bearing, delta.norm())?;
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

    #[test]
    fn bearing_rejects_zero_direction() {
        assert!(BearingRangeFactor::new(Vector3::zeros(), 1.0).is_err());
    }
}
