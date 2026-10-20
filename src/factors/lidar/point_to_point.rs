//! Pose-to-point correspondence factor (GTSAM `PoseToPointFactor` analogue).
//!
//! Constrains a body-to-world pose `T_wr` against a matched 3D point:
//!
//! ```text
//! r = T_wr · p_body − p_world_measured   (3D)
//! ```
//!
//! The correspondence is established upstream (ICP data association); the
//! factor only evaluates the single matched pair. Used for lidar/depth
//! odometry and as a building block for point-based loop closure.

use std::marker::PhantomData;

use apex_manifolds::capability::ActsOnPoints3;
use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::group_point::{act_point, expect_state_and_point};

/// Pose-to-point factor over `[T_wr, p_body]`, for any state that moves points.
#[derive(Clone)]
pub struct PoseToPointFactor<G: ActsOnPoints3> {
    /// 3D point measured in the world/target frame.
    pub measurement: Vector3<f64>,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> PoseToPointFactor<G> {
    /// Create the factor from the world-frame measured point.
    pub fn new(measurement: Vector3<f64>) -> Self {
        Self {
            measurement,
            group: PhantomData,
        }
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for PoseToPointFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let pose = G::from_param_slice(params[0]);
        let p_body = Vector3::new(params[1][0], params[1][1], params[1][2]);

        // `act` reports its own derivatives in the group's right convention:
        // asking the manifold keeps this factor on the group's conventions
        // instead of a second, hand-derived copy of them.
        let acted = act_point(&pose, &p_body);
        let r = acted.world - self.measurement;
        residual.copy_from_slice(r.as_slice());

        let Some(mut jac) = jacobian else { return };

        for row in 0..3 {
            for col in 0..G::DOF {
                *jac.rb_mut().get_mut(row, col) = acted.d_state[(row, col)];
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, G::DOF + col) = acted.d_point[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, G::DOF + 3)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_state_and_point::<G>(variables, "PoseToPointFactor")
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
    fn zero_residual_at_truth() -> TestResult<()> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, -0.3, 0.4),
            nalgebra::UnitQuaternion::from_euler_angles(0.05, 0.02, -0.01),
        ));
        let p_body = Vector3::new(0.4, -0.2, 2.5);
        let measured = pose.act(&p_body, None, None);
        let factor = PoseToPointFactor::<SE3>::new(measured);

        let mut residual = vec![0.0; 3];
        factor.linearize(
            &[pose.as_param_slice(), &[p_body.x, p_body.y, p_body.z]],
            &mut residual,
            None,
        );
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-12, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn finite_difference_jacobians() -> TestResult<()> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.2, 0.5, -0.1),
            nalgebra::UnitQuaternion::from_euler_angles(-0.04, 0.03, 0.06),
        ));
        let p_body = Vector3::new(0.3, 0.1, 1.8);
        let measured = pose.act(&p_body, None, None);
        let factor = PoseToPointFactor::<SE3>::new(measured);
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();
        let body_vec = [p_body.x, p_body.y, p_body.z];

        let mut r0 = vec![0.0; 3];
        let mut jac_buf = vec![0.0; 27];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 3, 9);
        factor.linearize(&[&pose_vec, &body_vec], &mut r0, Some(jac_mut));

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
            factor.linearize(&[&perturbed, &body_vec], &mut r_pert, None);
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let ana = jac_buf[col * 3 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "pose[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        for col in 0..3 {
            let mut plus = body_vec;
            let mut minus = body_vec;
            plus[col] += EPS;
            minus[col] -= EPS;
            let mut r_plus = vec![0.0; 3];
            let mut r_minus = vec![0.0; 3];
            factor.linearize(&[&pose_vec, &plus], &mut r_plus, None);
            factor.linearize(&[&pose_vec, &minus], &mut r_minus, None);
            for row in 0..3 {
                let fd = (r_plus[row] - r_minus[row]) / (2.0 * EPS);
                let ana = jac_buf[(6 + col) * 3 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "point[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }
}
