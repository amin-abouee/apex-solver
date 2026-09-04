//! Partial pose priors (GTSAM `PoseRotationPrior` / `PoseTranslationPrior`
//! analogues) — constrain one component of an SE3 pose, leaving the other
//! free. Used for loop-closure initialization (rotation-only loop edges) and
//! yaw anchoring.

use apex_manifolds::LieGroup;
use apex_manifolds::Tangent;
use apex_manifolds::se3::SE3;
use apex_manifolds::so3::SO3Tangent;
use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Rotation-only prior on an SE3 pose.
///
/// Residual (3D): `Log(R_meas⁻¹·R)` — the tangent-space rotation error;
/// translation is unconstrained.
#[derive(Clone)]
pub struct PoseRotationPrior {
    /// Prior rotation (as an SE3 whose attitude is used; position ignored).
    pub measured_rotation: SE3,
}

impl PoseRotationPrior {
    /// Create the rotation prior.
    pub fn new(measured_rotation: SE3) -> Self {
        Self { measured_rotation }
    }
}

impl Factor for PoseRotationPrior {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 1, "PoseRotationPrior expects one SE3 block");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");

        let pose = SE3::from_param_slice(params[0]);
        let r_prior = self.measured_rotation.rotation_so3();
        let r_cur = pose.rotation_so3();
        let rel = r_prior.inverse(None).compose(&r_cur, None, None);
        let theta = rel.log(None);
        residual[0..3].copy_from_slice(theta.coeffs().as_slice());

        let Some(mut jac) = jacobian else { return };
        // d Log(R₀ᵀ R exp(δθ))/dδθ = Jr⁻¹(θ); translation untouched.
        let jr_inv = SO3Tangent::new(theta.coeffs()).right_jacobian_inv();
        for row in 0..3 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = if col >= 3 {
                    jr_inv[(row, col - 3)]
                } else {
                    0.0
                };
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 1 || variables[0].as_param_slice().len() != SE3::REP_SIZE {
            return Err("PoseRotationPrior expects a single SE3 pose variable".into());
        }
        Ok(())
    }
}

/// Translation-only prior on an SE3 pose.
///
/// Residual (3D): `t − t_measured`; rotation is unconstrained.
#[derive(Clone)]
pub struct PoseTranslationPrior {
    /// Prior translation [m].
    pub measured_translation: Vector3<f64>,
}

impl PoseTranslationPrior {
    /// Create the translation prior.
    pub fn new(measured_translation: Vector3<f64>) -> Self {
        Self {
            measured_translation,
        }
    }
}

impl Factor for PoseTranslationPrior {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            1,
            "PoseTranslationPrior expects one SE3 block"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");

        let pose = SE3::from_param_slice(params[0]);
        let r = pose.translation() - self.measured_translation;
        residual.copy_from_slice(r.as_slice());

        let Some(mut jac) = jacobian else { return };
        // Right-plus retraction: t' = t + R·δρ → ∂t/∂δρ = R.
        let rotation = pose.rotation_so3().rotation_matrix();
        for row in 0..3 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = if col < 3 { rotation[(row, col)] } else { 0.0 };
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 1 || variables[0].as_param_slice().len() != SE3::REP_SIZE {
            return Err("PoseTranslationPrior expects a single SE3 pose variable".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3Tangent;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    fn sample_pose() -> SE3 {
        SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.4, -0.7, 1.1),
            nalgebra::UnitQuaternion::from_euler_angles(0.06, -0.03, 0.12),
        ))
    }

    #[test]
    fn rotation_prior_zero_residual_at_prior() -> TestResult<()> {
        let pose = sample_pose();
        let factor = PoseRotationPrior::new(pose.clone());
        let mut residual = vec![0.0; 3];
        factor.linearize(&[pose.as_param_slice()], &mut residual, None);
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-12, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn rotation_prior_fd_jacobians() -> TestResult<()> {
        let pose = sample_pose();
        let prior = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(9.0, 9.0, 9.0), // position must be ignored
            nalgebra::UnitQuaternion::from_euler_angles(-0.02, 0.05, 0.1),
        ));
        let factor = PoseRotationPrior::new(prior);
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();

        let mut r0 = vec![0.0; 3];
        let mut jac_buf = vec![0.0; 18];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 3, 6);
        factor.linearize(&[&pose_v], &mut r0, Some(jac_mut));

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
                    "rot J[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn translation_prior_zero_residual_and_fd() -> TestResult<()> {
        let pose = sample_pose();
        let factor = PoseTranslationPrior::new(pose.translation());
        let pose_v: Vec<f64> = pose.as_param_slice().to_vec();

        let mut residual = vec![0.0; 3];
        let mut jac_buf = vec![0.0; 18];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 3, 6);
        factor.linearize(&[&pose_v], &mut residual, Some(jac_mut));
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-12, "residual[{i}] = {r}");
        }

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
                let fd = (r_pert[row] - residual[row]) / EPS;
                let ana = jac_buf[col * 3 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "trans J[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn translation_prior_ignores_rotation_change() {
        // Rotating the pose about its own origin must not change the
        // translation residual.
        let pose = sample_pose();
        let factor = PoseTranslationPrior::new(pose.translation());
        let mut r1 = vec![0.0; 3];
        factor.linearize(&[pose.as_param_slice()], &mut r1, None);

        let mut tan = [0.0f64; 6];
        tan[4] = 0.1;
        let rotated = pose.right_plus(&SE3Tangent::from_slice(&tan), None, None);
        let mut r2 = vec![0.0; 3];
        factor.linearize(&[rotated.as_param_slice()], &mut r2, None);
        for i in 0..3 {
            assert!((r1[i] - r2[i]).abs() < 1e-9);
        }
    }
}
