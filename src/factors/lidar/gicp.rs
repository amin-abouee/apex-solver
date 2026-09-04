//! Generalized ICP (GICP) plane-to-plane factor.
//!
//! Follows Segal et al. (ICRA 2009): each correspondence aligns a source
//! **plane** (local covariance `C_body` degenerate along the surface normal)
//! with a target plane. The 3D residual
//!
//! ```text
//! e = T_wr · p_body − p_target        (3D)
//! ```
//!
//! is Mahalanobis-whitened by the combined covariance
//! `C = C_target + R·C_body·Rᵀ`, so tangential directions (flat in both
//! clouds) contribute almost nothing and the normal direction dominates —
//! plane-to-plane registration emerges from plain point-to-point with the
//! right metric.
//!
//! Following common practice (and GTSAM-external GICP factors), the combined
//! covariance is evaluated **once at construction** using a caller-supplied
//! rotation hint (typically the current state estimate) and then held fixed.
//! Callers rebuild correspondences (and this factor) every scan-matching
//! iteration, so the frozen covariance never goes stale; freezing it also
//! keeps the residual/Jacobian pair exactly consistent.
//!
//! Parameter blocks: `[T_wr (7), p_body (3)]` — 9 minimal DOF, 3D residual.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Minimum eigenvalue clamp used when whitening the combined covariance, so
/// near-flat tangential directions do not blow the inverse up.
const MIN_EIGENVALUE: f64 = 1e-9;

/// GICP plane-to-plane factor over `[T_wr, p_body]`.
#[derive(Clone)]
pub struct GicpFactor {
    /// Target point (world frame).
    pub target_point: Vector3<f64>,
    /// Target-plane local covariance (world frame), e.g. from the target
    /// cloud's local neighborhood.
    pub target_covariance: Matrix3<f64>,
    /// Source-point local covariance in the **body** frame.
    pub body_covariance: Matrix3<f64>,
    /// Combined covariance frozen at construction:
    /// `C_target + R_hint·C_body·R_hintᵀ`.
    pub combined: Matrix3<f64>,
}

impl GicpFactor {
    /// Create the factor. Covariances must be finite; degenerate (zero)
    /// covariances are clamped to a small isotropic floor when whitening.
    pub fn new(
        target_point: Vector3<f64>,
        target_covariance: Matrix3<f64>,
        body_covariance: Matrix3<f64>,
        rotation_hint: Matrix3<f64>,
    ) -> Result<Self, String> {
        for c in [&target_covariance, &body_covariance] {
            if c.iter().any(|v| !v.is_finite()) {
                return Err("GICP covariances must be finite".into());
            }
        }
        let rotated = rotation_hint * body_covariance * rotation_hint.transpose();
        let full = target_covariance + rotated;
        let combined = 0.5 * (full + full.transpose());
        Ok(Self {
            target_point,
            target_covariance,
            body_covariance,
            combined,
        })
    }

    /// `C^{-1/2}` via symmetric eigendecomposition with eigenvalue clamping.
    fn whiten(c: &Matrix3<f64>) -> Matrix3<f64> {
        let symm = nalgebra::SymmetricEigen::new(*c);
        let mut sqrt_inv_diag = Vector3::zeros();
        for i in 0..3 {
            let lam = symm.eigenvalues[i].max(MIN_EIGENVALUE);
            sqrt_inv_diag[i] = 1.0 / lam.sqrt();
        }
        symm.eigenvectors * Matrix3::from_diagonal(&sqrt_inv_diag) * symm.eigenvectors.transpose()
    }
}

impl Factor for GicpFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "GicpFactor expects [T_wr, p_body]");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be a 3D point");

        let pose = SE3::from_param_slice(params[0]);
        let p_body = Vector3::new(params[1][0], params[1][1], params[1][2]);

        let mut j_pose_act = SE3::zero_jacobian();
        let mut j_point_act = Matrix3::zeros();
        let predicted = pose.act(&p_body, Some(&mut j_pose_act), Some(&mut j_point_act));
        let e = predicted - self.target_point;

        // Whitening is applied internally (this factor *is* the noise model);
        // register it with NoiseModel::null().
        let c_inv_sqrt = Self::whiten(&self.combined);
        let r = c_inv_sqrt * e;
        residual.copy_from_slice(r.as_slice());

        let Some(mut jac) = jacobian else { return };

        // `act` supplies ∂e/∂(δρ, δθ) and ∂e/∂p_body in SE(3)'s right
        // convention; whitening left-multiplies by C^{-1/2} (the
        // rotation-dependence of C is dropped: standard small-angle
        // approximation, see module docs).
        let j_pose = c_inv_sqrt * j_pose_act.fixed_view::<3, 6>(0, 0);
        let j_point = c_inv_sqrt * j_point_act;

        for row in 0..3 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = j_pose[(row, col)];
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, 6 + col) = j_point[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != SE3::REP_SIZE
            || variables[1].as_param_slice().len() != 3
        {
            return Err("GicpFactor expects [SE3 pose, 3D point]".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    /// A plane-land plane covariance: flat in x/y, stiff along z.
    fn plane_cov_along_z() -> Matrix3<f64> {
        Matrix3::from_diagonal(&Vector3::new(1.0e-2, 1.0e-2, 1.0e-6))
    }

    #[test]
    fn zero_residual_at_truth() -> TestResult<()> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.3, 0.1, -0.2),
            nalgebra::UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03),
        ));
        let p_body = Vector3::new(0.5, 0.2, 1.4);
        let target = pose.act(&p_body, None, None);
        let factor = GicpFactor::new(
            target,
            plane_cov_along_z(),
            plane_cov_along_z(),
            pose.rotation_so3().rotation_matrix(),
        )?;

        let mut residual = vec![0.0; 3];
        factor.linearize(
            &[pose.as_param_slice(), &[p_body.x, p_body.y, p_body.z]],
            &mut residual,
            None,
        );
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-10, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn finite_difference_jacobians() -> TestResult<()> {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, -0.4, 0.2),
            nalgebra::UnitQuaternion::from_euler_angles(0.02, 0.04, -0.03),
        ));
        let p_body = Vector3::new(0.6, -0.3, 2.0);
        let target = pose.act(&p_body, None, None);
        let factor = GicpFactor::new(
            target + Vector3::new(0.01, -0.02, 0.03),
            plane_cov_along_z(),
            plane_cov_along_z(),
            pose.rotation_so3().rotation_matrix(),
        )?;
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();
        let body_vec = [p_body.x, p_body.y, p_body.z];

        let mut r0 = vec![0.0; 3];
        let mut jac_buf = vec![0.0; 27];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 3, 9);
        factor.linearize(&[&pose_vec, &body_vec], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-3;
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

    #[test]
    fn tangential_directions_are_suppressed() -> TestResult<()> {
        // A pure tangential error (in the plane) must be whitened to almost
        // nothing, while a normal-direction error of the same size must
        // survive — this is what makes GICP plane-to-plane.
        let pose = SE3::identity();
        let factor = GicpFactor::new(
            Vector3::new(0.0, 0.0, 1.0), // 1 m normal error
            plane_cov_along_z(),
            plane_cov_along_z(),
            Matrix3::identity(),
        )?;
        let mut residual = vec![0.0; 3];
        factor.linearize(
            &[pose.as_param_slice(), &[0.0f64, 0.0, 0.0]],
            &mut residual,
            None,
        );
        let normal_component = residual[2];
        assert!(
            normal_component.abs() > 100.0,
            "normal error should stay large: {normal_component}"
        );

        let tangential = GicpFactor::new(
            Vector3::new(1.0, 0.0, 0.0), // 1 m tangential error
            plane_cov_along_z(),
            plane_cov_along_z(),
            Matrix3::identity(),
        )?;
        let mut res_t = vec![0.0; 3];
        tangential.linearize(&[pose.as_param_slice(), &[0.0, 0.0, 0.0]], &mut res_t, None);
        assert!(
            res_t.iter().all(|r| r.abs() < 10.0),
            "tangential error should be suppressed: {res_t:?}"
        );
        Ok(())
    }

    #[test]
    fn rejects_non_finite_covariances() {
        let mut bad = plane_cov_along_z();
        bad[(0, 0)] = f64::NAN;
        assert!(
            GicpFactor::new(
                Vector3::zeros(),
                bad,
                plane_cov_along_z(),
                Matrix3::identity()
            )
            .is_err()
        );
        assert!(
            GicpFactor::new(
                Vector3::zeros(),
                plane_cov_along_z(),
                bad,
                Matrix3::identity()
            )
            .is_err()
        );
    }
}
