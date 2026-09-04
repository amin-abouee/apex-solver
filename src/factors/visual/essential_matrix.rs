//! Epipolar factors on the essential matrix (GTSAM `EssentialMatrixFactor` /
//! `EssentialMatrixConstraint` analogues).
//!
//! Two factors for 2D–2D matching:
//!
//! - [`EssentialMatrixFactor`]: scalar epipolar residual per point pair,
//!   over the **relative pose** variable `T_21` (`p₂ = R·p₁ + t`):
//!   `r = p₂ᵀ [t/‖t‖]× R p₁`. Scale-free by construction — only the
//!   translation *direction* is observable from epipolar geometry.
//! - [`EssentialMatrixConstraint`]: converts a measured essential matrix
//!   `(R_E, u_E)` into a 6D constraint on a relative pose: rotation rows
//!   `Log(R_Eᵀ R)` plus translation-direction rows `t/‖t‖ − u_E`
//!   (effectively rank-5: the direction difference is rank-2).

use apex_manifolds::LieGroup;
use apex_manifolds::Tangent;
use apex_manifolds::se3::SE3;
use apex_manifolds::so3::{SO3, SO3Tangent};
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Scalar epipolar residual per 2D–2D point pair, over the relative pose.
///
/// The point pairs are **normalized** camera coordinates (from
/// `CameraModel::unproject`), fixed at construction; the only variable is
/// the relative pose `T_21` with `p₂ = R·p₁ + t`.
#[derive(Clone)]
pub struct EssentialMatrixFactor {
    /// Normalized coordinates of points in camera 1.
    pub points_1: Vec<Vector3<f64>>,
    /// Normalized coordinates of the matched points in camera 2.
    pub points_2: Vec<Vector3<f64>>,
}

impl EssentialMatrixFactor {
    /// Create the factor from matched normalized point pairs.
    ///
    /// Returns an error when the pair lists disagree or are empty.
    pub fn new(points_1: Vec<Vector3<f64>>, points_2: Vec<Vector3<f64>>) -> Result<Self, String> {
        if points_1.is_empty() || points_1.len() != points_2.len() {
            return Err(format!(
                "EssentialMatrixFactor needs non-empty, equal-length point lists \
                 (got {} and {})",
                points_1.len(),
                points_2.len()
            ));
        }
        Ok(Self { points_1, points_2 })
    }

    /// Number of point pairs.
    pub fn num_points(&self) -> usize {
        self.points_1.len()
    }
}

impl Factor for EssentialMatrixFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        mut jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            1,
            "EssentialMatrixFactor expects 1 pose block"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");

        let pose = SE3::from_param_slice(params[0]);
        let rotation = pose.rotation_so3().rotation_matrix();
        let translation = pose.translation();
        let t_norm = translation.norm();

        for (i, (p1, p2)) in self.points_1.iter().zip(&self.points_2).enumerate() {
            if t_norm < f64::EPSILON {
                // Degenerate zero translation: the essential matrix is
                // undefined; bounded constant residual keeps the factor from
                // becoming a free cost reduction.
                residual[i] = 1.0;
                continue;
            }
            let u = translation / t_norm;
            let u_x = Matrix3::new(0.0, -u.z, u.y, u.z, 0.0, -u.x, -u.y, u.x, 0.0);
            let m = rotation * p1;
            let e_row = p2.transpose() * u_x * rotation; // 1×3, = p2ᵀ[u]×R
            residual[i] = (e_row * p1)[(0, 0)];

            let Some(jac) = jacobian.as_mut() else {
                continue;
            };

            // ∂r/∂δρ: ((I − uuᵀ)(m×p2))ᵀ R / ‖t‖
            let cross_mp2 = m.cross(p2);
            let proj = cross_mp2 - u * cross_mp2.dot(&u);
            let d_rho = (proj.transpose() * rotation) / t_norm;

            // ∂r/∂δθ: −(p2ᵀ[u]×R)·p̂₁
            let p1_x = Matrix3::new(0.0, -p1.z, p1.y, p1.z, 0.0, -p1.x, -p1.y, p1.x, 0.0);
            let d_theta = -(e_row * p1_x);

            for c in 0..3 {
                *jac.rb_mut().get_mut(i, c) = d_rho[(0, c)];
                *jac.rb_mut().get_mut(i, 3 + c) = d_theta[(0, c)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        self.points_1.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.points_1.len(), 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 1 || variables[0].as_param_slice().len() != SE3::REP_SIZE {
            return Err("EssentialMatrixFactor requires exactly one SE3 pose variable".into());
        }
        Ok(())
    }
}

/// Measured-essential-matrix constraint on a relative pose.
///
/// Residual (6D): `[Log(R_Eᵀ·R); t/‖t‖ − u_E]` — zero when the pose realizes
/// the measured essential matrix up to translation scale.
#[derive(Clone)]
pub struct EssentialMatrixConstraint {
    /// Measured rotation between the frames.
    pub rotation_e: SO3,
    /// Measured unit translation direction (world→? convention of T_21).
    pub direction_e: Vector3<f64>,
}

impl EssentialMatrixConstraint {
    /// Create the constraint from a measured rotation and translation
    /// direction. The direction is normalized; a near-zero direction is
    /// rejected.
    pub fn new(rotation_e: SO3, direction_e: Vector3<f64>) -> Result<Self, String> {
        let norm = direction_e.norm();
        if !(norm.is_finite() && norm > 1e-12) {
            return Err(
                "EssentialMatrixConstraint requires a non-zero translation direction".into(),
            );
        }
        Ok(Self {
            rotation_e,
            direction_e: direction_e / norm,
        })
    }
}

impl Factor for EssentialMatrixConstraint {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            1,
            "EssentialMatrixConstraint expects 1 pose block"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");

        let pose = SE3::from_param_slice(params[0]);
        let rotation = pose.rotation_so3();
        let translation = pose.translation();
        let t_norm = translation.norm();

        // Rotation rows: Log(R_Eᵀ R)
        let rel = self.rotation_e.inverse(None).compose(&rotation, None, None);
        let theta = rel.log(None);
        for (i, v) in theta.coeffs().iter().enumerate() {
            residual[i] = *v;
        }

        // Direction rows: t/‖t‖ − u_E
        if t_norm > f64::EPSILON {
            let u = translation / t_norm;
            let diff = u - self.direction_e;
            for (i, v) in diff.iter().enumerate() {
                residual[3 + i] = *v;
            }
        } else {
            for i in 0..3 {
                residual[3 + i] = 1.0;
            }
        }

        let Some(mut jac) = jacobian else { return };

        // ∂Log(R_Eᵀ R exp(δθ))/∂δθ = Jr⁻¹(θ); translation perturbation does
        // not affect the rotation.
        let jr_inv = SO3Tangent::new(theta.coeffs()).right_jacobian_inv();
        for r in 0..3 {
            for c in 0..3 {
                *jac.rb_mut().get_mut(r, 3 + c) = jr_inv[(r, c)];
            }
        }

        // ∂(t/‖t‖)/∂δρ = (I − uuᵀ)·R/‖t‖; rotation does not affect t.
        if t_norm > f64::EPSILON {
            let u = translation / t_norm;
            let rotation_m = pose.rotation_so3().rotation_matrix();
            let proj_u = Matrix3::identity() - u * u.transpose();
            let d_dir = (proj_u * rotation_m) / t_norm;
            for r in 0..3 {
                for c in 0..3 {
                    *jac.rb_mut().get_mut(3 + r, c) = d_dir[(r, c)];
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        6
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (6, 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 1 || variables[0].as_param_slice().len() != SE3::REP_SIZE {
            return Err("EssentialMatrixConstraint requires exactly one SE3 pose variable".into());
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

    fn truth_pose() -> SE3 {
        SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 0.2, -0.1),
            nalgebra::UnitQuaternion::from_euler_angles(0.02, 0.05, -0.03),
        ))
    }

    fn sample_points() -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>, SE3) {
        let pose = truth_pose();
        let p1s: Vec<Vector3<f64>> = (0..4)
            .map(|i| {
                Vector3::new(
                    0.1 + 0.2 * i as f64,
                    -0.1 + 0.1 * i as f64,
                    2.0 + 0.5 * i as f64,
                )
            })
            .collect();
        let p2s = p1s.iter().map(|p| pose.act(p, None, None)).collect();
        (p1s, p2s, pose)
    }

    #[test]
    fn zero_residual_at_truth_pose() -> TestResult<()> {
        let (p1s, p2s, pose) = sample_points();
        let factor = EssentialMatrixFactor::new(p1s, p2s)?;
        let mut residual = vec![0.0; factor.residual_dim()];
        factor.linearize(&[pose.as_param_slice()], &mut residual, None);
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-10, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn scale_invariance() -> TestResult<()> {
        // Doubling the translation must not change the residual.
        let (p1s, p2s, pose) = sample_points();
        let factor = EssentialMatrixFactor::new(p1s, p2s)?;
        let t = pose.translation();
        let scaled = SE3::new(t * 2.0, pose.rotation_quaternion());
        let mut r1 = vec![0.0; factor.residual_dim()];
        let mut r2 = vec![0.0; factor.residual_dim()];
        factor.linearize(&[pose.as_param_slice()], &mut r1, None);
        factor.linearize(&[scaled.as_param_slice()], &mut r2, None);
        for i in 0..r1.len() {
            assert!(
                (r1[i] - r2[i]).abs() < 1e-9,
                "row {i}: {} vs {}",
                r1[i],
                r2[i]
            );
        }
        Ok(())
    }

    #[test]
    fn finite_difference_jacobians() -> TestResult<()> {
        let (p1s, p2s, pose) = sample_points();
        let factor = EssentialMatrixFactor::new(p1s, p2s)?;
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();

        let (rows, cols) = factor.jacobian_shape();
        let mut r0 = vec![0.0; rows];
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
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
            let mut r_pert = vec![0.0; rows];
            factor.linearize(&[&perturbed], &mut r_pert, None);
            for row in 0..rows {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let ana = jac_buf[col * rows + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "J[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn constraint_zero_residual_at_truth() -> TestResult<()> {
        let pose = truth_pose();
        let r_e = pose.rotation_so3();
        let u_e = pose.translation().normalize();
        let constraint = EssentialMatrixConstraint::new(r_e, u_e)?;

        let mut residual = vec![0.0; 6];
        constraint.linearize(&[pose.as_param_slice()], &mut residual, None);
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-9, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn constraint_fd_jacobians() -> TestResult<()> {
        let pose = truth_pose();
        let r_e = pose.rotation_so3();
        let u_e = pose.translation().normalize();
        let constraint = EssentialMatrixConstraint::new(r_e, u_e)?;
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();

        let mut r0 = vec![0.0; 6];
        let mut jac_buf = vec![0.0; 36];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 6, 6);
        constraint.linearize(&[&pose_vec], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let perturbed: Vec<f64> = pose
                .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                .as_param_slice()
                .to_vec();
            let mut r_pert = vec![0.0; 6];
            constraint.linearize(&[&perturbed], &mut r_pert, None);
            for row in 0..6 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let ana = jac_buf[col * 6 + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "J[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_bad_construction() -> TestResult<()> {
        assert!(EssentialMatrixFactor::new(vec![], vec![]).is_err());
        assert!(EssentialMatrixFactor::new(vec![Vector3::zeros()], vec![]).is_err());
        assert!(EssentialMatrixConstraint::new(SO3::identity(), Vector3::zeros()).is_err());
        Ok(())
    }
}
