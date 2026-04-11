//! Bearing measurement factor for direction-only landmark observations.
//!
//! Implements a bearing factor that constrains the direction from a pose to a
//! 3D landmark, operating on the S² manifold (unit sphere).
//!
//! # Mathematical Formulation
//!
//! ```text
//! p_body = R_iᵀ (p_j − t_i)                    (point in body frame)
//! n_est  = p_body / ‖p_body‖                    (estimated bearing)
//! e_3d   = n_est − n_meas                       (3D bearing difference)
//! e      = sqrt_info · Eᵀ · e_3d   ∈ R²        (projected to tangent plane)
//! ```
//!
//! where `E` (3×2) is an orthonormal basis for the tangent plane at `n_meas`.
//!
//! # Parameter Layout (2 blocks, 9 DOF total)
//!
//! ```text
//! params[0]: T_i  — body pose in world frame (7D [tx,ty,tz,qw,qx,qy,qz], 6 DOF)
//! params[1]: p_j  — 3D landmark in world frame (3D, 3 DOF)
//! ```
//!
//! # Jacobians (2×9)
//!
//! Using right SE3 perturbation and chain rule through unit-vector normalization.

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

use apex_manifolds::se3::SE3;

use crate::factors::imu::helpers::cross_matrix;
use crate::factors::Factor;

/// Compute an orthonormal basis (3×2) for the tangent plane at unit vector `n`.
///
/// Picks the coordinate axis least aligned with `n`, crosses it with `n` to get
/// `e1`, then `e2 = n × e1`. Both are normalized.
fn tangent_basis(n: &Vector3<f64>) -> SMatrix<f64, 3, 2> {
    // Pick axis least aligned with n
    let abs_n = Vector3::new(n[0].abs(), n[1].abs(), n[2].abs());
    let axis = if abs_n[0] <= abs_n[1] && abs_n[0] <= abs_n[2] {
        Vector3::x()
    } else if abs_n[1] <= abs_n[2] {
        Vector3::y()
    } else {
        Vector3::z()
    };

    let mut e1 = n.cross(&axis);
    e1.normalize_mut();
    let e2 = n.cross(&e1);

    SMatrix::<f64, 3, 2>::from_columns(&[e1, e2])
}

/// Bearing factor: constrains the direction from a pose to a 3D landmark.
pub struct BearingFactor {
    /// Measured unit bearing vector in body frame.
    measured_bearing: Vector3<f64>,
    /// Square-root information matrix (2×2).
    sqrt_information: SMatrix<f64, 2, 2>,
    /// Precomputed tangent basis at `measured_bearing` (3×2).
    tangent_basis: SMatrix<f64, 3, 2>,
}

impl BearingFactor {
    /// Create a bearing factor.
    ///
    /// # Arguments
    /// * `measured_bearing` — unit vector in body frame (will be normalized)
    /// * `sqrt_information` — 2×2 square-root information matrix
    pub fn new(
        measured_bearing: Vector3<f64>,
        sqrt_information: SMatrix<f64, 2, 2>,
    ) -> Self {
        let n = measured_bearing.normalize();
        let basis = tangent_basis(&n);
        Self {
            measured_bearing: n,
            sqrt_information,
            tangent_basis: basis,
        }
    }

    /// Create with isotropic noise (scalar standard deviation in radians).
    pub fn new_isotropic(measured_bearing: Vector3<f64>, sigma: f64) -> Self {
        let sqrt_info = SMatrix::<f64, 2, 2>::identity() * (1.0 / sigma);
        Self::new(measured_bearing, sqrt_info)
    }
}

impl Factor for BearingFactor {
    fn get_dimension(&self) -> usize {
        2
    }

    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        debug_assert_eq!(params.len(), 2, "BearingFactor expects 2 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be 3D point");

        let t_i = SE3::from(params[0].clone());
        let p_j = Vector3::new(params[1][0], params[1][1], params[1][2]);

        let r_i = t_i.rotation_so3().rotation_matrix();
        let t_i_pos = t_i.translation();

        // Point in body frame
        let p_body = r_i.transpose() * (p_j - t_i_pos);
        let norm = p_body.norm();

        // Degenerate case: point at the pose origin
        if norm < 1e-16 {
            let zero_res = DVector::zeros(2);
            if compute_jacobian {
                return (zero_res, Some(DMatrix::zeros(2, 9)));
            }
            return (zero_res, None);
        }

        // Estimated bearing (unit vector)
        let n_est = p_body / norm;

        // 3D bearing error, projected to tangent plane at n_meas
        let e_3d = n_est - self.measured_bearing;
        let e_2d = self.tangent_basis.transpose() * e_3d;

        // Weighted residual
        let weighted = self.sqrt_information * e_2d;
        let residual = DVector::from_iterator(2, weighted.iter().copied());

        if !compute_jacobian {
            return (residual, None);
        }

        // ── Jacobians ─────────────────────────────────────────────────────────
        //
        // Chain rule: J = sqrt_info · E^T · dn/dp_body · dp_body/d(param)
        //
        // dn/dp_body = (I₃ - n·nᵀ) / ‖p_body‖   (unit vector normalization Jacobian)
        //
        // dp_body/d(δρ, δθ) with right perturbation of T_i:
        //   p_body → (I - [δθ]×) · (p_body - δρ)
        //   ≈ p_body - δρ - [δθ]× · p_body
        //   = p_body - δρ + [p_body]× · δθ
        //
        //   dp_body/dδρ = -I₃
        //   dp_body/dδθ = -[p_body]×
        //
        // dp_body/dp_j = R_iᵀ

        let dn_dp = (Matrix3::identity() - n_est * n_est.transpose()) / norm;

        // Common prefix: sqrt_info · E^T · dn/dp_body  (2×3)
        let prefix = self.sqrt_information * self.tangent_basis.transpose() * dn_dp;

        // J_T_i (2×6): prefix * [-I₃ | -[p_body]×]
        let p_body_hat = cross_matrix(&p_body);
        let j_rho = -prefix; // 2×3
        let j_theta = -prefix * p_body_hat; // 2×3

        // J_p_j (2×3): prefix * R_iᵀ
        let j_pj = prefix * r_i.transpose(); // 2×3

        // Assemble 2×9 Jacobian: [J_T_i(6) | J_p_j(3)]
        let mut j_full = SMatrix::<f64, 2, 9>::zeros();
        j_full.fixed_view_mut::<2, 3>(0, 0).copy_from(&j_rho);
        j_full.fixed_view_mut::<2, 3>(0, 3).copy_from(&j_theta);
        j_full.fixed_view_mut::<2, 3>(0, 6).copy_from(&j_pj);

        let jac = DMatrix::from_iterator(2, 9, j_full.iter().copied());

        (residual, Some(jac))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::se3::SE3Tangent;
    use nalgebra::{UnitQuaternion, Vector3};

    fn make_pose(tx: f64, ty: f64, tz: f64, q: UnitQuaternion<f64>) -> DVector<f64> {
        let q = q.quaternion();
        DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k])
    }

    fn perturb_se3(pose: &DVector<f64>, tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from(pose.clone());
        let tan = SE3Tangent::from(DVector::from_vec(tangent.to_vec()));
        DVector::from(se3.right_plus(&tan, None, None))
    }

    /// Compute the body-frame bearing from pose to point.
    fn compute_bearing(
        pose: &DVector<f64>,
        point: &Vector3<f64>,
    ) -> Vector3<f64> {
        let t_i = SE3::from(pose.clone());
        let r_i = t_i.rotation_so3().rotation_matrix();
        let t_pos = t_i.translation();
        let p_body = r_i.transpose() * (point - t_pos);
        p_body.normalize()
    }

    // ── Test 1: zero residual ───────────────────────────────────────────────

    #[test]
    fn zero_residual() {
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            0.5,
        );
        let pose = make_pose(1.0, 2.0, 0.5, q);
        let point = Vector3::new(5.0, 3.0, 1.0);
        let point_dv = DVector::from_iterator(3, point.iter().copied());

        let bearing = compute_bearing(&pose, &point);
        let factor = BearingFactor::new_isotropic(bearing, 1.0);
        let (r, _) = factor.linearize(&[pose, point_dv], false);

        for i in 0..2 {
            assert!(
                r[i].abs() < 1e-10,
                "residual[{i}] = {} should be zero",
                r[i]
            );
        }
    }

    // ── Test 2: simple geometry (point on z-axis) ───────────────────────────

    #[test]
    fn point_on_z_axis() {
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let point = DVector::from_vec(vec![0.0, 0.0, 10.0]);

        let bearing = Vector3::new(0.0, 0.0, 1.0);
        let factor = BearingFactor::new_isotropic(bearing, 1.0);
        let (r, _) = factor.linearize(&[pose, point], false);

        for i in 0..2 {
            assert!(r[i].abs() < 1e-10, "residual[{i}] = {}", r[i]);
        }
    }

    // ── Test 3: finite-difference Jacobian check ────────────────────────────

    #[test]
    fn finite_difference_jacobians() {
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.3, 0.7, 0.1)),
            0.4,
        );
        let pose = make_pose(1.0, -0.5, 2.0, q);
        let point = Vector3::new(5.0, 3.0, 4.0);
        let point_dv = DVector::from_iterator(3, point.iter().copied());

        // Use a slightly different bearing to get non-zero residual
        let bearing = compute_bearing(&pose, &(point + Vector3::new(0.1, -0.05, 0.02)));
        let factor = BearingFactor::new_isotropic(bearing, 0.5);

        let nominal = vec![pose.clone(), point_dv.clone()];
        let (r0, jac_opt) = factor.linearize(&nominal, true);
        let jac = jac_opt.expect("Jacobian must be computed");

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        // Block 0: T_i (6 DOF, cols 0–5)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[0] = perturb_se3(&pose, &tan);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..2 {
                let err = (fd[row] - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J_T_i[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, col)],
                    fd[row]
                );
            }
        }

        // Block 1: p_j (3D, cols 6–8)
        for col in 0..3 {
            let mut p = nominal.clone();
            p[1] = point_dv.clone();
            p[1][col] += EPS;
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..2 {
                let err = (fd[row] - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_p_j[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd[row]
                );
            }
        }
    }

    // ── Test 4: tangent basis orthonormality ────────────────────────────────

    #[test]
    fn tangent_basis_orthonormal() {
        let dirs = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0).normalize(),
            Vector3::new(-0.3, 0.7, 0.5).normalize(),
        ];

        for n in &dirs {
            let basis = tangent_basis(n);
            let e1 = basis.column(0);
            let e2 = basis.column(1);

            // Orthogonal to n
            assert!(e1.dot(n).abs() < 1e-12, "e1 not perpendicular to n");
            assert!(e2.dot(n).abs() < 1e-12, "e2 not perpendicular to n");

            // Orthonormal
            assert!((e1.norm() - 1.0).abs() < 1e-12);
            assert!((e2.norm() - 1.0).abs() < 1e-12);
            assert!(e1.dot(&e2).abs() < 1e-12, "e1 and e2 not orthogonal");
        }
    }

    // ── Test 5: dimension ───────────────────────────────────────────────────

    #[test]
    fn dimension_is_two() {
        let factor = BearingFactor::new_isotropic(Vector3::z(), 1.0);
        assert_eq!(factor.get_dimension(), 2);
    }
}
