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

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

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
    pub fn new(measured_bearing: Vector3<f64>, sqrt_information: SMatrix<f64, 2, 2>) -> Self {
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

/// Write zeros across a `2 × cols` Jacobian block.
fn zero_jacobian(jac: &mut faer::mat::MatMut<'_, f64>, cols: usize) {
    for row in 0..2 {
        for col in 0..cols {
            *jac.rb_mut().get_mut(row, col) = 0.0;
        }
    }
}

/// The part of the bearing residual that does not depend on how the pose is
/// parameterized.
///
/// [`BearingFactor`] and [`BearingFactorSe23`] differ only in which columns
/// their Jacobian has: the residual, the normalization Jacobian and the
/// body-frame point are functions of `R_i`, `t_i` and `p_j` alone. Deriving
/// them once here is what keeps the two from drifting apart.
struct BearingGeometry {
    /// Weighted residual `sqrt_info · Eᵀ · (n_est − n_meas)`.
    residual: SVector<f64, 2>,
    /// `sqrt_info · Eᵀ · ∂n/∂p_body`, the shared 2×3 Jacobian prefix.
    prefix: SMatrix<f64, 2, 3>,
    /// The landmark in the body frame.
    p_body: Vector3<f64>,
}

/// `None` when the landmark sits on the pose origin, where no bearing is defined.
fn bearing_geometry(
    measured_bearing: &Vector3<f64>,
    tangent_basis: &SMatrix<f64, 3, 2>,
    sqrt_information: &SMatrix<f64, 2, 2>,
    r_i: &Matrix3<f64>,
    t_i: &Vector3<f64>,
    p_j: &Vector3<f64>,
) -> Option<BearingGeometry> {
    let p_body = r_i.transpose() * (p_j - t_i);
    let norm = p_body.norm();
    if norm < 1e-16 {
        return None;
    }

    let n_est = p_body / norm;
    let e_2d = tangent_basis.transpose() * (n_est - measured_bearing);

    // ∂n/∂p_body = (I₃ − n·nᵀ) / ‖p_body‖, the unit-vector normalization Jacobian.
    let dn_dp = (Matrix3::identity() - n_est * n_est.transpose()) / norm;

    Some(BearingGeometry {
        residual: sqrt_information * e_2d,
        prefix: sqrt_information * tangent_basis.transpose() * dn_dp,
        p_body,
    })
}

impl Factor for BearingFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "BearingFactor expects 2 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be 3D point");

        let t_i = SE3::from_param_slice(params[0]);
        let p_j = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let r_i = t_i.rotation_so3().rotation_matrix();

        let Some(geometry) = bearing_geometry(
            &self.measured_bearing,
            &self.tangent_basis,
            &self.sqrt_information,
            &r_i,
            &t_i.translation(),
            &p_j,
        ) else {
            residual[0] = 0.0;
            residual[1] = 0.0;
            if let Some(mut jac) = jacobian {
                zero_jacobian(&mut jac, 9);
            }
            return;
        };

        residual[0] = geometry.residual[0];
        residual[1] = geometry.residual[1];

        let Some(mut jac) = jacobian else {
            return;
        };

        // Right perturbation T_i → T_i · Exp([δρ, δθ]):
        //   ∂p_body/∂δρ = −I₃,  ∂p_body/∂δθ = +[p_body]×  (note the sign),
        //   ∂p_body/∂p_j = R_iᵀ.
        let mut j_full = SMatrix::<f64, 2, 9>::zeros();
        j_full
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&(-geometry.prefix));
        j_full
            .fixed_view_mut::<2, 3>(0, 3)
            .copy_from(&(geometry.prefix * skew(&geometry.p_body)));
        j_full
            .fixed_view_mut::<2, 3>(0, 6)
            .copy_from(&(geometry.prefix * r_i.transpose()));

        for row in 0..2 {
            for col in 0..9 {
                *jac.rb_mut().get_mut(row, col) = j_full[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE, 3],
            "BearingFactor expects [SE3 pose, 3D point]",
        )
    }

    /// This factor whitens with the 2×2 sqrt-information supplied at construction,
    /// so it must be registered with `NoiseModel::null()`.
    fn whitens_internally(&self) -> bool {
        true
    }
}

/// Bearing factor over an `SE23` navigation state.
///
/// Identical measurement model to [`BearingFactor`] — same residual, same
/// tangent-plane projection — but attached to the `SE_2(3)` state a
/// visual-inertial window carries, so a bearing observation can constrain the
/// same variable the IMU factor does.
///
/// # Parameter layout (2 blocks, 12 DOF)
///
/// ```text
/// params[0]: SE23 state — 10D [tx,ty,tz,qw,qx,qy,qz,vx,vy,vz], 9 DOF
/// params[1]: p_j        — 3D landmark in world frame
/// ```
///
/// # Jacobian (2×12), columns `[ρ(3) | θ(3) | ν(3) | p_j(3)]`
///
/// The velocity columns are **exactly zero**, and that is the model, not an
/// omission: a bearing measures direction only, so it carries no information
/// about how fast the body is moving. Velocity is observable here only through
/// the IMU factors that share this state.
///
/// To first order `SE23 ⊞ δ` moves translation and rotation exactly as `SE3`
/// does — the group's left Jacobian is the identity at zero — so the `ρ` and `θ`
/// columns are the `SE3` ones unchanged.
pub struct BearingFactorSe23 {
    /// Measured unit bearing vector in body frame.
    measured_bearing: Vector3<f64>,
    /// Square-root information matrix (2×2).
    sqrt_information: SMatrix<f64, 2, 2>,
    /// Precomputed tangent basis at `measured_bearing` (3×2).
    tangent_basis: SMatrix<f64, 3, 2>,
}

impl BearingFactorSe23 {
    /// Create a bearing factor over an `SE23` state.
    ///
    /// # Arguments
    /// * `measured_bearing` — unit vector in body frame (will be normalized)
    /// * `sqrt_information` — 2×2 square-root information matrix
    pub fn new(measured_bearing: Vector3<f64>, sqrt_information: SMatrix<f64, 2, 2>) -> Self {
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

impl Factor for BearingFactorSe23 {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            2,
            "BearingFactorSe23 expects 2 parameter blocks"
        );
        debug_assert_eq!(params[0].len(), 10, "params[0] must be SE23 (10D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be 3D point");

        let state = SE23::from_param_slice(params[0]);
        let p_j = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let r_i = state.rotation_matrix();

        let Some(geometry) = bearing_geometry(
            &self.measured_bearing,
            &self.tangent_basis,
            &self.sqrt_information,
            &r_i,
            &state.translation(),
            &p_j,
        ) else {
            residual[0] = 0.0;
            residual[1] = 0.0;
            if let Some(mut jac) = jacobian {
                zero_jacobian(&mut jac, 12);
            }
            return;
        };

        residual[0] = geometry.residual[0];
        residual[1] = geometry.residual[1];

        let Some(mut jac) = jacobian else {
            return;
        };

        // Columns 6..9 (ν) stay zero — see the type-level comment.
        let mut j_full = SMatrix::<f64, 2, 12>::zeros();
        j_full
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&(-geometry.prefix));
        j_full
            .fixed_view_mut::<2, 3>(0, 3)
            .copy_from(&(geometry.prefix * skew(&geometry.p_body)));
        j_full
            .fixed_view_mut::<2, 3>(0, 9)
            .copy_from(&(geometry.prefix * r_i.transpose()));

        for row in 0..2 {
            for col in 0..12 {
                *jac.rb_mut().get_mut(row, col) = j_full[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 12)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, 3],
            "BearingFactorSe23 expects [SE23 state, 3D point]",
        )
    }

    /// Whitens with the 2×2 sqrt-information supplied at construction, so it
    /// must be registered with `NoiseModel::null()`.
    fn whitens_internally(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23Tangent;
    use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};

    fn make_pose(tx: f64, ty: f64, tz: f64, q: UnitQuaternion<f64>) -> DVector<f64> {
        let q = q.quaternion();
        DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k])
    }

    fn perturb_se3(pose: &[f64], tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from_param_slice(pose);
        let tan = SE3Tangent::from_slice(tangent);
        DVector::from_column_slice(se3.right_plus(&tan, None, None).as_param_slice())
    }

    /// Compute the body-frame bearing from pose to point.
    fn compute_bearing(pose: &[f64], point: &Vector3<f64>) -> Vector3<f64> {
        let t_i = SE3::from_param_slice(pose);
        let r_i = t_i.rotation_so3().rotation_matrix();
        let t_pos = t_i.translation();
        let p_body = r_i.transpose() * (point - t_pos);
        p_body.normalize()
    }

    fn compute_residual(factor: &BearingFactor, pose: &[f64], point: &[f64]) -> Vec<f64> {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(&[pose, point], &mut residual, None);
        residual
    }

    fn compute_with_jacobian(
        factor: &BearingFactor,
        pose: &[f64],
        point: &[f64],
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[pose, point], &mut residual, Some(jac_mut));
        let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
        (residual, jacobian)
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

        let bearing = compute_bearing(pose.as_slice(), &point);
        let factor = BearingFactor::new_isotropic(bearing, 1.0);
        let r = compute_residual(&factor, pose.as_slice(), point_dv.as_slice());

        for (i, ri) in r.iter().enumerate().take(2) {
            assert!(ri.abs() < 1e-10, "residual[{i}] = {} should be zero", ri);
        }
    }

    // ── Test 2: simple geometry (point on z-axis) ───────────────────────────

    #[test]
    fn point_on_z_axis() {
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let point = DVector::from_vec(vec![0.0, 0.0, 10.0]);

        let bearing = Vector3::new(0.0, 0.0, 1.0);
        let factor = BearingFactor::new_isotropic(bearing, 1.0);
        let r = compute_residual(&factor, pose.as_slice(), point.as_slice());

        for (i, ri) in r.iter().enumerate().take(2) {
            assert!(ri.abs() < 1e-10, "residual[{i}] = {}", ri);
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
        let bearing = compute_bearing(pose.as_slice(), &(point + Vector3::new(0.1, -0.05, 0.02)));
        let factor = BearingFactor::new_isotropic(bearing, 0.5);

        let (r0, jac) = compute_with_jacobian(&factor, pose.as_slice(), point_dv.as_slice());

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        // Block 0: T_i (6 DOF, cols 0–5)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_p = perturb_se3(pose.as_slice(), &tan);
            let r_pert = compute_residual(&factor, pose_p.as_slice(), point_dv.as_slice());
            for row in 0..2 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J_T_i[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, col)],
                    fd
                );
            }
        }

        // Block 1: p_j (3D, cols 6–8)
        for col in 0..3 {
            let mut point_p = point_dv.clone();
            point_p[col] += EPS;
            let r_pert = compute_residual(&factor, pose.as_slice(), point_p.as_slice());
            for row in 0..2 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let err = (fd - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_p_j[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd
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
        assert_eq!(factor.residual_dim(), 2);
    }
    // ── BearingFactorSe23 ───────────────────────────────────────────────────

    fn make_state(
        tx: f64,
        ty: f64,
        tz: f64,
        q: UnitQuaternion<f64>,
        v: Vector3<f64>,
    ) -> DVector<f64> {
        let q = q.quaternion();
        DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k, v.x, v.y, v.z])
    }

    fn perturb_se23(state: &[f64], tangent: &[f64; 9]) -> DVector<f64> {
        let se23 = SE23::from_param_slice(state);
        let tan = SE23Tangent::from_slice(tangent);
        DVector::from_column_slice(se23.right_plus(&tan, None, None).as_param_slice())
    }

    fn se23_residual(factor: &BearingFactorSe23, state: &[f64], point: &[f64]) -> Vec<f64> {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(&[state, point], &mut residual, None);
        residual
    }

    fn se23_with_jacobian(
        factor: &BearingFactorSe23,
        state: &[f64],
        point: &[f64],
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[state, point], &mut residual, Some(jac_mut));
        (residual, DMatrix::from_column_slice(rows, cols, &jac_buf))
    }

    fn se23_fixture() -> (BearingFactorSe23, DVector<f64>, DVector<f64>) {
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.3, -0.7, 0.4)),
            0.6,
        );
        let state = make_state(1.0, 2.0, 0.5, q, Vector3::new(0.4, -0.2, 0.1));
        let point = DVector::from_vec(vec![5.0, 3.0, 1.0]);

        let se23 = SE23::from_param_slice(state.as_slice());
        let p_body = se23.rotation_matrix().transpose()
            * (Vector3::new(point[0], point[1], point[2]) - se23.translation());
        // Offset the measurement from the truth so the residual, and therefore
        // the Jacobian, is exercised away from zero.
        let measured = (p_body.normalize() + Vector3::new(0.02, -0.015, 0.01)).normalize();
        (
            BearingFactorSe23::new_isotropic(measured, 0.5),
            state,
            point,
        )
    }

    #[test]
    fn se23_zero_residual_at_the_measured_bearing() {
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            0.5,
        );
        let state = make_state(1.0, 2.0, 0.5, q, Vector3::new(1.0, -1.0, 0.5));
        let point = DVector::from_vec(vec![5.0, 3.0, 1.0]);

        let se23 = SE23::from_param_slice(state.as_slice());
        let p_body = se23.rotation_matrix().transpose()
            * (Vector3::new(point[0], point[1], point[2]) - se23.translation());
        let factor = BearingFactorSe23::new_isotropic(p_body.normalize(), 1.0);

        let r = se23_residual(&factor, state.as_slice(), point.as_slice());
        for (i, ri) in r.iter().enumerate() {
            assert!(ri.abs() < 1e-10, "residual[{i}] = {ri} should be zero");
        }
    }

    #[test]
    fn se23_jacobian_matches_finite_differences() {
        let (factor, state, point) = se23_fixture();
        let (_, analytic) = se23_with_jacobian(&factor, state.as_slice(), point.as_slice());

        const EPS: f64 = 1e-7;

        for col in 0..9 {
            let mut plus = [0.0f64; 9];
            let mut minus = [0.0f64; 9];
            plus[col] = EPS;
            minus[col] = -EPS;
            let sp = perturb_se23(state.as_slice(), &plus);
            let sm = perturb_se23(state.as_slice(), &minus);
            let rp = se23_residual(&factor, sp.as_slice(), point.as_slice());
            let rm = se23_residual(&factor, sm.as_slice(), point.as_slice());
            for row in 0..2 {
                let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                assert!(
                    (fd - analytic[(row, col)]).abs() < 1e-5,
                    "state column {col}, row {row}: fd {fd} vs analytic {}",
                    analytic[(row, col)]
                );
            }
        }

        for axis in 0..3 {
            let mut pp = point.clone();
            let mut pm = point.clone();
            pp[axis] += EPS;
            pm[axis] -= EPS;
            let rp = se23_residual(&factor, state.as_slice(), pp.as_slice());
            let rm = se23_residual(&factor, state.as_slice(), pm.as_slice());
            for row in 0..2 {
                let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                let col = 9 + axis;
                assert!(
                    (fd - analytic[(row, col)]).abs() < 1e-5,
                    "point column {col}, row {row}: fd {fd} vs analytic {}",
                    analytic[(row, col)]
                );
            }
        }
    }

    /// A bearing measures direction only, so velocity carries no information.
    /// Asserted rather than assumed: a nonzero column here would silently let
    /// the optimizer move velocity to fit a reprojection.
    #[test]
    fn se23_velocity_columns_are_zero() {
        let (factor, state, point) = se23_fixture();
        let (_, jacobian) = se23_with_jacobian(&factor, state.as_slice(), point.as_slice());
        for row in 0..2 {
            for col in 6..9 {
                assert_eq!(
                    jacobian[(row, col)],
                    0.0,
                    "velocity column {col} is nonzero"
                );
            }
        }
    }

    /// The SE23 factor is a re-parameterization, not a new measurement model:
    /// for the same geometry it must produce the same residual as the SE3 one,
    /// and the same pose and landmark Jacobian columns.
    #[test]
    fn se23_agrees_with_se3_on_the_shared_blocks() {
        let (se23_factor, state, point) = se23_fixture();
        let se3_pose = DVector::from_vec(vec![
            state[0], state[1], state[2], state[3], state[4], state[5], state[6],
        ]);
        let se3_factor =
            BearingFactor::new(se23_factor.measured_bearing, se23_factor.sqrt_information);

        let (r23, j23) = se23_with_jacobian(&se23_factor, state.as_slice(), point.as_slice());
        let (r3, j3) = compute_with_jacobian(&se3_factor, se3_pose.as_slice(), point.as_slice());

        for row in 0..2 {
            assert!((r23[row] - r3[row]).abs() < 1e-12, "residual row {row}");
            for col in 0..6 {
                assert!(
                    (j23[(row, col)] - j3[(row, col)]).abs() < 1e-12,
                    "pose column {col}, row {row}"
                );
            }
            for axis in 0..3 {
                assert!(
                    (j23[(row, 9 + axis)] - j3[(row, 6 + axis)]).abs() < 1e-12,
                    "landmark column {axis}, row {row}"
                );
            }
        }
    }
}
