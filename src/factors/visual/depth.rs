//! Depth error factor for constraining the z-component of a projected landmark.
//!
//! Implements the OKVIS2-X `DepthErrorT` factor which constrains the depth
//! (z-coordinate in camera frame) of a homogeneous landmark point.
//!
//! # Mathematical Formulation
//!
//! ```text
//! hp_S  = T_SW · hp_W          (world → sensor)
//! hp_C  = T_CS · hp_S          (sensor → camera)
//! depth = hp_C[2] / hp_C[3]
//! e     = sqrt_info · (z_depth − depth)   ∈ R
//! ```
//!
//! `OneSidedDepthFactor` (ONESIDED=true) sets `e = 0` when `depth > z_depth`,
//! only penalizing points that are too close.
//!
//! # Parameter Layout (3 blocks, 16 DOF total)
//!
//! ```text
//! params[0]: T_WS — world-to-sensor pose (7D [tx,ty,tz,qw,qx,qy,qz], 6 DOF)
//! params[1]: hp_W — homogeneous landmark in world frame (4D)
//! params[2]: T_SC — sensor-to-camera extrinsics (7D, 6 DOF)
//! ```
//!
//! # Jacobians (1×16)
//!
//! Derived from OKVIS `DepthError.hpp`, using right SE3 perturbation model.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Matrix4, SMatrix, Vector3, Vector4};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

/// Depth constraint factor with compile-time one-sided toggle.
///
/// When `ONESIDED = false` (regular depth error), the factor always penalizes
/// the difference between measured and estimated depth.
///
/// When `ONESIDED = true`, the residual is set to zero if the estimated depth
/// exceeds the measurement (only penalizes points that are too close).
pub struct DepthFactor<const ONESIDED: bool> {
    /// Measured depth [m].
    measurement: f64,
    /// Square-root information (scalar, `w` s.t. `w² = σ⁻²`).
    sqrt_information: f64,
}

/// Regular depth error (penalizes both directions).
pub type RegularDepthFactor = DepthFactor<false>;

/// One-sided depth error (only penalizes points too close).
pub type OneSidedDepthFactor = DepthFactor<true>;

impl<const ONESIDED: bool> DepthFactor<ONESIDED> {
    /// Create a depth factor.
    ///
    /// # Arguments
    /// * `measurement` — measured depth [m]
    /// * `sqrt_information` — scalar square-root information (`1/σ`)
    pub fn new(measurement: f64, sqrt_information: f64) -> Self {
        Self {
            measurement,
            sqrt_information,
        }
    }

    /// Create from measurement and standard deviation.
    pub fn new_from_stdev(measurement: f64, stdev: f64) -> Self {
        Self::new(measurement, 1.0 / stdev)
    }
}

impl<const ONESIDED: bool> Factor for DepthFactor<ONESIDED> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 3, "DepthFactor expects 3 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(
            params[1].len(),
            4,
            "params[1] must be homogeneous point (4D)"
        );
        debug_assert_eq!(params[2].len(), 7, "params[2] must be SE3 (7D)");

        let t_ws = SE3::from_param_slice(params[0]);
        let hp_w = Vector4::new(params[1][0], params[1][1], params[1][2], params[1][3]);
        let t_sc = SE3::from_param_slice(params[2]);

        // Build 4×4 transformation matrices
        let t_sw: Matrix4<f64> = t_ws.inverse(None).matrix();
        let t_cs: Matrix4<f64> = t_sc.inverse(None).matrix();

        // Transform to camera frame
        let hp_s: Vector4<f64> = t_sw * hp_w;
        let hp_c: Vector4<f64> = t_cs * hp_s;

        let zero_out = |residual: &mut [f64], jacobian: Option<faer::mat::MatMut<'_, f64>>| {
            residual[0] = 0.0;
            if let Some(mut jac) = jacobian {
                for col in 0..16 {
                    *jac.rb_mut().get_mut(0, col) = 0.0;
                }
            }
        };

        // Check for degenerate homogeneous coordinate
        if hp_c[3].abs() < 1e-16 {
            zero_out(residual, jacobian);
            return;
        }

        let depth = hp_c[2] / hp_c[3];

        // One-sided: ignore if depth > measurement
        if ONESIDED && depth > self.measurement {
            zero_out(residual, jacobian);
            return;
        }

        let error = self.measurement - depth;
        residual[0] = self.sqrt_information * error;

        let Some(mut jac) = jacobian else {
            return;
        };

        // ── Jacobians (right SE3 perturbation, apex-solver convention) ──────────
        //
        // residual = sqrt_info * (z_depth − depth)
        // depth    = hp_C[2] / hp_C[3]
        //
        // Jh_weighted = [0, 0, sqrt_info/hp_C[3], 0]  (1×4, selects z-row)
        //
        // d(residual)/d(param) = −Jh_weighted · d(hp_C)/d(param)
        //
        // ── Right perturbation of T_WS ────────────────────────────────────────
        //   T_WS → T_WS · Exp([δρ, δθ])
        //   t_WS → t_WS + C_WS · δρ,  C_WS → C_WS · Exp(δθ)
        //   C_SW → Exp(−δθ) · C_SW ≈ (I − [δθ]×) · C_SW
        //
        //   hp_S[0:3] = C_SW · (hp_W[0:3] − t_WS · hp_W[3])
        //   δhp_S[0:3] = −[δθ]× · hp_S[0:3] − δρ · hp_W[3]
        //              = +[hp_S[0:3]]× · δθ − I · δρ · hp_W[3]
        //
        //   4×6 J_pose:  rows 0:3, col 0:3 = −I · hp_W[3]
        //                rows 0:3, col 3:6 = +[hp_S[0:3]]×
        //
        //   J_T_WS = −Jh_weighted · T_CS · J_pose   (1×6)
        //
        // ── hp_W (Euclidean, no manifold) ────────────────────────────────────
        //   hp_C = T_CS · T_SW · hp_W
        //   d(hp_C)/d(hp_W) = T_CW
        //   J_hp_W = −Jh_weighted · T_CW             (1×4)
        //
        // ── Right perturbation of T_SC ────────────────────────────────────────
        //   hp_C[0:3] = C_CS · (hp_S[0:3] − t_SC · hp_S[3])
        //   Under right perturbation of T_SC (same derivation as T_WS with hp_S):
        //   4×6 J_extr: rows 0:3, col 0:3 = −I · hp_S[3]
        //               rows 0:3, col 3:6 = +[hp_C[0:3]]×
        //
        //   J_T_SC = −Jh_weighted · J_extr            (1×6)

        // Jh_weighted (1×4 row vector)
        let jh =
            SMatrix::<f64, 1, 4>::from_row_slice(&[0.0, 0.0, self.sqrt_information / hp_c[3], 0.0]);

        // ── J_T_WS (1×6) ─────────────────────────────────────────────────────
        let hp_s_vec = Vector3::new(hp_s[0], hp_s[1], hp_s[2]);
        let mut j_pose = SMatrix::<f64, 4, 6>::zeros();
        // ∂hp_S[0:3]/∂δρ = −I · hp_W[3]
        j_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-Matrix3::identity() * hp_w[3]));
        // ∂hp_S[0:3]/∂δθ = [hp_S[0:3]]×
        j_pose
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&skew(&hp_s_vec));

        let j_t_ws: SMatrix<f64, 1, 6> = -jh * t_cs * j_pose;

        // ── J_hp_W (1×4) ─────────────────────────────────────────────────────
        //
        // When hp_W[3] is perturbed, hp_C[3] also changes (unlike pose perturbations).
        // Need full quotient rule: d(depth)/d(hp_C) = [0, 0, 1/hp_C[3], -depth/hp_C[3]]
        //
        // J_hp_W = -sqrt_info * d(depth)/d(hp_C) * d(hp_C)/d(hp_W)
        //        = -sqrt_info * [0, 0, 1/hp_C[3], -depth/hp_C[3]] * T_CW
        let t_cw: Matrix4<f64> = t_cs * t_sw;
        let jh_full = SMatrix::<f64, 1, 4>::from_row_slice(&[
            0.0,
            0.0,
            self.sqrt_information / hp_c[3],
            -self.sqrt_information * depth / hp_c[3],
        ]);
        let j_hp_w: SMatrix<f64, 1, 4> = -jh_full * t_cw;

        // ── J_T_SC (1×6) ─────────────────────────────────────────────────────
        let hp_c_vec = Vector3::new(hp_c[0], hp_c[1], hp_c[2]);
        let mut j_extr = SMatrix::<f64, 4, 6>::zeros();
        // ∂hp_C[0:3]/∂δρ_SC = −I · hp_S[3]
        j_extr
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-Matrix3::identity() * hp_s[3]));
        // ∂hp_C[0:3]/∂δθ_SC = [hp_C[0:3]]×
        j_extr
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&skew(&hp_c_vec));

        let j_t_sc: SMatrix<f64, 1, 6> = -jh * j_extr;

        // Assemble full 1×16 Jacobian: [J_T_WS(6) | J_hp_W(4) | J_T_SC(6)]
        let mut j_full = SMatrix::<f64, 1, 16>::zeros();
        j_full.fixed_view_mut::<1, 6>(0, 0).copy_from(&j_t_ws);
        j_full.fixed_view_mut::<1, 4>(0, 6).copy_from(&j_hp_w);
        j_full.fixed_view_mut::<1, 6>(0, 10).copy_from(&j_t_sc);

        for col in 0..16 {
            *jac.rb_mut().get_mut(0, col) = j_full[(0, col)];
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 16)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE, 4, SE3::REP_SIZE],
            "DepthFactor expects [SE3 T_WS, homogeneous point (4D), SE3 T_SC]",
        )
    }

    /// This factor whitens with the sqrt-information supplied at construction,
    /// so it must be registered with `NoiseModel::null()`.
    fn whitens_internally(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};

    fn identity_pose() -> DVector<f64> {
        DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    }

    fn make_pose(tx: f64, ty: f64, tz: f64, q: UnitQuaternion<f64>) -> DVector<f64> {
        let q = q.quaternion();
        DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k])
    }

    fn perturb_se3(pose: &[f64], tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from_param_slice(pose);
        let tan = SE3Tangent::from_slice(tangent);
        DVector::from_column_slice(se3.right_plus(&tan, None, None).as_param_slice())
    }

    /// Compute ground-truth depth for given params.
    fn compute_depth(t_ws: &[f64], hp_w: &[f64], t_sc: &[f64]) -> f64 {
        let ws = SE3::from_param_slice(t_ws);
        let sc = SE3::from_param_slice(t_sc);
        let t_sw = ws.inverse(None).matrix();
        let t_cs = sc.inverse(None).matrix();
        let hp = Vector4::new(hp_w[0], hp_w[1], hp_w[2], hp_w[3]);
        let hp_c = t_cs * t_sw * hp;
        hp_c[2] / hp_c[3]
    }

    fn compute_residual<const ONESIDED: bool>(
        factor: &DepthFactor<ONESIDED>,
        pose_ws: &[f64],
        hp_w: &[f64],
        pose_sc: &[f64],
    ) -> Vec<f64> {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(&[pose_ws, hp_w, pose_sc], &mut residual, None);
        residual
    }

    fn compute_with_jacobian<const ONESIDED: bool>(
        factor: &DepthFactor<ONESIDED>,
        pose_ws: &[f64],
        hp_w: &[f64],
        pose_sc: &[f64],
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[pose_ws, hp_w, pose_sc], &mut residual, Some(jac_mut));
        let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
        (residual, jacobian)
    }

    // ── Test 1: zero residual ───────────────────────────────────────────────

    #[test]
    fn zero_residual() {
        let q_ws = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.3,
        );
        let pose_ws = make_pose(1.0, 2.0, 0.5, q_ws);

        let q_sc = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            0.1,
        );
        let pose_sc = make_pose(0.05, 0.0, -0.02, q_sc);

        let hp_w = DVector::from_vec(vec![5.0, 3.0, 1.0, 1.0]);

        let depth = compute_depth(pose_ws.as_slice(), hp_w.as_slice(), pose_sc.as_slice());

        let factor = RegularDepthFactor::new_from_stdev(depth, 0.5);
        let r = compute_residual(
            &factor,
            pose_ws.as_slice(),
            hp_w.as_slice(),
            pose_sc.as_slice(),
        );

        assert!(r[0].abs() < 1e-10, "residual = {} should be zero", r[0]);
    }

    // ── Test 2: one-sided ignores depth > measurement ───────────────────────

    #[test]
    fn onesided_zero_when_depth_exceeds() {
        let pose_ws = identity_pose();
        let pose_sc = identity_pose();
        // Point at z=5 in camera frame (identity poses)
        let hp_w = DVector::from_vec(vec![0.0, 0.0, 5.0, 1.0]);

        // Measurement is 3.0, but depth is 5.0 → should be ignored
        let factor = OneSidedDepthFactor::new_from_stdev(3.0, 1.0);
        let r = compute_residual(
            &factor,
            pose_ws.as_slice(),
            hp_w.as_slice(),
            pose_sc.as_slice(),
        );

        assert!(r[0].abs() < 1e-16, "one-sided residual should be zero");
    }

    // ── Test 3: one-sided penalizes when depth < measurement ────────────────

    #[test]
    fn onesided_nonzero_when_depth_less() {
        let pose_ws = identity_pose();
        let pose_sc = identity_pose();
        let hp_w = DVector::from_vec(vec![0.0, 0.0, 2.0, 1.0]);

        // Measurement is 5.0, depth is 2.0 → should penalize
        let factor = OneSidedDepthFactor::new_from_stdev(5.0, 1.0);
        let r = compute_residual(
            &factor,
            pose_ws.as_slice(),
            hp_w.as_slice(),
            pose_sc.as_slice(),
        );

        assert!(
            (r[0] - 3.0).abs() < 1e-10,
            "residual = {} expected 3.0",
            r[0]
        );
    }

    // ── Test 4: finite-difference Jacobian check ────────────────────────────

    #[test]
    fn finite_difference_jacobians() {
        let q_ws = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.25,
        );
        let pose_ws = make_pose(1.0, -0.5, 0.3, q_ws);

        let q_sc = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            -0.15,
        );
        let pose_sc = make_pose(0.05, 0.02, -0.01, q_sc);

        let hp_w = DVector::from_vec(vec![3.0, 2.0, 4.0, 0.8]);

        let depth = compute_depth(pose_ws.as_slice(), hp_w.as_slice(), pose_sc.as_slice());
        let factor = RegularDepthFactor::new_from_stdev(depth + 0.1, 0.3);

        let (r0, jac) = compute_with_jacobian(
            &factor,
            pose_ws.as_slice(),
            hp_w.as_slice(),
            pose_sc.as_slice(),
        );

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        // Block 0: T_WS (6 DOF, cols 0–5)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_ws_p = perturb_se3(pose_ws.as_slice(), &tan);
            let r_pert = compute_residual(
                &factor,
                pose_ws_p.as_slice(),
                hp_w.as_slice(),
                pose_sc.as_slice(),
            );
            let fd = (r_pert[0] - r0[0]) / EPS;
            let err = (fd - jac[(0, col)]).abs();
            assert!(
                err < TOL,
                "J_T_WS[0,{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                jac[(0, col)],
                fd
            );
        }

        // Block 1: hp_W (4D, cols 6–9)
        for col in 0..4 {
            let mut hp_w_p = hp_w.clone();
            hp_w_p[col] += EPS;
            let r_pert = compute_residual(
                &factor,
                pose_ws.as_slice(),
                hp_w_p.as_slice(),
                pose_sc.as_slice(),
            );
            let fd = (r_pert[0] - r0[0]) / EPS;
            let err = (fd - jac[(0, 6 + col)]).abs();
            assert!(
                err < TOL,
                "J_hp_W[0,{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                jac[(0, 6 + col)],
                fd
            );
        }

        // Block 2: T_SC (6 DOF, cols 10–15)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let pose_sc_p = perturb_se3(pose_sc.as_slice(), &tan);
            let r_pert = compute_residual(
                &factor,
                pose_ws.as_slice(),
                hp_w.as_slice(),
                pose_sc_p.as_slice(),
            );
            let fd = (r_pert[0] - r0[0]) / EPS;
            let err = (fd - jac[(0, 10 + col)]).abs();
            assert!(
                err < TOL,
                "J_T_SC[0,{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                jac[(0, 10 + col)],
                fd
            );
        }
    }

    // ── Test 5: with w != 1 ─────────────────────────────────────────────────

    #[test]
    fn nonunit_homogeneous_w() {
        let pose_ws = identity_pose();
        let pose_sc = identity_pose();
        // hp = [0, 0, 10, 2] → p = [0, 0, 5], depth = 5
        let hp_w = DVector::from_vec(vec![0.0, 0.0, 10.0, 2.0]);

        let depth = compute_depth(pose_ws.as_slice(), hp_w.as_slice(), pose_sc.as_slice());
        assert!((depth - 5.0).abs() < 1e-10);

        let factor = RegularDepthFactor::new_from_stdev(depth, 1.0);
        let r = compute_residual(
            &factor,
            pose_ws.as_slice(),
            hp_w.as_slice(),
            pose_sc.as_slice(),
        );
        assert!(r[0].abs() < 1e-10);
    }
}
