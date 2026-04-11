//! Essential matrix factor for two-view epipolar constraints.
//!
//! Implements the epipolar constraint between two poses given a set of
//! normalized bearing correspondences. Each correspondence contributes a
//! scalar residual `x₂ᵀ E x₁ = 0`.
//!
//! # Mathematical Formulation
//!
//! ```text
//! T_rel = T₁⁻¹ · T₂
//! E     = [t_rel]× · R_rel           (essential matrix)
//! eₖ   = wₖ · x₂ₖᵀ · E · x₁ₖ      (per-correspondence epipolar error)
//! ```
//!
//! # Parameter Layout (2 blocks, 12 DOF total)
//!
//! ```text
//! params[0]: T₁ — first camera pose (7D, 6 DOF)
//! params[1]: T₂ — second camera pose (7D, 6 DOF)
//! ```
//!
//! # Jacobians (N×12)
//!
//! Per correspondence k:
//! ```text
//! ∂eₖ/∂δ_rel = [(R·x₁ × x₂)ᵀ·R  |  −x₂ᵀ·E·[x₁]×]   (1×6)
//! ∂eₖ/∂T₁    = ∂eₖ/∂δ_rel · (−Adj(T_rel⁻¹))
//! ∂eₖ/∂T₂    = ∂eₖ/∂δ_rel
//! ```

use nalgebra::{DMatrix, DVector, Matrix3, Matrix6, SMatrix, Vector3};

use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;

use crate::factors::imu::helpers::cross_matrix;
use crate::factors::Factor;

/// Essential matrix factor for epipolar constraints.
///
/// Given N pairs of normalized bearing vectors `(x₁ₖ, x₂ₖ)` observed in two
/// camera frames, this factor enforces the epipolar constraint `x₂ᵀ E x₁ = 0`
/// for all pairs simultaneously.
pub struct EssentialMatrixFactor {
    /// Normalized bearing vectors in frame 1 (homogeneous 3D).
    correspondences_1: Vec<Vector3<f64>>,
    /// Normalized bearing vectors in frame 2 (homogeneous 3D).
    correspondences_2: Vec<Vector3<f64>>,
    /// Per-correspondence scalar weights (square-root information).
    sqrt_information: Vec<f64>,
}

impl EssentialMatrixFactor {
    /// Create an essential matrix factor.
    ///
    /// # Arguments
    /// * `correspondences_1` — normalized bearing vectors in frame 1
    /// * `correspondences_2` — normalized bearing vectors in frame 2
    /// * `sqrt_information` — per-correspondence scalar weights
    ///
    /// # Panics
    /// Panics if the three vectors have different lengths.
    pub fn new(
        correspondences_1: Vec<Vector3<f64>>,
        correspondences_2: Vec<Vector3<f64>>,
        sqrt_information: Vec<f64>,
    ) -> Self {
        assert_eq!(
            correspondences_1.len(),
            correspondences_2.len(),
            "Correspondence count mismatch"
        );
        assert_eq!(
            correspondences_1.len(),
            sqrt_information.len(),
            "Weight count mismatch"
        );
        Self {
            correspondences_1,
            correspondences_2,
            sqrt_information,
        }
    }

    /// Create with uniform unit weights.
    pub fn new_uniform(
        correspondences_1: Vec<Vector3<f64>>,
        correspondences_2: Vec<Vector3<f64>>,
    ) -> Self {
        let n = correspondences_1.len();
        Self::new(correspondences_1, correspondences_2, vec![1.0; n])
    }
}

impl Factor for EssentialMatrixFactor {
    fn get_dimension(&self) -> usize {
        self.correspondences_1.len()
    }

    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        debug_assert_eq!(
            params.len(),
            2,
            "EssentialMatrixFactor expects 2 parameter blocks"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 7, "params[1] must be SE3 (7D)");

        let n = self.correspondences_1.len();

        let t_1 = SE3::from(params[0].clone());
        let t_2 = SE3::from(params[1].clone());

        // Relative pose: T_rel = T_1^{-1} * T_2
        let t_1_inv = t_1.inverse(None);
        let t_rel = t_1_inv.compose(&t_2, None, None);

        let r_rel: Matrix3<f64> = t_rel.rotation_so3().rotation_matrix();
        let t_rel_vec: Vector3<f64> = t_rel.translation();

        // Essential matrix: E = [t]× · R
        let t_hat: Matrix3<f64> = cross_matrix(&t_rel_vec);
        let e_matrix: Matrix3<f64> = t_hat * r_rel;

        // Compute residuals
        let mut residual = DVector::zeros(n);
        for k in 0..n {
            let x1 = &self.correspondences_1[k];
            let x2 = &self.correspondences_2[k];
            residual[k] = self.sqrt_information[k] * x2.dot(&(e_matrix * x1));
        }

        if !compute_jacobian {
            return (residual, None);
        }

        // ── Jacobians ─────────────────────────────────────────────────────────
        //
        // Per correspondence k, the Jacobian w.r.t. right perturbation of T_rel:
        //
        // de_k/d(δρ_rel):
        //   E → E + [R·δρ]× · R  (first order in δρ)
        //   δe_k = x2^T · [R·δρ]× · R · x1 = x2^T · (R·δρ × R·x1)
        //        = (R·x1 × x2)^T · R · δρ
        //   So: de_k/d(δρ) = (R·x1 × x2)^T · R   (1×3)
        //
        // de_k/d(δθ_rel):
        //   R → R·(I + [δθ]×),  t stays same
        //   E → [t]× · R · (I + [δθ]×) = E + [t]× · R · [δθ]×
        //   δe_k = x2^T · [t]× · R · [δθ]× · x1 = x2^T · E · (δθ × x1)
        //        = x2^T · E · (-[x1]× · δθ)
        //   So: de_k/d(δθ) = -x2^T · E · [x1]×   (1×3)
        //
        // Chain to T_1 and T_2 (right perturbation):
        //   T_rel = T_1^{-1} · T_2
        //   Under T_1 → T_1 · Exp(δ_1):
        //     T_1^{-1} → Exp(-δ_1) · T_1^{-1}
        //     T_rel → Exp(-δ_1) · T_rel
        //     Right perturbation of T_rel: δ_rel = -Adj(T_rel^{-1}) · δ_1
        //     So: d(δ_rel)/d(δ_1) = -Adj(T_rel^{-1})
        //
        //   Under T_2 → T_2 · Exp(δ_2):
        //     T_rel → T_1^{-1} · T_2 · Exp(δ_2) = T_rel · Exp(δ_2)
        //     So: d(δ_rel)/d(δ_2) = I_6

        let t_rel_inv = t_rel.inverse(None);
        let adj_rel_inv: Matrix6<f64> = t_rel_inv.adjoint();

        let mut jac = DMatrix::zeros(n, 12);

        for k in 0..n {
            let x1 = &self.correspondences_1[k];
            let x2 = &self.correspondences_2[k];
            let w = self.sqrt_information[k];

            // de_k/d(δ_rel) (1×6)
            let r_x1 = r_rel * x1;
            let de_drho: Vector3<f64> = r_x1.cross(x2); // (R·x1 × x2)
            let de_drho_row: SMatrix<f64, 1, 3> =
                SMatrix::from_iterator((de_drho.transpose() * r_rel).iter().copied());

            let x1_hat = cross_matrix(x1);
            let de_dtheta_row: SMatrix<f64, 1, 3> =
                SMatrix::from_iterator((-x2.transpose() * e_matrix * x1_hat).iter().copied());

            let mut de_drel = SMatrix::<f64, 1, 6>::zeros();
            de_drel.fixed_view_mut::<1, 3>(0, 0).copy_from(&de_drho_row);
            de_drel
                .fixed_view_mut::<1, 3>(0, 3)
                .copy_from(&de_dtheta_row);

            // Chain to T_1: J_T1 = w * de_drel * (-Adj(T_rel^{-1}))
            let j_t1 = w * de_drel * (-adj_rel_inv);
            // Chain to T_2: J_T2 = w * de_drel
            let j_t2 = w * de_drel;

            for col in 0..6 {
                jac[(k, col)] = j_t1[(0, col)];
                jac[(k, 6 + col)] = j_t2[(0, col)];
            }
        }

        (residual, Some(jac))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3Tangent;
    use nalgebra::{UnitQuaternion, Vector3};

    fn make_pose(tx: f64, ty: f64, tz: f64, q: UnitQuaternion<f64>) -> DVector<f64> {
        let q = q.quaternion();
        DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k])
    }

    fn identity_pose() -> DVector<f64> {
        DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    }

    fn perturb_se3(pose: &DVector<f64>, tangent: &[f64; 6]) -> DVector<f64> {
        let se3 = SE3::from(pose.clone());
        let tan = SE3Tangent::from(DVector::from_vec(tangent.to_vec()));
        DVector::from(se3.right_plus(&tan, None, None))
    }

    /// Generate synthetic correspondences from 3D points observed by two poses.
    fn generate_correspondences(
        t_1: &DVector<f64>,
        t_2: &DVector<f64>,
        points_w: &[Vector3<f64>],
    ) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let se3_1 = SE3::from(t_1.clone());
        let se3_2 = SE3::from(t_2.clone());
        let r1 = se3_1.rotation_so3().rotation_matrix();
        let t1 = se3_1.translation();
        let r2 = se3_2.rotation_so3().rotation_matrix();
        let t2 = se3_2.translation();

        let mut c1 = Vec::new();
        let mut c2 = Vec::new();

        for p in points_w {
            let p_c1 = r1.transpose() * (p - t1);
            let p_c2 = r2.transpose() * (p - t2);
            c1.push(p_c1.normalize());
            c2.push(p_c2.normalize());
        }

        (c1, c2)
    }

    // ── Test 1: zero residual with perfect correspondences ──────────────────

    #[test]
    fn zero_residual() {
        let q1 = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.1,
        );
        let q2 = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            -0.2,
        );

        let t_1 = make_pose(0.0, 0.0, 0.0, q1);
        let t_2 = make_pose(1.0, 0.0, 0.5, q2);

        let points = vec![
            Vector3::new(5.0, 1.0, 3.0),
            Vector3::new(-2.0, 3.0, 8.0),
            Vector3::new(1.0, -1.0, 6.0),
            Vector3::new(3.0, 4.0, 10.0),
        ];

        let (c1, c2) = generate_correspondences(&t_1, &t_2, &points);
        let factor = EssentialMatrixFactor::new_uniform(c1, c2);
        let (r, _) = factor.linearize(&[t_1, t_2], false);

        for k in 0..r.len() {
            assert!(
                r[k].abs() < 1e-10,
                "residual[{k}] = {} should be zero",
                r[k]
            );
        }
    }

    // ── Test 2: pure translation ────────────────────────────────────────────

    #[test]
    fn pure_translation() {
        let t_1 = identity_pose();
        let t_2 = make_pose(2.0, 0.0, 0.0, UnitQuaternion::identity());

        let points = vec![
            Vector3::new(3.0, 1.0, 5.0),
            Vector3::new(-1.0, 2.0, 4.0),
            Vector3::new(0.0, -3.0, 7.0),
        ];

        let (c1, c2) = generate_correspondences(&t_1, &t_2, &points);
        let factor = EssentialMatrixFactor::new_uniform(c1, c2);
        let (r, _) = factor.linearize(&[t_1, t_2], false);

        for k in 0..r.len() {
            assert!(r[k].abs() < 1e-10, "residual[{k}] = {}", r[k]);
        }
    }

    // ── Test 3: finite-difference Jacobian check ────────────────────────────

    #[test]
    fn finite_difference_jacobians() {
        let q1 = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.3, 0.7, 0.1)),
            0.15,
        );
        let q2 = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.5, 0.2, 0.8)),
            -0.25,
        );

        let t_1 = make_pose(0.5, -0.2, 0.1, q1);
        let t_2 = make_pose(1.5, 0.3, -0.4, q2);

        let points = vec![
            Vector3::new(5.0, 1.0, 8.0),
            Vector3::new(-2.0, 3.0, 6.0),
            Vector3::new(1.0, -1.0, 10.0),
        ];

        // Use slightly perturbed correspondences so residuals are non-zero
        let (c1, c2) = generate_correspondences(&t_1, &t_2, &points);
        let mut c2_noisy = c2;
        c2_noisy[0] += Vector3::new(0.001, -0.002, 0.0005);
        c2_noisy[0].normalize_mut();

        let weights = vec![1.0, 0.5, 2.0];
        let factor = EssentialMatrixFactor::new(c1.clone(), c2_noisy.clone(), weights.clone());

        let nominal = vec![t_1.clone(), t_2.clone()];
        let (r0, jac_opt) = factor.linearize(&nominal, true);
        let jac = jac_opt.expect("Jacobian must be computed");

        let n = r0.len();
        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        // Block 0: T_1 (6 DOF, cols 0–5)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[0] = perturb_se3(&t_1, &tan);
            let factor2 =
                EssentialMatrixFactor::new(c1.clone(), c2_noisy.clone(), weights.clone());
            let (r_pert, _) = factor2.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..n {
                let err = (fd[row] - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J_T1[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, col)],
                    fd[row]
                );
            }
        }

        // Block 1: T_2 (6 DOF, cols 6–11)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[1] = perturb_se3(&t_2, &tan);
            let factor2 =
                EssentialMatrixFactor::new(c1.clone(), c2_noisy.clone(), weights.clone());
            let (r_pert, _) = factor2.linearize(&p, false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..n {
                let err = (fd[row] - jac[(row, 6 + col)]).abs();
                assert!(
                    err < TOL,
                    "J_T2[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, 6 + col)],
                    fd[row]
                );
            }
        }
    }

    // ── Test 4: dimension matches correspondence count ──────────────────────

    #[test]
    fn dimension_matches_count() {
        let n = 5;
        let c1: Vec<_> = (0..n)
            .map(|i| Vector3::new(i as f64, 1.0, 5.0).normalize())
            .collect();
        let c2 = c1.clone();
        let factor = EssentialMatrixFactor::new_uniform(c1, c2);
        assert_eq!(factor.get_dimension(), n);
    }

    // ── Test 5: single correspondence ───────────────────────────────────────

    #[test]
    fn single_correspondence() {
        let t_1 = identity_pose();
        let t_2 = make_pose(1.0, 0.0, 0.0, UnitQuaternion::identity());

        let point = Vector3::new(0.5, 1.0, 5.0);
        let (c1, c2) = generate_correspondences(&t_1, &t_2, &[point]);

        let factor = EssentialMatrixFactor::new_uniform(c1, c2);
        let (r, _) = factor.linearize(&[t_1, t_2], false);

        assert_eq!(r.len(), 1);
        assert!(r[0].abs() < 1e-10);
    }
}
