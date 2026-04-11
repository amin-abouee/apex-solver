//! ICP-style alignment factor for point-to-field registration.
//!
//! Implements the OKVIS2-X `SubmapIcpError` factor: aligns a query point from
//! frame B against a distance/occupancy field defined in frame A.
//!
//! # Mathematical Formulation
//!
//! ```text
//! T_AB = T_WA⁻¹ · T_WB
//! p_A  = T_AB · p_B                         (transform query point to field frame)
//! e    = sqrt_info · field(p_A) / ‖∇field(p_A)‖   ∈ R
//! ```
//!
//! # Parameter Layout (2 blocks, 12 DOF total)
//!
//! ```text
//! params[0]: T_WA — reference submap/field pose (7D, 6 DOF)
//! params[1]: T_WB — query point cloud pose (7D, 6 DOF)
//! ```
//!
//! # Jacobians (1×12)
//!
//! ```text
//! ∂e/∂p_A  = sqrt_info · ∇field / ‖∇field‖   (1×3)
//!
//! ∂p_A/∂T_WA = [−C_WAᵀ | C_WAᵀ · [C_WB·p_B + t_WB − t_WA]×]   (3×6)
//! ∂p_A/∂T_WB = [ C_WAᵀ | −C_WAᵀ · [C_WB·p_B]×]                 (3×6)
//! ```

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

use apex_manifolds::se3::SE3;

use crate::factors::imu::helpers::cross_matrix;
use crate::factors::Factor;

/// Trait for querying a distance/occupancy field.
///
/// Implementors provide the field value and its spatial gradient at a query point.
pub trait DistanceField: Send + Sync {
    /// Query the field at point `p` (in the field's own frame).
    ///
    /// Returns `Some((value, gradient))` if the point is inside the field domain,
    /// or `None` if outside.
    fn query(&self, point: &Vector3<f64>) -> Option<(f64, Vector3<f64>)>;
}

/// ICP factor: aligns a single point from frame B against a field in frame A.
pub struct IcpFactor<F: DistanceField> {
    /// The distance/occupancy field defined in frame A.
    field: F,
    /// Query point in frame B.
    point_b: Vector3<f64>,
    /// Measurement noise standard deviation [m].
    sigma_measurement: f64,
}

impl<F: DistanceField> IcpFactor<F> {
    /// Create an ICP factor.
    ///
    /// # Arguments
    /// * `field` — distance field in frame A
    /// * `point_b` — 3D point in frame B
    /// * `sigma_measurement` — measurement noise standard deviation
    pub fn new(field: F, point_b: Vector3<f64>, sigma_measurement: f64) -> Self {
        Self {
            field,
            point_b,
            sigma_measurement,
        }
    }
}

impl<F: DistanceField> Factor for IcpFactor<F> {
    fn get_dimension(&self) -> usize {
        1
    }

    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        debug_assert_eq!(params.len(), 2, "IcpFactor expects 2 parameter blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 7, "params[1] must be SE3 (7D)");

        let t_wa = SE3::from(params[0].clone());
        let t_wb = SE3::from(params[1].clone());

        let c_wa: Matrix3<f64> = t_wa.rotation_so3().rotation_matrix();
        let c_wb: Matrix3<f64> = t_wb.rotation_so3().rotation_matrix();
        let t_wa_pos = t_wa.translation();
        let t_wb_pos = t_wb.translation();

        // Transform point to frame A: p_A = C_WA^T * (C_WB * p_B + t_WB - t_WA)
        let p_w = c_wb * self.point_b + t_wb_pos;
        let p_a = c_wa.transpose() * (p_w - t_wa_pos);

        let zero_res = DVector::zeros(1);
        let zero_jac = || DMatrix::zeros(1, 12);

        // Query the field
        let (field_val, gradient) = match self.field.query(&p_a) {
            Some(v) => v,
            None => {
                if compute_jacobian {
                    return (zero_res, Some(zero_jac()));
                }
                return (zero_res, None);
            }
        };

        let grad_norm = gradient.norm();
        if grad_norm < 1e-3 {
            if compute_jacobian {
                return (zero_res, Some(zero_jac()));
            }
            return (zero_res, None);
        }

        // Compute total sigma (measurement + map uncertainty)
        // Following OKVIS: sigma_map = |log_odd_min| / (3 * grad_norm)
        // For generic use, we just use sigma_measurement directly
        let sqrt_info = 1.0 / self.sigma_measurement;

        // Weighted residual
        let weighted_error = sqrt_info * field_val / grad_norm;
        let residual = DVector::from_element(1, weighted_error);

        if !compute_jacobian {
            return (residual, None);
        }

        // ── Jacobians (from OKVIS SubmapIcpError.cpp) ─────────────────────────
        //
        // de/dp_A = sqrt_info * grad / grad_norm   (1×3)
        //
        // Using right perturbation of T_WA:
        //   t_WA → t_WA + C_WA · δρ_A
        //   C_WA → C_WA · (I + [δθ_A]×)
        //
        //   p_A = C_WA^T * (p_W - t_WA)
        //       → (I - [δθ_A]×) * C_WA^T * (p_W - t_WA - C_WA*δρ_A)
        //       ≈ p_A - δρ_A + [δθ_A]× * p_A    (wait, OKVIS derivation...)
        //
        //   Actually from OKVIS code directly:
        //   dp_A/dT_WA = [-C_WA^T | C_WA^T * [p_W - t_WA]×]
        //              = [-C_WA^T | C_WA^T * [C_WB*p_B + t_WB - t_WA]×]
        //
        // For T_WB:
        //   dp_A/dT_WB = [C_WA^T | -C_WA^T * [C_WB*p_B]×]

        let de_dp_a: SMatrix<f64, 1, 3> =
            SMatrix::<f64, 1, 3>::from_iterator((sqrt_info * gradient / grad_norm).iter().copied());

        let c_wa_t = c_wa.transpose();
        let diff = p_w - t_wa_pos; // C_WB*p_B + t_WB - t_WA

        // dp_A/dT_WA (3×6)
        let mut dpa_dtwa = SMatrix::<f64, 3, 6>::zeros();
        dpa_dtwa
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-c_wa_t));
        dpa_dtwa
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(c_wa_t * cross_matrix(&diff)));

        // dp_A/dT_WB (3×6)
        let c_wb_pb = c_wb * self.point_b;
        let mut dpa_dtwb = SMatrix::<f64, 3, 6>::zeros();
        dpa_dtwb
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&c_wa_t);
        dpa_dtwb
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-c_wa_t * cross_matrix(&c_wb_pb)));

        // Chain rule
        let j_twa: SMatrix<f64, 1, 6> = de_dp_a * dpa_dtwa;
        let j_twb: SMatrix<f64, 1, 6> = de_dp_a * dpa_dtwb;

        // Assemble 1×12 Jacobian
        let mut j_full = SMatrix::<f64, 1, 12>::zeros();
        j_full.fixed_view_mut::<1, 6>(0, 0).copy_from(&j_twa);
        j_full.fixed_view_mut::<1, 6>(0, 6).copy_from(&j_twb);

        let jac = DMatrix::from_iterator(1, 12, j_full.iter().copied());

        (residual, Some(jac))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::se3::SE3Tangent;
    use nalgebra::{UnitQuaternion, Vector3};

    /// Simple planar distance field: f(p) = n·p - d, grad = n.
    struct PlanarField {
        normal: Vector3<f64>,
        offset: f64,
    }

    impl DistanceField for PlanarField {
        fn query(&self, point: &Vector3<f64>) -> Option<(f64, Vector3<f64>)> {
            let val = self.normal.dot(point) - self.offset;
            Some((val, self.normal))
        }
    }

    /// Sphere field: f(p) = ||p - center|| - radius, grad = (p - center).normalize()
    struct SphereField {
        center: Vector3<f64>,
        radius: f64,
    }

    impl DistanceField for SphereField {
        fn query(&self, point: &Vector3<f64>) -> Option<(f64, Vector3<f64>)> {
            let diff = point - self.center;
            let dist = diff.norm();
            if dist < 1e-16 {
                return None;
            }
            Some((dist - self.radius, diff / dist))
        }
    }

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

    // ── Test 1: zero residual when point lies on the plane ──────────────────

    #[test]
    fn zero_residual_on_plane() {
        let field = PlanarField {
            normal: Vector3::z(),
            offset: 5.0,
        };

        // Frames aligned, point on the z=5 plane
        let t_wa = identity_pose();
        let t_wb = identity_pose();
        let point_b = Vector3::new(1.0, 2.0, 5.0); // on the plane

        let factor = IcpFactor::new(field, point_b, 1.0);
        let (r, _) = factor.linearize(&[t_wa, t_wb], false);

        assert!(r[0].abs() < 1e-10, "residual = {} should be zero", r[0]);
    }

    // ── Test 2: non-zero residual ───────────────────────────────────────────

    #[test]
    fn nonzero_residual_off_plane() {
        let field = PlanarField {
            normal: Vector3::z(),
            offset: 5.0,
        };

        let t_wa = identity_pose();
        let t_wb = identity_pose();
        let point_b = Vector3::new(1.0, 2.0, 7.0); // 2m above the plane

        let factor = IcpFactor::new(field, point_b, 1.0);
        let (r, _) = factor.linearize(&[t_wa, t_wb], false);

        // field_val = 7 - 5 = 2, grad_norm = 1, sqrt_info = 1
        assert!((r[0] - 2.0).abs() < 1e-10, "residual = {}", r[0]);
    }

    // ── Test 3: finite-difference Jacobian with planar field ────────────────

    #[test]
    fn finite_difference_jacobians_planar() {
        let field = PlanarField {
            normal: Vector3::new(0.0, 0.3, 0.9).normalize(),
            offset: 3.0,
        };

        let q_a = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.2,
        );
        let q_b = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            -0.15,
        );

        let t_wa = make_pose(1.0, -0.5, 0.3, q_a);
        let t_wb = make_pose(2.0, 1.0, -0.2, q_b);
        let point_b = Vector3::new(0.5, -0.3, 4.0);

        let factor = IcpFactor::new(field, point_b, 0.5);

        let nominal = vec![t_wa.clone(), t_wb.clone()];
        let (r0, jac_opt) = factor.linearize(&nominal, true);
        let jac = jac_opt.expect("Jacobian must be computed");

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        // Block 0: T_WA (6 DOF, cols 0–5)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[0] = perturb_se3(&t_wa, &tan);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (r_pert[0] - r0[0]) / EPS;
            let err = (fd - jac[(0, col)]).abs();
            assert!(
                err < TOL,
                "J_T_WA[0,{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                jac[(0, col)],
                fd
            );
        }

        // Block 1: T_WB (6 DOF, cols 6–11)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let mut p = nominal.clone();
            p[1] = perturb_se3(&t_wb, &tan);
            let (r_pert, _) = factor.linearize(&p, false);
            let fd = (r_pert[0] - r0[0]) / EPS;
            let err = (fd - jac[(0, 6 + col)]).abs();
            assert!(
                err < TOL,
                "J_T_WB[0,{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                jac[(0, 6 + col)],
                fd
            );
        }
    }

    // ── Test 4: finite-difference Jacobian with sphere field ────────────────

    #[test]
    fn finite_difference_jacobians_sphere() {
        let field = SphereField {
            center: Vector3::new(2.0, 1.0, 3.0),
            radius: 1.0,
        };

        let q_a = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            0.1,
        );
        let q_b = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            -0.2,
        );

        let t_wa = make_pose(0.5, 0.0, 0.0, q_a);
        let t_wb = make_pose(-0.5, 1.0, 0.5, q_b);
        let point_b = Vector3::new(1.0, 0.5, 2.0);

        let factor = IcpFactor::new(field, point_b, 0.3);

        let nominal = vec![t_wa.clone(), t_wb.clone()];
        let (r0, jac_opt) = factor.linearize(&nominal, true);
        let jac = jac_opt.expect("Jacobian must be computed");

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        // Both blocks
        for block in 0..2 {
            let pose = if block == 0 { &t_wa } else { &t_wb };
            let col_offset = block * 6;
            for col in 0..6 {
                let mut tan = [0.0f64; 6];
                tan[col] = EPS;
                let mut p = nominal.clone();
                p[block] = perturb_se3(pose, &tan);
                // Need fresh field for each evaluation
                let field2 = SphereField {
                    center: Vector3::new(2.0, 1.0, 3.0),
                    radius: 1.0,
                };
                let factor2 = IcpFactor::new(field2, point_b, 0.3);
                let (r_pert, _) = factor2.linearize(&p, false);
                let fd = (r_pert[0] - r0[0]) / EPS;
                let err = (fd - jac[(0, col_offset + col)]).abs();
                assert!(
                    err < TOL,
                    "J_block{block}[0,{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(0, col_offset + col)],
                    fd
                );
            }
        }
    }

    // ── Test 5: dimension ───────────────────────────────────────────────────

    #[test]
    fn dimension_is_one() {
        let field = PlanarField {
            normal: Vector3::z(),
            offset: 0.0,
        };
        let factor = IcpFactor::new(field, Vector3::zeros(), 1.0);
        assert_eq!(factor.get_dimension(), 1);
    }
}
