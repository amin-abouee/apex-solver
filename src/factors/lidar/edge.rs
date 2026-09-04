//! LOAM-style point-to-edge LiDAR factor.
//!
//! Implements the edge-feature residual from Zhang & Singh (2014, "LOAM: Lidar
//! Odometry and Mapping in Real-time"), following the same
//! precomputed-correspondence pattern as [`IcpFactor`](super::distance_field::IcpFactor):
//! the caller supplies an already-matched edge (point + direction) alongside
//! the query point — no nearest-neighbor search happens inside the factor.
//!
//! ```text
//! T_AB = T_WA⁻¹ · T_WB
//! p_A  = T_AB · p_B                                  (query point, frame A)
//! e    = sqrt_info · (I − d·dᵀ) · (p_A − q)   ∈ R³   (perpendicular offset from the edge line)
//! ```
//!
//! where `q` is a point on the matched edge line and `d` its unit direction. The
//! vector-rejection residual is used (rather than the scalar cross-product
//! magnitude `‖(p_A − q) × d‖`) to match this crate's vector-residual convention
//! and to avoid the Jacobian singularity the magnitude form has at its own zero.
//!
//! The companion point-to-plane residual lives in [`plane`](super::plane).
//!
//! # Parameter Layout (2 blocks, 12 DOF total)
//!
//! ```text
//! params[0]: T_WA — reference/map pose (7D, 6 DOF)
//! params[1]: T_WB — query point cloud pose (7D, 6 DOF)
//! ```

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector3};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;

use crate::factors::Factor;
use crate::factors::common::math::skew;

/// Point-to-edge (point-to-line) LOAM factor.
///
/// Aligns a query point from frame B against a matched edge line (point +
/// unit direction) defined in frame A.
pub struct LidarEdgeFactor {
    /// Query point in frame B.
    point_b: Vector3<f64>,
    /// A point on the matched edge line, in frame A.
    edge_point: Vector3<f64>,
    /// Unit direction vector of the matched edge line, in frame A.
    edge_direction: Vector3<f64>,
    /// Square-root information matrix (3×3, `W` s.t. `Wᵀ W = Σ⁻¹`).
    sqrt_information: SMatrix<f64, 3, 3>,
}

impl LidarEdgeFactor {
    /// Create a point-to-edge factor.
    ///
    /// # Arguments
    /// * `point_b` — query point in frame B
    /// * `edge_point` — a point on the matched edge line, in frame A
    /// * `edge_direction` — direction of the matched edge line, in frame A
    ///   (normalized internally; need not be unit length on input)
    /// * `sqrt_information` — 3×3 square-root information matrix
    pub fn new(
        point_b: Vector3<f64>,
        edge_point: Vector3<f64>,
        edge_direction: Vector3<f64>,
        sqrt_information: SMatrix<f64, 3, 3>,
    ) -> Self {
        Self {
            point_b,
            edge_point,
            edge_direction: edge_direction.normalize(),
            sqrt_information,
        }
    }

    /// Create with isotropic noise (scalar standard deviation).
    pub fn new_isotropic(
        point_b: Vector3<f64>,
        edge_point: Vector3<f64>,
        edge_direction: Vector3<f64>,
        sigma: f64,
    ) -> Self {
        let sqrt_info = SMatrix::<f64, 3, 3>::identity() * (1.0 / sigma);
        Self::new(point_b, edge_point, edge_direction, sqrt_info)
    }
}

impl Factor for LidarEdgeFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            2,
            "LidarEdgeFactor expects 2 parameter blocks"
        );
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 7, "params[1] must be SE3 (7D)");

        let t_wa = SE3::from_param_slice(params[0]);
        let t_wb = SE3::from_param_slice(params[1]);

        let c_wa: Matrix3<f64> = t_wa.rotation_so3().rotation_matrix();
        let c_wb: Matrix3<f64> = t_wb.rotation_so3().rotation_matrix();
        let t_wa_pos = t_wa.translation();
        let t_wb_pos = t_wb.translation();

        // Transform point to frame A: p_A = C_WA^T * (C_WB * p_B + t_WB - t_WA)
        let p_w = c_wb * self.point_b + t_wb_pos;
        let p_a = c_wa.transpose() * (p_w - t_wa_pos);

        // Perpendicular projector onto the plane orthogonal to the edge direction:
        // P = I - d*d^T  (dᵀd = 1 since edge_direction is normalized)
        let d = self.edge_direction;
        let projector = Matrix3::identity() - d * d.transpose();

        let e_vec = projector * (p_a - self.edge_point);
        let weighted = self.sqrt_information * e_vec;
        for i in 0..3 {
            residual[i] = weighted[i];
        }

        let Some(mut jac) = jacobian else {
            return;
        };

        // ── Jacobians (right SE3 perturbation, apex-solver convention) ─────────
        //
        // de/dp_A = sqrt_info * P   (3×3), where P = I - d*d^T is symmetric
        //
        // p_A = C_WA^T * (p_W - t_WA),  p_W = C_WB * p_B + t_WB
        // Same chain rule as IcpFactor (see icp_factor.rs for the full derivation):
        //   ∂p_A/∂δρ_WA = −I₃,             ∂p_A/∂δθ_WA = +[p_A]×
        //   ∂p_A/∂δρ_WB = C_WA^T·C_WB,     ∂p_A/∂δθ_WB = −C_WA^T·C_WB·[p_B]×

        let de_dp_a = self.sqrt_information * projector; // 3×3

        let c_wa_t = c_wa.transpose();
        let c_wa_t_c_wb = c_wa_t * c_wb;

        // dp_A/dT_WA (3×6): [-I | [p_A]×]
        let mut dpa_dtwa = SMatrix::<f64, 3, 6>::zeros();
        dpa_dtwa
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-Matrix3::identity()));
        dpa_dtwa.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(&p_a));

        // dp_A/dT_WB (3×6): [C_WA^T·C_WB | -C_WA^T·C_WB·[p_B]×]
        let mut dpa_dtwb = SMatrix::<f64, 3, 6>::zeros();
        dpa_dtwb
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&c_wa_t_c_wb);
        dpa_dtwb
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-c_wa_t_c_wb * skew(&self.point_b)));

        // Chain rule
        let j_twa: SMatrix<f64, 3, 6> = de_dp_a * dpa_dtwa;
        let j_twb: SMatrix<f64, 3, 6> = de_dp_a * dpa_dtwb;

        // Assemble 3×12 Jacobian
        let mut j_full = SMatrix::<f64, 3, 12>::zeros();
        j_full.fixed_view_mut::<3, 6>(0, 0).copy_from(&j_twa);
        j_full.fixed_view_mut::<3, 6>(0, 6).copy_from(&j_twb);

        for row in 0..3 {
            for col in 0..12 {
                *jac.rb_mut().get_mut(row, col) = j_full[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::common::test_utils::{
        compute_residual, compute_with_jacobian, identity_pose, make_pose, perturb_se3,
    };
    use nalgebra::UnitQuaternion;

    // ── LidarEdgeFactor ──────────────────────────────────────────────────────
    #[test]
    fn edge_zero_residual_when_point_on_line() {
        // Edge along z-axis through origin; point exactly on it.
        let t_wa = identity_pose();
        let t_wb = identity_pose();
        let point_b = Vector3::new(0.0, 0.0, 5.0);

        let factor = LidarEdgeFactor::new_isotropic(point_b, Vector3::zeros(), Vector3::z(), 1.0);
        let r = compute_residual(&factor, t_wa.as_slice(), t_wb.as_slice());
        let norm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "residual norm = {norm} should be zero");
    }

    #[test]
    fn edge_nonzero_residual_perpendicular_offset() {
        let t_wa = identity_pose();
        let t_wb = identity_pose();
        // Point offset by (3, 4, 0) perpendicular to a z-axis edge.
        let point_b = Vector3::new(3.0, 4.0, 5.0);

        let factor = LidarEdgeFactor::new_isotropic(point_b, Vector3::zeros(), Vector3::z(), 1.0);
        let r = compute_residual(&factor, t_wa.as_slice(), t_wb.as_slice());
        let norm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Perpendicular distance from (3,4,5) to the z-axis is 5.
        assert!(
            (norm - 5.0).abs() < 1e-10,
            "residual norm = {norm}, expected 5.0"
        );
    }

    #[test]
    fn edge_residual_orthogonal_to_direction() {
        // The residual vector must always be perpendicular to the edge direction.
        let t_wa = identity_pose();
        let t_wb = identity_pose();
        let direction = Vector3::new(1.0, 2.0, 3.0).normalize();
        let point_b = Vector3::new(-1.0, 2.0, 6.0);

        let factor =
            LidarEdgeFactor::new_isotropic(point_b, Vector3::new(0.5, -0.5, 1.0), direction, 1.0);
        let r = compute_residual(&factor, t_wa.as_slice(), t_wb.as_slice());
        let r_vec = Vector3::new(r[0], r[1], r[2]);
        assert!(
            r_vec.dot(&direction).abs() < 1e-10,
            "residual not orthogonal to edge direction: dot = {}",
            r_vec.dot(&direction)
        );
    }

    #[test]
    fn edge_finite_difference_jacobians_axis_aligned() {
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

        let factor =
            LidarEdgeFactor::new_isotropic(point_b, Vector3::new(0.0, 0.0, 2.0), Vector3::z(), 0.5);

        let (r0, jac) = compute_with_jacobian(&factor, t_wa.as_slice(), t_wb.as_slice());

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let t_wa_p = perturb_se3(t_wa.as_slice(), &tan);
            let r_pert = compute_residual(&factor, t_wa_p.as_slice(), t_wb.as_slice());
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                crate::factors::common::test_utils::assert_close(
                    jac[(row, col)],
                    fd,
                    TOL,
                    &format!("J_T_WA[{row},{col}]"),
                );
            }
        }

        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let t_wb_p = perturb_se3(t_wb.as_slice(), &tan);
            let r_pert = compute_residual(&factor, t_wa.as_slice(), t_wb_p.as_slice());
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                crate::factors::common::test_utils::assert_close(
                    jac[(row, 6 + col)],
                    fd,
                    TOL,
                    &format!("J_T_WB[{row},{col}]"),
                );
            }
        }
    }

    #[test]
    fn edge_finite_difference_jacobians_arbitrary_orientation() {
        let q_a = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.3, 0.5, 0.1)),
            0.35,
        );
        let q_b = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.1, -0.2, 0.9)),
            -0.25,
        );
        let t_wa = make_pose(0.4, 0.9, -0.6, q_a);
        let t_wb = make_pose(-0.3, 1.2, 0.5, q_b);
        let point_b = Vector3::new(1.5, -0.7, 2.3);

        let edge_direction = Vector3::new(0.4, -0.6, 0.7);
        let edge_point = Vector3::new(0.2, 0.4, -0.1);

        let factor = LidarEdgeFactor::new_isotropic(point_b, edge_point, edge_direction, 0.3);

        let (r0, jac) = compute_with_jacobian(&factor, t_wa.as_slice(), t_wb.as_slice());

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-4;

        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let t_wa_p = perturb_se3(t_wa.as_slice(), &tan);
            let r_pert = compute_residual(&factor, t_wa_p.as_slice(), t_wb.as_slice());
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                crate::factors::common::test_utils::assert_close(
                    jac[(row, col)],
                    fd,
                    TOL,
                    &format!("J_T_WA[{row},{col}]"),
                );
            }
        }

        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let t_wb_p = perturb_se3(t_wb.as_slice(), &tan);
            let r_pert = compute_residual(&factor, t_wa.as_slice(), t_wb_p.as_slice());
            for row in 0..3 {
                let fd = (r_pert[row] - r0[row]) / EPS;
                crate::factors::common::test_utils::assert_close(
                    jac[(row, 6 + col)],
                    fd,
                    TOL,
                    &format!("J_T_WB[{row},{col}]"),
                );
            }
        }
    }

    #[test]
    fn edge_dimension_is_three() {
        let factor =
            LidarEdgeFactor::new_isotropic(Vector3::zeros(), Vector3::zeros(), Vector3::z(), 1.0);
        assert_eq!(factor.residual_dim(), 3);
        assert_eq!(factor.jacobian_shape(), (3, 12));
    }
}
