//! Inverse-depth landmark reprojection factor (GTSAM `InvDepthFactor3` analogue).
//!
//! The landmark is parametrized in the **anchor camera `i`** by its anchor
//! pixel `(u, v)` and inverse depth `d = 1/z_i`:
//!
//! ```text
//! X_i = unproject(u, v) / d          (camera-i frame, depth 1/d)
//! p_w = T_wc,i⁻¹ · X_i
//! p_cam,j = T_wc,j · p_w
//! r = project_j(p_cam,j) − uv_measured   (2D)
//! ```
//!
//! This parametrization keeps early, low-parallax landmarks well-conditioned
//! where a Euclidean 3D point blows up.
//!
//! Parameter blocks: `[pose_i (7), anchor (3), pose_j (7)]` — 15 minimal DOF,
//! 2D residual. The camera model (intrinsics) is fixed at construction.

use apex_camera_models::{CameraModel, NUMERICAL_DERIVATIVE_EPS};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::projection_factor::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::Factor;

/// Inverse-depth landmark reprojection factor over `[pose_i, anchor, pose_j]`.
#[derive(Clone)]
pub struct InverseDepthFactor<CAM: CameraModel> {
    /// Measured pixel in camera `j`.
    pub measurement: Vector2<f64>,
    /// Fixed camera model (shared intrinsics).
    pub camera: CAM,
    /// Log warnings for cheirality violations.
    pub verbose_cheirality: bool,
}

impl<CAM: CameraModel> InverseDepthFactor<CAM> {
    /// Create the factor from the measured pixel in camera `j` and the fixed
    /// camera model.
    pub fn new(measurement: Vector2<f64>, camera: CAM) -> Self {
        Self {
            measurement,
            camera,
            verbose_cheirality: false,
        }
    }

    /// Enable verbose cheirality warnings.
    pub fn with_verbose_cheirality(mut self) -> Self {
        self.verbose_cheirality = true;
        self
    }

    /// Backproject the anchor `(u, v, d)` into camera-i coordinates.
    ///
    /// Returns `None` when `d ≤ 0` (inverse depth must be positive — the
    /// anchor observation must have been in front of the anchor camera).
    fn anchor_point(&self, u: f64, v: f64, d: f64) -> Option<Vector3<f64>> {
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        let ray = self
            .camera
            .unproject(&Vector2::new(u, v))
            .ok()?;
        Some(ray / d)
    }

    /// ∂unproject/∂(u, v) by central differences — the `CameraModel` trait
    /// does not expose an analytic pixel-space backprojection Jacobian.
    fn unproject_jacobian(&self, u: f64, v: f64) -> SMatrix<f64, 3, 2> {
        let mut jac = SMatrix::<f64, 3, 2>::zeros();
        for (col, (du, dv)) in [(0usize, (NUMERICAL_DERIVATIVE_EPS, 0.0)), (1, (0.0, NUMERICAL_DERIVATIVE_EPS))] {
            let plus = self
                .camera
                .unproject(&Vector2::new(u + du, v + dv))
                .unwrap_or_else(|_| Vector3::zeros());
            let minus = self
                .camera
                .unproject(&Vector2::new(u - du, v - dv))
                .unwrap_or_else(|_| Vector3::zeros());
            jac.fixed_columns_mut::<1>(col)
                .copy_from(&((plus - minus) / (2.0 * NUMERICAL_DERIVATIVE_EPS)));
        }
        jac
    }
}

impl<CAM: CameraModel> Factor for InverseDepthFactor<CAM> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 3, "InverseDepthFactor expects 3 blocks");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be anchor (u,v,d)");
        debug_assert_eq!(params[2].len(), 7, "params[2] must be SE3 (7D)");

        let pose_i = SE3::from_param_slice(params[0]);
        let (u, v, d) = (params[1][0], params[1][1], params[1][2]);
        let pose_j = SE3::from_param_slice(params[2]);

        let Some(x_i) = self.anchor_point(u, v, d) else {
            // Non-positive inverse depth: the anchor itself was behind its
            // camera. Bounded constant penalty with no gradient — the same
            // rationale as the cheirality fallback, so an invalid anchor can
            // never be a cheap way to reduce cost.
            warn!("InverseDepthFactor: non-positive inverse depth d={d}");
            residual[0] = CHEIRALITY_BASE_PENALTY;
            residual[1] = CHEIRALITY_BASE_PENALTY;
            if let Some(mut jac) = jacobian {
                for r in 0..2 {
                    for c in 0..15 {
                        *jac.rb_mut().get_mut(r, c) = 0.0;
                    }
                }
            }
            return;
        };

        // p_w = T_i⁻¹ X_i ; p_cj = T_j p_w
        let p_w = pose_i
            .inverse(None)
            .act(&x_i, None, None);
        let p_cj = pose_j.act(&p_w, None, None);

        let uv_j = match self.camera.project(&p_cj) {
            Ok(uv) => uv,
            Err(apex_camera_models::CameraModelError::PointBehindCamera { z, min_z }) => {
                if self.verbose_cheirality {
                    warn!("InverseDepthFactor: point behind camera j (z={z})");
                }
                let depth_deficit = (min_z - z).max(0.0);
                let penalty = CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * depth_deficit;
                residual[0] = penalty;
                residual[1] = penalty;
                if let Some(mut jac) = jacobian {
                    // ∂penalty/∂z_cj = −CHEIRALITY_DEPTH_SCALE. z_cj depends
                    // on all three blocks; reuse the same chain as the valid
                    // path (third row of each point-Jacobian chain).
                    let d_pen = -CHEIRALITY_DEPTH_SCALE;
                    let r_j = pose_j.rotation_so3().rotation_matrix();
                    let r_i = pose_i.rotation_so3().rotation_matrix();
                    // pose_j: z-row of [R_j | −R_j p̂_w]
                    for c in 0..3 {
                        *jac.rb_mut().get_mut(0, 9 + c) = d_pen * r_j[(2, c)];
                        *jac.rb_mut().get_mut(1, 9 + c) = d_pen * r_j[(2, c)];
                    }
                    for c in 0..3 {
                        let pc = p_w;
                        let col = -r_j * Matrix3::new(
                            0.0, -pc.z, pc.y, pc.z, 0.0, -pc.x, -pc.y, pc.x, 0.0,
                        ).column(c);
                        *jac.rb_mut().get_mut(0, 12 + c) = d_pen * col[2];
                        *jac.rb_mut().get_mut(1, 12 + c) = d_pen * col[2];
                    }
                    // pose_i: z-row of [−R_j | R_j p̂_w]
                    for c in 0..3 {
                        *jac.rb_mut().get_mut(0, c) = d_pen * (-r_j[(2, c)]);
                        *jac.rb_mut().get_mut(1, c) = d_pen * (-r_j[(2, c)]);
                    }
                    for c in 0..3 {
                        let skew = Matrix3::new(
                            0.0, -p_w.z, p_w.y, p_w.z, 0.0, -p_w.x, -p_w.y, p_w.x, 0.0,
                        );
                        let col = r_j * skew.column(c);
                        *jac.rb_mut().get_mut(0, 3 + c) = d_pen * col[2];
                        *jac.rb_mut().get_mut(1, 3 + c) = d_pen * col[2];
                    }
                    // anchor: dz/du, dz/dv via unproject Jacobian; dz/dd = z/d
                    let d_unproj = self.unproject_jacobian(u, v);
                    let d_pw_d_xi = r_i.transpose();
                    for c in 0..2 {
                        let dxi = d_unproj.column(c) / d;
                        let dz = (r_j * d_pw_d_xi * dxi).z;
                        *jac.rb_mut().get_mut(0, 6 + c) = d_pen * dz;
                        *jac.rb_mut().get_mut(1, 6 + c) = d_pen * dz;
                    }
                    let dz_dd = (r_j * d_pw_d_xi * (-x_i / d)).z;
                    *jac.rb_mut().get_mut(0, 8) = d_pen * dz_dd;
                    *jac.rb_mut().get_mut(1, 8) = d_pen * dz_dd;
                }
                return;
            }
            Err(cam_err) => {
                if self.verbose_cheirality {
                    warn!("InverseDepthFactor: invalid projection: {cam_err}");
                }
                residual[0] = CHEIRALITY_BASE_PENALTY;
                residual[1] = CHEIRALITY_BASE_PENALTY;
                return;
            }
        };

        residual[0] = uv_j.x - self.measurement.x;
        residual[1] = uv_j.y - self.measurement.y;

        let Some(mut jac) = jacobian else {
            return;
        };

        let cam_jac = self.camera.jacobian_point(&p_cj); // 2×3
        let r_i = pose_i.rotation_so3().rotation_matrix();
        let r_j = pose_j.rotation_so3().rotation_matrix();

        // pose_j block: J_cam · [R_j | −R_j p̂_w]  (2×6)
        let skew_w = Matrix3::new(
            0.0, -p_w.z, p_w.y, p_w.z, 0.0, -p_w.x, -p_w.y, p_w.x, 0.0,
        );
        let mut d_pcj_d_pose_j = SMatrix::<f64, 3, 6>::zeros();
        d_pcj_d_pose_j.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_j);
        d_pcj_d_pose_j
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-r_j * skew_w));
        let j_pose_j = cam_jac.clone() * d_pcj_d_pose_j;

        // pose_i block: J_cam · [−R_j | R_j p̂_w]  (2×6)
        let mut d_pcj_d_pose_i = SMatrix::<f64, 3, 6>::zeros();
        d_pcj_d_pose_i
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-r_j));
        d_pcj_d_pose_i
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(r_j * skew_w));
        let j_pose_i = cam_jac.clone() * d_pcj_d_pose_i;

        // anchor block: J_cam · R_j · R_iᵀ · [∂X/∂u, ∂X/∂v, ∂X/∂d]  (2×3)
        let d_pw_d_xi = r_i.transpose();
        let d_unproj = self.unproject_jacobian(u, v);
        let mut d_xi_d_anchor = SMatrix::<f64, 3, 3>::zeros();
        d_xi_d_anchor
            .fixed_columns_mut::<2>(0)
            .copy_from(&(d_unproj / d));
        d_xi_d_anchor
            .fixed_columns_mut::<1>(2)
            .copy_from(&(-x_i / d));
        let j_anchor = cam_jac * r_j * d_pw_d_xi * d_xi_d_anchor;

        for r in 0..2 {
            for c in 0..6 {
                *jac.rb_mut().get_mut(r, c) = j_pose_i[(r, c)];
            }
            for c in 0..3 {
                *jac.rb_mut().get_mut(r, 6 + c) = j_anchor[(r, c)];
            }
            for c in 0..6 {
                *jac.rb_mut().get_mut(r, 9 + c) = j_pose_j[(r, c)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 15)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 3 {
            return Err(format!(
                "InverseDepthFactor expects 3 variables, got {}",
                variables.len()
            ));
        }
        for (idx, expected) in [(0usize, SE3::REP_SIZE), (1, 3), (2, SE3::REP_SIZE)] {
            if variables[idx].as_param_slice().len() != expected {
                return Err(format!(
                    "InverseDepthFactor block {idx} requires {expected} parameters"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_camera_models::PinholeCamera;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    fn camera() -> PinholeCamera {
        PinholeCamera::from([500.0, 500.0, 320.0, 240.0])
    }

    fn truth_setup() -> TestResult<(InverseDepthFactor<PinholeCamera>, SE3, [f64; 3], SE3)> {
        let cam = camera();
        let pose_i = SE3::identity();
        let pose_j = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(-0.5, 0.0, 0.0),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.05, 0.0),
        ));

        // Anchor pixel + true inverse depth.
        let anchor_uv = Vector2::new(360.0, 200.0);
        let ray = cam.unproject(&anchor_uv)?;
        let depth = 4.0;
        let x_i = ray * depth;
        let p_w = pose_i.inverse(None).act(&x_i, None, None);
        let p_cj = pose_j.act(&p_w, None, None);
        let uv_j = cam.project(&p_cj)?;

        let factor = InverseDepthFactor::new(uv_j, cam);
        let anchor = [anchor_uv.x, anchor_uv.y, 1.0 / depth];
        Ok((factor, pose_i, anchor, pose_j))
    }

    #[test]
    fn zero_residual_at_truth() -> TestResult<()> {
        let (factor, pose_i, anchor, pose_j) = truth_setup()?;
        let mut residual = vec![0.0; 2];
        factor.linearize(
            &[
                pose_i.as_param_slice(),
                &anchor,
                pose_j.as_param_slice(),
            ],
            &mut residual,
            None,
        );
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-9, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn finite_difference_jacobians() -> TestResult<()> {
        let (factor, pose_i, anchor, pose_j) = truth_setup()?;
        let pi_vec: Vec<f64> = pose_i.as_param_slice().to_vec();
        let pj_vec: Vec<f64> = pose_j.as_param_slice().to_vec();

        let (rows, cols) = factor.jacobian_shape();
        let mut r0 = vec![0.0; rows];
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[&pi_vec, &anchor, &pj_vec], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-3;

        let blocks: [(usize, usize); 3] = [(0, 6), (1, 3), (2, 6)];
        for &(block, len) in &blocks {
            for col in 0..len {
                let mut r_pert = vec![0.0; rows];
                match block {
                    0 => {
                        let mut tan = [0.0f64; 6];
                        tan[col] = EPS;
                        let perturbed: Vec<f64> = pose_i
                            .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                            .as_param_slice()
                            .to_vec();
                        factor.linearize(&[&perturbed, &anchor, &pj_vec], &mut r_pert, None);
                    }
                    1 => {
                        let mut a = anchor;
                        a[col] += EPS;
                        factor.linearize(&[&pi_vec, &a, &pj_vec], &mut r_pert, None);
                    }
                    _ => {
                        let mut tan = [0.0f64; 6];
                        tan[col] = EPS;
                        let perturbed: Vec<f64> = pose_j
                            .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                            .as_param_slice()
                            .to_vec();
                        factor.linearize(&[&pi_vec, &anchor, &perturbed], &mut r_pert, None);
                    }
                }
                for row in 0..rows {
                    let fd = (r_pert[row] - r0[row]) / EPS;
                    let col_off = match block {
                        0 => 0,
                        1 => 6,
                        _ => 9,
                    };
                    let ana = jac_buf[(col_off + col) * rows + row];
                    assert!(
                        (fd - ana).abs() < TOL,
                        "block {block}[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn non_positive_inverse_depth_is_penalized() -> TestResult<()> {
        let (factor, pose_i, mut anchor, pose_j) = truth_setup()?;
        anchor[2] = -0.25;
        let mut residual = vec![0.0; 2];
        factor.linearize(
            &[pose_i.as_param_slice(), &anchor, pose_j.as_param_slice()],
            &mut residual,
            None,
        );
        assert!(residual[0] >= CHEIRALITY_BASE_PENALTY);
        assert!(residual[1] >= CHEIRALITY_BASE_PENALTY);
        Ok(())
    }

    #[test]
    fn large_depth_landmark_stays_well_conditioned() -> TestResult<()> {
        // The whole point of the inverse-depth parametrization: a landmark
        // 100 m away (z=100, d=0.01) still yields a finite residual and
        // finite Jacobian.
        let cam = camera();
        let anchor_uv = Vector2::new(340.0, 250.0);
        let ray = cam.unproject(&anchor_uv)?;
        let depth = 100.0;
        let x_i = ray * depth;
        let pose_i = SE3::identity();
        let pose_j = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(-1.0, 0.1, 0.0),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.01, 0.0),
        ));
        let p_w = pose_i.inverse(None).act(&x_i, None, None);
        let p_cj = pose_j.act(&p_w, None, None);
        let uv_j = cam.project(&p_cj)?;
        let factor = InverseDepthFactor::new(uv_j, cam);
        let anchor = [anchor_uv.x, anchor_uv.y, 1.0 / depth];

        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0; rows];
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(
            &[pose_i.as_param_slice(), &anchor, pose_j.as_param_slice()],
            &mut residual,
            Some(jac_mut),
        );
        assert!(residual.iter().all(|r| r.abs() < 1e-9));
        assert!(jac_buf.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
