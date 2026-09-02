//! Structure-less smart projection factor (GTSAM `SmartProjectionPose3Factor`
//! analogue).
//!
//! Connects N **pose** variables to a single unobserved 3D landmark that is
//! re-triangulated from the current pose estimates at every linearization
//! (DLT on the unprojected bearing rays). The landmark is eliminated exactly
//! at the linearization point via the implicit Schur complement.
//!
//! Writing `A_i = ∂uv_i/∂pose_i / σ_i` (2×6) and `B_i = ∂uv_i/∂point / σ_i`
//! (2×3), the output Jacobian is the **dense implicit** Jacobian
//!
//! ```text
//! J(i,j) = δ_ij·A_i − B_i·M_j,     M_j = (Σ_k B_kᵀB_k)⁻¹ B_jᵀA_j  (3×6)
//! ```
//!
//! Because the triangulated point `p*(T)` satisfies the first-order
//! optimality condition `Σ B_iᵀ r_i = 0`, this dense `J` is exactly the
//! total derivative of the re-triangulated residual — and `JᵀJ` equals the
//! point-marginalized (Schur) Hessian `H_pp − H_pl H_ll⁻¹ H_lp` of the
//! explicit `[poses, point]` system. No landmark variable ever enters the
//! graph.
//!
//! # Noise handling
//!
//! The elimination must happen in the *whitened* system, so the factor
//! whitens internally with per-observation `sigma` (default 1.0) and must be
//! registered with `NoiseModel::null()`. Whitening again externally would
//! break the Schur structure for non-uniform noise.
//!
//! # Degeneracy handling
//!
//! - Triangulation rank failure (e.g. pure-rotation two-view): constant
//!   bounded penalty with zero Jacobian — never a free cost reduction.
//! - Cheirality violation of the triangulated point: the same smooth penalty
//!   as the monocular [`ProjectionFactor`], with a real gradient pushing the
//!   poses back toward validity.
//!
//! [`ProjectionFactor`]: crate::factors::projection_factor::ProjectionFactor

use apex_camera_models::{CameraModel, CameraModelError};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, Matrix3, SMatrix, SVector, Vector2, Vector3};
use std::sync::atomic::{AtomicU8, Ordering};
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::projection_factor::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::Factor;

/// Failure mode of the internal triangulation, exposed for graph-building
/// code to decide on outlier rejection / keyframing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangulationStatus {
    /// A well-conditioned point was triangulated in front of every camera.
    Ok,
    /// The DLT system was rank-deficient (degenerate configuration).
    RankDeficient,
    /// The triangulated point ended up behind at least one camera.
    CheiralityViolation,
    /// A projection was invalid for a non-cheirality reason.
    InvalidProjection,
}

/// Structure-less multi-view reprojection factor over N poses.
pub struct SmartProjectionFactor<CAM: CameraModel> {
    /// Pixel observations, one per connected pose.
    pub observations: Vec<Vector2<f64>>,
    /// Shared camera model (fixed intrinsics).
    pub camera: CAM,
    /// Per-observation sigmas used for the internal whitening (default 1.0).
    pub sigmas: Vec<f64>,
    /// Log warnings for degeneracy handling.
    pub verbose: bool,
    /// Outcome of the most recent internal triangulation (0 = Ok,
    /// 1 = RankDeficient, 2 = CheiralityViolation, 3 = InvalidProjection).
    /// Stored atomically because `Factor::linearize` takes `&self` from
    /// parallel residual-evaluation threads.
    status: AtomicU8,
}

impl<CAM: CameraModel> SmartProjectionFactor<CAM> {
    /// Create the factor from pixel observations and a shared camera model.
    pub fn new(observations: Vec<Vector2<f64>>, camera: CAM) -> Result<Self, String> {
        let n = observations.len();
        if n == 0 {
            return Err("SmartProjectionFactor requires at least one observation".into());
        }
        Ok(Self {
            observations,
            camera,
            sigmas: vec![1.0; n],
            verbose: false,
            status: AtomicU8::new(0),
        })
    }

    /// Set per-observation sigmas for the internal whitening.
    pub fn with_sigmas(mut self, sigmas: Vec<f64>) -> Result<Self, String> {
        if sigmas.len() != self.observations.len()
            || sigmas.iter().any(|s| !(s.is_finite() && *s > 0.0))
        {
            return Err(
                "sigmas must be positive, finite, and match the observation count".into(),
            );
        }
        self.sigmas = sigmas;
        Ok(self)
    }

    /// Enable verbose degeneracy warnings.
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Number of connected poses / observations.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Outcome of the most recent internal triangulation.
    pub fn status(&self) -> TriangulationStatus {
        match self.status.load(Ordering::Relaxed) {
            1 => TriangulationStatus::RankDeficient,
            2 => TriangulationStatus::CheiralityViolation,
            3 => TriangulationStatus::InvalidProjection,
            _ => TriangulationStatus::Ok,
        }
    }

    fn set_status(&self, status: TriangulationStatus) {
        let code = match status {
            TriangulationStatus::Ok => 0u8,
            TriangulationStatus::RankDeficient => 1,
            TriangulationStatus::CheiralityViolation => 2,
            TriangulationStatus::InvalidProjection => 3,
        };
        self.status.store(code, Ordering::Relaxed);
    }

    /// Triangulate the landmark from the current pose estimates by DLT on the
    /// unprojected bearing rays. Returns `None` when the system is
    /// rank-deficient or the point lands at infinity.
    fn triangulate(&self, poses: &[SE3]) -> Option<Vector3<f64>> {
        let n = poses.len();
        let mut a = DMatrix::<f64>::zeros(3 * n, 4);
        for (i, pose) in poses.iter().enumerate() {
            let ray = self.camera.unproject(&self.observations[i]).ok()?;
            let r = pose.rotation_so3().rotation_matrix();
            let t = pose.translation();
            let d_x = Matrix3::new(
                0.0, -ray.z, ray.y, ray.z, 0.0, -ray.x, -ray.y, ray.x, 0.0,
            );
            let block = d_x * r;
            let rhs = d_x * t;
            for row in 0..3 {
                for col in 0..4 {
                    a[(3 * i + row, col)] = if col < 3 { block[(row, col)] } else { rhs[row] };
                }
            }
        }

        let svd = a.svd(true, true);
        let v_t = svd.v_t?;
        let s = svd.singular_values;
        // For consistent measurements the *smallest* singular value is ~0 by
        // construction (its singular vector is the solution). The system is
        // rank-deficient (degenerate geometry) when the *second-smallest*
        // value also collapses — the null space becomes 2-dimensional.
        let s_max = s[0];
        let s_second_min = s[2];
        if !(s_second_min.is_finite() && s_max > 0.0 && s_second_min > 1e-8 * s_max) {
            return None;
        }

        let mut x = SMatrix::<f64, 4, 1>::from_iterator((0..4).map(|row| v_t[(3, row)]));
        if x[3] < 0.0 {
            x = -x;
        }
        if x[3].abs() < 1e-12 {
            return None; // point at infinity
        }
        let mut point = Vector3::new(x[0] / x[3], x[1] / x[3], x[2] / x[3]);

        // Gauss-Newton refinement: the DLT point minimizes *algebraic* error,
        // but the implicit-Schur Jacobian assumes the point satisfies the
        // geometric first-order condition Σ B_iᵀ r_i = 0. A few refinement
        // steps close that gap.
        for _ in 0..5 {
            let mut hll = SMatrix::<f64, 3, 3>::zeros();
            let mut g = SVector::<f64, 3>::zeros();
            let mut converged = true;
            for (i, pose) in poses.iter().enumerate() {
                let p_cam = pose.act(&point, None, None);
                let Ok(uv) = self.camera.project(&p_cam) else {
                    converged = false;
                    break;
                };
                let r = uv - self.observations[i];
                let d_uv_d_pc = self.camera.jacobian_point(&p_cam);
                let rot = pose.rotation_so3().rotation_matrix();
                let b = d_uv_d_pc * rot;
                hll += b.transpose() * b;
                g += b.transpose() * r;
            }
            if !converged {
                break;
            }
            let Some(hll_inv) = hll.try_inverse() else {
                break;
            };
            let delta = -(hll_inv * g);
            point += delta;
            if delta.norm() < 1e-12 {
                break;
            }
        }
        Some(point)
    }

    /// Whitened per-view blocks: `(r_i, A_i, B_i)` with
    /// `A_i = ∂uv_i/∂pose_i / σ_i` and `B_i = ∂uv_i/∂point / σ_i`.
    fn view_blocks(
        &self,
        i: usize,
        pose: &SE3,
        point: &Vector3<f64>,
        p_cam: &Vector3<f64>,
    ) -> (SVector<f64, 2>, SMatrix<f64, 2, 6>, SMatrix<f64, 2, 3>) {
        let inv_sigma = 1.0 / self.sigmas[i];
        let uv = self
            .camera
            .project(p_cam)
            .unwrap_or_else(|_| Vector2::zeros());
        let r = (uv - self.observations[i]) * inv_sigma;
        let d_uv_d_pc = self.camera.jacobian_point(p_cam);
        let rot = pose.rotation_so3().rotation_matrix();
        let p_x = Matrix3::new(
            0.0, -point.z, point.y, point.z, 0.0, -point.x, -point.y, point.x, 0.0,
        );
        let mut d_pc_d_pose = SMatrix::<f64, 3, 6>::zeros();
        d_pc_d_pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
        d_pc_d_pose
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-rot * p_x));
        let a = (d_uv_d_pc.clone() * d_pc_d_pose) * inv_sigma;
        let b = (d_uv_d_pc * rot) * inv_sigma;
        (r, a, b)
    }

    /// Write the smooth cheirality penalty for every observation, with the
    /// gradient flowing into each pose through `∂z_i/∂pose_i`.
    fn write_cheirality_penalty(
        &self,
        poses: &[SE3],
        point: &Vector3<f64>,
        depth_deficit: f64,
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let penalty = CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * depth_deficit;
        for i in 0..self.observations.len() {
            residual[2 * i] = penalty;
            residual[2 * i + 1] = penalty;
        }
        let Some(mut jac) = jacobian else { return };
        let d_pen_d_z = -CHEIRALITY_DEPTH_SCALE;
        for (i, pose) in poses.iter().enumerate() {
            let rot = pose.rotation_so3().rotation_matrix();
            let p_x = Matrix3::new(
                0.0, -point.z, point.y, point.z, 0.0, -point.x, -point.y, point.x, 0.0,
            );
            for c in 0..3 {
                let d_t = d_pen_d_z * rot[(2, c)];
                let d_r = d_pen_d_z * (-rot * p_x).row(2)[c];
                *jac.rb_mut().get_mut(2 * i, 6 * i + c) = d_t;
                *jac.rb_mut().get_mut(2 * i + 1, 6 * i + c) = d_t;
                *jac.rb_mut().get_mut(2 * i, 6 * i + 3 + c) = d_r;
                *jac.rb_mut().get_mut(2 * i + 1, 6 * i + 3 + c) = d_r;
            }
        }
    }

    /// Write a constant bounded penalty (zero Jacobian) — used when no
    /// principled gradient exists (rank-deficient triangulation, invalid
    /// projection).
    fn write_constant_penalty(
        &self,
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        for i in 0..self.observations.len() {
            residual[2 * i] = CHEIRALITY_BASE_PENALTY;
            residual[2 * i + 1] = CHEIRALITY_BASE_PENALTY;
        }
        if let Some(mut jac) = jacobian {
            let rows = 2 * self.observations.len();
            let cols = 6 * self.observations.len();
            for r in 0..rows {
                for c in 0..cols {
                    *jac.rb_mut().get_mut(r, c) = 0.0;
                }
            }
        }
    }
}

impl<CAM: CameraModel> Clone for SmartProjectionFactor<CAM> {
    fn clone(&self) -> Self {
        Self {
            observations: self.observations.clone(),
            camera: self.camera.clone(),
            sigmas: self.sigmas.clone(),
            verbose: self.verbose,
            status: AtomicU8::new(self.status.load(Ordering::Relaxed)),
        }
    }
}

impl<CAM: CameraModel> Factor for SmartProjectionFactor<CAM> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let n = self.observations.len();
        debug_assert_eq!(
            params.len(),
            n,
            "SmartProjectionFactor expects one pose per observation"
        );
        for (i, p) in params.iter().enumerate() {
            debug_assert_eq!(p.len(), 7, "params[{i}] must be SE3 (7D)");
        }

        let poses: Vec<SE3> = params.iter().map(|p| SE3::from_param_slice(p)).collect();

        let Some(point) = self.triangulate(&poses) else {
            if self.verbose {
                warn!("SmartProjectionFactor: rank-deficient triangulation (degenerate configuration)");
            }
            self.set_status(TriangulationStatus::RankDeficient);
            self.write_constant_penalty(residual, jacobian);
            return;
        };

        // Cheirality check on the triangulated point.
        let min_z = poses
            .iter()
            .map(|pose| pose.act(&point, None, None).z)
            .fold(f64::INFINITY, f64::min);
        if min_z <= apex_camera_models::MIN_DEPTH {
            if self.verbose {
                warn!("SmartProjectionFactor: cheirality violation (min z = {min_z})");
            }
            self.set_status(TriangulationStatus::CheiralityViolation);
            let depth_deficit = (apex_camera_models::MIN_DEPTH - min_z).max(0.0);
            self.write_cheirality_penalty(&poses, &point, depth_deficit, residual, jacobian);
            return;
        }

        // Whitened per-view residuals and Jacobian blocks.
        let mut r_w: Vec<SVector<f64, 2>> = Vec::with_capacity(n);
        let mut a_list: Vec<SMatrix<f64, 2, 6>> = Vec::with_capacity(n);
        let mut b_list: Vec<SMatrix<f64, 2, 3>> = Vec::with_capacity(n);
        for (i, pose) in poses.iter().enumerate() {
            let p_cam = pose.act(&point, None, None);
            match self.camera.project(&p_cam) {
                Ok(uv) => {
                    let inv_sigma = 1.0 / self.sigmas[i];
                    r_w.push((uv - self.observations[i]) * inv_sigma);
                    let (_, a, b) = self.view_blocks(i, pose, &point, &p_cam);
                    a_list.push(a);
                    b_list.push(b);
                }
                Err(CameraModelError::PointBehindCamera { z, min_z }) => {
                    if self.verbose {
                        warn!("SmartProjectionFactor: view {i} behind camera (z={z}, min_z={min_z})");
                    }
                    self.set_status(TriangulationStatus::CheiralityViolation);
                    let depth_deficit = (min_z - z).max(0.0);
                    self.write_cheirality_penalty(&poses, &point, depth_deficit, residual, jacobian);
                    return;
                }
                Err(cam_err) => {
                    if self.verbose {
                        warn!("SmartProjectionFactor: invalid projection in view {i}: {cam_err}");
                    }
                    self.set_status(TriangulationStatus::InvalidProjection);
                    self.write_constant_penalty(residual, jacobian);
                    return;
                }
            }
        }
        self.set_status(TriangulationStatus::Ok);

        for i in 0..n {
            residual[2 * i] = r_w[i][0];
            residual[2 * i + 1] = r_w[i][1];
        }

        // Implicit Schur: M_j = Hll⁻¹ B_jᵀ A_j, J(i,j) = δ_ij A_i − B_i M_j.
        let mut hll = SMatrix::<f64, 3, 3>::zeros();
        for b in &b_list {
            hll += b.transpose() * b;
        }
        let Some(hll_inv) = hll.try_inverse() else {
            if self.verbose {
                warn!("SmartProjectionFactor: singular point Hessian — treating as degenerate");
            }
            self.set_status(TriangulationStatus::RankDeficient);
            self.write_constant_penalty(residual, jacobian);
            return;
        };
        let Some(mut jac) = jacobian else { return };
        let m_blocks: Vec<SMatrix<f64, 3, 6>> = (0..n)
            .map(|j| hll_inv * (b_list[j].transpose() * a_list[j]))
            .collect();

        for (j, m_j) in m_blocks.iter().enumerate() {
            for (i, b_i) in b_list.iter().enumerate() {
                let block = if i == j {
                    a_list[i] - b_i * m_j
                } else {
                    -b_i * m_j
                };
                for r in 0..2 {
                    for c in 0..6 {
                        *jac.rb_mut().get_mut(2 * i + r, 6 * j + c) = block[(r, c)];
                    }
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2 * self.observations.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2 * self.observations.len(), 6 * self.observations.len())
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != self.observations.len() {
            return Err(format!(
                "SmartProjectionFactor expects {} pose variables, got {}",
                self.observations.len(),
                variables.len()
            ));
        }
        for (i, v) in variables.iter().enumerate() {
            if v.as_param_slice().len() != SE3::REP_SIZE {
                return Err(format!(
                    "SmartProjectionFactor variable {i} must be SE3 (7 parameters)"
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

    /// Three-view setup looking at a point 4 m ahead.
    fn three_view_setup()
    -> TestResult<(SmartProjectionFactor<PinholeCamera>, Vec<SE3>, Vector3<f64>)> {
        let cam = camera();
        let point = Vector3::new(0.2, -0.1, 4.0);
        let poses: Vec<SE3> = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.5, 0.0, 0.0], [0.0, 0.02, 0.0]),
            ([-0.4, 0.1, 0.0], [0.0, -0.01, 0.01]),
        ]
        .into_iter()
        .map(|(t, r)| {
            SE3::from_isometry(nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(t[0], t[1], t[2]),
                nalgebra::UnitQuaternion::from_euler_angles(r[0], r[1], r[2]),
            ))
        })
        .collect();

        let obs: Vec<Vector2<f64>> = poses
            .iter()
            .map(|pose| cam.project(&pose.act(&point, None, None)))
            .collect::<Result<_, _>>()?;

        let factor = SmartProjectionFactor::new(obs, cam)?;
        Ok((factor, poses, point))
    }

    #[test]
    fn zero_residual_and_ok_status_at_truth() -> TestResult<()> {
        let (factor, poses, _) = three_view_setup()?;
        let params: Vec<Vec<f64>> = poses.iter().map(|p| p.as_param_slice().to_vec()).collect();
        let slices: Vec<&[f64]> = params.iter().map(|p| p.as_slice()).collect();
        let mut residual = vec![0.0; factor.residual_dim()];
        factor.linearize(&slices, &mut residual, None);
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-9, "residual[{i}] = {r}");
        }
        assert_eq!(factor.status(), TriangulationStatus::Ok);
        Ok(())
    }

    #[test]
    fn triangulation_recovers_point() -> TestResult<()> {
        let (factor, poses, point) = three_view_setup()?;
        let recovered = factor
            .triangulate(&poses)
            .ok_or("triangulation should succeed")?;
        assert!(
            (recovered - point).norm() < 1e-6,
            "recovered {recovered}, truth {point}"
        );
        Ok(())
    }

    #[test]
    fn implicit_schur_matches_explicit_point_elimination() -> TestResult<()> {
        // J_impᵀ J_imp must equal the Schur complement of the explicit
        // [poses, point] system: H_pp − H_pl H_ll⁻¹ H_lpᵀ.
        let (factor, poses, _) = three_view_setup()?;
        let params: Vec<Vec<f64>> = poses.iter().map(|p| p.as_param_slice().to_vec()).collect();
        let slices: Vec<&[f64]> = params.iter().map(|p| p.as_slice()).collect();

        let n = poses.len();
        let point = factor
            .triangulate(&poses)
            .ok_or("triangulation should succeed")?;

        // Explicit whitened (σ=1) blocks, recomputed independently.
        let mut h_pp = DMatrix::<f64>::zeros(6 * n, 6 * n);
        let mut h_pl = DMatrix::<f64>::zeros(6 * n, 3);
        for (i, pose) in poses.iter().enumerate() {
            let p_cam = pose.act(&point, None, None);
            let d_uv_d_pc = factor.camera.jacobian_point(&p_cam);
            let rot = pose.rotation_so3().rotation_matrix();
            let p_x = Matrix3::new(
                0.0, -point.z, point.y, point.z, 0.0, -point.x, -point.y, point.x, 0.0,
            );
            let mut d_pc_d_pose = SMatrix::<f64, 3, 6>::zeros();
            d_pc_d_pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
            d_pc_d_pose.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-rot * p_x));
            let a = d_uv_d_pc * d_pc_d_pose;
            let b = d_uv_d_pc * rot;
            for r_i in 0..6 {
                for c_i in 0..6 {
                    h_pp[(6 * i + r_i, 6 * i + c_i)] += (a.transpose() * a)[(r_i, c_i)];
                }
                for c_i in 0..3 {
                    h_pl[(6 * i + r_i, c_i)] += (a.transpose() * b)[(r_i, c_i)];
                }
            }
        }
        let mut h_ll = SMatrix::<f64, 3, 3>::zeros();
        for (i, pose) in poses.iter().enumerate() {
            let p_cam = pose.act(&point, None, None);
            let d_uv_d_pc = factor.camera.jacobian_point(&p_cam);
            let rot = pose.rotation_so3().rotation_matrix();
            let b = d_uv_d_pc * rot;
            h_ll += b.transpose() * b;
        }
        let h_ll_inv = h_ll.try_inverse().ok_or("Hll singular")?;
        let schur = &h_pp - &h_pl * DMatrix::from_fn(3, 3, |r, c| h_ll_inv[(r, c)]) * h_pl.transpose();

        // Factor output.
        let (rows, cols) = factor.jacobian_shape();
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        let mut residual = vec![0.0; rows];
        factor.linearize(&slices, &mut residual, Some(jac_mut));
        let jac = DMatrix::from_column_slice(rows, cols, &jac_buf);
        let marginalized = jac.transpose() * &jac;

        assert!(
            (marginalized - &schur).norm() < 1e-8 * schur.norm().max(1.0),
            "J_impᵀJ_imp does not match the explicit Schur complement"
        );
        Ok(())
    }

    #[test]
    fn pure_rotation_two_view_is_rank_deficient() -> TestResult<()> {
        // Identical poses (no parallax): the point cannot be triangulated.
        let cam = camera();
        let point = Vector3::new(0.2, -0.1, 4.0);
        let pose = SE3::identity();
        let obs = cam.project(&pose.act(&point, None, None))?;
        let factor = SmartProjectionFactor::new(vec![obs, obs], cam)?;
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();
        let mut residual = vec![0.0; factor.residual_dim()];
        factor.linearize(&[&pose_vec, &pose_vec], &mut residual, None);
        assert_eq!(factor.status(), TriangulationStatus::RankDeficient);
        for r in residual {
            assert!(r >= CHEIRALITY_BASE_PENALTY, "expected constant penalty");
        }
        Ok(())
    }

    #[test]
    fn cheirality_violation_is_penalized() -> TestResult<()> {
        // pose_b sits 3 m past the point, so the triangulated point lands
        // behind its camera.
        let cam = camera();
        let point = Vector3::new(0.2, -0.1, 1.0);
        let pose_a = SE3::identity();
        let pose_b = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, -3.0),
            nalgebra::UnitQuaternion::identity(),
        ));
        let obs_a = cam.project(&pose_a.act(&point, None, None))?;
        let obs_b = Vector2::new(320.0, 240.0); // principal-point ray
        let factor = SmartProjectionFactor::new(vec![obs_a, obs_b], cam)?;
        let a_vec: Vec<f64> = pose_a.as_param_slice().to_vec();
        let b_vec: Vec<f64> = pose_b.as_param_slice().to_vec();
        let mut residual = vec![0.0; factor.residual_dim()];
        factor.linearize(&[&a_vec, &b_vec], &mut residual, None);
        assert_eq!(factor.status(), TriangulationStatus::CheiralityViolation);
        assert!(residual.iter().all(|r| r.is_finite() && *r >= CHEIRALITY_BASE_PENALTY));
        Ok(())
    }

    #[test]
    fn fd_jacobians_match_retriangulated_residual() -> TestResult<()> {
        // The dense implicit Jacobian is the total derivative of the residual
        // under re-triangulation — verify against finite differences.
        let (factor, poses, _) = three_view_setup()?;
        let params: Vec<Vec<f64>> = poses.iter().map(|p| p.as_param_slice().to_vec()).collect();
        let slices: Vec<&[f64]> = params.iter().map(|p| p.as_slice()).collect();

        let (rows, cols) = factor.jacobian_shape();
        let mut r0 = vec![0.0; rows];
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&slices, &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-3;
        for (view, pose) in poses.iter().enumerate() {
            for col in 0..6 {
                let mut tan = [0.0f64; 6];
                tan[col] = EPS;
                let perturbed: Vec<f64> = pose
                    .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                    .as_param_slice()
                    .to_vec();
                let mut perturbed_params = params.clone();
                perturbed_params[view] = perturbed;
                let pert_slices: Vec<&[f64]> =
                    perturbed_params.iter().map(|p| p.as_slice()).collect();
                let mut r_pert = vec![0.0; rows];
                factor.linearize(&pert_slices, &mut r_pert, None);
                for row in 0..rows {
                    let fd = (r_pert[row] - r0[row]) / EPS;
                    let ana = jac_buf[(6 * view + col) * rows + row];
                    assert!(
                        (fd - ana).abs() < TOL,
                        "view {view} J[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn non_uniform_sigmas_whiten_internally() -> TestResult<()> {
        // Scaling a sigma by k must scale the corresponding residual rows
        // by 1/k.
        let (factor, poses, _) = three_view_setup()?;
        let params: Vec<Vec<f64>> = poses.iter().map(|p| p.as_param_slice().to_vec()).collect();
        let slices: Vec<&[f64]> = params.iter().map(|p| p.as_slice()).collect();

        let rows = factor.residual_dim();
        let mut r1 = vec![0.0; rows];
        factor.linearize(&slices, &mut r1, None);

        let weighted = factor.clone().with_sigmas(vec![1.0, 1.0, 2.0])?;
        let mut r2 = vec![0.0; rows];
        weighted.linearize(&slices, &mut r2, None);

        for row in 0..rows {
            let expected_scale = if row / 2 == 2 { 0.5 } else { 1.0 };
            assert!(
                (r2[row] - expected_scale * r1[row]).abs() < 1e-9,
                "row {row}: {} vs {}",
                r2[row],
                r1[row]
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_empty_observations() {
        assert!(SmartProjectionFactor::new(Vec::new(), camera()).is_err());
        assert!(SmartProjectionFactor::new(vec![Vector2::zeros()], camera())
            .unwrap_or_else(|e| panic!("{e}"))
            .with_sigmas(vec![1.0, 1.0])
            .is_err());
    }
}
