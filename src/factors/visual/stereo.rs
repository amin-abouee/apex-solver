//! Rectified stereo reprojection factor (GTSAM `GenericStereoFactor3D` analogue).
//!
//! Projects a 3D landmark through a rectified stereo pair and compares
//! against a `StereoPoint2` measurement `(uL, uR, v)`:
//!
//! ```text
//! uL = cx + fx·x/z     uR = cx + fx·(x−b)/z     v = cy + fy·y/z
//! ```
//!
//! Parameter blocks: `[pose X ∈ G, landmark (3)]` — `G::DOF + 3` minimal DOF, 3D
//! residual `[uL, uR, v]`. `G` is any group that moves points ([`ActsOnPoints3`]:
//! `SE3` by default, `SE23`, `Sim3`, …); the pose is world-to-camera. Calibration is fixed at construction. Cheirality violations
//! (`z ≤ 0`) are handled with the same smooth penalty as the monocular
//! [`ProjectionFactor`](crate::factors::visual::projection::ProjectionFactor).

use apex_camera_models::{ProjectionError, ProjectionResult};
use std::marker::PhantomData;

use apex_manifolds::capability::ActsOnPoints3;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::common::cheirality::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::common::group_point::{MAX_ACT_DOF, act_point, expect_state_and_point};
use crate::factors::{Factor, FactorResult};

/// Rectified stereo calibration: focal lengths, principal point, baseline.
#[derive(Clone, Debug)]
pub struct StereoCalibration {
    /// Horizontal focal length [px].
    pub fx: f64,
    /// Vertical focal length [px].
    pub fy: f64,
    /// Principal point [px].
    pub cx: f64,
    /// Principal point y [px].
    pub cy: f64,
    /// Stereo baseline [m] (left-to-right camera translation).
    pub baseline: f64,
}

impl StereoCalibration {
    /// Create and validate a rectified stereo calibration.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, baseline: f64) -> FactorResult<Self> {
        use crate::factors::common::validate::{expect_finite, expect_positive};
        const OWNER: &str = "StereoCalibration";
        expect_positive(OWNER, "fx", fx)?;
        expect_positive(OWNER, "fy", fy)?;
        expect_finite(OWNER, "principal point", [cx, cy].iter())?;
        expect_positive(OWNER, "baseline", baseline)?;
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            baseline,
        })
    }

    /// Project a camera-frame point to `(uL, uR, v)`; errors behind the
    /// cheirality boundary surface as [`ProjectionError::PointBehindCamera`].
    pub fn project(&self, p_cam: &Vector3<f64>) -> ProjectionResult<Vector3<f64>> {
        let z = p_cam.z;
        if !(z.is_finite()) || z <= apex_camera_models::MIN_DEPTH {
            return Err(ProjectionError::PointBehindCamera {
                z,
                min_z: apex_camera_models::MIN_DEPTH,
            });
        }
        let x = p_cam.x;
        let y = p_cam.y;
        Ok(Vector3::new(
            self.cx + self.fx * x / z,
            self.cx + self.fx * (x - self.baseline) / z,
            self.cy + self.fy * y / z,
        ))
    }
}

/// Rectified stereo reprojection factor over `[pose, landmark]`.
#[derive(Clone)]
pub struct StereoFactor<G: ActsOnPoints3 = SE3> {
    /// Measured `(uL, uR, v)` in the rectified stereo pair.
    pub measurement: Vector3<f64>,
    /// Fixed stereo calibration.
    pub calibration: StereoCalibration,
    /// Log warnings for cheirality violations.
    pub verbose_cheirality: bool,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> StereoFactor<G> {
    /// Create a stereo factor from a `(uL, uR, v)` measurement and calibration.
    pub fn new(measurement: Vector3<f64>, calibration: StereoCalibration) -> Self {
        Self {
            measurement,
            calibration,
            verbose_cheirality: false,
            group: PhantomData,
        }
    }

    /// Enable verbose cheirality warnings.
    pub fn with_verbose_cheirality(mut self) -> Self {
        self.verbose_cheirality = true;
        self
    }

    /// World-to-camera projection Jacobians shared by the valid and cheirality
    /// paths: `(∂p_cam/∂pose_tangent (3 × G::DOF), ∂p_cam/∂p_world (3×3))` for
    /// the world-to-camera pose convention with right-plus retractions.
    ///
    /// Taken straight from the group's own `act`, so this factor uses the
    /// group's right-perturbation convention rather than a second,
    /// hand-derived copy of it.
    fn point_jacobians(
        pose: &G,
        p_world: &Vector3<f64>,
    ) -> (SMatrix<f64, 3, MAX_ACT_DOF>, Matrix3<f64>) {
        let moved = act_point(pose, p_world);
        (moved.d_state, moved.d_point)
    }
}

impl<G: ActsOnPoints3> Factor for StereoFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "StereoFactor expects [pose, landmark]");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be a 3D point");

        let pose = G::from_param_slice(params[0]);
        let p_world = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let cal = &self.calibration;

        let p_cam = pose.act(&p_world, None, None);

        let projection = match cal.project(&p_cam) {
            Ok(uv) => uv,
            Err(ProjectionError::PointBehindCamera { z, min_z }) => {
                if self.verbose_cheirality {
                    warn!("StereoFactor: point behind camera (z={z}, min_z={min_z})");
                }
                // Same smooth penalty rationale as the monocular projection
                // factor: strictly worse than any in-image residual, with a
                // real gradient pushing z back toward validity.
                let depth_deficit = (min_z - z).max(0.0);
                let penalty = CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * depth_deficit;
                residual[0] = penalty;
                residual[1] = penalty;
                residual[2] = penalty;
                if let Some(mut jac) = jacobian {
                    let d_pen_d_zcam = -CHEIRALITY_DEPTH_SCALE;
                    let (d_pc_d_pose, rotation) = Self::point_jacobians(&pose, &p_world);
                    for c in 0..G::DOF {
                        let d = d_pen_d_zcam * d_pc_d_pose[(2, c)];
                        for r in 0..3 {
                            *jac.rb_mut().get_mut(r, c) = d;
                        }
                    }
                    for c in 0..3 {
                        let d = d_pen_d_zcam * rotation[(2, c)];
                        for r in 0..3 {
                            *jac.rb_mut().get_mut(r, G::DOF + c) = d;
                        }
                    }
                }
                return;
            }
            Err(cam_err) => {
                if self.verbose_cheirality {
                    warn!("StereoFactor: invalid projection: {cam_err}");
                }
                // Model-specific numerical singularity: bounded constant
                // residual, no principled gradient.
                residual[0] = CHEIRALITY_BASE_PENALTY;
                residual[1] = CHEIRALITY_BASE_PENALTY;
                residual[2] = CHEIRALITY_BASE_PENALTY;
                return;
            }
        };

        residual[0] = projection.x - self.measurement.x;
        residual[1] = projection.y - self.measurement.y;
        residual[2] = projection.z - self.measurement.z;

        let Some(mut jac) = jacobian else {
            return;
        };

        let z = p_cam.z;
        let x = p_cam.x;
        let y = p_cam.y;
        let b = cal.baseline;
        let fx = cal.fx;
        let fy = cal.fy;
        let x_r = x - b;

        // ∂(uL, uR, v)/∂(x, y, z)
        let mut d_uv_d_pc = SMatrix::<f64, 3, 3>::zeros();
        d_uv_d_pc[(0, 0)] = fx / z;
        d_uv_d_pc[(0, 2)] = -fx * x / (z * z);
        d_uv_d_pc[(1, 0)] = fx / z;
        d_uv_d_pc[(1, 2)] = -fx * x_r / (z * z);
        d_uv_d_pc[(2, 1)] = fy / z;
        d_uv_d_pc[(2, 2)] = -fy * y / (z * z);

        let (d_pc_d_pose, rotation) = Self::point_jacobians(&pose, &p_world);
        let d_uv_d_pose = d_uv_d_pc * d_pc_d_pose;
        let d_uv_d_landmark = d_uv_d_pc * rotation;

        for r in 0..3 {
            for c in 0..G::DOF {
                *jac.rb_mut().get_mut(r, c) = d_uv_d_pose[(r, c)];
            }
            for c in 0..3 {
                *jac.rb_mut().get_mut(r, G::DOF + c) = d_uv_d_landmark[(r, c)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, G::DOF + 3)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> FactorResult<()> {
        expect_state_and_point::<G>(variables, "StereoFactor")
    }
}

/// Convenience alias for the measured `StereoPoint2` triple.
pub type StereoPoint2 = SVector<f64, 3>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::common::conformance::{Block, finite_diff_jacobian_parity};
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sim3::Sim3;
    use apex_manifolds::so3::SO3;
    use apex_manifolds::{LieGroup, Tangent};

    fn stereo_cal() -> StereoCalibration {
        StereoCalibration::new(500.0, 450.0, 320.0, 240.0, 0.2).unwrap_or_else(|e| panic!("{e}"))
    }

    fn truth_setup() -> (StereoFactor, SE3, Vector3<f64>) {
        let cal = stereo_cal();
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, -0.05, 0.02),
            nalgebra::UnitQuaternion::from_euler_angles(0.03, -0.02, 0.01),
        ));
        let landmark = Vector3::new(0.5, 0.2, 4.0);
        let p_cam = pose.act(&landmark, None, None);
        let measurement = cal
            .project(&p_cam)
            .unwrap_or_else(|e| panic!("truth point must project: {e}"));
        let factor = StereoFactor::new(measurement, cal);
        (factor, pose, landmark)
    }

    #[test]
    fn zero_residual_at_truth() {
        let (factor, pose, landmark) = truth_setup();
        let mut residual = vec![0.0; 3];
        factor.linearize(
            &[pose.as_param_slice(), &[landmark.x, landmark.y, landmark.z]],
            &mut residual,
            None,
        );
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-10, "residual[{i}] = {r}");
        }
    }

    #[test]
    fn finite_difference_jacobians_at_nonzero_residual() {
        // Evaluate the Jacobian at a state with a large non-zero residual:
        // the measurement is offset from the truth projection.
        let (factor0, pose, landmark) = truth_setup();
        let factor = StereoFactor::<SE3>::new(
            factor0.measurement + Vector3::new(1.0, -0.5, 0.3),
            factor0.calibration.clone(),
        );
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();
        let lm_vec = [landmark.x, landmark.y, landmark.z];

        let (rows, cols) = factor.jacobian_shape();
        let mut r0 = vec![0.0; rows];
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[&pose_vec, &lm_vec], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;

        // Pose block (6 DOF)
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let tan6 = SE3Tangent::from_slice(&tan);
            let perturbed: Vec<f64> = pose.right_plus(&tan6, None, None).as_param_slice().to_vec();
            let mut r_pert = vec![0.0; rows];
            factor.linearize(&[&perturbed, &lm_vec], &mut r_pert, None);
            for row in 0..rows {
                let fd = (r_pert[row] - r0[row]) / EPS;
                let ana = jac_buf[col * rows + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "pose[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }

        // Landmark block (3 DOF)
        for col in 0..3 {
            let mut plus = lm_vec;
            let mut minus = lm_vec;
            plus[col] += EPS;
            minus[col] -= EPS;
            let mut r_plus = vec![0.0; rows];
            let mut r_minus = vec![0.0; rows];
            factor.linearize(&[&pose_vec, &plus], &mut r_plus, None);
            factor.linearize(&[&pose_vec, &minus], &mut r_minus, None);
            for row in 0..rows {
                let fd = (r_plus[row] - r_minus[row]) / (2.0 * EPS);
                let ana = jac_buf[(6 + col) * rows + row];
                assert!(
                    (fd - ana).abs() < TOL,
                    "lm[{row},{col}]: analytical={ana:.6} fd={fd:.6}"
                );
            }
        }
    }

    /// A pose of group `G` a little away from the identity, on every axis.
    fn offset_pose<G: ActsOnPoints3>() -> G {
        let step: Vec<f64> = (0..G::DOF)
            .map(|axis| 0.02 * (axis as f64 + 1.0) * if axis % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        <G::TangentVector as Tangent<G>>::from_slice(&step).exp(None)
    }

    /// The analytic Jacobian matches a central difference for group `G`, both
    /// on the valid path and with the point behind the camera (cheirality).
    fn parity_for<G: ActsOnPoints3 + 'static>() -> Result<(), String> {
        let pose = offset_pose::<G>();
        for landmark in [Vector3::new(0.5, 0.2, 4.0), Vector3::new(0.3, -0.1, -1.5)] {
            let measurement = stereo_cal()
                .project(&pose.act(&landmark, None, None))
                .unwrap_or_else(|_| Vector3::new(330.0, 300.0, 250.0));
            let factor =
                StereoFactor::<G>::new(measurement + Vector3::new(1.0, -0.5, 0.3), stereo_cal());
            let blocks = [Block::lie(&pose), Block::euclidean(landmark.as_slice())];
            finite_diff_jacobian_parity(&factor, &blocks, 1e-4)?;
        }
        Ok(())
    }

    #[test]
    fn jacobian_matches_finite_differences_for_every_point_group() -> Result<(), String> {
        parity_for::<SE3>()?;
        parity_for::<SE23>()?;
        parity_for::<Sim3>()?;
        parity_for::<SO3>()
    }

    #[test]
    fn cheirality_violation_gets_penalty() {
        let cal = stereo_cal();
        let factor = StereoFactor::<SE3>::new(Vector3::new(320.0, 320.0, 240.0), cal)
            .with_verbose_cheirality();
        let pose = SE3::identity();
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();

        let eval = |z: f64| -> f64 {
            let mut residual = vec![0.0; 3];
            factor.linearize(&[&pose_vec, &[0.0, 0.0, z]], &mut residual, None);
            residual[0]
        };

        let behind = eval(-0.5);
        let behind_far = eval(-2.0);
        assert!(behind >= CHEIRALITY_BASE_PENALTY);
        assert!(
            behind < behind_far,
            "penalty must grow with violation depth"
        );
    }

    #[test]
    fn calibration_validation_rejects_bad_params() {
        assert!(StereoCalibration::new(0.0, 450.0, 320.0, 240.0, 0.2).is_err());
        assert!(StereoCalibration::new(500.0, 450.0, 320.0, 240.0, -0.1).is_err());
        assert!(StereoCalibration::new(500.0, f64::NAN, 320.0, 240.0, 0.2).is_err());
    }
}
