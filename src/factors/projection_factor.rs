//! Generic projection factor for bundle adjustment and SfM.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix2xX, Matrix3xX, Vector3};
use std::convert::TryFrom;
use std::marker::PhantomData;
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::{Factor, OptimizeParams};
use apex_camera_models::{CameraModel, CameraModelError};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;

/// Baseline cheirality-violation penalty (pixels-equivalent). Chosen to
/// comfortably dominate any plausible in-image residual — an observation is
/// always a finite pixel coordinate, and even a badly-fit-but-valid
/// projection is bounded by roughly the image extent, so a small multiple
/// of a typical image diagonal (a few thousand pixels) is already an
/// unreachable residual for a valid projection — so becoming invalid is
/// never a cheaper escape hatch than fitting the data. Deliberately *not*
/// orders of magnitude larger than that: mixing genuinely huge and
/// pixel-scale residuals/Jacobian entries in the same least-squares problem
/// ill-conditions the normal equations (observed as Cholesky/linear-solve
/// failures in practice).
const CHEIRALITY_BASE_PENALTY: f64 = 1.0e4;

/// Penalty growth rate per metre of depth violation, added on top of
/// [`CHEIRALITY_BASE_PENALTY`] so the penalty keeps increasing — and keeps
/// providing a gradient pointing back toward validity — the further behind
/// the camera a point ends up. Kept modest for the same conditioning reason
/// as [`CHEIRALITY_BASE_PENALTY`].
const CHEIRALITY_DEPTH_SCALE: f64 = 1.0e3;

/// Trait for optimization configuration.
///
/// This trait allows accessing the compile-time boolean flags for
/// parameter optimization (pose, landmarks, intrinsics).
pub trait OptimizationConfig: Send + Sync + 'static {
    const POSE: bool;
    const LANDMARK: bool;
    const INTRINSIC: bool;
}

impl<const P: bool, const L: bool, const I: bool> OptimizationConfig for OptimizeParams<P, L, I> {
    const POSE: bool = P;
    const LANDMARK: bool = L;
    const INTRINSIC: bool = I;
}

/// Generic projection factor for bundle adjustment and structure from motion.
///
/// This factor computes reprojection errors between observed 2D image points
/// and projected 3D landmarks. It supports flexible optimization configurations
/// via generic types implementing `OptimizationConfig`.
///
/// # Type Parameters
///
/// - `CAM`: Camera model implementing [`CameraModel`] trait
/// - `OP`: Optimization configuration (e.g., [`BundleAdjustment`](crate::factors::BundleAdjustment))
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use apex_solver::factors::projection_factor::ProjectionFactor;
/// use apex_solver::factors::BundleAdjustment;
/// use apex_camera_models::PinholeCamera;
/// use nalgebra::{Matrix2xX, Vector2};
///
/// let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
/// let observations = Matrix2xX::from_columns(&[
///     Vector2::new(100.0, 150.0),
///     Vector2::new(200.0, 250.0),
/// ]);
///
/// // Bundle adjustment: optimize pose + landmarks (intrinsics fixed)
/// let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
///     ProjectionFactor::new(observations, camera);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ProjectionFactor<CAM, OP>
where
    CAM: CameraModel,
    OP: OptimizationConfig,
{
    /// 2D observations in image coordinates (2×N for N observations)
    pub observations: Matrix2xX<f64>,

    /// Camera model with intrinsic parameters
    pub camera: CAM,

    /// Fixed pose (required when POSE = false)
    pub fixed_pose: Option<SE3>,

    /// Fixed landmarks (required when LANDMARK = false), 3×N matrix
    pub fixed_landmarks: Option<Matrix3xX<f64>>,

    /// Log warnings for cheirality exceptions (points behind camera)
    pub verbose_cheirality: bool,

    /// Phantom data for optimization type
    _phantom: PhantomData<OP>,
}

impl<CAM, OP> ProjectionFactor<CAM, OP>
where
    CAM: CameraModel,
    OP: OptimizationConfig,
{
    /// Create a new projection factor.
    ///
    /// # Arguments
    ///
    /// * `observations` - 2D image measurements (2×N matrix)
    /// * `camera` - Camera model with intrinsics
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use apex_solver::factors::projection_factor::ProjectionFactor;
    /// # use apex_solver::factors::BundleAdjustment;
    /// # use apex_camera_models::PinholeCamera;
    /// # use nalgebra::{Matrix2xX, Vector2};
    /// # let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
    /// # let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
    /// let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
    ///     ProjectionFactor::new(observations, camera);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(observations: Matrix2xX<f64>, camera: CAM) -> Self {
        Self {
            observations,
            camera,
            fixed_pose: None,
            fixed_landmarks: None,
            verbose_cheirality: false,
            _phantom: PhantomData,
        }
    }

    /// Set fixed pose (required when POSE = false).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use apex_solver::factors::projection_factor::ProjectionFactor;
    /// # use apex_solver::factors::BundleAdjustment;
    /// # use apex_camera_models::PinholeCamera;
    /// # use apex_solver::manifold::se3::SE3;
    /// # use nalgebra::{Matrix2xX, Vector2};
    /// # let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
    /// # let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
    /// # let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> = ProjectionFactor::new(observations, camera);
    /// let factor = factor.with_fixed_pose(SE3::identity());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_fixed_pose(mut self, pose: SE3) -> Self {
        self.fixed_pose = Some(pose);
        self
    }

    /// Set fixed landmarks (required when LANDMARK = false).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use apex_solver::factors::projection_factor::ProjectionFactor;
    /// # use apex_solver::factors::BundleAdjustment;
    /// # use apex_camera_models::PinholeCamera;
    /// # use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
    /// # let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
    /// # let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
    /// # let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> = ProjectionFactor::new(observations, camera);
    /// # let landmarks = Matrix3xX::from_columns(&[Vector3::new(0.1, 0.2, 1.0)]);
    /// let factor = factor.with_fixed_landmarks(landmarks);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_fixed_landmarks(mut self, landmarks: Matrix3xX<f64>) -> Self {
        self.fixed_landmarks = Some(landmarks);
        self
    }

    /// Enable verbose cheirality warnings.
    ///
    /// When enabled, logs warnings when landmarks project behind the camera.
    pub fn with_verbose_cheirality(mut self) -> Self {
        self.verbose_cheirality = true;
        self
    }

    /// Get number of observations.
    pub fn num_observations(&self) -> usize {
        self.observations.ncols()
    }

    /// Internal evaluation function that writes residuals and Jacobians directly
    /// into the provided buffers — no temporary allocations.
    /// `landmarks` is a flat column-major `[x0, y0, z0, x1, y1, z1, …]` buffer.
    ///
    /// That is the layout of both sources — the optimizer's parameter slice and
    /// `Matrix3xX::as_slice` — so neither caller has to build an owned matrix
    /// on the hot path.
    fn evaluate_internal(
        &self,
        pose: &SE3,
        landmarks: &[f64],
        camera: &CAM,
        residual: &mut [f64],
        mut jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let n = self.observations.ncols();

        // Process each observation
        for i in 0..n {
            let observation = self.observations.column(i);
            let p_world =
                Vector3::new(landmarks[3 * i], landmarks[3 * i + 1], landmarks[3 * i + 2]);

            // Transform point to camera frame
            // World-to-camera convention: pose is T_wc where p_cam = R * p_world + t
            // This matches BAL dataset format and ReprojectionFactor
            // pose.act() computes exactly: R * p_world + t = p_cam
            let p_cam = pose.act(&p_world, None, None);

            // Project point (includes all validity checks)
            let uv = match camera.project(&p_cam) {
                Ok(proj) => proj,
                Err(CameraModelError::PointBehindCamera { z, min_z }) => {
                    if self.verbose_cheirality {
                        warn!(
                            "Point {} behind camera (z={}, min_z={}): applying cheirality penalty",
                            i, z, min_z
                        );
                    }
                    self.write_cheirality_penalty(
                        i,
                        z,
                        min_z,
                        &p_world,
                        pose,
                        camera,
                        residual,
                        jacobian.as_mut(),
                    );
                    continue;
                }
                Err(cam_err) => {
                    if self.verbose_cheirality {
                        warn!("Invalid projection for point {}: {}", i, cam_err);
                    }
                    // Invalid projection for a reason other than cheirality
                    // (e.g. a model-specific numerical singularity): no
                    // principled penalty gradient is available, so fall
                    // back to a zero residual as before.
                    residual[i * 2] = 0.0;
                    residual[i * 2 + 1] = 0.0;
                    // Jacobian rows remain zero
                    continue;
                }
            };

            // Compute residual
            residual[i * 2] = uv.x - observation.x;
            residual[i * 2 + 1] = uv.y - observation.y;

            // Compute Jacobians if requested
            if let Some(ref mut jac) = jacobian {
                let mut col_offset = 0;

                // Jacobian w.r.t. pose (world-to-camera convention)
                if OP::POSE {
                    let (d_uv_d_pcam, d_pcam_d_pose) = camera.jacobian_pose(&p_world, pose);
                    let d_uv_d_pose = d_uv_d_pcam * d_pcam_d_pose;
                    for r in 0..2 {
                        for c in 0..6 {
                            *jac.rb_mut().get_mut(i * 2 + r, col_offset + c) = d_uv_d_pose[(r, c)];
                        }
                    }
                    col_offset += 6;
                }

                // Jacobian w.r.t. landmarks (world-to-camera convention)
                if OP::LANDMARK {
                    // For this landmark (3 DOF)
                    let d_uv_d_pcam = camera.jacobian_point(&p_cam);
                    // p_cam = R * p_world + t
                    // ∂p_cam/∂p_world = R
                    // ∂uv/∂p_world = ∂uv/∂p_cam * R
                    let rotation = pose.rotation_so3().rotation_matrix();
                    let d_uv_d_landmark = d_uv_d_pcam * rotation;

                    for r in 0..2 {
                        for c in 0..3 {
                            *jac.rb_mut().get_mut(i * 2 + r, col_offset + i * 3 + c) =
                                d_uv_d_landmark[(r, c)];
                        }
                    }
                }

                // Update column offset for intrinsics (if landmarks are optimized)
                if OP::LANDMARK {
                    col_offset += n * 3;
                }

                // Jacobian w.r.t. intrinsics (shared across all observations)
                if OP::INTRINSIC {
                    let d_uv_d_intrinsics = camera.jacobian_intrinsics(&p_cam);
                    for r in 0..2 {
                        for c in 0..CAM::INTRINSIC_DIM {
                            *jac.rb_mut().get_mut(i * 2 + r, col_offset + c) =
                                d_uv_d_intrinsics[(r, c)];
                        }
                    }
                }
            }
        }
    }

    /// Writes a smooth cheirality-violation penalty for observation `i`,
    /// used in place of the normal reprojection residual when the point
    /// fails `camera.project`'s `PointBehindCamera` check.
    ///
    /// A hard zero residual/Jacobian there (the previous behaviour) makes
    /// "point behind camera" a free way to reduce total cost — and worse
    /// than free, since a valid-but-grazing-incidence point can have a very
    /// large residual, so pushing it just past the cheirality boundary
    /// (residual → 0) is actually *cheaper* than fitting it. That gives the
    /// optimizer a standing incentive to make points invalid rather than
    /// fit them, which is backwards for a residual meant to be minimized.
    ///
    /// Instead this returns a residual that (a) is unconditionally larger
    /// than any plausible in-image residual, so becoming invalid is never
    /// attractive, and (b) grows with how far behind the camera the point
    /// is, with a real gradient — built from `∂z_cam/∂pose` and
    /// `∂z_cam/∂p_world`, both well-defined for any point regardless of
    /// cheirality — that pushes the optimizer back toward `z_cam > min_z`.
    /// The intrinsics block is left at zero: `z_cam` does not depend on the
    /// intrinsic parameters.
    ///
    /// # Rank property (deliberate)
    ///
    /// Both rows carry the same scalar penalty, so their Jacobian rows are
    /// identical and the block is rank-1. This is exact — the residual *is*
    /// the same scalar in both rows — and intentional: the factor's row
    /// layout is fixed at 2 rows per observation, and a violating point
    /// contributes one scalar constraint however it is laid out. Under LM
    /// damping the resulting singular normal block is harmless; under plain
    /// Gauss–Newton a solve made only of cheirality blocks would be
    /// rank-deficient by construction.
    #[allow(clippy::too_many_arguments)]
    fn write_cheirality_penalty(
        &self,
        i: usize,
        z: f64,
        min_z: f64,
        p_world: &Vector3<f64>,
        pose: &SE3,
        camera: &CAM,
        residual: &mut [f64],
        jacobian: Option<&mut faer::mat::MatMut<'_, f64>>,
    ) {
        let depth_deficit = (min_z - z).max(0.0);
        let penalty = CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * depth_deficit;
        residual[i * 2] = penalty;
        residual[i * 2 + 1] = penalty;

        let Some(jac) = jacobian else { return };

        // ∂penalty/∂z_cam = -CHEIRALITY_DEPTH_SCALE (increasing z_cam
        // shrinks the deficit).
        let d_penalty_d_zcam = -CHEIRALITY_DEPTH_SCALE;
        let mut col_offset = 0;

        if OP::POSE {
            // `d_pcam_d_pose`'s 3rd row is ∂z_cam/∂(pose tangent) — a pure
            // rotation/skew(p_world) quantity (see the default
            // `CameraModel::jacobian_pose` body) independent of the camera
            // model's own projection formula, so it is exactly as valid
            // here as it is behind the cheirality boundary. The first
            // tuple element (∂uv/∂p_cam) is intentionally unused: it is
            // not defined in a meaningful way for an invalid projection.
            let (_, d_pcam_d_pose) = camera.jacobian_pose(p_world, pose);
            for c in 0..6 {
                let d = d_penalty_d_zcam * d_pcam_d_pose[(2, c)];
                *jac.rb_mut().get_mut(i * 2, col_offset + c) = d;
                *jac.rb_mut().get_mut(i * 2 + 1, col_offset + c) = d;
            }
            col_offset += 6;
        }

        if OP::LANDMARK {
            // z_cam = (R p_world + t).z, so ∂z_cam/∂p_world = R's 3rd row.
            let rotation = pose.rotation_so3().rotation_matrix();
            for c in 0..3 {
                let d = d_penalty_d_zcam * rotation[(2, c)];
                *jac.rb_mut().get_mut(i * 2, col_offset + i * 3 + c) = d;
                *jac.rb_mut().get_mut(i * 2 + 1, col_offset + i * 3 + c) = d;
            }
        }
        // Intrinsics block (if present) is left at zero: z_cam does not
        // depend on the intrinsic parameters.
    }
}

// Factor trait implementation with generic dispatch
impl<CAM, OP> Factor for ProjectionFactor<CAM, OP>
where
    CAM: CameraModel,
    for<'a> CAM: TryFrom<&'a [f64]>,
    OP: OptimizationConfig,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let mut param_idx = 0;

        let pose: SE3 = if OP::POSE {
            let p = SE3::from_param_slice(params[param_idx]);
            param_idx += 1;
            p
        } else {
            self.fixed_pose.clone().unwrap_or_else(SE3::identity)
        };

        // Both landmark sources are already column-major triples, so this
        // borrows rather than materializing a `Matrix3xX` per call — this
        // factor is evaluated once per observation per iteration.
        let landmarks: &[f64] = if OP::LANDMARK {
            let flat = params[param_idx];
            param_idx += 1;
            flat
        } else {
            self.fixed_landmarks
                .as_ref()
                .map_or(&[][..], |fixed| fixed.as_slice())
        };

        // Decode intrinsics only when they are being optimized; otherwise (and
        // on a decode failure) fall back to the constructor-time camera by
        // reference instead of cloning it.
        let decoded_camera: Option<CAM> = if OP::INTRINSIC {
            CAM::try_from(params[param_idx]).ok()
        } else {
            None
        };
        let camera: &CAM = decoded_camera.as_ref().unwrap_or(&self.camera);

        let n = self.observations.ncols();
        debug_assert_eq!(
            landmarks.len(),
            3 * n,
            "Number of landmarks ({}) must match observations ({})",
            landmarks.len() / 3,
            n
        );

        // Write directly into caller-provided buffers — zero temporary allocation.
        self.evaluate_internal(&pose, landmarks, camera, residual, jacobian);
    }

    fn residual_dim(&self) -> usize {
        self.observations.ncols() * 2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        let n = self.observations.ncols();
        let mut cols = 0;
        if OP::POSE {
            cols += 6;
        }
        if OP::LANDMARK {
            cols += n * 3;
        }
        if OP::INTRINSIC {
            cols += CAM::INTRINSIC_DIM;
        }
        (n * 2, cols)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        let mut idx = 0;

        if OP::POSE {
            let pose = variables.get(idx).ok_or_else(|| {
                "ProjectionFactor expects a pose variable as its first parameter".to_string()
            })?;
            if pose.as_param_slice().len() != SE3::REP_SIZE {
                return Err(format!(
                    "pose variable holds {} parameters, ProjectionFactor requires {} (SE3)",
                    pose.as_param_slice().len(),
                    SE3::REP_SIZE
                ));
            }
            idx += 1;
        }

        if OP::LANDMARK {
            let landmarks = variables
                .get(idx)
                .ok_or_else(|| "ProjectionFactor expects a landmark variable".to_string())?;
            let expected = 3 * self.observations.ncols();
            if landmarks.as_param_slice().len() != expected {
                return Err(format!(
                    "landmark variable holds {} parameters but the factor's {} observations \
                     reference {} landmarks (3 coordinates each)",
                    landmarks.as_param_slice().len(),
                    self.observations.ncols(),
                    self.observations.ncols()
                ));
            }
        } else if self
            .fixed_landmarks
            .as_ref()
            .is_none_or(|l| l.ncols() != self.observations.ncols())
        {
            return Err(format!(
                "fixed landmarks must be set and match the {} observations",
                self.observations.ncols()
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::{BundleAdjustment, OnlyIntrinsics, SelfCalibration};
    use apex_camera_models::PinholeCamera;
    use nalgebra::{DMatrix, DVector, Vector2, Vector3};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn call_linearize(
        factor: &impl Factor,
        params: &[DVector<f64>],
        with_jacobian: bool,
    ) -> (Vec<f64>, Option<DMatrix<f64>>) {
        let param_slices: Vec<&[f64]> = params.iter().map(|p| p.as_slice()).collect();
        let mut residual = vec![0.0f64; factor.residual_dim()];
        if with_jacobian {
            let (rows, cols) = factor.jacobian_shape();
            let mut jac_buf = vec![0.0f64; rows * cols];
            let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
            factor.linearize(&param_slices, &mut residual, Some(jac_mut));
            let jac = DMatrix::from_column_slice(rows, cols, &jac_buf);
            (residual, Some(jac))
        } else {
            factor.linearize(&param_slices, &mut residual, None);
            (residual, None)
        }
    }

    #[test]
    fn test_projection_factor_creation() -> TestResult {
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);

        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        assert_eq!(factor.num_observations(), 1);
        assert_eq!(factor.residual_dim(), 2);

        Ok(())
    }

    #[test]
    fn test_bundle_adjustment_factor() -> TestResult {
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);

        let p_world = Vector3::new(0.1, 0.2, 1.0);
        let pose = SE3::identity();

        let p_cam = pose.act(&p_world, None, None);
        let uv = camera.project(&p_cam)?;

        let observations = Matrix2xX::from_columns(&[uv]);

        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        let pose_vec = DVector::from_column_slice(pose.as_param_slice());
        let landmarks_vec = DVector::from_vec(vec![p_world.x, p_world.y, p_world.z]);
        let params = vec![pose_vec, landmarks_vec];

        let (residual, jacobian) = call_linearize(&factor, &params, true);

        let res_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-10, "Residual: {:?}", residual);

        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 9);

        Ok(())
    }

    #[test]
    fn test_self_calibration_factor() -> TestResult {
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let p_world = Vector3::new(0.1, 0.2, 1.0);
        let pose = SE3::identity();

        let p_cam = pose.act(&p_world, None, None);
        let uv = camera.project(&p_cam)?;

        let observations = Matrix2xX::from_columns(&[uv]);
        let factor: ProjectionFactor<PinholeCamera, SelfCalibration> =
            ProjectionFactor::new(observations, camera);

        let pose_vec = DVector::from_column_slice(pose.as_param_slice());
        let landmarks_vec = DVector::from_vec(vec![p_world.x, p_world.y, p_world.z]);
        let intrinsics_vec = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
        let params = vec![pose_vec, landmarks_vec, intrinsics_vec];

        let (residual, jacobian) = call_linearize(&factor, &params, true);

        let res_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-10);

        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 13);

        Ok(())
    }

    #[test]
    fn test_calibration_factor() -> TestResult {
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let pose = SE3::identity();
        let p_world = Vector3::new(0.1, 0.2, 1.0);

        let p_cam = pose.act(&p_world, None, None);
        let uv = camera.project(&p_cam)?;

        let observations = Matrix2xX::from_columns(&[uv]);
        let landmarks = Matrix3xX::from_columns(&[p_world]);

        let factor: ProjectionFactor<PinholeCamera, OnlyIntrinsics> =
            ProjectionFactor::new(observations, camera)
                .with_fixed_pose(pose)
                .with_fixed_landmarks(landmarks);

        let intrinsics_vec = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
        let params = vec![intrinsics_vec];

        let (residual, jacobian) = call_linearize(&factor, &params, true);

        let res_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-10);

        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 4);

        Ok(())
    }

    #[test]
    fn test_invalid_projection_handling() -> TestResult {
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);

        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera).with_verbose_cheirality();

        let pose = SE3::identity();
        let pose_vec = DVector::from_column_slice(pose.as_param_slice());
        let landmarks_vec = DVector::from_vec(vec![0.0, 0.0, -1.0]);
        let params = vec![pose_vec, landmarks_vec];

        let (residual, _) = call_linearize(&factor, &params, false);

        // A point behind the camera must NOT be a free (zero-residual) way
        // to reduce cost: see `write_cheirality_penalty`. The point is 1m
        // behind the camera (min_z is ~0), so the penalty is at least the
        // base penalty.
        assert!(
            residual[0] >= CHEIRALITY_BASE_PENALTY,
            "residual[0] = {}",
            residual[0]
        );
        assert!(
            residual[1] >= CHEIRALITY_BASE_PENALTY,
            "residual[1] = {}",
            residual[1]
        );

        Ok(())
    }

    #[test]
    fn test_cheirality_penalty_grows_with_depth_violation() -> TestResult {
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);
        let pose = SE3::identity();
        let pose_vec = DVector::from_column_slice(pose.as_param_slice());

        let residual_at = |z: f64| -> f64 {
            let landmarks_vec = DVector::from_vec(vec![0.0, 0.0, z]);
            let params = vec![pose_vec.clone(), landmarks_vec];
            let (residual, _) = call_linearize(&factor, &params, false);
            residual[0]
        };

        // Further behind the camera => strictly larger penalty, so the
        // optimizer always has a gradient pointing back toward validity
        // rather than a flat or decreasing cost.
        let r_close = residual_at(-0.01);
        let r_mid = residual_at(-0.5);
        let r_far = residual_at(-2.0);
        assert!(r_close < r_mid, "{r_close} vs {r_mid}");
        assert!(r_mid < r_far, "{r_mid} vs {r_far}");

        // And it must always exceed any plausible valid residual.
        assert!(r_close >= CHEIRALITY_BASE_PENALTY);

        Ok(())
    }

    #[test]
    fn test_cheirality_penalty_jacobian_numerical() -> TestResult {
        // Numerically verify the pose and landmark Jacobians written by
        // `write_cheirality_penalty` against finite differences of the
        // penalty residual itself, the same style used for the camera
        // models' own Jacobian tests.
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, -0.2, 0.3),
            nalgebra::UnitQuaternion::from_euler_angles(0.05, -0.1, 0.2),
        ));
        let landmark = Vector3::new(0.2, -0.1, -0.5); // behind the camera

        let eval = |pose: &SE3, landmark: &Vector3<f64>| -> f64 {
            let pose_vec = DVector::from_column_slice(pose.as_param_slice());
            let landmarks_vec = DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]);
            let params = vec![pose_vec, landmarks_vec];
            let (residual, _) = call_linearize(&factor, &params, false);
            residual[0]
        };

        let params = vec![
            DVector::from_column_slice(pose.as_param_slice()),
            DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]),
        ];
        let (_, jacobian) = call_linearize(&factor, &params, true);
        let jac = jacobian.ok_or("Jacobian should be Some")?;

        // ∂residual/∂landmark (columns 6..9), numerically.
        let eps = 1e-6;
        for c in 0..3 {
            let mut plus = landmark;
            let mut minus = landmark;
            plus[c] += eps;
            minus[c] -= eps;
            let num = (eval(&pose, &plus) - eval(&pose, &minus)) / (2.0 * eps);
            let ana = jac[(0, 6 + c)];
            assert!(
                (num - ana).abs() < 1e-2,
                "landmark col {c}: numerical={num}, analytical={ana}"
            );
        }

        Ok(())
    }
}
