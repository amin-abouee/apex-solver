//! Generic projection factor for bundle adjustment and SfM.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix2xX, Matrix3xX, Vector3};
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::warn;

use crate::core::variable::ManifoldVariable;
use crate::factors::{Factor, OptimizeParams};
use apex_camera_models::{CameraModel, CameraModelError};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;

use crate::factors::common::cheirality::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};

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
/// use apex_solver::factors::visual::projection::ProjectionFactor;
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
//
// NOTE: `Clone` is manual because of the atomic fallback counter below.
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

    /// How often intrinsics decoding failed and evaluation fell back to the
    /// constructor-time camera. A nonzero count after a solve means the
    /// intrinsics blocks are degenerate — inspect, don't ignore.
    intrinsics_fallbacks: AtomicUsize,

    /// Phantom data for optimization type
    _phantom: PhantomData<OP>,
}

impl<CAM, OP> Clone for ProjectionFactor<CAM, OP>
where
    CAM: CameraModel + Clone,
    OP: OptimizationConfig,
{
    fn clone(&self) -> Self {
        Self {
            observations: self.observations.clone(),
            camera: self.camera.clone(),
            fixed_pose: self.fixed_pose.clone(),
            fixed_landmarks: self.fixed_landmarks.clone(),
            verbose_cheirality: self.verbose_cheirality,
            intrinsics_fallbacks: AtomicUsize::new(
                self.intrinsics_fallbacks.load(Ordering::Relaxed),
            ),
            _phantom: PhantomData,
        }
    }
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
    /// # use apex_solver::factors::visual::projection::ProjectionFactor;
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
            intrinsics_fallbacks: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }

    /// Set fixed pose (required when POSE = false).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use apex_solver::factors::visual::projection::ProjectionFactor;
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
    /// # use apex_solver::factors::visual::projection::ProjectionFactor;
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

    /// How often intrinsics decoding failed and evaluation silently fell back
    /// to the constructor-time camera. Query after a solve: nonzero means the
    /// intrinsics blocks went degenerate mid-optimization.
    pub fn intrinsics_fallback_count(&self) -> usize {
        self.intrinsics_fallbacks.load(Ordering::Relaxed)
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
                    self.write_projection_barrier(
                        i,
                        min_z - z,
                        &Vector3::new(0.0, 0.0, -1.0),
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
                    // Invalid for a reason other than an explicit
                    // `PointBehindCamera`: zero residual, zero Jacobian rows.
                    //
                    // Charging this arm was implemented and measured, then
                    // reverted. Routing every out-of-domain observation to a
                    // `CHEIRALITY_BASE_PENALTY` barrier put the *initial*
                    // cost of BAL t257 at 3.29e12 and dubrovnik-135 at
                    // 2.20e10 — against goldens of 6.86e4 and 1.89e5 — and
                    // the LM solve then stopped making progress at all
                    // (final/initial = 1 − 1e-7): 5 of the 8 BA benchmark ids
                    // regressed, 1 improved, 2 were unchanged. Halving the
                    // penalty 1000x did not help, so it is not a
                    // conditioning artifact but an irreducible barrier cost
                    // at the initial guess. See
                    // `benchmarking_results/experiment_007_projection_domain_barrier_REJECTED.md`
                    // and scenario S12 in `docs/factor_correctness_audit.md`.
                    //
                    // Consequence: `fov`, `bal_pinhole` and `double_sphere`
                    // report their cheirality failure as
                    // `ProjectionOutOfBounds`, so for those models this arm
                    // still lets a behind-camera point go free — the
                    // incentive S12 describes stays open, and the model-side
                    // [`CameraModel::projection_deficit`] metadata is kept
                    // ready for a formulation that survives the benchmark
                    // goldens.
                    residual[i * 2] = 0.0;
                    residual[i * 2 + 1] = 0.0;
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

    /// Writes a smooth cheirality barrier for observation `i`, used in
    /// place of the normal reprojection residual when `camera.project`
    /// fails with `PointBehindCamera { z, min_z }` — `deficit = min_z - z`,
    /// gradient `-e_z`, i.e. a z-forward depth deficit.
    ///
    /// # Scope note (one call site, deliberately)
    ///
    /// Routing the *other* camera errors through the model's own
    /// [`CameraModel::projection_deficit`] — which describes that model's
    /// domain (the FOV plane, the BAL `z < -MIN_DEPTH` half-space, the
    /// double-sphere cone, …) instead of a z-forward one — was implemented,
    /// measured and then reverted; the call site records the numbers, and
    /// `benchmarking_results/experiment_007_projection_domain_barrier_REJECTED.md`
    /// has the full per-dataset table. Only the cheirality arm charges here.
    ///
    /// # Why not zero
    ///
    /// A hard zero residual/Jacobian for an invalid projection (the
    /// behaviour before the cheirality fix, and still the behaviour for
    /// every error class other than `PointBehindCamera`) makes "invalid" a free way to reduce
    /// total cost — and worse than free, since a valid-but-grazing-incidence
    /// point can have a very large residual, so pushing it just past a
    /// validity boundary (residual → 0) is actually *cheaper* than fitting
    /// it. That gives the optimizer a standing incentive to make points
    /// invalid rather than fit them, which is backwards for a residual meant
    /// to be minimized. Because `fov`, `bal_pinhole` and `double_sphere`
    /// report their cheirality failure as `ProjectionOutOfBounds`, that
    /// incentive was live for them specifically: the penalty only ever fired
    /// on the `PointBehindCamera` arm.
    ///
    /// Instead this returns a residual that (a) is unconditionally larger
    /// than any plausible in-image residual, so becoming invalid is never
    /// attractive, and (b) grows with the depth deficit, with a real
    /// gradient — built from `∂deficit/∂p_cam`, `∂p_cam/∂pose` and
    /// `∂p_cam/∂p_world = R`, all well-defined for any point regardless of
    /// whether the projection itself is — that pushes the optimizer back
    /// across the boundary.
    ///
    /// # Degenerate failures
    ///
    /// A non-positive deficit clamps to zero, which leaves only the
    /// constant base penalty and no Jacobian. With the cheirality arm the
    /// deficit is `min_z - z ≥ 0` by construction, so this is defensive;
    /// where there is no direction "out" of a failure to write down, a
    /// wrong-signed gradient would be worse than none.
    ///
    /// # Intrinsics block
    ///
    /// Left at zero. For a depth deficit based on `z_cam` that is exact —
    /// depth does not depend on intrinsics.
    ///
    /// # Rank property (deliberate)
    ///
    /// Both rows carry the same scalar penalty, so their Jacobian rows are
    /// identical and the block is rank-1. This is exact — the residual *is*
    /// the same scalar in both rows — and intentional: the factor's row
    /// layout is fixed at 2 rows per observation, and a violating point
    /// contributes one scalar constraint however it is laid out. Under LM
    /// damping the resulting singular normal block is harmless; under plain
    /// Gauss–Newton a solve made only of barrier blocks would be
    /// rank-deficient by construction.
    #[allow(clippy::too_many_arguments)]
    fn write_projection_barrier(
        &self,
        i: usize,
        deficit: f64,
        d_deficit_d_pcam: &Vector3<f64>,
        p_world: &Vector3<f64>,
        pose: &SE3,
        camera: &CAM,
        residual: &mut [f64],
        jacobian: Option<&mut faer::mat::MatMut<'_, f64>>,
    ) {
        // Clamp before use: `deficit > 0` is the precondition for both the
        // growth term and the gradient below.
        let deficit = deficit.max(0.0);
        let penalty = CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * deficit;
        residual[i * 2] = penalty;
        residual[i * 2 + 1] = penalty;

        if deficit <= 0.0 {
            return;
        }

        let Some(jac) = jacobian else { return };

        // ∂penalty/∂p_cam = CHEIRALITY_DEPTH_SCALE · ∂deficit/∂p_cam.
        let d_penalty_d_pcam = *d_deficit_d_pcam * CHEIRALITY_DEPTH_SCALE;
        let mut col_offset = 0;

        if OP::POSE {
            // `d_pcam_d_pose` is `∂p_cam/∂(pose tangent)`, a pure
            // rotation/skew(p_world) quantity (see the default
            // `CameraModel::jacobian_pose` body) that is independent of the
            // camera model's own projection formula, so it is exactly as
            // valid here as it is on a valid projection. The first tuple
            // element (∂uv/∂p_cam) is intentionally unused: it is not
            // defined in a meaningful way for an invalid projection.
            let (_, d_pcam_d_pose) = camera.jacobian_pose(p_world, pose);
            for c in 0..6 {
                let d = (0..3)
                    .map(|r| d_penalty_d_pcam[r] * d_pcam_d_pose[(r, c)])
                    .sum::<f64>();
                *jac.rb_mut().get_mut(i * 2, col_offset + c) = d;
                *jac.rb_mut().get_mut(i * 2 + 1, col_offset + c) = d;
            }
            col_offset += 6;
        }

        if OP::LANDMARK {
            // p_cam = R p_world + t, so ∂p_cam/∂p_world = R.
            let rotation = pose.rotation_so3().rotation_matrix();
            for c in 0..3 {
                let d = (0..3)
                    .map(|r| d_penalty_d_pcam[r] * rotation[(r, c)])
                    .sum::<f64>();
                *jac.rb_mut().get_mut(i * 2, col_offset + i * 3 + c) = d;
                *jac.rb_mut().get_mut(i * 2 + 1, col_offset + i * 3 + c) = d;
            }
        }
        // Intrinsics block (if present) stays zero — see the doc comment.
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
        // reference instead of cloning it. Failures are counted: a nonzero
        // `intrinsics_fallback_count` after a solve means degenerate
        // intrinsics blocks, not a working self-calibration.
        let decoded_camera: Option<CAM> = if OP::INTRINSIC {
            CAM::try_from(params[param_idx]).ok()
        } else {
            None
        };
        if OP::INTRINSIC && decoded_camera.is_none() {
            self.intrinsics_fallbacks.fetch_add(1, Ordering::Relaxed);
        }
        let camera: &CAM = decoded_camera.as_ref().unwrap_or(&self.camera);

        let n = self.observations.ncols();
        // Hard assert, not debug-only: a stride mismatch silently misprojects
        // every landmark, and `profile.test` inherits release, so a
        // `debug_assert` would never fire under `cargo test`.
        assert_eq!(
            landmarks.len(),
            3 * n,
            "landmark buffer length {} is not 3×N observations ({n}); \
             the buffer must be flat column-major [x0,y0,z0,…]",
            landmarks.len()
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
    use crate::factors::{
        BundleAdjustment, LandmarksAndIntrinsics, OnlyIntrinsics, OnlyLandmarks, OnlyPose,
        PoseAndIntrinsics, SelfCalibration,
    };
    use apex_camera_models::{DoubleSphereCamera, PinholeCamera};
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use nalgebra::{DMatrix, DVector, Vector2, Vector3};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// `[fx, fy, cx, cy]` — layout expected by `PinholeCamera::try_from`,
    /// which is what the factor decodes the intrinsics block with.
    const PINHOLE_INTRINSICS: [f64; 4] = [500.0, 500.0, 320.0, 240.0];

    /// `[fx, fy, cx, cy, xi, alpha]` — layout expected by
    /// `DoubleSphereCamera::try_from`.
    const DOUBLE_SPHERE_INTRINSICS: [f64; 6] = [500.0, 500.0, 320.0, 240.0, 0.0, 0.3];

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

    /// Compares one analytic Jacobian column against the central difference
    /// of the residuals along the parameter direction that column represents.
    ///
    /// The tolerance is deliberately relative-and-offset (`tol·(1+|ana|)`):
    /// projection Jacobian entries span several orders of magnitude between
    /// focal-length and principal-point columns, so a single absolute
    /// tolerance would be vacuous at one end and unachievable at the other.
    fn assert_fd_agrees(
        label: &str,
        col: usize,
        r_plus: &[f64],
        r_minus: &[f64],
        jac: &DMatrix<f64>,
        eps: f64,
        tol: f64,
    ) {
        for (row, (plus, minus)) in r_plus.iter().zip(r_minus.iter()).enumerate() {
            let fd = (plus - minus) / (2.0 * eps);
            let ana = jac[(row, col)];
            assert!(
                (fd - ana).abs() <= tol * (1.0 + ana.abs()),
                "{label} column {col}, row {row}: finite difference {fd} \
                 vs analytic {ana} (diff {})",
                (fd - ana).abs()
            );
        }
    }

    /// Finite-difference check of every column of `factor`'s Jacobian, for
    /// one concrete parameter vector.
    ///
    /// The column blocks are `[pose(6)] [landmarks(3N)] [intrinsics(D)]`
    /// (see [`Factor::jacobian_shape`]) and each is perturbed on the space
    /// the block actually lives on: pose columns through `right_plus` with a
    /// unit `se(3)` tangent — the convention `CameraModel::jacobian_pose`
    /// documents, so a convention mismatch between the perturbation and the
    /// analytic formula shows up as a sign/row swap rather than passing —
    /// and landmark/intrinsic columns by a plain Euclidean ±`eps`.
    ///
    /// This validates the *assembly*: block offsets, which rows each block
    /// writes, and the product `∂uv/∂p_cam · ∂p_cam/∂ξ`.
    fn assert_jacobian_matches_fd<CAM, OP>(
        factor: &ProjectionFactor<CAM, OP>,
        params: &[DVector<f64>],
        eps: f64,
        tol: f64,
    ) -> TestResult
    where
        CAM: CameraModel + for<'a> TryFrom<&'a [f64]>,
        OP: OptimizationConfig,
    {
        let (analytic, jacobian) = call_linearize(factor, params, true);
        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), analytic.len());
        // Non-vacuity guard: a factor that wrote no Jacobian at all would
        // otherwise agree with zero finite differences and pass.
        assert!(
            jac.norm() > 1.0,
            "analytic Jacobian norm {} — nothing to check",
            jac.norm()
        );

        let residual_at = |p: &[DVector<f64>]| -> Vec<f64> { call_linearize(factor, p, false).0 };
        let mut col = 0usize;

        if OP::POSE {
            let base = SE3::from_param_slice(params[0].as_slice());
            for c in 0..6 {
                let mut tangent = [0.0f64; 6];
                tangent[c] = eps;
                let plus = base.right_plus(&SE3Tangent::from_slice(&tangent), None, None);
                tangent[c] = -eps;
                let minus = base.right_plus(&SE3Tangent::from_slice(&tangent), None, None);

                let mut p_plus = params.to_vec();
                p_plus[0] = DVector::from_column_slice(plus.as_param_slice());
                let mut p_minus = params.to_vec();
                p_minus[0] = DVector::from_column_slice(minus.as_param_slice());

                assert_fd_agrees(
                    "pose",
                    col + c,
                    &residual_at(&p_plus),
                    &residual_at(&p_minus),
                    &jac,
                    eps,
                    tol,
                );
            }
            col += 6;
        }

        if OP::LANDMARK {
            // Landmarks occupy a single flat parameter block, and the factor
            // writes observation `i`'s three columns at `offset + 3i`, so the
            // parameter order and the column order coincide exactly.
            let idx = usize::from(OP::POSE);
            for c in 0..params[idx].len() {
                let mut p_plus = params.to_vec();
                p_plus[idx][c] += eps;
                let mut p_minus = params.to_vec();
                p_minus[idx][c] -= eps;

                assert_fd_agrees(
                    "landmark",
                    col + c,
                    &residual_at(&p_plus),
                    &residual_at(&p_minus),
                    &jac,
                    eps,
                    tol,
                );
            }
            col += params[idx].len();
        }

        if OP::INTRINSIC {
            let idx = params.len() - 1;
            for c in 0..params[idx].len() {
                let mut p_plus = params.to_vec();
                p_plus[idx][c] += eps;
                let mut p_minus = params.to_vec();
                p_minus[idx][c] -= eps;

                assert_fd_agrees(
                    "intrinsics",
                    col + c,
                    &residual_at(&p_plus),
                    &residual_at(&p_minus),
                    &jac,
                    eps,
                    tol,
                );
            }
        }

        Ok(())
    }

    /// Builds a fully-observed scene (three landmarks, exact reprojection
    /// observations) for one camera model and optimization mode, then runs
    /// [`assert_jacobian_matches_fd`] on it.
    ///
    /// Landmarks are specified in the *camera* frame — comfortably in front
    /// of the camera and near the image centre — and carried to world
    /// coordinates, which guarantees every projection succeeds and that no
    /// finite-difference stencil crosses a validity boundary. Crossing one
    /// would compare the analytic reprojection Jacobian against the
    /// cheirality penalty's derivative, which is a different quantity by
    /// design.
    fn run_fd_scenario<CAM, OP>(camera: CAM, intrinsics: &[f64]) -> TestResult
    where
        CAM: CameraModel + for<'a> TryFrom<&'a [f64]>,
        OP: OptimizationConfig,
    {
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.3, -0.4, 0.2),
            nalgebra::UnitQuaternion::from_euler_angles(0.2, -0.15, 0.35),
        ));

        let p_cams = [
            Vector3::new(0.4, -0.3, 4.0),
            Vector3::new(-1.2, 0.8, 6.0),
            Vector3::new(0.05, 0.1, 5.0),
        ];
        let to_world = pose.inverse(None);
        let p_worlds: Vec<Vector3<f64>> =
            p_cams.iter().map(|p| to_world.act(p, None, None)).collect();

        let mut observations = Vec::with_capacity(p_cams.len());
        for p_cam in &p_cams {
            observations.push(camera.project(p_cam)?);
        }

        let mut factor =
            ProjectionFactor::<CAM, OP>::new(Matrix2xX::from_columns(&observations), camera);
        if !OP::POSE {
            factor = factor.with_fixed_pose(pose.clone());
        }
        if !OP::LANDMARK {
            factor = factor.with_fixed_landmarks(Matrix3xX::from_columns(&p_worlds));
        }

        // Parameter blocks appear in `OptimizationConfig` order: pose,
        // landmarks, intrinsics — the same order the Jacobian columns use.
        let mut params: Vec<DVector<f64>> = Vec::new();
        if OP::POSE {
            params.push(DVector::from_column_slice(pose.as_param_slice()));
        }
        if OP::LANDMARK {
            params.push(DVector::from_vec(
                p_worlds.iter().flat_map(|p| [p.x, p.y, p.z]).collect(),
            ));
        }
        if OP::INTRINSIC {
            params.push(DVector::from_column_slice(intrinsics));
        }

        assert_jacobian_matches_fd(&factor, &params, 1e-6, 1e-5)?;

        // A nonzero count means the intrinsics block decoded to the
        // constructor-time camera instead of `params`, in which case the
        // intrinsics columns above would be comparing against a stale model.
        assert_eq!(
            factor.intrinsics_fallback_count(),
            0,
            "intrinsics decoding fell back to the constructor-time camera \
             during the finite-difference sweep"
        );

        Ok(())
    }

    #[test]
    fn fd_bundle_adjustment_pinhole() -> TestResult {
        run_fd_scenario::<PinholeCamera, BundleAdjustment>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_self_calibration_pinhole() -> TestResult {
        run_fd_scenario::<PinholeCamera, SelfCalibration>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_only_pose_pinhole() -> TestResult {
        run_fd_scenario::<PinholeCamera, OnlyPose>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_only_landmarks_pinhole() -> TestResult {
        run_fd_scenario::<PinholeCamera, OnlyLandmarks>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_only_intrinsics_pinhole() -> TestResult {
        run_fd_scenario::<PinholeCamera, OnlyIntrinsics>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_pose_and_intrinsics_pinhole() -> TestResult {
        // No landmark block: the intrinsics columns must sit directly after
        // the pose block rather than at `6 + 3N`.
        run_fd_scenario::<PinholeCamera, PoseAndIntrinsics>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_landmarks_and_intrinsics_pinhole() -> TestResult {
        // No pose block: the intrinsics columns must sit at `3N` rather
        // than after an implicit pose block.
        run_fd_scenario::<PinholeCamera, LandmarksAndIntrinsics>(
            PinholeCamera::from(PINHOLE_INTRINSICS),
            &PINHOLE_INTRINSICS,
        )
    }

    #[test]
    fn fd_bundle_adjustment_double_sphere() -> TestResult {
        run_fd_scenario::<DoubleSphereCamera, BundleAdjustment>(
            DoubleSphereCamera::from(DOUBLE_SPHERE_INTRINSICS),
            &DOUBLE_SPHERE_INTRINSICS,
        )
    }

    #[test]
    fn fd_self_calibration_double_sphere() -> TestResult {
        // The double-sphere model exercises `jacobian_intrinsics`'s
        // distortion columns (xi, alpha), which the pinhole runs never touch.
        run_fd_scenario::<DoubleSphereCamera, SelfCalibration>(
            DoubleSphereCamera::from(DOUBLE_SPHERE_INTRINSICS),
            &DOUBLE_SPHERE_INTRINSICS,
        )
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
        // to reduce cost: see `write_projection_barrier`. The point is 1m
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
        // `write_projection_barrier` against finite differences of the
        // penalty residuals themselves, the same style used for the camera
        // models' own Jacobian tests. Both residual rows and both parameter
        // blocks (pose and landmark) are covered: a penalty whose gradient
        // only matched in one row or in Euclidean coordinates would still
        // look correct to a narrower check.
        let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, -0.2, 0.3),
            nalgebra::UnitQuaternion::from_euler_angles(0.05, -0.1, 0.2),
        ));
        let landmark = Vector3::new(0.2, -0.1, -0.5); // behind the camera

        let eval = |pose: &SE3, landmark: &Vector3<f64>| -> Vec<f64> {
            let pose_vec = DVector::from_column_slice(pose.as_param_slice());
            let landmarks_vec = DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]);
            let params = vec![pose_vec, landmarks_vec];
            call_linearize(&factor, &params, false).0
        };

        let params = vec![
            DVector::from_column_slice(pose.as_param_slice()),
            DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]),
        ];
        let (_, jacobian) = call_linearize(&factor, &params, true);
        let jac = jacobian.ok_or("Jacobian should be Some")?;

        let eps = 1e-6;

        // ∂penalty/∂(pose tangent), columns 0..6. The perturbation is the
        // same right-plus `se(3)` exponential the analytic row is derived
        // for, so a left/right convention error fails rather than passes.
        for c in 0..6 {
            let mut tangent = [0.0f64; 6];
            tangent[c] = eps;
            let plus = pose.right_plus(&SE3Tangent::from_slice(&tangent), None, None);
            tangent[c] = -eps;
            let minus = pose.right_plus(&SE3Tangent::from_slice(&tangent), None, None);

            let r_plus = eval(&plus, &landmark);
            let r_minus = eval(&minus, &landmark);
            for (row, (p, m)) in r_plus.iter().zip(r_minus.iter()).enumerate() {
                let num = (p - m) / (2.0 * eps);
                let ana = jac[(row, c)];
                assert!(
                    (num - ana).abs() < 1e-4 * (1.0 + ana.abs()),
                    "pose col {c}, row {row}: numerical={num}, analytical={ana}"
                );
            }
        }

        // ∂penalty/∂landmark, columns 6..9.
        for c in 0..3 {
            let mut plus = landmark;
            let mut minus = landmark;
            plus[c] += eps;
            minus[c] -= eps;
            let r_plus = eval(&pose, &plus);
            let r_minus = eval(&pose, &minus);
            for (row, (p, m)) in r_plus.iter().zip(r_minus.iter()).enumerate() {
                let num = (p - m) / (2.0 * eps);
                let ana = jac[(row, 6 + c)];
                assert!(
                    (num - ana).abs() < 1e-4 * (1.0 + ana.abs()),
                    "landmark col {c}, row {row}: numerical={num}, analytical={ana}"
                );
            }
        }

        // Both rows carry the same scalar penalty, so their Jacobian rows
        // are identical — the deliberate rank-1 property documented on
        // `write_projection_barrier`.
        for c in 0..jac.ncols() {
            assert_eq!(
                jac[(0, c)],
                jac[(1, c)],
                "penalty Jacobian rows differ at column {c}"
            );
        }

        Ok(())
    }
}
