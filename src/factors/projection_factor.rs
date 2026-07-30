//! Generic projection factor for bundle adjustment and SfM.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix2xX, Matrix3xX};
use std::convert::TryFrom;
use std::marker::PhantomData;
use tracing::warn;

use crate::factors::{Factor, OptimizeParams};
use apex_camera_models::CameraModel;
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;

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
    fn evaluate_internal(
        &self,
        pose: &SE3,
        landmarks: &Matrix3xX<f64>,
        camera: &CAM,
        residual: &mut [f64],
        mut jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let n = self.observations.ncols();

        // Process each observation
        for i in 0..n {
            let observation = self.observations.column(i);
            let p_world = landmarks.column(i).into_owned();

            // Transform point to camera frame
            // World-to-camera convention: pose is T_wc where p_cam = R * p_world + t
            // This matches BAL dataset format and ReprojectionFactor
            // pose.act() computes exactly: R * p_world + t = p_cam
            let p_cam = pose.act(&p_world, None, None);

            // Project point (includes all validity checks)
            let uv = match camera.project(&p_cam) {
                Ok(proj) => proj,
                Err(cam_err) => {
                    if self.verbose_cheirality {
                        warn!("Invalid projection for point {}: {}", i, cam_err);
                    }
                    // Invalid projection: use zero residual (matches Ceres convention)
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

        let landmarks: Matrix3xX<f64> = if OP::LANDMARK {
            let flat = params[param_idx];
            let n = flat.len() / 3;
            param_idx += 1;
            Matrix3xX::from_fn(n, |r, c| flat[c * 3 + r])
        } else {
            self.fixed_landmarks
                .clone()
                .unwrap_or_else(|| Matrix3xX::zeros(0))
        };

        let camera: CAM = if OP::INTRINSIC {
            CAM::try_from(params[param_idx])
                .ok()
                .unwrap_or_else(|| self.camera.clone())
        } else {
            self.camera.clone()
        };

        let n = self.observations.ncols();
        assert_eq!(
            landmarks.ncols(),
            n,
            "Number of landmarks ({}) must match observations ({})",
            landmarks.ncols(),
            n
        );

        // Write directly into caller-provided buffers — zero temporary allocation.
        self.evaluate_internal(&pose, &landmarks, &camera, residual, jacobian);
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

        assert!(residual[0].abs() < 1e-10);
        assert!(residual[1].abs() < 1e-10);

        Ok(())
    }
}
