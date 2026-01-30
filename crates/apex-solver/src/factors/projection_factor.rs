//! Generic projection factor for bundle adjustment and SfM.

use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3, Matrix3xX, Vector3};
use std::marker::PhantomData;
use tracing::warn;

use crate::factors::Factor;
use apex_camera_models::{CameraModel, OptimizeParams};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;

/// Compute skew-symmetric matrix from vector.
/// [v]× such that [v]× * w = v × w (cross product)
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Trait for optimization configuration.
///
/// This trait allows accessing the compile-time boolean flags for
/// parameter optimization (pose, landmarks, intrinsics).
pub trait OptimizationConfig: Send + Sync + 'static + Clone + Copy + Default {
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
/// - `OP`: Optimization configuration (e.g., [`BundleAdjustment`](apex_camera_models::BundleAdjustment))
///
/// # Examples
///
/// ```
/// use apex_solver::factors::projection_factor::ProjectionFactor;
/// use apex_solver::factors::camera::{PinholeCamera, BundleAdjustment};
/// use nalgebra::{Matrix2xX, Vector2};
///
/// let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
/// let observations = Matrix2xX::from_columns(&[
///     Vector2::new(100.0, 150.0),
///     Vector2::new(200.0, 250.0),
/// ]);
///
/// // Bundle adjustment: optimize pose + landmarks (intrinsics fixed)
/// let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
///     ProjectionFactor::new(observations, camera);
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
    /// # use apex_solver::factors::projection_factor::ProjectionFactor;
    /// # use apex_solver::factors::camera::{PinholeCamera, BundleAdjustment};
    /// # use nalgebra::{Matrix2xX, Vector2};
    /// # let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    /// # let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
    /// let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
    ///     ProjectionFactor::new(observations, camera);
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
    /// # use apex_solver::factors::projection_factor::ProjectionFactor;
    /// # use apex_solver::factors::camera::{PinholeCamera, BundleAdjustment};
    /// # use apex_solver::manifold::se3::SE3;
    /// # use nalgebra::{Matrix2xX, Vector2};
    /// # let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    /// # let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
    /// # let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> = ProjectionFactor::new(observations, camera);
    /// let factor = factor.with_fixed_pose(SE3::identity());
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
    /// # use apex_solver::factors::projection_factor::ProjectionFactor;
    /// # use apex_solver::factors::camera::{PinholeCamera, BundleAdjustment};
    /// # use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
    /// # let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    /// # let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);
    /// # let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> = ProjectionFactor::new(observations, camera);
    /// # let landmarks = Matrix3xX::from_columns(&[Vector3::new(0.1, 0.2, 1.0)]);
    /// let factor = factor.with_fixed_landmarks(landmarks);
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

    /// Internal evaluation function that computes residuals and Jacobians.
    fn evaluate_internal(
        &self,
        pose: &SE3,
        landmarks: &Matrix3xX<f64>,
        camera: &CAM,
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let n = self.observations.ncols();
        let residual_dim = n * 2;

        // Allocate residuals
        let mut residuals = DVector::zeros(residual_dim);

        // Calculate total Jacobian dimension
        let mut jacobian_cols = 0;
        if OP::POSE {
            jacobian_cols += 6; // SE3 tangent space
        }
        if OP::LANDMARK {
            jacobian_cols += n * 3; // 3D landmarks
        }
        if OP::INTRINSIC {
            jacobian_cols += CAM::INTRINSIC_DIM;
        }

        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, jacobian_cols))
        } else {
            None
        };

        // Process each observation
        for i in 0..n {
            let observation = self.observations.column(i);
            let p_world = landmarks.column(i).into_owned();

            // Transform point to camera frame
            // World-to-camera convention: pose is T_wc where p_cam = R * p_world + t
            // This matches BAL dataset format and ReprojectionFactor
            // pose.act() computes exactly: R * p_world + t = p_cam
            let p_cam = pose.act(&p_world, None, None);

            // Check validity and project
            if !camera.is_valid_point(&p_cam) {
                if self.verbose_cheirality {
                    warn!(
                        "Point {} behind camera or invalid: p_cam = ({}, {}, {})",
                        i, p_cam.x, p_cam.y, p_cam.z
                    );
                }
                // Invalid projection: use zero residual (matches Ceres convention)
                residuals[i * 2] = 0.0;
                residuals[i * 2 + 1] = 0.0;
                // Jacobian rows remain zero
                continue;
            }

            // Project point
            let uv = match camera.project(&p_cam) {
                Some(proj) => proj,
                None => {
                    if self.verbose_cheirality {
                        warn!("Projection failed for point {}", i);
                    }
                    residuals[i * 2] = 0.0;
                    residuals[i * 2 + 1] = 0.0;
                    continue;
                }
            };

            // Compute residual
            residuals[i * 2] = uv.x - observation.x;
            residuals[i * 2 + 1] = uv.y - observation.y;

            // Compute Jacobians if requested
            if let Some(ref mut jac) = jacobian_matrix {
                let mut col_offset = 0;

                // Jacobian w.r.t. pose (world-to-camera convention)
                // This matches ReprojectionFactor exactly
                if OP::POSE {
                    // Jacobian of projection w.r.t. point in camera frame
                    let d_uv_d_pcam = camera.jacobian_point(&p_cam);

                    // Jacobian of p_cam w.r.t. pose for world-to-camera convention
                    // p_cam = R * p_world + t
                    //
                    // Using right perturbation: pose' = pose ∘ Exp([δρ; δθ])
                    // For small perturbations:
                    //   R' = R * Exp(δθ) ≈ R * (I + [δθ]×)
                    //   t' = t + R * δρ
                    //
                    // So:
                    //   p_cam' = R' * p_world + t' = R*(I+[δθ]×)*p_world + t + R*δρ
                    //          = R*p_world + t + R*[δθ]×*p_world + R*δρ
                    //          = p_cam + R*(δρ + [δθ]× * p_world)
                    //          = p_cam + R*(δρ - [p_world]× * δθ)
                    //          = p_cam + R*δρ - R*[p_world]×*δθ
                    //
                    // ∂p_cam/∂δρ = R
                    // ∂p_cam/∂δθ = -R * [p_world]×
                    let rotation = pose.rotation_so3().rotation_matrix();
                    let p_world_skew = skew_symmetric(&p_world);

                    let d_pcam_d_pose = nalgebra::SMatrix::<f64, 3, 6>::from_fn(|r, c| {
                        if c < 3 {
                            // Translation part: R
                            rotation[(r, c)]
                        } else {
                            // Rotation part: -R * [p_world]×
                            let col = c - 3;
                            let mut sum = 0.0;
                            for k in 0..3 {
                                sum += rotation[(r, k)] * p_world_skew[(k, col)];
                            }
                            -sum
                        }
                    });

                    // Chain rule: ∂uv/∂pose = ∂uv/∂p_cam * ∂p_cam/∂pose
                    let d_uv_d_pose = d_uv_d_pcam * d_pcam_d_pose;

                    for r in 0..2 {
                        for c in 0..6 {
                            jac[(i * 2 + r, col_offset + c)] = d_uv_d_pose[(r, c)];
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
                            jac[(i * 2 + r, col_offset + i * 3 + c)] = d_uv_d_landmark[(r, c)];
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
                            jac[(i * 2 + r, col_offset + c)] = d_uv_d_intrinsics[(r, c)];
                        }
                    }
                }
            }
        }

        (residuals, jacobian_matrix)
    }
}

// Factor trait implementation with generic dispatch
impl<CAM, OP> Factor for ProjectionFactor<CAM, OP>
where
    CAM: CameraModel,
    OP: OptimizationConfig,
{
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let mut param_idx = 0;

        // Get pose (from params if optimized, else from fixed_pose)
        let pose: SE3 = if OP::POSE {
            if param_idx >= params.len() {
                panic!("Missing pose parameter");
            }
            let p = SE3::from(params[param_idx].clone());
            param_idx += 1;
            p
        } else {
            self.fixed_pose.clone().unwrap_or_else(|| {
                panic!("Fixed pose required when POSE = false. Use with_fixed_pose() when creating ProjectionFactor with OptimizeParams<false, _, _>")
            })
        };

        // Get landmarks (3×N)
        let landmarks: Matrix3xX<f64> = if OP::LANDMARK {
            if param_idx >= params.len() {
                panic!("Missing landmark parameters");
            }
            let flat = &params[param_idx];
            let n = flat.len() / 3;
            param_idx += 1;
            Matrix3xX::from_fn(n, |r, c| flat[c * 3 + r])
        } else {
            self.fixed_landmarks.clone().unwrap_or_else(|| {
                panic!("Fixed landmarks required when LANDMARK = false. Use with_fixed_landmarks() when creating ProjectionFactor with OptimizeParams<_, false, _>")
            })
        };

        // Get camera intrinsics
        let camera: CAM = if OP::INTRINSIC {
            if param_idx >= params.len() {
                panic!("Missing intrinsic parameters");
            }
            CAM::from_params(params[param_idx].as_slice())
        } else {
            self.camera.clone()
        };

        // Verify dimensions
        let n = self.observations.ncols();
        assert_eq!(
            landmarks.ncols(),
            n,
            "Number of landmarks ({}) must match observations ({})",
            landmarks.ncols(),
            n
        );

        // Compute residuals and Jacobians
        self.evaluate_internal(&pose, &landmarks, &camera, compute_jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.observations.ncols() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_camera_models::{BundleAdjustment, OnlyIntrinsics, PinholeCamera, SelfCalibration};
    use nalgebra::{Vector2, Vector3};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_projection_factor_creation() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);

        // Bundle adjustment: optimize pose + landmarks (intrinsics fixed)
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        assert_eq!(factor.num_observations(), 1);
        assert_eq!(factor.get_dimension(), 2);
    }

    #[test]
    fn test_bundle_adjustment_factor() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);

        // Create known landmark and observation
        // World-to-camera convention: p_cam = R * p_world + t
        // With identity pose: p_cam = p_world
        let p_world = Vector3::new(0.1, 0.2, 1.0);
        let pose = SE3::identity();

        // Project to get observation using world-to-camera convention
        let p_cam = pose.act(&p_world, None, None);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        let observations = Matrix2xX::from_columns(&[uv]);
        let _landmarks = Matrix3xX::from_columns(&[p_world]);

        // Bundle adjustment: optimize pose + landmarks (intrinsics fixed)
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        // Linearize
        let pose_vec: DVector<f64> = pose.clone().into();
        let landmarks_vec = DVector::from_vec(vec![p_world.x, p_world.y, p_world.z]);
        let params = vec![pose_vec, landmarks_vec];

        let (residual, jacobian) = factor.linearize(&params, true);

        // Residual should be near zero
        assert!(residual.norm() < 1e-10, "Residual: {:?}", residual);

        // Jacobian dimensions: 2×(6+3) = 2×9
        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 9);

        Ok(())
    }

    #[test]
    fn test_self_calibration_factor() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let p_world = Vector3::new(0.1, 0.2, 1.0);
        let pose = SE3::identity();

        // Get observation using world-to-camera convention
        let p_cam = pose.act(&p_world, None, None);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        let observations = Matrix2xX::from_columns(&[uv]);
        let factor: ProjectionFactor<PinholeCamera, SelfCalibration> =
            ProjectionFactor::new(observations, camera);

        // Linearize with all parameters
        let pose_vec: DVector<f64> = pose.into();
        let landmarks_vec = DVector::from_vec(vec![p_world.x, p_world.y, p_world.z]);
        let intrinsics_vec = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
        let params = vec![pose_vec, landmarks_vec, intrinsics_vec];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert!(residual.norm() < 1e-10);

        // Jacobian dimensions: 2×(6+3+4) = 2×13
        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 13);

        Ok(())
    }

    #[test]
    fn test_calibration_factor() -> TestResult {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let pose = SE3::identity();
        let p_world = Vector3::new(0.1, 0.2, 1.0);

        // World-to-camera convention
        let p_cam = pose.act(&p_world, None, None);
        let uv = camera.project(&p_cam).ok_or("Projection failed")?;

        let observations = Matrix2xX::from_columns(&[uv]);
        let landmarks = Matrix3xX::from_columns(&[p_world]);

        let factor: ProjectionFactor<PinholeCamera, OnlyIntrinsics> =
            ProjectionFactor::new(observations, camera)
                .with_fixed_pose(pose)
                .with_fixed_landmarks(landmarks);

        // Only intrinsics are optimized
        let intrinsics_vec = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
        let params = vec![intrinsics_vec];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert!(residual.norm() < 1e-10);

        // Jacobian dimensions: 2×4 (only intrinsics)
        let jac = jacobian.ok_or("Jacobian should be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 4);

        Ok(())
    }

    #[test]
    fn test_invalid_projection_handling() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let observations = Matrix2xX::from_columns(&[Vector2::new(100.0, 150.0)]);

        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(observations, camera).with_verbose_cheirality();

        let pose = SE3::identity();
        // Landmark behind camera
        let _landmarks = Matrix3xX::from_columns(&[Vector3::new(0.0, 0.0, -1.0)]);

        let pose_vec: DVector<f64> = pose.into();
        let landmarks_vec = DVector::from_vec(vec![0.0, 0.0, -1.0]);
        let params = vec![pose_vec, landmarks_vec];

        let (residual, _) = factor.linearize(&params, false);

        // Invalid projections should have zero residual (Ceres convention)
        assert!(residual[0].abs() < 1e-10);
        assert!(residual[1].abs() < 1e-10);
    }
}
