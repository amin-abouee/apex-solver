//! Computer vision factor types for bundle adjustment and camera-based optimization
//!
//! This module provides factor types specifically designed for computer vision
//! applications, including bundle adjustment, visual SLAM, and structure from motion.
//! These factors handle camera projection models and 3D-2D correspondences.
//!
//! # Factor Types
//!
//! - `ProjectionFactor`: 3D point to 2D image projection constraint
//! - `ReprojectionFactor`: Reprojection error for bundle adjustment
//! - `StereoFactor`: Stereo camera projection constraint
//! - `BearingFactor`: Bearing-only measurement constraint
//!
//! # Mathematical Background
//!
//! ## Projection Factor
//! A projection factor constrains a 3D point to project to a specific 2D image location:
//! ```text
//! r(T, p) = π(T ⊕ p) - z
//! ```
//! where `T` is the camera pose, `p` is the 3D point, `π` is the projection function,
//! and `z` is the observed 2D measurement.
//!
//! ## Camera Model
//! The standard pinhole camera model is used:
//! ```text
//! π([X, Y, Z]ᵀ) = [fx * X/Z + cx, fy * Y/Z + cy]ᵀ
//! ```

use nalgebra::{DVector, DMatrix, Vector2, Vector3, Matrix2};
use crate::core::types::{ApexResult, ApexError};
use crate::factors::Factor;

/// Camera intrinsic parameters
///
/// Represents the intrinsic calibration parameters of a pinhole camera model.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Radial distortion coefficient k1
    pub k1: f64,
    /// Radial distortion coefficient k2
    pub k2: f64,
    /// Tangential distortion coefficient p1
    pub p1: f64,
    /// Tangential distortion coefficient p2
    pub p2: f64,
}

impl CameraIntrinsics {
    /// Create new camera intrinsics without distortion
    pub fn new_simple(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Create new camera intrinsics with distortion
    pub fn new_with_distortion(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
        }
    }

    /// Project a 3D point to 2D image coordinates
    pub fn project(&self, point_3d: &Vector3<f64>) -> ApexResult<Vector2<f64>> {
        if point_3d.z <= 0.0 {
            return Err(ApexError::InvalidInput(
                "Cannot project point with non-positive Z coordinate".to_string(),
            ));
        }

        let x = point_3d.x / point_3d.z;
        let y = point_3d.y / point_3d.z;

        // Apply distortion if present
        let (x_distorted, y_distorted) = if self.k1 != 0.0 || self.k2 != 0.0 || self.p1 != 0.0 || self.p2 != 0.0 {
            let r2 = x * x + y * y;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            let tangential_x = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let tangential_y = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            
            (x * radial + tangential_x, y * radial + tangential_y)
        } else {
            (x, y)
        };

        let u = self.fx * x_distorted + self.cx;
        let v = self.fy * y_distorted + self.cy;

        Ok(Vector2::new(u, v))
    }

    /// Compute projection Jacobian with respect to 3D point
    pub fn projection_jacobian(&self, point_3d: &Vector3<f64>) -> ApexResult<Matrix2<f64>> {
        if point_3d.z <= 0.0 {
            return Err(ApexError::InvalidInput(
                "Cannot compute Jacobian for point with non-positive Z coordinate".to_string(),
            ));
        }

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;
        let z_inv = 1.0 / z;
        let z_inv2 = z_inv * z_inv;

        // For simplicity, assume no distortion for Jacobian computation
        // TODO: Add distortion Jacobian computation
        let mut jacobian = Matrix2::zeros();
        jacobian[(0, 0)] = self.fx * z_inv;
        jacobian[(0, 1)] = 0.0;
        jacobian[(0, 2)] = -self.fx * x * z_inv2;
        jacobian[(1, 0)] = 0.0;
        jacobian[(1, 1)] = self.fy * z_inv;
        jacobian[(1, 2)] = -self.fy * y * z_inv2;

        Ok(jacobian)
    }
}

/// Projection factor for 3D point to 2D image projection
///
/// This factor constrains a 3D point to project to a specific 2D image location
/// given a camera pose and intrinsic parameters. It's fundamental for bundle
/// adjustment and visual SLAM applications.
///
/// # Mathematical Formulation
/// For camera pose `T`, 3D point `p`, and 2D observation `z`:
/// ```text
/// r(T, p) = π(T ⊕ p) - z
/// ```
/// where `π` is the camera projection function.
#[derive(Debug, Clone)]
pub struct ProjectionFactor {
    /// Unique identifier for this factor
    id: usize,
    /// Key of the camera pose variable
    pose_key: usize,
    /// Key of the 3D point variable
    point_key: usize,
    /// 2D image observation
    observation: Vector2<f64>,
    /// Camera intrinsic parameters
    intrinsics: CameraIntrinsics,
    /// Information matrix (inverse covariance)
    information: Matrix2<f64>,
}

impl ProjectionFactor {
    /// Create a new projection factor
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this factor
    /// * `pose_key` - Key of the camera pose variable
    /// * `point_key` - Key of the 3D point variable
    /// * `observation` - 2D image observation
    /// * `intrinsics` - Camera intrinsic parameters
    /// * `information` - Information matrix (inverse covariance)
    pub fn new(
        id: usize,
        pose_key: usize,
        point_key: usize,
        observation: Vector2<f64>,
        intrinsics: CameraIntrinsics,
        information: Matrix2<f64>,
    ) -> Self {
        Self {
            id,
            pose_key,
            point_key,
            observation,
            intrinsics,
            information,
        }
    }

    /// Create a new projection factor with identity information matrix
    pub fn new_identity(
        id: usize,
        pose_key: usize,
        point_key: usize,
        observation: Vector2<f64>,
        intrinsics: CameraIntrinsics,
    ) -> Self {
        let information = Matrix2::identity();
        Self::new(id, pose_key, point_key, observation, intrinsics, information)
    }

    /// Create a new projection factor with isotropic information matrix
    pub fn new_isotropic(
        id: usize,
        pose_key: usize,
        point_key: usize,
        observation: Vector2<f64>,
        intrinsics: CameraIntrinsics,
        precision: f64,
    ) -> Self {
        let information = Matrix2::identity() * precision;
        Self::new(id, pose_key, point_key, observation, intrinsics, information)
    }

    /// Get the observation
    pub fn observation(&self) -> &Vector2<f64> {
        &self.observation
    }

    /// Get the camera intrinsics
    pub fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// Get the information matrix
    pub fn information(&self) -> &Matrix2<f64> {
        &self.information
    }

    /// Set the observation
    pub fn set_observation(&mut self, observation: Vector2<f64>) {
        self.observation = observation;
    }

    /// Set the camera intrinsics
    pub fn set_intrinsics(&mut self, intrinsics: CameraIntrinsics) {
        self.intrinsics = intrinsics;
    }

    /// Set the information matrix
    pub fn set_information(&mut self, information: Matrix2<f64>) {
        self.information = information;
    }
}

impl Factor for ProjectionFactor {
    fn id(&self) -> usize {
        self.id
    }

    fn key(&self) -> usize {
        self.id
    }

    fn variable_keys(&self) -> &[usize] {
        // Return both pose and point keys
        // Note: This is a simplified implementation
        // TODO: Store variable keys as a member field
        &[self.pose_key, self.point_key]
    }

    fn linearize(&self, variables: &[&dyn std::any::Any]) -> ApexResult<(DVector<f64>, DMatrix<f64>)> {
        if variables.len() != 2 {
            return Err(ApexError::InvalidInput(format!(
                "ProjectionFactor expects exactly 2 variables, got {}",
                variables.len()
            )));
        }

        // For now, return a placeholder implementation
        // TODO: Implement proper projection linearization with Jacobians
        let residual_dim = 2; // 2D projection residual
        let jacobian_cols = 6 + 3; // SE(3) pose (6 DOF) + 3D point (3 DOF)
        let residual = DVector::zeros(residual_dim);
        let jacobian = DMatrix::zeros(residual_dim, jacobian_cols);
        
        Ok((residual, jacobian))
    }

    fn evaluate(&self, variables: &[&dyn std::any::Any]) -> ApexResult<DVector<f64>> {
        if variables.len() != 2 {
            return Err(ApexError::InvalidInput(format!(
                "ProjectionFactor expects exactly 2 variables, got {}",
                variables.len()
            )));
        }

        // For now, return a placeholder implementation
        // TODO: Implement proper projection evaluation
        let residual_dim = 2;
        let residual = DVector::zeros(residual_dim);
        
        Ok(residual)
    }
}

/// Reprojection factor (alias for ProjectionFactor)
///
/// This is a semantic alias for ProjectionFactor to make the intent clearer
/// in bundle adjustment contexts.
pub type ReprojectionFactor = ProjectionFactor;

/// Stereo projection factor for stereo camera systems
///
/// This factor constrains a 3D point to project to specific 2D locations
/// in both left and right camera images of a stereo system.
#[derive(Debug, Clone)]
pub struct StereoFactor {
    /// Unique identifier for this factor
    id: usize,
    /// Key of the camera pose variable
    pose_key: usize,
    /// Key of the 3D point variable
    point_key: usize,
    /// 2D observation in left camera
    left_observation: Vector2<f64>,
    /// 2D observation in right camera
    right_observation: Vector2<f64>,
    /// Left camera intrinsic parameters
    left_intrinsics: CameraIntrinsics,
    /// Right camera intrinsic parameters
    right_intrinsics: CameraIntrinsics,
    /// Baseline between cameras (meters)
    baseline: f64,
    /// Information matrix (inverse covariance) for 4D observation
    information: DMatrix<f64>,
}

impl StereoFactor {
    /// Create a new stereo factor
    pub fn new(
        id: usize,
        pose_key: usize,
        point_key: usize,
        left_observation: Vector2<f64>,
        right_observation: Vector2<f64>,
        left_intrinsics: CameraIntrinsics,
        right_intrinsics: CameraIntrinsics,
        baseline: f64,
        information: DMatrix<f64>,
    ) -> Self {
        Self {
            id,
            pose_key,
            point_key,
            left_observation,
            right_observation,
            left_intrinsics,
            right_intrinsics,
            baseline,
            information,
        }
    }

    /// Create a new stereo factor with identity information matrix
    pub fn new_identity(
        id: usize,
        pose_key: usize,
        point_key: usize,
        left_observation: Vector2<f64>,
        right_observation: Vector2<f64>,
        left_intrinsics: CameraIntrinsics,
        right_intrinsics: CameraIntrinsics,
        baseline: f64,
    ) -> Self {
        let information = DMatrix::identity(4, 4);
        Self::new(
            id,
            pose_key,
            point_key,
            left_observation,
            right_observation,
            left_intrinsics,
            right_intrinsics,
            baseline,
            information,
        )
    }
}

impl Factor for StereoFactor {
    fn id(&self) -> usize {
        self.id
    }

    fn key(&self) -> usize {
        self.id
    }

    fn variable_keys(&self) -> &[usize] {
        // Return both pose and point keys
        // Note: This is a simplified implementation
        &[self.pose_key, self.point_key]
    }

    fn linearize(&self, variables: &[&dyn std::any::Any]) -> ApexResult<(DVector<f64>, DMatrix<f64>)> {
        if variables.len() != 2 {
            return Err(ApexError::InvalidInput(format!(
                "StereoFactor expects exactly 2 variables, got {}",
                variables.len()
            )));
        }

        // For now, return a placeholder implementation
        // TODO: Implement proper stereo projection linearization
        let residual_dim = 4; // 2D projection residual for each camera
        let jacobian_cols = 6 + 3; // SE(3) pose (6 DOF) + 3D point (3 DOF)
        let residual = DVector::zeros(residual_dim);
        let jacobian = DMatrix::zeros(residual_dim, jacobian_cols);
        
        Ok((residual, jacobian))
    }

    fn evaluate(&self, variables: &[&dyn std::any::Any]) -> ApexResult<DVector<f64>> {
        if variables.len() != 2 {
            return Err(ApexError::InvalidInput(format!(
                "StereoFactor expects exactly 2 variables, got {}",
                variables.len()
            )));
        }

        // For now, return a placeholder implementation
        // TODO: Implement proper stereo projection evaluation
        let residual_dim = 4;
        let residual = DVector::zeros(residual_dim);
        
        Ok(residual)
    }
}
