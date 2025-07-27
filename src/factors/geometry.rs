//! Geometric factor types for pose graph optimization
//!
//! This module provides factor types specifically designed for geometric
//! constraints in robotics and computer vision applications. These factors
//! work with manifold types like SE(2), SE(3), SO(2), and SO(3).
//!
//! # Factor Types
//!
//! - `BetweenFactor`: Relative pose constraint between two poses
//! - `RelativePoseFactor`: Specialized between factor for pose graphs
//! - `SE2BetweenFactor`: SE(2)-specific between factor
//! - `SE3BetweenFactor`: SE(3)-specific between factor
//!
//! # Mathematical Background
//!
//! ## Between Factor
//! A between factor constrains the relative transformation between two poses:
//! ```text
//! r(T_i, T_j) = log((T_i^(-1) ⊕ T_j)^(-1) ⊕ z_ij)
//! ```
//! where `T_i`, `T_j` are poses and `z_ij` is the measured relative transformation.
//!
//! The Jacobians are computed using the adjoint representation:
//! ```text
//! ∂r/∂T_i = -Adj(z_ij^(-1) ⊕ (T_i^(-1) ⊕ T_j))
//! ∂r/∂T_j = I
//! ```

use nalgebra::{DVector, DMatrix};
use crate::core::types::{ApexResult, ApexError};
use crate::manifold::{LieGroup, se2::SE2, se3::SE3, so2::SO2, so3::SO3};
use crate::factors::Factor;

/// Generic between factor for relative pose constraints
///
/// This factor implements a constraint between two poses on the same manifold,
/// representing a relative measurement between them. It's the fundamental
/// building block for pose graph optimization.
///
/// # Type Parameters
/// - `M`: The manifold type (SE2, SE3, SO2, SO3, etc.)
///
/// # Mathematical Formulation
/// For poses `T_i`, `T_j` and relative measurement `z_ij`:
/// ```text
/// r(T_i, T_j) = log((T_i^(-1) ⊕ T_j)^(-1) ⊕ z_ij)
/// ```
#[derive(Debug, Clone)]
pub struct BetweenFactor<M: LieGroup> {
    /// Unique identifier for this factor
    id: usize,
    /// Keys of the two poses this factor constrains
    pose_keys: [usize; 2],
    /// Relative measurement between poses
    measurement: M,
    /// Information matrix (inverse covariance)
    information: DMatrix<f64>,
}

impl<M: LieGroup> BetweenFactor<M> {
    /// Create a new between factor
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this factor
    /// * `pose_keys` - Keys of the two poses [from_pose, to_pose]
    /// * `measurement` - Relative measurement from pose_i to pose_j
    /// * `information` - Information matrix (inverse covariance)
    pub fn new(
        id: usize,
        pose_keys: [usize; 2],
        measurement: M,
        information: DMatrix<f64>,
    ) -> Self {
        Self {
            id,
            pose_keys,
            measurement,
            information,
        }
    }

    /// Get the measurement
    pub fn measurement(&self) -> &M {
        &self.measurement
    }

    /// Get the information matrix
    pub fn information(&self) -> &DMatrix<f64> {
        &self.information
    }

    /// Get the pose keys
    pub fn pose_keys(&self) -> [usize; 2] {
        self.pose_keys
    }

    /// Set the measurement
    pub fn set_measurement(&mut self, measurement: M) {
        self.measurement = measurement;
    }

    /// Set the information matrix
    pub fn set_information(&mut self, information: DMatrix<f64>) {
        self.information = information;
    }
}

impl<M: LieGroup + Send + Sync + 'static> Factor for BetweenFactor<M>
where
    M::TangentVector: Into<DVector<f64>>,
    M::JacobianMatrix: Into<DMatrix<f64>>,
{
    fn id(&self) -> usize {
        self.id
    }

    fn key(&self) -> usize {
        self.id
    }

    fn variable_keys(&self) -> &[usize] {
        &self.pose_keys
    }

    fn linearize(&self, variables: &[&dyn std::any::Any]) -> ApexResult<(DVector<f64>, DMatrix<f64>)> {
        if variables.len() != 2 {
            return Err(ApexError::InvalidInput(format!(
                "BetweenFactor expects exactly 2 variables, got {}",
                variables.len()
            )));
        }

        // For now, return a placeholder implementation
        // TODO: Implement proper manifold-aware linearization with Jacobians
        let residual_dim = M::DOF;
        let jacobian_cols = 2 * M::DOF;
        let residual = DVector::zeros(residual_dim);
        let jacobian = DMatrix::zeros(residual_dim, jacobian_cols);
        
        Ok((residual, jacobian))
    }

    fn evaluate(&self, variables: &[&dyn std::any::Any]) -> ApexResult<DVector<f64>> {
        if variables.len() != 2 {
            return Err(ApexError::InvalidInput(format!(
                "BetweenFactor expects exactly 2 variables, got {}",
                variables.len()
            )));
        }

        // For now, return a placeholder implementation
        // TODO: Implement proper manifold-aware evaluation
        let residual_dim = M::DOF;
        let residual = DVector::zeros(residual_dim);
        
        Ok(residual)
    }
}

/// Specialized between factor for SE(2) pose graphs
///
/// This factor is optimized for 2D pose graph optimization problems,
/// providing efficient computation for SE(2) manifold operations.
pub type SE2BetweenFactor = BetweenFactor<SE2>;

impl SE2BetweenFactor {
    /// Create a new SE(2) between factor with identity information
    pub fn new_identity(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SE2,
    ) -> Self {
        let information = DMatrix::identity(3, 3);
        Self::new(id, pose_keys, measurement, information)
    }

    /// Create a new SE(2) between factor with diagonal information
    pub fn new_diagonal(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SE2,
        translation_precision: f64,
        rotation_precision: f64,
    ) -> Self {
        let mut information = DMatrix::zeros(3, 3);
        information[(0, 0)] = translation_precision;
        information[(1, 1)] = translation_precision;
        information[(2, 2)] = rotation_precision;
        Self::new(id, pose_keys, measurement, information)
    }
}

/// Specialized between factor for SE(3) pose graphs
///
/// This factor is optimized for 3D pose graph optimization problems,
/// providing efficient computation for SE(3) manifold operations.
pub type SE3BetweenFactor = BetweenFactor<SE3>;

impl SE3BetweenFactor {
    /// Create a new SE(3) between factor with identity information
    pub fn new_identity(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SE3,
    ) -> Self {
        let information = DMatrix::identity(6, 6);
        Self::new(id, pose_keys, measurement, information)
    }

    /// Create a new SE(3) between factor with diagonal information
    pub fn new_diagonal(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SE3,
        translation_precision: f64,
        rotation_precision: f64,
    ) -> Self {
        let mut information = DMatrix::zeros(6, 6);
        // Translation components (x, y, z)
        for i in 0..3 {
            information[(i, i)] = translation_precision;
        }
        // Rotation components (rx, ry, rz)
        for i in 3..6 {
            information[(i, i)] = rotation_precision;
        }
        Self::new(id, pose_keys, measurement, information)
    }

    /// Create a new SE(3) between factor with separate translation/rotation precision
    pub fn new_split_precision(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SE3,
        translation_precision: &[f64; 3],
        rotation_precision: &[f64; 3],
    ) -> Self {
        let mut information = DMatrix::zeros(6, 6);
        // Translation components
        for i in 0..3 {
            information[(i, i)] = translation_precision[i];
        }
        // Rotation components
        for i in 0..3 {
            information[(i + 3, i + 3)] = rotation_precision[i];
        }
        Self::new(id, pose_keys, measurement, information)
    }
}

/// Relative pose factor (alias for BetweenFactor)
///
/// This is a semantic alias for BetweenFactor to make the intent clearer
/// in pose graph optimization contexts.
pub type RelativePoseFactor<M> = BetweenFactor<M>;

/// Specialized relative pose factor for SE(2)
pub type SE2RelativePoseFactor = SE2BetweenFactor;

/// Specialized relative pose factor for SE(3)
pub type SE3RelativePoseFactor = SE3BetweenFactor;

/// Specialized between factor for SO(2) rotations
pub type SO2BetweenFactor = BetweenFactor<SO2>;

impl SO2BetweenFactor {
    /// Create a new SO(2) between factor with scalar precision
    pub fn new_scalar(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SO2,
        precision: f64,
    ) -> Self {
        let information = DMatrix::from_element(1, 1, precision);
        Self::new(id, pose_keys, measurement, information)
    }
}

/// Specialized between factor for SO(3) rotations
pub type SO3BetweenFactor = BetweenFactor<SO3>;

impl SO3BetweenFactor {
    /// Create a new SO(3) between factor with identity information
    pub fn new_identity(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SO3,
    ) -> Self {
        let information = DMatrix::identity(3, 3);
        Self::new(id, pose_keys, measurement, information)
    }

    /// Create a new SO(3) between factor with diagonal information
    pub fn new_diagonal(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SO3,
        precision: f64,
    ) -> Self {
        let mut information = DMatrix::zeros(3, 3);
        information.fill_diagonal(precision);
        Self::new(id, pose_keys, measurement, information)
    }

    /// Create a new SO(3) between factor with axis-specific precision
    pub fn new_axis_precision(
        id: usize,
        pose_keys: [usize; 2],
        measurement: SO3,
        axis_precision: &[f64; 3],
    ) -> Self {
        let mut information = DMatrix::zeros(3, 3);
        for i in 0..3 {
            information[(i, i)] = axis_precision[i];
        }
        Self::new(id, pose_keys, measurement, information)
    }
}
