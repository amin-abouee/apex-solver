//! Manifold variables for geometric optimization
//!
//! This module provides variable types that live on manifolds,
//! integrating with the existing manifold operations from the manifold module.

use crate::manifold::se2::SE2;
use crate::manifold::se3::{SE3, SE3Tangent};
use crate::manifold::so2::SO2;
use crate::manifold::so3::{SO3, SO3Tangent};
use crate::variables::{Variable, VariableDomain, VariableState};
use nalgebra::{DVector, Isometry2, Isometry3, UnitComplex, UnitQuaternion, Vector2, Vector3};

/// SE(3) variable for 3D poses (position + orientation)
#[derive(Debug, Clone)]
pub struct SE3Variable {
    id: usize,
    pose: SE3,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl SE3Variable {
    /// Create a new SE(3) variable
    pub fn new(id: usize, pose: SE3) -> Self {
        Self {
            id,
            pose,
            state: VariableState::Free,
            domain: VariableDomain::Manifold {
                manifold_type: "SE3".to_string(),
            },
            name: None,
        }
    }

    /// Create SE(3) variable from translation and rotation
    pub fn from_translation_rotation(
        id: usize,
        translation: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
    ) -> Self {
        let isometry = Isometry3::from_parts(translation.into(), rotation);
        let pose = SE3::from_isometry(isometry);
        Self::new(id, pose)
    }

    /// Create identity SE(3) variable
    pub fn identity_with_id(id: usize) -> Self {
        Self::new(id, SE3::identity())
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the SE(3) pose
    pub fn pose(&self) -> &SE3 {
        &self.pose
    }

    /// Set the SE(3) pose
    pub fn set_pose(&mut self, pose: SE3) {
        self.pose = pose;
    }

    /// Get translation component
    pub fn translation(&self) -> Vector3<f64> {
        self.pose.translation()
    }

    /// Get rotation component
    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.pose.rotation_quaternion()
    }
}

impl Variable for SE3Variable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        6 // SE(3) degrees of freedom
    }

    fn identity() -> Self {
        Self::new(0, SE3::identity())
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            pose: self.pose.inverse(None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            pose: self.pose.compose(&other.pose, None, None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(delta.len(), 6, "SE(3) variable expects 6-dimensional delta");
        let tangent = SE3Tangent::new(
            Vector3::new(delta[0], delta[1], delta[2]),
            Vector3::new(delta[3], delta[4], delta[5]),
        );
        let pose = tangent.exp(None);
        Self::new(0, pose)
    }

    fn log(&self) -> DVector<f64> {
        let tangent = self.pose.log(None);
        let rho = tangent.rho();
        let theta = tangent.theta();
        DVector::from_vec(vec![rho.x, rho.y, rho.z, theta.x, theta.y, theta.z])
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn domain(&self) -> &VariableDomain {
        &self.domain
    }

    fn set_domain(&mut self, domain: VariableDomain) {
        self.domain = domain;
    }

    fn state(&self) -> VariableState {
        self.state
    }

    fn set_state(&mut self, state: VariableState) {
        self.state = state;
    }

    fn is_valid(&self) -> bool {
        self.pose.is_valid(1e-6)
    }
}

/// SO(3) variable for 3D rotations
#[derive(Debug, Clone)]
pub struct SO3Variable {
    id: usize,
    rotation: SO3,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl SO3Variable {
    /// Create a new SO(3) variable
    pub fn new(id: usize, rotation: SO3) -> Self {
        Self {
            id,
            rotation,
            state: VariableState::Free,
            domain: VariableDomain::Manifold {
                manifold_type: "SO3".to_string(),
            },
            name: None,
        }
    }

    /// Create SO(3) variable from quaternion
    pub fn from_quaternion(id: usize, quaternion: UnitQuaternion<f64>) -> Self {
        let rotation = SO3::from_quaternion(quaternion);
        Self::new(id, rotation)
    }

    /// Create identity SO(3) variable
    pub fn identity_with_id(id: usize) -> Self {
        Self::new(id, SO3::identity())
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the SO(3) rotation
    pub fn rotation(&self) -> &SO3 {
        &self.rotation
    }

    /// Set the SO(3) rotation
    pub fn set_rotation(&mut self, rotation: SO3) {
        self.rotation = rotation;
    }

    /// Get as quaternion
    pub fn quaternion(&self) -> UnitQuaternion<f64> {
        self.rotation.to_quaternion()
    }
}

impl Variable for SO3Variable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        3 // SO(3) degrees of freedom
    }

    fn identity() -> Self {
        Self::new(0, SO3::identity())
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            rotation: self.rotation.inverse(None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            rotation: self.rotation.compose(&other.rotation, None, None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(delta.len(), 3, "SO(3) variable expects 3-dimensional delta");
        let tangent = SO3Tangent::new(Vector3::new(delta[0], delta[1], delta[2]));
        let rotation = tangent.exp(None);
        Self::new(0, rotation)
    }

    fn log(&self) -> DVector<f64> {
        let tangent = self.rotation.log(None);
        let coeffs = tangent.coeffs();
        DVector::from_vec(vec![coeffs.x, coeffs.y, coeffs.z])
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn domain(&self) -> &VariableDomain {
        &self.domain
    }

    fn set_domain(&mut self, domain: VariableDomain) {
        self.domain = domain;
    }

    fn state(&self) -> VariableState {
        self.state
    }

    fn set_state(&mut self, state: VariableState) {
        self.state = state;
    }

    fn is_valid(&self) -> bool {
        self.rotation.is_valid(1e-6)
    }
}

/// SE(2) variable for 2D poses (position + orientation)
#[derive(Debug, Clone)]
pub struct SE2Variable {
    id: usize,
    pose: SE2,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl SE2Variable {
    /// Create a new SE(2) variable
    pub fn new(id: usize, pose: SE2) -> Self {
        Self {
            id,
            pose,
            state: VariableState::Free,
            domain: VariableDomain::Manifold {
                manifold_type: "SE2".to_string(),
            },
            name: None,
        }
    }

    /// Create SE(2) variable from translation and rotation
    pub fn from_translation_rotation(
        id: usize,
        translation: Vector2<f64>,
        rotation: UnitComplex<f64>,
    ) -> Self {
        let isometry = Isometry2::from_parts(translation.into(), rotation);
        let pose = SE2::from_isometry(isometry);
        Self::new(id, pose)
    }

    /// Create identity SE(2) variable
    pub fn identity_with_id(id: usize) -> Self {
        Self::new(id, SE2::identity())
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the SE(2) pose
    pub fn pose(&self) -> &SE2 {
        &self.pose
    }

    /// Set the SE(2) pose
    pub fn set_pose(&mut self, pose: SE2) {
        self.pose = pose;
    }

    /// Get translation component
    pub fn translation(&self) -> Vector2<f64> {
        self.pose.translation()
    }

    /// Get rotation component
    pub fn rotation(&self) -> UnitComplex<f64> {
        self.pose.rotation_complex()
    }
}

impl Variable for SE2Variable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        3 // SE(2) degrees of freedom
    }

    fn identity() -> Self {
        Self::new(0, SE2::identity())
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            pose: self.pose.inverse(None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            pose: self.pose.compose(&other.pose, None, None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(delta.len(), 3, "SE(2) variable expects 3-dimensional delta");
        // For SE(2), we need to create the tangent vector and apply exp
        // This is a simplified implementation - in practice you'd use the proper SE2 tangent
        let translation = Vector2::new(delta[0], delta[1]);
        let rotation = UnitComplex::from_angle(delta[2]);
        let isometry = Isometry2::from_parts(translation.into(), rotation);
        let pose = SE2::from_isometry(isometry);
        Self::new(0, pose)
    }

    fn log(&self) -> DVector<f64> {
        // For SE(2), extract translation and rotation angle
        let translation = self.pose.translation();
        let angle = self.pose.rotation_angle();
        DVector::from_vec(vec![translation.x, translation.y, angle])
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn domain(&self) -> &VariableDomain {
        &self.domain
    }

    fn set_domain(&mut self, domain: VariableDomain) {
        self.domain = domain;
    }

    fn state(&self) -> VariableState {
        self.state
    }

    fn set_state(&mut self, state: VariableState) {
        self.state = state;
    }

    fn is_valid(&self) -> bool {
        self.pose.is_valid(1e-6)
    }
}

/// SO(2) variable for 2D rotations
#[derive(Debug, Clone)]
pub struct SO2Variable {
    id: usize,
    rotation: SO2,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl SO2Variable {
    /// Create a new SO(2) variable
    pub fn new(id: usize, rotation: SO2) -> Self {
        Self {
            id,
            rotation,
            state: VariableState::Free,
            domain: VariableDomain::Manifold {
                manifold_type: "SO2".to_string(),
            },
            name: None,
        }
    }

    /// Create SO(2) variable from angle
    pub fn from_angle(id: usize, angle: f64) -> Self {
        let rotation = SO2::from_angle(angle);
        Self::new(id, rotation)
    }

    /// Create identity SO(2) variable
    pub fn identity_with_id(id: usize) -> Self {
        Self::new(id, SO2::identity())
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the SO(2) rotation
    pub fn rotation(&self) -> &SO2 {
        &self.rotation
    }

    /// Set the SO(2) rotation
    pub fn set_rotation(&mut self, rotation: SO2) {
        self.rotation = rotation;
    }

    /// Get angle
    pub fn angle(&self) -> f64 {
        self.rotation.angle()
    }
}

impl Variable for SO2Variable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        1 // SO(2) degrees of freedom
    }

    fn identity() -> Self {
        Self::new(0, SO2::identity())
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            rotation: self.rotation.inverse(None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            rotation: self.rotation.compose(&other.rotation, None, None),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(delta.len(), 1, "SO(2) variable expects 1-dimensional delta");
        let rotation = SO2::from_angle(delta[0]);
        Self::new(0, rotation)
    }

    fn log(&self) -> DVector<f64> {
        DVector::from_element(1, self.rotation.angle())
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn domain(&self) -> &VariableDomain {
        &self.domain
    }

    fn set_domain(&mut self, domain: VariableDomain) {
        self.domain = domain;
    }

    fn state(&self) -> VariableState {
        self.state
    }

    fn set_state(&mut self, state: VariableState) {
        self.state = state;
    }

    fn is_valid(&self) -> bool {
        self.rotation.is_valid(1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_se3_variable_basic_operations() {
        let pose = SE3::identity();
        let var = SE3Variable::new(1, pose);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 6); // SE(3) degrees of freedom
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = SE3Variable::identity();
        assert_eq!(identity.dim(), 6);

        let inverse = var.inverse();
        assert_eq!(inverse.dim(), 6);

        let other = SE3Variable::new(2, SE3::identity());
        let composed = var.compose(&other);
        assert_eq!(composed.dim(), 6);

        // Test exp/log operations
        let delta = DVector::zeros(6);
        let exp_result = SE3Variable::exp(&delta);
        assert_eq!(exp_result.dim(), 6);

        let log_result = var.log();
        assert_eq!(log_result.len(), 6);

        // Test plus/minus operations
        let plus_result = var.plus(&delta);
        assert_eq!(plus_result.dim(), 6);

        let minus_result = var.minus(&other);
        assert_eq!(minus_result.len(), 6);
    }

    #[test]
    fn test_so3_variable_basic_operations() {
        let rotation = SO3::identity();
        let var = SO3Variable::new(1, rotation);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 3); // SO(3) degrees of freedom
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = SO3Variable::identity();
        assert_eq!(identity.dim(), 3);

        let inverse = var.inverse();
        assert_eq!(inverse.dim(), 3);

        let other = SO3Variable::new(2, SO3::identity());
        let composed = var.compose(&other);
        assert_eq!(composed.dim(), 3);

        // Test exp/log operations
        let delta = DVector::zeros(3);
        let exp_result = SO3Variable::exp(&delta);
        assert_eq!(exp_result.dim(), 3);

        let log_result = var.log();
        assert_eq!(log_result.len(), 3);

        // Test plus/minus operations
        let plus_result = var.plus(&delta);
        assert_eq!(plus_result.dim(), 3);

        let minus_result = var.minus(&other);
        assert_eq!(minus_result.len(), 3);
    }

    #[test]
    fn test_se2_variable_basic_operations() {
        let pose = SE2::identity();
        let var = SE2Variable::new(1, pose);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 3); // SE(2) degrees of freedom
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = SE2Variable::identity();
        assert_eq!(identity.dim(), 3);

        let inverse = var.inverse();
        assert_eq!(inverse.dim(), 3);

        let other = SE2Variable::new(2, SE2::identity());
        let composed = var.compose(&other);
        assert_eq!(composed.dim(), 3);

        // Test exp/log operations
        let delta = DVector::zeros(3);
        let exp_result = SE2Variable::exp(&delta);
        assert_eq!(exp_result.dim(), 3);

        let log_result = var.log();
        assert_eq!(log_result.len(), 3);
    }

    #[test]
    fn test_so2_variable_basic_operations() {
        let rotation = SO2::identity();
        let var = SO2Variable::new(1, rotation);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 1); // SO(2) degrees of freedom
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = SO2Variable::identity();
        assert_eq!(identity.dim(), 1);

        let inverse = var.inverse();
        assert_eq!(inverse.dim(), 1);

        let other = SO2Variable::new(2, SO2::identity());
        let composed = var.compose(&other);
        assert_eq!(composed.dim(), 1);

        // Test exp/log operations
        let delta = DVector::zeros(1);
        let exp_result = SO2Variable::exp(&delta);
        assert_eq!(exp_result.dim(), 1);

        let log_result = var.log();
        assert_eq!(log_result.len(), 1);
    }
}
