//! Geometric variables for computer vision and robotics
//!
//! This module provides variable types for common geometric entities
//! used in computer vision and robotics applications.

use crate::variables::{Variable, VariableDomain, VariableState};
use nalgebra::{DVector, Point2, Point3};

/// A 2D point variable for landmark positions, image features, etc.
#[derive(Debug, Clone)]
pub struct Point2DVariable {
    id: usize,
    point: Point2<f64>,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl Point2DVariable {
    /// Create a new 2D point variable
    pub fn new(id: usize, point: Point2<f64>) -> Self {
        Self {
            id,
            point,
            state: VariableState::Free,
            domain: VariableDomain::Unconstrained,
            name: None,
        }
    }

    /// Create a 2D point variable from coordinates
    pub fn from_coords(id: usize, x: f64, y: f64) -> Self {
        Self::new(id, Point2::new(x, y))
    }

    /// Create a 2D point variable with bounds
    pub fn with_bounds(
        id: usize,
        point: Point2<f64>,
        min_point: Point2<f64>,
        max_point: Point2<f64>,
    ) -> Self {
        let domain = VariableDomain::Box {
            lower: DVector::from_vec(vec![min_point.x, min_point.y]),
            upper: DVector::from_vec(vec![max_point.x, max_point.y]),
        };
        Self {
            id,
            point,
            state: VariableState::Free,
            domain,
            name: None,
        }
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the 2D point
    pub fn point(&self) -> &Point2<f64> {
        &self.point
    }

    /// Set the 2D point
    pub fn set_point(&mut self, point: Point2<f64>) {
        self.point = point;
    }

    /// Get x coordinate
    pub fn x(&self) -> f64 {
        self.point.x
    }

    /// Get y coordinate
    pub fn y(&self) -> f64 {
        self.point.y
    }

    /// Set coordinates
    pub fn set_coords(&mut self, x: f64, y: f64) {
        self.point = Point2::new(x, y);
    }
}

impl Variable for Point2DVariable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        2
    }

    fn identity() -> Self {
        Self::new(0, Point2::origin())
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            point: Point2::new(-self.point.x, -self.point.y),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            point: Point2::new(self.point.x + other.point.x, self.point.y + other.point.y),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(
            delta.len(),
            2,
            "Point2D variable expects 2-dimensional delta"
        );
        Self::new(0, Point2::new(delta[0], delta[1]))
    }

    fn log(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.point.x, self.point.y])
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
        match &self.domain {
            VariableDomain::Unconstrained => true,
            VariableDomain::Box { lower, upper } => {
                self.point.x >= lower[0]
                    && self.point.x <= upper[0]
                    && self.point.y >= lower[1]
                    && self.point.y <= upper[1]
            }
            VariableDomain::Manifold { .. } => true,
        }
    }
}

/// A 3D point variable for landmark positions, 3D features, etc.
#[derive(Debug, Clone)]
pub struct Point3DVariable {
    id: usize,
    point: Point3<f64>,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl Point3DVariable {
    /// Create a new 3D point variable
    pub fn new(id: usize, point: Point3<f64>) -> Self {
        Self {
            id,
            point,
            state: VariableState::Free,
            domain: VariableDomain::Unconstrained,
            name: None,
        }
    }

    /// Create a 3D point variable from coordinates
    pub fn from_coords(id: usize, x: f64, y: f64, z: f64) -> Self {
        Self::new(id, Point3::new(x, y, z))
    }

    /// Create a 3D point variable with bounds
    pub fn with_bounds(
        id: usize,
        point: Point3<f64>,
        min_point: Point3<f64>,
        max_point: Point3<f64>,
    ) -> Self {
        let domain = VariableDomain::Box {
            lower: DVector::from_vec(vec![min_point.x, min_point.y, min_point.z]),
            upper: DVector::from_vec(vec![max_point.x, max_point.y, max_point.z]),
        };
        Self {
            id,
            point,
            state: VariableState::Free,
            domain,
            name: None,
        }
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the 3D point
    pub fn point(&self) -> &Point3<f64> {
        &self.point
    }

    /// Set the 3D point
    pub fn set_point(&mut self, point: Point3<f64>) {
        self.point = point;
    }

    /// Get x coordinate
    pub fn x(&self) -> f64 {
        self.point.x
    }

    /// Get y coordinate
    pub fn y(&self) -> f64 {
        self.point.y
    }

    /// Get z coordinate
    pub fn z(&self) -> f64 {
        self.point.z
    }

    /// Set coordinates
    pub fn set_coords(&mut self, x: f64, y: f64, z: f64) {
        self.point = Point3::new(x, y, z);
    }
}

impl Variable for Point3DVariable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        3
    }

    fn identity() -> Self {
        Self::new(0, Point3::origin())
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            point: Point3::new(-self.point.x, -self.point.y, -self.point.z),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            point: Point3::new(
                self.point.x + other.point.x,
                self.point.y + other.point.y,
                self.point.z + other.point.z,
            ),
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(
            delta.len(),
            3,
            "Point3D variable expects 3-dimensional delta"
        );
        Self::new(0, Point3::new(delta[0], delta[1], delta[2]))
    }

    fn log(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.point.x, self.point.y, self.point.z])
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
        match &self.domain {
            VariableDomain::Unconstrained => true,
            VariableDomain::Box { lower, upper } => {
                self.point.x >= lower[0]
                    && self.point.x <= upper[0]
                    && self.point.y >= lower[1]
                    && self.point.y <= upper[1]
                    && self.point.z >= lower[2]
                    && self.point.z <= upper[2]
            }
            VariableDomain::Manifold { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_variable_basic_operations() {
        let point = Point2::new(1.0, 2.0);
        let var = Point2DVariable::new(1, point);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 2);
        assert_eq!(var.x(), 1.0);
        assert_eq!(var.y(), 2.0);
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = Point2DVariable::identity();
        assert_eq!(identity.x(), 0.0);
        assert_eq!(identity.y(), 0.0);

        let inverse = var.inverse();
        assert_eq!(inverse.x(), -1.0);
        assert_eq!(inverse.y(), -2.0);

        let other = Point2DVariable::from_coords(2, 3.0, 4.0);
        let composed = var.compose(&other);
        assert_eq!(composed.x(), 4.0);
        assert_eq!(composed.y(), 6.0);

        // Test exp/log operations
        let delta = DVector::from_vec(vec![0.5, 1.0]);
        let exp_result = Point2DVariable::exp(&delta);
        assert_eq!(exp_result.x(), 0.5);
        assert_eq!(exp_result.y(), 1.0);

        let log_result = var.log();
        assert_eq!(log_result[0], 1.0);
        assert_eq!(log_result[1], 2.0);
    }

    #[test]
    fn test_point3d_variable_basic_operations() {
        let point = Point3::new(1.0, 2.0, 3.0);
        let var = Point3DVariable::new(1, point);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 3);
        assert_eq!(var.x(), 1.0);
        assert_eq!(var.y(), 2.0);
        assert_eq!(var.z(), 3.0);
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = Point3DVariable::identity();
        assert_eq!(identity.x(), 0.0);
        assert_eq!(identity.y(), 0.0);
        assert_eq!(identity.z(), 0.0);

        let inverse = var.inverse();
        assert_eq!(inverse.x(), -1.0);
        assert_eq!(inverse.y(), -2.0);
        assert_eq!(inverse.z(), -3.0);

        let other = Point3DVariable::from_coords(2, 4.0, 5.0, 6.0);
        let composed = var.compose(&other);
        assert_eq!(composed.x(), 5.0);
        assert_eq!(composed.y(), 7.0);
        assert_eq!(composed.z(), 9.0);

        // Test exp/log operations
        let delta = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let exp_result = Point3DVariable::exp(&delta);
        assert_eq!(exp_result.x(), 0.1);
        assert_eq!(exp_result.y(), 0.2);
        assert_eq!(exp_result.z(), 0.3);

        let log_result = var.log();
        assert_eq!(log_result[0], 1.0);
        assert_eq!(log_result[1], 2.0);
        assert_eq!(log_result[2], 3.0);
    }
}
