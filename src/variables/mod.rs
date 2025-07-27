//! Variable implementations for the factor graph
//!
//! This module provides a comprehensive set of variable types for optimization problems,
//! particularly focused on computer vision and robotics applications. The module is
//! organized into submodules for different categories of variables.
//!
//! # Module Structure
//!
//! - `euclidean`: Basic Euclidean space variables (scalars, vectors)
//! - `geometry`: Geometric variables (points, poses, rotations)
//! - `camera`: Camera-related variables (intrinsics, poses)
//! - `manifold`: Variables living on manifolds (SE3, SO3, etc.)

use crate::core::types::ApexResult;
use nalgebra::DVector;
use std::fmt;

pub mod euclidean;
pub mod geometry;
pub mod manifold;

// Re-export commonly used types
pub use euclidean::{ScalarVariable, VectorVariable};
pub use geometry::{Point2DVariable, Point3DVariable};
pub use manifold::{SE2Variable, SE3Variable, SO2Variable, SO3Variable};

/// State of a variable in the factor graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableState {
    /// Variable is free to be optimized
    Free,
    /// Variable is fixed/observed and should not be optimized
    Fixed,
    /// Variable is marginalized (removed from optimization but constraints preserved)
    Marginalized,
}

/// Domain constraints for a variable
#[derive(Debug, Clone)]
pub enum VariableDomain {
    /// No constraints (unconstrained)
    Unconstrained,
    /// Box constraints: lower and upper bounds for each dimension
    Box {
        lower: DVector<f64>,
        upper: DVector<f64>,
    },
    /// Manifold constraints (e.g., SO(3), SE(3))
    Manifold { manifold_type: String },
}

/// Simplified Variable trait
///
/// This trait represents variables as elements of Lie groups with the essential operations:
/// - identity: Identity element of the group
/// - inverse: Inverse of the group element
/// - compose: Composition of two group elements
/// - exp: Exponential map from tangent space to group
/// - log: Logarithm map from group to tangent space
/// - plus/minus: Manifold operations for optimization
pub trait Variable: fmt::Debug + Clone + Send + Sync {
    /// Get the unique identifier of this variable
    fn id(&self) -> usize;

    /// Get the dimension of the tangent space (degrees of freedom)
    fn dim(&self) -> usize;

    /// Identity element of the group
    fn identity() -> Self
    where
        Self: Sized;

    /// Inverse of the group element
    fn inverse(&self) -> Self;

    /// Composition of two group elements
    fn compose(&self, other: &Self) -> Self;

    /// Exponential map: tangent space -> group
    fn exp(delta: &DVector<f64>) -> Self
    where
        Self: Sized;

    /// Logarithm map: group -> tangent space
    fn log(&self) -> DVector<f64>;

    /// Manifold plus operation: self ⊕ δ = self ∘ exp(δ)
    fn plus(&self, delta: &DVector<f64>) -> Self {
        self.compose(&Self::exp(delta))
    }

    /// Manifold minus operation: self ⊖ other = log(other⁻¹ ∘ self)
    fn minus(&self, other: &Self) -> DVector<f64> {
        other.inverse().compose(self).log()
    }

    /// Get a human-readable name for this variable (optional)
    fn name(&self) -> Option<&str> {
        None
    }

    /// Get the domain constraints of the variable
    fn domain(&self) -> &VariableDomain;

    /// Set the domain constraints of the variable
    fn set_domain(&mut self, domain: VariableDomain);

    /// Get the state of the variable
    fn state(&self) -> VariableState;

    /// Set the state of the variable
    fn set_state(&mut self, state: VariableState);

    /// Check if the variable is valid (satisfies constraints)
    fn is_valid(&self) -> bool {
        true // Default: always valid
    }
}
