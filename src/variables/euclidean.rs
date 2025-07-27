//! Euclidean space variables
//!
//! This module provides variable types for standard Euclidean spaces,
//! including scalar variables, vector variables, and general n-dimensional variables.

use crate::variables::{Variable, VariableDomain, VariableState};
use nalgebra::DVector;

/// A scalar variable (1-dimensional)
#[derive(Debug, Clone)]
pub struct ScalarVariable {
    id: usize,
    value: f64,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl ScalarVariable {
    /// Create a new scalar variable
    pub fn new(id: usize, initial_value: f64) -> Self {
        Self {
            id,
            value: initial_value,
            state: VariableState::Free,
            domain: VariableDomain::Unconstrained,
            name: None,
        }
    }

    /// Create a scalar variable with bounds
    pub fn with_bounds(id: usize, initial_value: f64, lower: f64, upper: f64) -> Self {
        let domain = VariableDomain::Box {
            lower: DVector::from_element(1, lower),
            upper: DVector::from_element(1, upper),
        };
        Self {
            id,
            value: initial_value,
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

    /// Get the scalar value
    pub fn scalar_value(&self) -> f64 {
        self.value
    }

    /// Set the scalar value
    pub fn set_scalar_value(&mut self, value: f64) {
        self.value = value;
    }
}

impl Variable for ScalarVariable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        1
    }

    fn identity() -> Self {
        Self::new(0, 0.0)
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            value: -self.value,
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            id: self.id,
            value: self.value + other.value,
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        assert_eq!(
            delta.len(),
            1,
            "Scalar variable expects 1-dimensional delta"
        );
        Self::new(0, delta[0])
    }

    fn log(&self) -> DVector<f64> {
        DVector::from_element(1, self.value)
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
                self.value >= lower[0] && self.value <= upper[0]
            }
            VariableDomain::Manifold { .. } => true,
        }
    }
}

/// A general n-dimensional vector variable
#[derive(Debug, Clone)]
pub struct VectorVariable {
    id: usize,
    value: DVector<f64>,
    state: VariableState,
    domain: VariableDomain,
    name: Option<String>,
}

impl VectorVariable {
    /// Create a new vector variable
    pub fn new(id: usize, initial_value: DVector<f64>) -> Self {
        Self {
            id,
            value: initial_value,
            state: VariableState::Free,
            domain: VariableDomain::Unconstrained,
            name: None,
        }
    }

    /// Create a vector variable with box constraints
    pub fn with_bounds(
        id: usize,
        initial_value: DVector<f64>,
        lower: DVector<f64>,
        upper: DVector<f64>,
    ) -> Result<Self, String> {
        if initial_value.len() != lower.len() || initial_value.len() != upper.len() {
            return Err(
                "Initial value, lower, and upper bounds must have the same dimension".to_string(),
            );
        }

        let domain = VariableDomain::Box { lower, upper };
        Ok(Self {
            id,
            value: initial_value,
            state: VariableState::Free,
            domain,
            name: None,
        })
    }

    /// Set a name for this variable
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the vector value
    pub fn vector_value(&self) -> &DVector<f64> {
        &self.value
    }

    /// Set the vector value
    pub fn set_vector_value(&mut self, value: DVector<f64>) -> Result<(), String> {
        if value.len() != self.value.len() {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.value.len(),
                value.len()
            ));
        }
        self.value = value;
        Ok(())
    }
}

impl Variable for VectorVariable {
    fn id(&self) -> usize {
        self.id
    }

    fn dim(&self) -> usize {
        self.value.len()
    }

    fn identity() -> Self {
        Self::new(0, DVector::zeros(3)) // Default to 3D for identity
    }

    fn inverse(&self) -> Self {
        Self {
            id: self.id,
            value: -&self.value,
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        assert_eq!(
            self.value.len(),
            other.value.len(),
            "Vector dimensions must match for composition"
        );
        Self {
            id: self.id,
            value: &self.value + &other.value,
            state: self.state,
            domain: self.domain.clone(),
            name: self.name.clone(),
        }
    }

    fn exp(delta: &DVector<f64>) -> Self {
        Self::new(0, delta.clone())
    }

    fn log(&self) -> DVector<f64> {
        self.value.clone()
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
            VariableDomain::Box { lower, upper } => self
                .value
                .iter()
                .zip(lower.iter())
                .zip(upper.iter())
                .all(|((&val, &low), &up)| val >= low && val <= up),
            VariableDomain::Manifold { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_variable_basic_operations() {
        let var = ScalarVariable::new(1, 5.0);

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 1);
        assert_eq!(var.scalar_value(), 5.0);
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = ScalarVariable::identity();
        assert_eq!(identity.scalar_value(), 0.0);

        let inverse = var.inverse();
        assert_eq!(inverse.scalar_value(), -5.0);

        let other = ScalarVariable::new(2, 3.0);
        let composed = var.compose(&other);
        assert_eq!(composed.scalar_value(), 8.0);

        // Test exp/log operations
        let delta = DVector::from_element(1, 2.0);
        let exp_result = ScalarVariable::exp(&delta);
        assert_eq!(exp_result.scalar_value(), 2.0);

        let log_result = var.log();
        assert_eq!(log_result[0], 5.0);

        // Test plus/minus operations
        let plus_result = var.plus(&delta);
        assert_eq!(plus_result.scalar_value(), 7.0);

        let minus_result = var.minus(&other);
        assert_eq!(minus_result[0], 2.0);
    }

    #[test]
    fn test_scalar_variable_with_bounds() {
        let var = ScalarVariable::with_bounds(1, 5.0, 0.0, 10.0);
        assert!(var.is_valid());

        let mut invalid_var = ScalarVariable::with_bounds(2, 15.0, 0.0, 10.0);
        assert!(!invalid_var.is_valid());
    }

    #[test]
    fn test_vector_variable_basic_operations() {
        let initial_value = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let var = VectorVariable::new(1, initial_value.clone());

        // Test basic properties
        assert_eq!(var.id(), 1);
        assert_eq!(var.dim(), 3);
        assert_eq!(var.vector_value(), &initial_value);
        assert_eq!(var.state(), VariableState::Free);
        assert!(var.is_valid());

        // Test Lie group operations
        let identity = VectorVariable::identity();
        assert_eq!(identity.dim(), 3); // Default 3D identity

        let inverse = var.inverse();
        assert_eq!(inverse.vector_value()[0], -1.0);
        assert_eq!(inverse.vector_value()[1], -2.0);
        assert_eq!(inverse.vector_value()[2], -3.0);

        let other_value = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let other = VectorVariable::new(2, other_value);
        let composed = var.compose(&other);
        assert_eq!(composed.vector_value()[0], 5.0);
        assert_eq!(composed.vector_value()[1], 7.0);
        assert_eq!(composed.vector_value()[2], 9.0);

        // Test exp/log operations
        let delta = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let exp_result = VectorVariable::exp(&delta);
        assert_eq!(exp_result.vector_value(), &delta);

        let log_result = var.log();
        assert_eq!(log_result, initial_value);
    }

    #[test]
    fn test_variable_state_management() {
        let mut var = ScalarVariable::new(1, 5.0);

        // Test initial state
        assert_eq!(var.state(), VariableState::Free);

        // Test state changes
        var.set_state(VariableState::Fixed);
        assert_eq!(var.state(), VariableState::Fixed);

        var.set_state(VariableState::Marginalized);
        assert_eq!(var.state(), VariableState::Marginalized);
    }

    #[test]
    fn test_variable_domain_management() {
        let mut var = ScalarVariable::new(1, 5.0);

        // Test initial domain
        match var.domain() {
            VariableDomain::Unconstrained => {}
            _ => panic!("Expected unconstrained domain"),
        }

        // Test domain change
        let new_domain = VariableDomain::Box {
            lower: DVector::from_element(1, 0.0),
            upper: DVector::from_element(1, 10.0),
        };
        var.set_domain(new_domain);

        match var.domain() {
            VariableDomain::Box { lower, upper } => {
                assert_eq!(lower[0], 0.0);
                assert_eq!(upper[0], 10.0);
            }
            _ => panic!("Expected box domain"),
        }
    }
}
