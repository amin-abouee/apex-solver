//! Simplified Graph container for factors
//!
//! This module provides a simplified Graph container inspired by factrs,
//! which stores factors in a simple vector for efficient iteration.

use crate::core::graph::{Factor, FactorId, VariableId, VariableSafe};
use crate::core::types::{ApexError, ApexResult};
use crate::core::values::{Key, Values};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fmt;

/// Simplified Graph container for factors
///
/// This is a simplified version inspired by factrs that stores factors
/// as trait objects in a vector for efficient iteration.
#[derive(Debug)]
pub struct Graph {
    factors: Vec<Box<dyn Factor>>,
    factor_map: HashMap<FactorId, usize>, // Maps factor ID to index in factors vector
}

impl Graph {
    /// Create a new empty Graph
    pub fn new() -> Self {
        Self {
            factors: Vec::new(),
            factor_map: HashMap::new(),
        }
    }

    /// Get the number of factors
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Add a factor to the graph
    pub fn add_factor(&mut self, factor: Box<dyn Factor>) -> ApexResult<()> {
        let factor_id = factor.id();

        // Check for duplicate factor IDs
        if self.factor_map.contains_key(&factor_id) {
            return Err(ApexError::InvalidInput(format!(
                "Factor with ID {} already exists",
                factor_id
            )));
        }

        let index = self.factors.len();
        self.factors.push(factor);
        self.factor_map.insert(factor_id, index);

        Ok(())
    }

    /// Get a factor by ID
    pub fn get_factor(&self, factor_id: FactorId) -> Option<&dyn Factor> {
        let index = self.factor_map.get(&factor_id)?;
        self.factors.get(*index).map(|f| f.as_ref())
    }

    /// Get a mutable reference to a factor by ID
    pub fn get_factor_mut(&mut self, factor_id: FactorId) -> Option<&mut dyn Factor> {
        let index = self.factor_map.get(&factor_id)?;
        self.factors.get_mut(*index).map(|f| f.as_mut())
    }

    /// Remove a factor by ID
    pub fn remove_factor(&mut self, factor_id: FactorId) -> Option<Box<dyn Factor>> {
        let index = *self.factor_map.get(&factor_id)?;
        self.factor_map.remove(&factor_id);

        // Remove from vector and update indices in map
        let factor = self.factors.remove(index);

        // Update indices for factors that were shifted
        for (_, idx) in self.factor_map.iter_mut() {
            if *idx > index {
                *idx -= 1;
            }
        }

        Some(factor)
    }

    /// Iterate over all factors
    pub fn factors(&self) -> impl Iterator<Item = &dyn Factor> {
        self.factors.iter().map(|f| f.as_ref())
    }

    /// Iterate over all factors mutably
    pub fn factors_mut(&mut self) -> impl Iterator<Item = &mut dyn Factor> {
        self.factors.iter_mut().map(|f| f.as_mut())
    }

    /// Get all factor IDs
    pub fn factor_ids(&self) -> impl Iterator<Item = FactorId> + '_ {
        self.factor_map.keys().copied()
    }

    /// Get all variable IDs referenced by factors
    pub fn variable_ids(&self) -> Vec<VariableId> {
        let mut var_ids = std::collections::HashSet::new();
        for factor in &self.factors {
            for &var_id in factor.variable_ids() {
                var_ids.insert(var_id);
            }
        }
        var_ids.into_iter().collect()
    }

    /// Compute total error for all factors given values
    pub fn error(&self, values: &Values) -> ApexResult<f64> {
        let mut total_error = 0.0;

        for factor in &self.factors {
            let variables = self.collect_variables(factor.as_ref(), values)?;
            let factor_error = factor.error(&variables)?;
            total_error += factor_error;
        }

        Ok(total_error)
    }

    /// Linearize the graph to get Jacobian and residual
    pub fn linearize(&self, values: &Values) -> ApexResult<(DMatrix<f64>, DVector<f64>)> {
        // Collect all variable IDs and create ordering
        let var_ids = self.variable_ids();
        let var_id_to_index: HashMap<VariableId, usize> =
            var_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Calculate dimensions
        let mut total_manifold_dim = 0;
        let mut total_residual_dim = 0;

        for &var_id in &var_ids {
            let key = Key::new(var_id, "unknown"); // We'll need to improve this
            if let Some(var) = values.get_unchecked(&key) {
                total_manifold_dim += var.dim();
            }
        }

        for factor in &self.factors {
            total_residual_dim += factor.residual_dimension();
        }

        // Build Jacobian and residual
        let mut jacobian = DMatrix::zeros(total_residual_dim, total_manifold_dim);
        let mut residual = DVector::zeros(total_residual_dim);

        let mut current_row = 0;

        for factor in &self.factors {
            let variables = self.collect_variables(factor.as_ref(), values)?;
            let factor_residual = factor.residual(&variables)?;
            let factor_jacobian = factor.jacobian(&variables)?;

            let residual_dim = factor.residual_dimension();

            // Copy residual
            residual
                .rows_mut(current_row, residual_dim)
                .copy_from(&factor_residual);

            // Copy Jacobian blocks
            let mut current_col = 0;
            for &var_id in factor.variable_ids() {
                if let Some(&var_index) = var_id_to_index.get(&var_id) {
                    let key = Key::new(var_id, "unknown");
                    if let Some(var) = values.get_unchecked(&key) {
                        let var_dim = var.dim();
                        let j_block =
                            factor_jacobian.view((0, current_col), (residual_dim, var_dim));

                        // Find column offset for this variable
                        let mut col_offset = 0;
                        for i in 0..var_index {
                            let other_key = Key::new(var_ids[i], "unknown");
                            if let Some(other_var) = values.get_unchecked(&other_key) {
                                col_offset += other_var.dim();
                            }
                        }

                        jacobian
                            .view_mut((current_row, col_offset), (residual_dim, var_dim))
                            .copy_from(&j_block);

                        current_col += var_dim;
                    }
                }
            }

            current_row += residual_dim;
        }

        Ok((jacobian, residual))
    }

    /// Helper function to collect variables for a factor
    fn collect_variables(
        &self,
        factor: &dyn Factor,
        values: &Values,
    ) -> ApexResult<Vec<&dyn VariableSafe>> {
        let mut variables = Vec::new();

        for &var_id in factor.variable_ids() {
            let key = Key::new(var_id, "unknown"); // We'll need to improve this
            let var = values
                .get_unchecked(&key)
                .ok_or_else(|| ApexError::InvalidInput(format!("Variable {} not found", var_id)))?;
            variables.push(var);
        }

        Ok(variables)
    }

    /// Get graph statistics
    pub fn statistics(&self) -> GraphStatistics {
        let num_factors = self.len();
        let variable_ids = self.variable_ids();
        let num_variables = variable_ids.len();

        let total_residual_dimension = self.factors.iter().map(|f| f.residual_dimension()).sum();

        GraphStatistics {
            num_factors,
            num_variables,
            total_residual_dimension,
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph ({} factors):", self.len())?;
        for factor in &self.factors {
            writeln!(
                f,
                "  Factor {}: {} (variables: {:?})",
                factor.id(),
                factor.factor_type(),
                factor.variable_ids()
            )?;
        }
        Ok(())
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub num_factors: usize,
    pub num_variables: usize,
    pub total_residual_dimension: usize,
}

impl fmt::Display for GraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph Statistics:")?;
        writeln!(f, "  Factors: {}", self.num_factors)?;
        writeln!(f, "  Variables: {}", self.num_variables)?;
        writeln!(
            f,
            "  Total residual dimension: {}",
            self.total_residual_dimension
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic_operations() {
        let mut graph = Graph::new();
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_graph_display() {
        let graph = Graph::new();
        let display_str = format!("{}", graph);
        assert!(display_str.contains("Graph (0 factors)"));
    }
}
