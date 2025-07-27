use crate::core::types::{ApexError, ApexResult};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fmt;

/// Statistics about a factor graph
#[derive(Debug, Clone)]
pub struct FactorGraphStatistics {
    pub num_variables: usize,
    pub num_factors: usize,
    pub num_free_variables: usize,
    pub num_fixed_variables: usize,
    pub total_parameter_dimension: usize,
    pub total_manifold_dimension: usize,
    pub total_residual_dimension: usize,
}

impl fmt::Display for FactorGraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FactorGraph Statistics:\n\
             Variables: {} (free: {}, fixed: {})\n\
             Factors: {}\n\
             Parameter dimensions: {} (manifold: {})\n\
             Residual dimension: {}",
            self.num_variables,
            self.num_free_variables,
            self.num_fixed_variables,
            self.num_factors,
            self.total_parameter_dimension,
            self.total_manifold_dimension,
            self.total_residual_dimension
        )
    }
}

/// Unique identifier for variables in the factor graph
pub type VariableId = usize;

/// Unique identifier for factors in the factor graph
pub type FactorId = usize;

/// Enhanced factor graph representing a nonlinear least squares problem
#[derive(Default)]
pub struct FactorGraph {
    variables: HashMap<VariableId, Box<dyn Variable>>,
    factors: HashMap<FactorId, Box<dyn Factor>>,
    variable_ordering: Vec<VariableId>,
    factor_ordering: Vec<FactorId>,
    next_variable_id: VariableId,
    next_factor_id: FactorId,
}

impl FactorGraph {
    /// Creates a new, empty factor graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable to the factor graph and return its ID
    pub fn add_variable(&mut self, variable: Box<dyn Variable>) -> VariableId {
        let id = variable.id();
        self.variable_ordering.push(id);
        self.variables.insert(id, variable);
        self.next_variable_id = self.next_variable_id.max(id + 1);
        id
    }

    /// Add a standard variable with automatic ID assignment
    pub fn add_standard_variable(
        &mut self,
        initial_value: DVector<f64>,
        dimension: usize,
        manifold_dimension: Option<usize>,
    ) -> VariableId {
        let id = self.next_variable_id;
        let variable = StandardVariable::new(id, initial_value, dimension, manifold_dimension);
        self.add_variable(Box::new(variable))
    }

    /// Add a factor to the factor graph and return its ID
    pub fn add_factor(&mut self, factor: Box<dyn Factor>) -> ApexResult<FactorId> {
        let id = factor.id();

        // Validate that all referenced variables exist
        for &var_id in factor.variable_ids() {
            if !self.variables.contains_key(&var_id) {
                return Err(ApexError::InvalidInput(format!(
                    "Factor {} references non-existent variable {}",
                    id, var_id
                )));
            }
        }

        self.factor_ordering.push(id);
        self.factors.insert(id, factor);
        self.next_factor_id = self.next_factor_id.max(id + 1);
        Ok(id)
    }

    /// Remove a variable from the factor graph
    pub fn remove_variable(&mut self, id: VariableId) -> ApexResult<()> {
        // Check if any factors depend on this variable
        for factor in self.factors.values() {
            if factor.variable_ids().contains(&id) {
                return Err(ApexError::InvalidInput(format!(
                    "Cannot remove variable {} - it is referenced by factor {}",
                    id,
                    factor.id()
                )));
            }
        }

        self.variables.remove(&id);
        self.variable_ordering.retain(|&x| x != id);
        Ok(())
    }

    /// Remove a factor from the factor graph
    pub fn remove_factor(&mut self, id: FactorId) -> ApexResult<()> {
        if self.factors.remove(&id).is_none() {
            return Err(ApexError::InvalidInput(format!(
                "Factor {} does not exist",
                id
            )));
        }
        self.factor_ordering.retain(|&x| x != id);
        Ok(())
    }

    /// Get a variable by ID
    pub fn variable(&self, id: VariableId) -> Option<&dyn Variable> {
        self.variables.get(&id).map(|v| v.as_ref())
    }

    /// Get a mutable variable by ID
    pub fn variable_mut(&mut self, id: VariableId) -> Option<&mut dyn Variable> {
        self.variables.get_mut(&id).map(|v| v.as_mut())
    }

    /// Get a factor by ID
    pub fn factor(&self, id: FactorId) -> Option<&dyn Factor> {
        self.factors.get(&id).map(|f| f.as_ref())
    }

    /// Get all variable IDs in order
    pub fn variable_ids(&self) -> &[VariableId] {
        &self.variable_ordering
    }

    /// Get all factor IDs in order
    pub fn factor_ids(&self) -> &[FactorId] {
        &self.factor_ordering
    }

    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Get the number of factors
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Get variables connected to a specific factor
    pub fn variables_for_factor(&self, factor_id: FactorId) -> ApexResult<Vec<&dyn Variable>> {
        let factor = self
            .factors
            .get(&factor_id)
            .ok_or_else(|| ApexError::InvalidInput(format!("Factor {} not found", factor_id)))?;

        let mut variables = Vec::new();
        for &var_id in factor.variable_ids() {
            let variable = self
                .variables
                .get(&var_id)
                .ok_or_else(|| ApexError::InvalidInput(format!("Variable {} not found", var_id)))?;
            variables.push(variable.as_ref());
        }
        Ok(variables)
    }

    /// Get factors connected to a specific variable
    pub fn factors_for_variable(&self, variable_id: VariableId) -> Vec<&dyn Factor> {
        self.factors
            .values()
            .filter(|factor| factor.variable_ids().contains(&variable_id))
            .map(|f| f.as_ref())
            .collect()
    }

    /// Calculate the total squared error of the factor graph
    pub fn total_error(&self) -> ApexResult<f64> {
        let mut total_error = 0.0;

        for factor in self.factors.values() {
            let variables = self.variables_for_factor(factor.id())?;
            let residual = factor.residual(&variables)?;
            let information = factor.information_matrix();

            // Compute weighted squared error: r^T * Ω * r
            let weighted_residual = information * &residual;
            total_error += residual.dot(&weighted_residual);
        }

        Ok(total_error)
    }

    /// Get the total dimension of all variables
    pub fn total_variable_dimension(&self) -> usize {
        self.variables.values().map(|v| v.dimension()).sum()
    }

    /// Get the total manifold dimension of all variables
    pub fn total_manifold_dimension(&self) -> usize {
        self.variables
            .values()
            .map(|v| v.manifold_dimension())
            .sum()
    }

    /// Get the total residual dimension of all factors
    pub fn total_residual_dimension(&self) -> usize {
        self.factors.values().map(|f| f.residual_dimension()).sum()
    }

    /// Build the sparse Jacobian matrix and dense residual vector for linearization
    pub fn linearize(&self) -> ApexResult<(SparseColMat<usize, f64>, DVector<f64>)> {
        let num_cols = self.total_manifold_dimension();
        let num_rows = self.total_residual_dimension();

        let mut residuals = Vec::new();
        let mut triplets = Vec::new();
        let mut current_row = 0;

        for &factor_id in &self.factor_ordering {
            let factor = &self.factors[&factor_id];
            let variables = self.variables_for_factor(factor_id)?;

            let (r_i, j_i) = factor.residual_and_jacobian(&variables)?;
            let information = factor.information_matrix();

            // Apply information matrix weighting: sqrt(Ω) * r and sqrt(Ω) * J
            let sqrt_info = information
                .clone()
                .cholesky()
                .ok_or_else(|| {
                    ApexError::LinearAlgebra(
                        "Information matrix is not positive definite".to_string(),
                    )
                })?
                .l();

            let weighted_residual = &sqrt_info * &r_i;
            let weighted_jacobian = &sqrt_info * &j_i;

            let residual_dim = weighted_residual.nrows();
            residuals.extend_from_slice(weighted_residual.as_slice());

            let mut current_block_col = 0;
            for &var_id in factor.variable_ids() {
                let var = &self.variables[&var_id];
                let var_manifold_dim = var.manifold_dimension();
                let col_offset = self.variable_manifold_col_offset(var_id)?;

                let j_block = weighted_jacobian
                    .view((0, current_block_col), (residual_dim, var_manifold_dim));

                for r in 0..residual_dim {
                    for c in 0..var_manifold_dim {
                        let value = j_block[(r, c)];
                        if value.abs() > 1e-15 {
                            // Only add non-zero entries
                            triplets.push(faer::sparse::Triplet::new(
                                current_row + r,
                                col_offset + c,
                                value,
                            ));
                        }
                    }
                }
                current_block_col += var_manifold_dim;
            }
            current_row += residual_dim;
        }

        let r = DVector::from_vec(residuals);
        let J =
            SparseColMat::try_new_from_triplets(num_rows, num_cols, &triplets).map_err(|e| {
                ApexError::LinearAlgebra(format!("Failed to create sparse matrix: {:?}", e))
            })?;

        Ok((J, r))
    }

    /// Update variables with a given delta vector (manifold retraction)
    pub fn update(&mut self, delta: &DVector<f64>) -> ApexResult<()> {
        if delta.len() != self.total_manifold_dimension() {
            return Err(ApexError::InvalidInput(format!(
                "Delta dimension {} does not match total manifold dimension {}",
                delta.len(),
                self.total_manifold_dimension()
            )));
        }

        let mut current_offset = 0;
        for &var_id in &self.variable_ordering {
            let var = self.variables.get_mut(&var_id).unwrap();

            // Skip fixed variables
            if var.state() == VariableState::Fixed {
                continue;
            }

            let manifold_dim = var.manifold_dimension();
            let delta_i = delta.rows(current_offset, manifold_dim);

            // Use manifold retraction operation
            var.retract(&delta_i)?;

            // Project to domain if constrained
            var.project_to_domain()?;

            current_offset += manifold_dim;
        }

        Ok(())
    }

    /// Get the starting column index for a variable in the Jacobian (manifold coordinates)
    fn variable_manifold_col_offset(&self, var_id: VariableId) -> ApexResult<usize> {
        let mut offset = 0;
        for &id in &self.variable_ordering {
            if id == var_id {
                return Ok(offset);
            }
            let var = self
                .variables
                .get(&id)
                .ok_or_else(|| ApexError::InvalidInput(format!("Variable {} not found", id)))?;

            // Skip fixed variables in the manifold parameterization
            if var.state() != VariableState::Fixed {
                offset += var.manifold_dimension();
            }
        }
        Err(ApexError::InvalidInput(format!(
            "Variable {} not found in ordering",
            var_id
        )))
    }

    /// Get all free (optimizable) variables
    pub fn free_variables(&self) -> Vec<&dyn Variable> {
        self.variables
            .values()
            .filter(|v| v.state() == VariableState::Free)
            .map(|v| v.as_ref())
            .collect()
    }

    /// Get all fixed variables
    pub fn fixed_variables(&self) -> Vec<&dyn Variable> {
        self.variables
            .values()
            .filter(|v| v.state() == VariableState::Fixed)
            .map(|v| v.as_ref())
            .collect()
    }

    /// Check if the factor graph is valid (all constraints satisfied)
    pub fn is_valid(&self) -> bool {
        // Check that all variables satisfy their domain constraints
        for variable in self.variables.values() {
            if !variable.is_valid() {
                return false;
            }
        }

        // Check that all factors reference existing variables
        for factor in self.factors.values() {
            for &var_id in factor.variable_ids() {
                if !self.variables.contains_key(&var_id) {
                    return false;
                }
            }
        }

        true
    }

    /// Get statistics about the factor graph
    pub fn statistics(&self) -> FactorGraphStatistics {
        let num_free_vars = self.free_variables().len();
        let num_fixed_vars = self.fixed_variables().len();
        let total_params = self.total_variable_dimension();
        let total_manifold_params = self.total_manifold_dimension();
        let total_residuals = self.total_residual_dimension();

        FactorGraphStatistics {
            num_variables: self.num_variables(),
            num_factors: self.num_factors(),
            num_free_variables: num_free_vars,
            num_fixed_variables: num_fixed_vars,
            total_parameter_dimension: total_params,
            total_manifold_dimension: total_manifold_params,
            total_residual_dimension: total_residuals,
        }
    }
}
