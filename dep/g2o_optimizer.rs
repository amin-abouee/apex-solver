//! G2O optimization module
//!
//! This module provides functionality to convert G2O graph data into optimization problems
//! and solve them using the apex-solver optimization framework.

use nalgebra as na;
use std::collections::HashMap;

use crate::core::factors::{BetweenFactorSE2, BetweenFactorSE3, Factor, PriorFactor};
use crate::core::{ApexResult, Optimizable};
use crate::io::G2oGraph;
use crate::optimizer::{AnySolver, OptimizerConfig, SolverResult};

/// G2O optimization problem that can be solved using apex-solver
pub struct G2oProblem {
    /// The loaded G2O graph data
    pub graph: G2oGraph,
    /// Initial parameter values extracted from vertices
    pub initial_parameters: HashMap<String, na::DVector<f64>>,
    /// Factors created from edges
    pub factors: Vec<Box<dyn Factor>>,
    /// Variable names for SE2 vertices (format: "se2_{id}")
    pub se2_variable_names: Vec<String>,
    /// Variable names for SE3 vertices (format: "se3_{id}")
    pub se3_variable_names: Vec<String>,
}

impl G2oProblem {
    /// Create a new G2O optimization problem from a loaded graph
    pub fn from_graph(graph: G2oGraph) -> ApexResult<Self> {
        let mut initial_parameters = HashMap::new();
        let mut factors: Vec<Box<dyn Factor>> = Vec::new();
        let mut se2_variable_names = Vec::new();
        let mut se3_variable_names = Vec::new();

        // Extract initial parameters from vertices
        for (id, vertex) in &graph.vertices_se2 {
            let var_name = format!("se2_{}", id);
            let params = na::dvector![vertex.theta(), vertex.x(), vertex.y()];
            initial_parameters.insert(var_name.clone(), params);
            se2_variable_names.push(var_name);
        }

        for (id, vertex) in &graph.vertices_se3 {
            let var_name = format!("se3_{}", id);
            let translation = vertex.translation();
            let rotation = vertex.rotation();
            let params = na::dvector![
                translation.x,
                translation.y,
                translation.z,
                rotation.coords.x,
                rotation.coords.y,
                rotation.coords.z,
                rotation.coords.w
            ];
            initial_parameters.insert(var_name.clone(), params);
            se3_variable_names.push(var_name);
        }

        // Create factors from edges
        for edge in &graph.edges_se2 {
            let measurement = &edge.measurement;
            let factor = BetweenFactorSE2::new(
                measurement.x(),
                measurement.y(),
                measurement.angle(),
                edge.information,
            );
            factors.push(Box::new(factor));
        }

        for edge in &graph.edges_se3 {
            let measurement = &edge.measurement;
            let translation = measurement.translation();
            let rotation = measurement.rotation_quaternion();
            let factor = BetweenFactorSE3::new(
                translation.x,
                translation.y,
                translation.z,
                rotation.coords.x,
                rotation.coords.y,
                rotation.coords.z,
                rotation.coords.w,
                edge.information,
            );
            factors.push(Box::new(factor));
        }

        // Add a weak prior factor to fix the first pose (prevent gauge freedom)
        // Use a small weight to avoid over-constraining the problem
        if !se2_variable_names.is_empty() {
            let first_se2_var = &se2_variable_names[0];
            if let Some(first_params) = initial_parameters.get(first_se2_var) {
                // Create a weak prior by scaling down the target
                let weak_prior = first_params * 0.001; // Very weak prior
                let prior_factor = PriorFactor { v: weak_prior };
                factors.push(Box::new(prior_factor));
            }
        } else if !se3_variable_names.is_empty() {
            let first_se3_var = &se3_variable_names[0];
            if let Some(first_params) = initial_parameters.get(first_se3_var) {
                // Create a weak prior by scaling down the target
                let weak_prior = first_params * 0.001; // Very weak prior
                let prior_factor = PriorFactor { v: weak_prior };
                factors.push(Box::new(prior_factor));
            }
        }

        Ok(Self {
            graph,
            initial_parameters,
            factors,
            se2_variable_names,
            se3_variable_names,
        })
    }

    /// Get the total number of parameters
    pub fn parameter_count(&self) -> usize {
        self.initial_parameters.values().map(|v| v.len()).sum()
    }

    /// Get the total number of residuals
    pub fn residual_count(&self) -> usize {
        // SE2 edges contribute 3 residuals each, SE3 edges contribute 6 residuals each
        // Plus prior factors
        self.graph.edges_se2.len() * 3
            + self.graph.edges_se3.len() * 6
            + if !self.se2_variable_names.is_empty() {
                3
            } else if !self.se3_variable_names.is_empty() {
                7
            } else {
                0
            }
    }

    /// Solve the optimization problem
    pub fn solve(
        &self,
        config: OptimizerConfig,
    ) -> ApexResult<SolverResult<HashMap<String, na::DVector<f64>>>> {
        let mut solver = AnySolver::new(config);
        solver.solve(self, self.initial_parameters.clone())
    }

    /// Get statistics about the problem
    pub fn statistics(&self) -> G2oProblemStatistics {
        G2oProblemStatistics {
            se2_vertices: self.graph.vertices_se2.len(),
            se3_vertices: self.graph.vertices_se3.len(),
            se2_edges: self.graph.edges_se2.len(),
            se3_edges: self.graph.edges_se3.len(),
            total_parameters: self.parameter_count(),
            total_residuals: self.residual_count(),
        }
    }
}

/// Statistics about a G2O optimization problem
#[derive(Debug, Clone)]
pub struct G2oProblemStatistics {
    pub se2_vertices: usize,
    pub se3_vertices: usize,
    pub se2_edges: usize,
    pub se3_edges: usize,
    pub total_parameters: usize,
    pub total_residuals: usize,
}

impl std::fmt::Display for G2oProblemStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "G2O Problem Statistics:")?;
        writeln!(f, "  SE2 vertices: {}", self.se2_vertices)?;
        writeln!(f, "  SE3 vertices: {}", self.se3_vertices)?;
        writeln!(f, "  SE2 edges: {}", self.se2_edges)?;
        writeln!(f, "  SE3 edges: {}", self.se3_edges)?;
        writeln!(f, "  Total parameters: {}", self.total_parameters)?;
        writeln!(f, "  Total residuals: {}", self.total_residuals)?;
        Ok(())
    }
}

impl Optimizable for G2oProblem {
    type Parameters = HashMap<String, na::DVector<f64>>;
    type Residuals = na::DVector<f64>;
    type Jacobian = na::DMatrix<f64>;

    fn weights(&self) -> faer::Mat<f64> {
        let mut weights = faer::Mat::zeros(self.residual_count(), self.residual_count());
        let mut current_row = 0;
        for edge in &self.graph.edges_se2 {
            for i in 0..3 {
                weights[(current_row + i, current_row + i)] = edge.information[(i, i)];
            }
            current_row += 3;
        }
        for edge in &self.graph.edges_se3 {
            for i in 0..6 {
                weights[(current_row + i, current_row + i)] = edge.information[(i, i)];
            }
            current_row += 6;
        }

        weights
    }

    fn evaluate(&self, parameters: &Self::Parameters) -> ApexResult<Self::Residuals> {
        let (residuals, _) = self.evaluate_with_jacobian(parameters)?;
        Ok(residuals)
    }

    fn evaluate_with_jacobian(
        &self,
        parameters: &Self::Parameters,
    ) -> ApexResult<(Self::Residuals, Self::Jacobian)> {
        let total_residuals = self.residual_count();
        let total_parameters = self.parameter_count();

        let mut residuals = na::DVector::<f64>::zeros(total_residuals);
        let mut jacobian = na::DMatrix::<f64>::zeros(total_residuals, total_parameters);

        let mut residual_idx = 0;
        let mut param_col_map = HashMap::new();
        let mut col_idx = 0;

        // Build parameter column mapping
        for var_name in &self.se2_variable_names {
            param_col_map.insert(var_name.clone(), col_idx);
            col_idx += 3; // SE2 has 3 parameters
        }
        for var_name in &self.se3_variable_names {
            param_col_map.insert(var_name.clone(), col_idx);
            col_idx += 7; // SE3 has 7 parameters
        }

        // Evaluate SE2 edge factors
        for edge in &self.graph.edges_se2 {
            let from_var = format!("se2_{}", edge.from);
            let to_var = format!("se2_{}", edge.to);

            if let (Some(from_params), Some(to_params)) =
                (parameters.get(&from_var), parameters.get(&to_var))
            {
                let measurement = &edge.measurement;
                let factor = BetweenFactorSE2::new(
                    measurement.x(),
                    measurement.y(),
                    measurement.angle(),
                    edge.information,
                );

                let params_vec = vec![from_params.clone(), to_params.clone()];
                let (res, jac) = factor.linearize(&params_vec);

                // Copy residuals
                residuals.rows_mut(residual_idx, 3).copy_from(&res);

                // Copy Jacobian blocks
                if let Some(&from_col) = param_col_map.get(&from_var) {
                    jacobian
                        .view_mut((residual_idx, from_col), (3, 3))
                        .copy_from(&jac.view((0, 0), (3, 3)));
                }
                if let Some(&to_col) = param_col_map.get(&to_var) {
                    jacobian
                        .view_mut((residual_idx, to_col), (3, 3))
                        .copy_from(&jac.view((0, 3), (3, 3)));
                }

                residual_idx += 3;
            }
        }

        // Evaluate SE3 edge factors
        for edge in &self.graph.edges_se3 {
            let from_var = format!("se3_{}", edge.from);
            let to_var = format!("se3_{}", edge.to);

            if let (Some(from_params), Some(to_params)) =
                (parameters.get(&from_var), parameters.get(&to_var))
            {
                let measurement = &edge.measurement;
                let translation = measurement.translation();
                let rotation = measurement.rotation_quaternion();
                let factor = BetweenFactorSE3::new(
                    translation.x,
                    translation.y,
                    translation.z,
                    rotation.coords.x,
                    rotation.coords.y,
                    rotation.coords.z,
                    rotation.coords.w,
                    edge.information,
                );

                let params_vec = vec![from_params.clone(), to_params.clone()];
                let (res, jac) = factor.linearize(&params_vec);

                // Copy residuals
                residuals.rows_mut(residual_idx, 6).copy_from(&res);

                // Copy Jacobian blocks
                if let Some(&from_col) = param_col_map.get(&from_var) {
                    jacobian
                        .view_mut((residual_idx, from_col), (6, 7))
                        .copy_from(&jac.view((0, 0), (6, 7)));
                }
                if let Some(&to_col) = param_col_map.get(&to_var) {
                    jacobian
                        .view_mut((residual_idx, to_col), (6, 7))
                        .copy_from(&jac.view((0, 7), (6, 7)));
                }

                residual_idx += 6;
            }
        }

        // Add prior factor
        if !self.se2_variable_names.is_empty() {
            let first_var = &self.se2_variable_names[0];
            if let Some(params) = parameters.get(first_var) {
                let initial_params = self.initial_parameters.get(first_var).unwrap();
                let prior_factor = PriorFactor {
                    v: initial_params.clone(),
                };
                let (res, jac) = prior_factor.linearize(&[params.clone()]);

                residuals.rows_mut(residual_idx, 3).copy_from(&res);
                if let Some(&col) = param_col_map.get(first_var) {
                    jacobian
                        .view_mut((residual_idx, col), (3, 3))
                        .copy_from(&jac);
                }
            }
        } else if !self.se3_variable_names.is_empty() {
            let first_var = &self.se3_variable_names[0];
            if let Some(params) = parameters.get(first_var) {
                let initial_params = self.initial_parameters.get(first_var).unwrap();
                let prior_factor = PriorFactor {
                    v: initial_params.clone(),
                };
                let (res, jac) = prior_factor.linearize(&[params.clone()]);

                residuals.rows_mut(residual_idx, 7).copy_from(&res);
                if let Some(&col) = param_col_map.get(first_var) {
                    jacobian
                        .view_mut((residual_idx, col), (7, 7))
                        .copy_from(&jac);
                }
            }
        }

        Ok((residuals, jacobian))
    }

    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }

    fn residual_count(&self) -> usize {
        self.residual_count()
    }

    fn cost(&self, parameters: &Self::Parameters) -> ApexResult<f64> {
        let residuals = self.evaluate(parameters)?;
        Ok(0.5 * residuals.norm_squared())
    }
}
