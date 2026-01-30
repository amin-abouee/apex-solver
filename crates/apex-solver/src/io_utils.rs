//! Utility functions for working with I/O types and optimization results.
//!
//! This module provides conversion utilities between apex-solver's core types
//! and apex-io's graph representation.

use crate::core::problem::VariableEnum;
use apex_io::{Graph, VertexSE2, VertexSE3};
use std::collections::HashMap;

/// Create a new graph from optimized variables, keeping the original edges.
///
/// This is useful for saving optimization results: vertices are updated with
/// optimized poses, while edges (constraints) remain the same.
///
/// # Arguments
///
/// * `variables` - HashMap of optimized variable values from solver
/// * `original_edges` - Reference to original graph to copy edges from
///
/// # Returns
///
/// A new Graph with optimized vertices and original edges
///
/// # Example
///
/// ```
/// use apex_solver::{LevenbergMarquardt, Solver};
/// use apex_solver::io_utils::graph_from_optimized_variables;
/// use apex_io::{G2oLoader, GraphLoader};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Load graph
/// let graph = G2oLoader::load("data/sphere2500.g2o")?;
///
/// // Build and solve problem
/// // let problem = ...;
/// // let result = solver.minimize(&problem, &initial_values)?;
///
/// // Create output graph with optimized poses
/// // let optimized_graph = graph_from_optimized_variables(&result.values, &graph);
/// # Ok(())
/// # }
/// ```
pub fn graph_from_optimized_variables(
    variables: &HashMap<String, VariableEnum>,
    original_edges: &Graph,
) -> Graph {
    use VariableEnum;

    let mut graph = Graph::new();

    // Copy edges from original (they don't change during optimization)
    graph.edges_se2 = original_edges.edges_se2.clone();
    graph.edges_se3 = original_edges.edges_se3.clone();

    // Convert optimized variables back to vertices
    for (var_name, var) in variables {
        // Extract vertex ID from variable name (format: "x{id}")
        if let Some(id_str) = var_name.strip_prefix('x') {
            if let Ok(id) = id_str.parse::<usize>() {
                match var {
                    VariableEnum::SE2(v) => {
                        let vertex = VertexSE2 {
                            id,
                            pose: v.value.clone(),
                        };
                        graph.vertices_se2.insert(id, vertex);
                    }
                    VariableEnum::SE3(v) => {
                        let vertex = VertexSE3 {
                            id,
                            pose: v.value.clone(),
                        };
                        graph.vertices_se3.insert(id, vertex);
                    }
                    _ => {
                        // Skip other manifold types (SO2, SO3, Rn)
                        // These are not commonly used in SLAM graphs
                    }
                }
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use apex_io::{EdgeSE3, Graph, VertexSE3};
    use apex_manifolds::se3::SE3;
    use nalgebra::{Matrix6, Vector3};

    #[test]
    fn test_graph_from_optimized_variables() {
        // Create original graph
        let mut original_graph = Graph::new();
        original_graph.vertices_se3.insert(
            0,
            VertexSE3::new(0, Vector3::zeros(), nalgebra::UnitQuaternion::identity()),
        );
        original_graph.vertices_se3.insert(
            1,
            VertexSE3::new(
                1,
                Vector3::new(1.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::identity(),
            ),
        );
        original_graph.edges_se3.push(EdgeSE3::new(
            0,
            1,
            Vector3::new(1.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
            Matrix6::identity(),
        ));

        // Create optimized variables
        let mut optimized_vars = HashMap::new();
        optimized_vars.insert(
            "x0".to_string(),
            VariableEnum::SE3(Variable::new(SE3::identity())),
        );
        optimized_vars.insert(
            "x1".to_string(),
            VariableEnum::SE3(Variable::new(SE3::new(
                Vector3::new(2.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::identity(),
            ))),
        );

        // Convert to graph
        let result_graph = graph_from_optimized_variables(&optimized_vars, &original_graph);

        // Verify
        assert_eq!(result_graph.vertices_se3.len(), 2);
        assert_eq!(result_graph.edges_se3.len(), 1);
        assert_eq!(result_graph.vertices_se3[&1].x(), 2.0);
    }
}
