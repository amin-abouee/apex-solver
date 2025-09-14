use std::collections::HashMap;

use apex_solver::core::factors::BetweenFactorSE2;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APEX-SOLVER SE2 POSE LOADING EXAMPLE ===");
    println!("Loading 2D poses from M3500.g2o dataset");

    // Load the G2O graph file
    let graph = G2oLoader::load("data/M3500.g2o")?;

    println!("Successfully loaded SE2 graph:");
    println!("  SE2 vertices: {}", graph.vertices_se2.len());
    println!("  SE2 edges: {}", graph.edges_se2.len());

    // Display first few vertices
    println!("\nFirst 10 SE2 vertices:");
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in vertex_ids.iter().take(10) {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            println!(
                "  x{}: theta={:.6}, x={:.6}, y={:.6}",
                id,
                vertex.theta(),
                vertex.x(),
                vertex.y()
            );
        }
    }

    // Display first few edges
    println!("\nFirst 10 SE2 edges (relative transformations):");
    for edge in graph.edges_se2.iter().take(10) {
        println!(
            "  Edge {}->{}: dx={:.6}, dy={:.6}, dtheta={:.6}",
            edge.from,
            edge.to,
            edge.measurement.translation().x,
            edge.measurement.translation().y,
            edge.measurement.angle()
        );
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE2 vertices as variables
    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            // Format: [theta, x, y]
            let se2_data = na::dvector![vertex.theta(), vertex.x(), vertex.y()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Add SE2 between factors
    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let dx = edge.measurement.translation().x;
        let dy = edge.measurement.translation().y;
        let dtheta = edge.measurement.angle();

        let between_factor = BetweenFactorSE2::new(dx, dy, dtheta);

        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            None, // No loss function
        );
    }

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!("  Between factors: {}", graph.edges_se2.len());

    // Initialize problem variables
    let variables = problem.initialize_variables(&initial_values);
    println!("  Variable dimensions:");
    for (name, var) in variables.iter().take(3) {
        println!("    {}: {} dimensions", name, var.get_size());
    }

    // Display graph statistics
    println!("\nGraph statistics:");
    let total_poses = vertex_ids.len();
    let total_constraints = graph.edges_se2.len();
    let dof_before = total_poses * 3; // Each SE2 pose has 3 DOF
    let dof_after = dof_before - 3; // Fix one pose (3 DOF removed)
    let constraint_dim = total_constraints * 3; // Each between factor has 3 constraints

    println!("  Total poses: {}", total_poses);
    println!("  Total constraints: {}", total_constraints);
    println!("  DOF before fixing: {}", dof_before);
    println!("  DOF after fixing first pose: {}", dof_after);
    println!("  Total constraint dimensions: {}", constraint_dim);

    if constraint_dim >= dof_after {
        println!("  System is over-constrained (good for optimization)");
    } else {
        println!("  System is under-constrained (may need more constraints)");
    }

    println!("\n=== SE2 POSE LOADING COMPLETE ===");
    println!("Graph successfully loaded and ready for optimization!");

    Ok(())
}
