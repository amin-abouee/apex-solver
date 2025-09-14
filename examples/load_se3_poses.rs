use std::collections::HashMap;

use apex_solver::core::factors::BetweenFactorSE3;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APEX-SOLVER SE3 POSE LOADING EXAMPLE ===");
    println!("Loading 3D poses from sphere2500.g2o dataset");

    // Load the G2O graph file
    let graph = G2oLoader::load("data/sphere2500.g2o")?;

    println!("Successfully loaded SE3 graph:");
    println!("  SE3 vertices: {}", graph.vertices_se3.len());
    println!("  SE3 edges: {}", graph.edges_se3.len());

    // Display first few vertices
    println!("\nFirst 5 SE3 vertices:");
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in vertex_ids.iter().take(5) {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            println!(
                "  x{}: quat=[{:.6},{:.6},{:.6},{:.6}] trans=[{:.6},{:.6},{:.6}]",
                id, quat.i, quat.j, quat.k, quat.w, trans.x, trans.y, trans.z
            );
        }
    }

    // Display first few edges
    println!("\nFirst 5 SE3 edges (relative transformations):");
    for edge in graph.edges_se3.iter().take(5) {
        let quat = edge.measurement.rotation_quaternion();
        let trans = edge.measurement.translation();
        println!(
            "  Edge {}->{}: quat=[{:.6},{:.6},{:.6},{:.6}] trans=[{:.6},{:.6},{:.6}]",
            edge.from, edge.to, quat.i, quat.j, quat.k, quat.w, trans.x, trans.y, trans.z
        );
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE3 vertices as variables
    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            // Extract quaternion and translation from SE3
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            // Format: [qx, qy, qz, qw, tx, ty, tz]
            let se3_data = na::dvector![quat.i, quat.j, quat.k, quat.w, trans.x, trans.y, trans.z];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Add SE3 between factors
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactorSE3::new(relative_pose);

        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            None, // No loss function
        );
    }

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!("  Between factors: {}", graph.edges_se3.len());

    // Initialize problem variables
    let variables = problem.initialize_variables(&initial_values);
    println!("  Variable dimensions:");
    for (name, var) in variables.iter().take(3) {
        println!("    {}: {} dimensions", name, var.get_size());
    }

    println!("\n=== SE3 POSE LOADING COMPLETE ===");
    println!("Graph successfully loaded and ready for optimization!");

    Ok(())
}
