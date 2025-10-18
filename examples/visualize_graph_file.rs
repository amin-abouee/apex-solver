use apex_solver::io::{
    Graph, load_graph, rerun_conversions::IntoRerunPosition2D,
    rerun_conversions::IntoRerunTransform,
};
use clap::Parser;
use rerun::{
    RecordingStreamBuilder, Transform3D,
    archetypes::{Pinhole, Points2D},
    components::Color,
};
use std::path::PathBuf;

/// Visualize a graph from a G2O/TORO file using Rerun
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the graph file to visualize
    #[arg(short = 'f', long, default_value = "data/parking-garage.g2o")]
    file_path: PathBuf,

    /// Scale factor for visualization
    #[arg(short, long, default_value_t = 0.1)]
    scale: f32,

    /// Height for SE2 poses (Z coordinate)
    #[arg(long, default_value_t = 0.0)]
    se2_height: f32,

    /// Size of camera frustums for SE3 visualization and points for SE2 visualization
    #[arg(long, default_value_t = 0.5)]
    frustum_size: f32,

    /// Field of view (in degrees) for camera frustums
    #[arg(long, default_value_t = 30.0)]
    fov_degrees: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading graph from: {}", args.file_path.display());
    let graph = load_graph(&args.file_path)?;

    // Initialize Rerun
    let rec = RecordingStreamBuilder::new("apex-solver-graph-visualization").spawn()?;

    // Print statistics
    println!("Graph loaded successfully:");
    println!("  - SE2 vertices: {}", graph.vertices_se2.len());
    println!("  - SE3 vertices: {}", graph.vertices_se3.len());
    println!("  - Total vertices: {}", graph.vertex_count());

    // Visualize SE3 vertices as camera frustums
    if !graph.vertices_se3.is_empty() {
        visualize_se3_poses(
            &graph,
            &rec,
            args.scale,
            args.frustum_size,
            args.fov_degrees,
        )?;
    }

    // Visualize SE2 vertices as 2D points
    if !graph.vertices_se2.is_empty() {
        visualize_se2_poses(&graph, &rec, args.scale, args.se2_height, args.frustum_size)?;
    }

    if graph.vertices_se3.is_empty() && graph.vertices_se2.is_empty() {
        println!("No poses found in the graph file.");
        Ok(())
    } else {
        // Keep the program running until user interrupts
        println!("Visualization ready! The Rerun viewer should open automatically.");
        println!("Press Ctrl+C to exit.");

        #[allow(unreachable_code)]
        {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            Ok(())
        }
    }
}

/// Visualize SE3 poses as camera frustums
fn visualize_se3_poses(
    graph: &Graph,
    rec: &rerun::RecordingStream,
    scale: f32,
    _frustum_size: f32,
    fov_degrees: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Visualizing {} SE3 poses as camera frustums...",
        graph.vertices_se3.len()
    );

    for (id, vertex) in &graph.vertices_se3 {
        // Use conversion trait for clean code
        let (position, rotation) = vertex.into_rerun_transform(scale);

        // Create transform from position and rotation
        let transform = Transform3D::from_translation_rotation(position, rotation);

        // Create entity path for this pose
        let entity_path = format!("se3_poses/{id}");

        // Log transform first
        rec.log(entity_path.as_str(), &transform)?;

        // Log pinhole camera with field of view
        let fov_radians = fov_degrees.to_radians();
        let pinhole = Pinhole::from_fov_and_aspect_ratio(fov_radians, 1.0);

        rec.log(entity_path.as_str(), &pinhole)?;
    }

    Ok(())
}

/// Visualize SE2 poses as 2D points
fn visualize_se2_poses(
    graph: &Graph,
    rec: &rerun::RecordingStream,
    scale: f32,
    _height: f32,
    point_size: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Visualizing {} SE2 poses as 2D points...",
        graph.vertices_se2.len()
    );

    // Collect all positions using conversion trait
    let positions: Vec<[f32; 2]> = graph
        .vertices_se2
        .values()
        .map(|vertex| vertex.into_rerun_position_2d(scale))
        .collect();

    // Create colors for all points
    let colors = vec![Color::from_rgb(255, 170, 0); positions.len()];

    // Log all points as a batch
    if !positions.is_empty() {
        rec.log(
            "se2_poses",
            &Points2D::new(positions)
                .with_colors(colors)
                .with_radii([point_size * scale]),
        )?;
    }

    Ok(())
}
