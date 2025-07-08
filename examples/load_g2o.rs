use apex_solver::{G2oLoader};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading all G2O files from the data directory...\n");

    // Read all .g2o files from the data directory
    let data_dir = Path::new("data");
    let mut g2o_files = Vec::new();
    
    if data_dir.exists() && data_dir.is_dir() {
        for entry in fs::read_dir(data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("g2o") {
                g2o_files.push(path);
            }
        }
    } else {
        eprintln!("Error: data directory not found!");
        return Ok(());
    }

    // Sort files for consistent output
    g2o_files.sort();

    if g2o_files.is_empty() {
        println!("No .g2o files found in the data directory.");
        return Ok(());
    }

    println!("Found {} G2O files:", g2o_files.len());
    for file in &g2o_files {
        println!("  - {}", file.display());
    }
    println!();

    // Load and analyze each file
    let mut total_vertices = 0;
    let mut total_edges = 0;
    let mut total_se2_vertices = 0;
    let mut total_se3_vertices = 0;
    let mut total_se2_edges = 0;
    let mut total_se3_edges = 0;
    let mut successful_loads = 0;

    for file_path in &g2o_files {
        let filename = file_path.file_name().unwrap().to_string_lossy();
        println!("Loading {}:", filename);

        match G2oLoader::load(file_path.to_str().unwrap()) {
            Ok(graph) => {
                let vertices = graph.vertex_count();
                let edges = graph.edge_count();
                let se2_vertices = graph.vertices_se2.len();
                let se3_vertices = graph.vertices_se3.len();
                let se2_edges = graph.edges_se2.len();
                let se3_edges = graph.edges_se3.len();

                println!("Successfully loaded!");
                println!("Statistics:");
                println!("  - SE2 vertices: {}", se2_vertices);
                println!("  - SE3 vertices: {}", se3_vertices);
                println!("  - SE2 edges: {}", se2_edges);
                println!("  - SE3 edges: {}", se3_edges);
                println!("  - Total vertices: {}", vertices);
                println!("  - Total edges: {}", edges);

                // Show first vertex if available
                if let Some(vertex_0) = graph.vertices_se2.get(&0) {
                    println!("  - First SE2 vertex: id={}, x={:.3}, y={:.3}, θ={:.3}", 
                             vertex_0.id, vertex_0.x, vertex_0.y, vertex_0.theta);
                } else if let Some(vertex_0) = graph.vertices_se3.get(&0) {
                    println!("  - First SE3 vertex: id={}, translation=({:.3}, {:.3}, {:.3})", 
                             vertex_0.id, vertex_0.translation.x, vertex_0.translation.y, vertex_0.translation.z);
                }

                // Accumulate totals
                total_vertices += vertices;
                total_edges += edges;
                total_se2_vertices += se2_vertices;
                total_se3_vertices += se3_vertices;
                total_se2_edges += se2_edges;
                total_se3_edges += se3_edges;
                successful_loads += 1;
            }
            Err(e) => {
                println!("  ❌ Failed to load: {}", e);
            }
        }
        println!();
    }

    // Display summary statistics
    println!("🎯 SUMMARY STATISTICS:");
    println!("  Files processed: {}/{}", successful_loads, g2o_files.len());
    println!("  Total SE2 vertices: {}", total_se2_vertices);
    println!("  Total SE3 vertices: {}", total_se3_vertices);
    println!("  Total SE2 edges: {}", total_se2_edges);
    println!("  Total SE3 edges: {}", total_se3_edges);
    println!("  Grand total vertices: {}", total_vertices);
    println!("  Grand total edges: {}", total_edges);

    if successful_loads == g2o_files.len() {
        println!("\n✅ All files loaded successfully!");
    } else {
        println!("\n⚠️  {} out of {} files failed to load.", 
                 g2o_files.len() - successful_loads, g2o_files.len());
    }

    Ok(())
} 