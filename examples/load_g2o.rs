use apex_solver::load_graph;
use std::fs;
use std::path::{Path, PathBuf};

/// Statistics for a single loaded graph file
#[derive(Debug, Default)]
struct FileStatistics {
    vertices: usize,
    edges: usize,
    se2_vertices: usize,
    se3_vertices: usize,
    tum_vertices: usize,
    se2_edges: usize,
    se3_edges: usize,
}

/// Summary statistics for all loaded files
#[derive(Debug, Default)]
struct SummaryStatistics {
    files_processed: usize,
    total_files: usize,
    total_vertices: usize,
    total_edges: usize,
    total_se2_vertices: usize,
    total_se3_vertices: usize,
    total_tum_vertices: usize,
    total_se2_edges: usize,
    total_se3_edges: usize,
    successful_loads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading all graph files from the data directory...\n");

    let graph_files = find_graph_files()?;

    if graph_files.is_empty() {
        display_no_files_message();
        return Ok(());
    }

    display_found_files(&graph_files);

    let summary = process_all_files(&graph_files);

    display_summary(&summary);

    Ok(())
}

/// Find and collect all supported graph files from the data directory
fn find_graph_files() -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let data_dir = Path::new("data");
    let mut graph_files = Vec::new();

    if !data_dir.exists() || !data_dir.is_dir() {
        eprintln!("Error: data directory not found!");
        return Ok(graph_files);
    }

    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
            if is_supported_format(extension) {
                graph_files.push(path);
            }
        }
    }

    // Sort files for consistent output
    graph_files.sort();
    Ok(graph_files)
}

/// Check if the file extension represents a supported format
fn is_supported_format(extension: &str) -> bool {
    matches!(
        extension.to_lowercase().as_str(),
        "g2o" | "graph" | "txt" | "csv"
    )
}

/// Get the format name from file extension
fn get_format_name(extension: &str) -> &'static str {
    match extension.to_lowercase().as_str() {
        "g2o" => "G2O",
        "graph" => "TORO",
        "txt" | "csv" => "TUM",
        _ => "Unknown",
    }
}

/// Display message when no files are found
fn display_no_files_message() {
    println!("No supported graph files found in the data directory.");
    println!("Supported formats: .g2o (G2O), .graph (TORO), .txt/.csv (TUM)");
}

/// Display the list of found graph files
fn display_found_files(graph_files: &[PathBuf]) {
    println!("Found {} graph files:", graph_files.len());
    for file in graph_files {
        let extension = file
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("unknown");
        let format = get_format_name(extension);
        println!("  - {} ({})", file.display(), format);
    }
    println!();
}

/// Process all graph files and return summary statistics
fn process_all_files(graph_files: &[PathBuf]) -> SummaryStatistics {
    let mut summary = SummaryStatistics {
        total_files: graph_files.len(),
        ..Default::default()
    };

    for file_path in graph_files {
        match load_and_analyze_file(file_path) {
            Ok(stats) => {
                display_file_statistics(file_path, &stats);
                accumulate_statistics(&mut summary, &stats);
                summary.successful_loads += 1;
            }
            Err(e) => {
                display_load_error(file_path, &e);
            }
        }
        summary.files_processed += 1;
        println!();
    }

    summary
}

/// Load and analyze a single graph file
fn load_and_analyze_file(file_path: &Path) -> Result<FileStatistics, Box<dyn std::error::Error>> {
    let graph = load_graph(file_path)?;

    let stats = FileStatistics {
        vertices: graph.vertex_count(),
        edges: graph.edge_count(),
        se2_vertices: graph.vertices_se2.len(),
        se3_vertices: graph.vertices_se3.len(),
        tum_vertices: graph.vertices_tum.len(),
        se2_edges: graph.edges_se2.len(),
        se3_edges: graph.edges_se3.len(),
    };

    Ok(stats)
}

/// Display statistics for a successfully loaded file
fn display_file_statistics(file_path: &Path, stats: &FileStatistics) {
    let filename = file_path.file_name().unwrap().to_string_lossy();
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("unknown");
    let format = get_format_name(extension);

    println!("Loading {} ({}):", filename, format);
    println!("Successfully loaded!");
    println!("Statistics:");
    println!("  - SE2 vertices: {}", stats.se2_vertices);
    println!("  - SE3 vertices: {}", stats.se3_vertices);

    if stats.tum_vertices > 0 {
        println!("  - TUM vertices: {}", stats.tum_vertices);
    }

    println!("  - SE2 edges: {}", stats.se2_edges);
    println!("  - SE3 edges: {}", stats.se3_edges);
    println!("  - Total vertices: {}", stats.vertices);
    println!("  - Total edges: {}", stats.edges);

    // Show first vertex information if available
    display_first_vertex_info(file_path);
}

/// Display information about the first vertex in the graph
fn display_first_vertex_info(file_path: &Path) {
    // Re-load the graph to access vertex data (could be optimized by passing the graph)
    if let Ok(graph) = load_graph(file_path) {
        if let Some(vertex_0) = graph.vertices_se2.get(&0) {
            println!(
                "  - First SE2 vertex: id={}, x={:.3}, y={:.3}, θ={:.3}",
                vertex_0.id, vertex_0.x, vertex_0.y, vertex_0.theta
            );
        } else if let Some(vertex_0) = graph.vertices_se3.get(&0) {
            println!(
                "  - First SE3 vertex: id={}, translation=({:.3}, {:.3}, {:.3}), rotation=({:.3}, {:.3}, {:.3}, {:.3})",
                vertex_0.id,
                vertex_0.translation.x,
                vertex_0.translation.y,
                vertex_0.translation.z,
                vertex_0.rotation.coords.w,
                vertex_0.rotation.coords.x,
                vertex_0.rotation.coords.y,
                vertex_0.rotation.coords.z,
            );
        } else if !graph.vertices_tum.is_empty() {
            let vertex_0 = &graph.vertices_tum[0];
            println!(
                "  - First TUM vertex: timestamp={:.3}, translation=({:.3}, {:.3}, {:.3}), rotation=({:.3}, {:.3}, {:.3}, {:.3})",
                vertex_0.timestamp,
                vertex_0.translation.x,
                vertex_0.translation.y,
                vertex_0.translation.z,
                vertex_0.rotation.coords.w,
                vertex_0.rotation.coords.x,
                vertex_0.rotation.coords.y,
                vertex_0.rotation.coords.z,
            );
        }
    }
}

/// Display error message for failed file load
fn display_load_error(file_path: &Path, error: &Box<dyn std::error::Error>) {
    let filename = file_path.file_name().unwrap().to_string_lossy();
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("unknown");
    let format = get_format_name(extension);

    println!("Loading {} ({}):", filename, format);
    println!("  ❌ Failed to load: {}", error);
}

/// Accumulate statistics from a single file into the summary
fn accumulate_statistics(summary: &mut SummaryStatistics, stats: &FileStatistics) {
    summary.total_vertices += stats.vertices;
    summary.total_edges += stats.edges;
    summary.total_se2_vertices += stats.se2_vertices;
    summary.total_se3_vertices += stats.se3_vertices;
    summary.total_tum_vertices += stats.tum_vertices;
    summary.total_se2_edges += stats.se2_edges;
    summary.total_se3_edges += stats.se3_edges;
}

/// Display the final summary statistics
fn display_summary(summary: &SummaryStatistics) {
    println!("🎯 SUMMARY STATISTICS:");
    println!(
        "  Files processed: {}/{}",
        summary.successful_loads, summary.total_files
    );
    println!("  Total SE2 vertices: {}", summary.total_se2_vertices);
    println!("  Total SE3 vertices: {}", summary.total_se3_vertices);

    if summary.total_tum_vertices > 0 {
        println!("  Total TUM vertices: {}", summary.total_tum_vertices);
    }

    println!("  Total SE2 edges: {}", summary.total_se2_edges);
    println!("  Total SE3 edges: {}", summary.total_se3_edges);
    println!("  Grand total vertices: {}", summary.total_vertices);
    println!("  Grand total edges: {}", summary.total_edges);

    if summary.successful_loads == summary.total_files {
        println!("\n✅ All files loaded successfully!");
    } else {
        let failed_count = summary.total_files - summary.successful_loads;
        println!(
            "\n⚠️  {} out of {} files failed to load.",
            failed_count, summary.total_files
        );
    }
}
