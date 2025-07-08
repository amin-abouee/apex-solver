use apex_solver::{G2oLoader, G2oError};

fn main() -> Result<(), G2oError> {
    println!("Loading G2O files from the data directory...\n");

    // Load M3500.g2o (SE2 poses)
    println!("Loading M3500.g2o (2D pose graph):");
    let m3500 = G2oLoader::load("data/M3500.g2o")?;
    println!("  - SE2 vertices: {}", m3500.vertices_se2.len());
    println!("  - SE3 vertices: {}", m3500.vertices_se3.len());
    println!("  - SE2 edges: {}", m3500.edges_se2.len());
    println!("  - SE3 edges: {}", m3500.edges_se3.len());
    
    if let Some(vertex_0) = m3500.vertices_se2.get(&0) {
        println!("  - First vertex: id={}, x={:.3}, y={:.3}, θ={:.3}", 
                 vertex_0.id, vertex_0.x, vertex_0.y, vertex_0.theta);
    }

    // Load parking-garage.g2o (SE3 poses)
    println!("\nLoading parking-garage.g2o (3D pose graph):");
    let parking = G2oLoader::load("data/parking-garage.g2o")?;
    println!("  - SE2 vertices: {}", parking.vertices_se2.len());
    println!("  - SE3 vertices: {}", parking.vertices_se3.len());
    println!("  - SE2 edges: {}", parking.edges_se2.len());
    println!("  - SE3 edges: {}", parking.edges_se3.len());
    

    // Load sphere2500.g2o (SE3 poses)
    println!("\nLoading sphere2500.g2o (3D pose graph):");
    let sphere = G2oLoader::load("data/sphere2500.g2o")?;
    println!("  - SE2 vertices: {}", sphere.vertices_se2.len());
    println!("  - SE3 vertices: {}", sphere.vertices_se3.len());
    println!("  - SE2 edges: {}", sphere.edges_se2.len());
    println!("  - SE3 edges: {}", sphere.edges_se3.len());

    println!("\n✅ All files loaded successfully!");
    println!("\n📊 Total statistics:");
    println!("  - Total vertices: {}", 
             m3500.vertex_count() + parking.vertex_count() + sphere.vertex_count());
    println!("  - Total edges: {}", 
             m3500.edge_count() + parking.edge_count() + sphere.edge_count());

    Ok(())
} 