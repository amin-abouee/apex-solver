# apex-io

File I/O utilities for pose graphs (G2O, TORO, BAL formats) with SE2/SE3 support.

## Overview

This library provides parsers and writers for common SLAM and Structure-from-Motion file formats, enabling easy loading and saving of pose graph optimization problems.

## Supported Formats

- **G2O**: General Graph Optimization format (SE2 and SE3 variants)
- **TORO**: Tree-based netwORk Optimizer format (2D SLAM)
- **BAL**: Bundle Adjustment in the Large format (Structure-from-Motion)

## Features

- Fast memory-mapped file reading
- Parallel parsing with rayon
- Type-safe graph representation
- Optional Rerun visualization support

## Installation

```toml
[dependencies]
apex-io = "1.0.0"
```

For visualization features:

```toml
[dependencies]
apex-io = { version = "1.0.0", features = ["visualization"] }
```

## Usage

### Loading a G2O File

```rust
use apex_io::{G2oLoader, GraphLoader};

let graph = G2oLoader::load("path/to/file.g2o")?;
println!("Loaded {} vertices and {} edges", 
    graph.vertex_count(), graph.edge_count());
```

### Auto-detect Format

```rust
use apex_io::load_graph;

// Automatically detects format from extension
let graph = load_graph("path/to/file.g2o")?;
```

### Working with Vertices and Edges

```rust
use apex_io::{Graph, VertexSE3, EdgeSE3};

let graph = load_graph("sphere2500.g2o")?;

// Iterate over SE3 vertices
for (id, vertex) in &graph.vertices_se3 {
    println!("Vertex {}: {:?}", id, vertex.translation());
}

// Iterate over SE3 edges
for edge in &graph.edges_se3 {
    println!("Edge from {} to {}", edge.from, edge.to);
}
```

## Dependencies

- `apex-manifolds`: Lie group types (SE2, SE3)
- `nalgebra`: Linear algebra
- `memmap2`: Fast file reading
- `rayon`: Parallel parsing

## License

Apache-2.0
