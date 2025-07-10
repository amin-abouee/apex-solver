# apex-solver

A Rust-based library for efficient non-linear least squares optimization, with a focus on computer vision applications such as bundle adjustment, graph-based pose optimization, and SLAM. Designed for high performance, flexibility, and scalability in visual-inertial systems and 3D reconstruction tasks.

## Features

- **Manifold Operations**: Complete Lie group implementations for SE(2), SE(3), SO(2), SO(3), and more
- **Multiple File Format Support**: Load and process G2O, TORO, and TUM trajectory files
- **High Performance**: Written in Rust with focus on zero-cost abstractions and memory safety
- **Visualization Support**: Built-in visualization capabilities using Rerun
- **Comprehensive Testing**: Extensive test coverage for all mathematical operations

## Modules

### 🧮 Manifold Module (`src/manifold/`)

The manifold module provides robust implementations of Lie groups commonly used in robotics and computer vision. Inspired by the [manif](https://github.com/artivis/manif) C++ library, it offers:

**Key Features:**
- **SE(3)**: Special Euclidean group for 3D rigid body transformations (6-DOF poses)
- **SO(3)**: Special Orthogonal group for 3D rotations
- **SE(2)**: 2D rigid transformations (3-DOF poses)
- **SO(2)**: 2D rotations

**Mathematical Operations:**
- Analytic Jacobian computations for all operations
- Right and left perturbation models
- Composition and inverse operations
- Exponential and logarithmic maps
- Tangent space operations
- Adjoint representations

**Design Philosophy:**
Each manifold represents a Lie group with its associated tangent space (Lie algebra). Operations are automatically differentiated with respect to perturbations on the local tangent space, making it ideal for optimization algorithms that require precise gradients.

### 📁 IO Module (`src/io/`)

The IO module provides comprehensive support for loading and processing various graph file formats commonly used in SLAM and computer vision research:

**Supported Formats:**
- **G2O Format** (`.g2o`): Industry-standard format for pose graph optimization
- **TORO Format** (`.graph`): Tree-based network optimizer format
- **TUM Format** (`.txt`, `.csv`): TUM RGB-D dataset trajectory format

**Data Structures:**
- **Vertices**: SE(2) poses (x, y, θ), SE(3) poses (translation + quaternion), TUM trajectories with timestamps
- **Edges**: Constraint measurements between poses with full information matrices
- **Unified Graph**: Single data structure (`G2oGraph`) that can hold mixed vertex and edge types

**Key Features:**
- Automatic format detection based on file extension
- Robust error handling with detailed error messages
- Memory-efficient parsing for large datasets
- Support for mixed 2D/3D graphs in a single file

## Examples

### 📊 `load_graph_file.rs` - Comprehensive Graph Analysis

This example demonstrates how to load and analyze graph files from multiple formats. It provides detailed statistics and insights into the structure of pose graphs.

**Features:**
- **Multi-format Support**: Automatically detects and loads G2O, TORO, and TUM files
- **Detailed Statistics**: Reports vertex counts, edge counts, and data distribution
- **Batch Processing**: Processes all supported files in the `data/` directory
- **Error Handling**: Graceful handling of corrupted or unsupported files
- **First Vertex Display**: Shows the actual pose data of the first vertex for verification

**Sample Output:**
```
Found 15 graph files:
  - parking-garage.g2o (G2O)
  - M3500.g2o (G2O)
  ...

Loading parking-garage.g2o (G2O):
Successfully loaded!
Statistics:
  - SE3 vertices: 1661
  - SE2 vertices: 0
  - SE3 edges: 6275
  - Total vertices: 1661
  - Total edges: 6275
  - First SE3 vertex: id=0, translation=(0.000, 0.000, 0.000), rotation=(0.000, 0.000, 0.000, 1.000)
```

### 🎨 `visualize_graph_file.rs` - Interactive 3D Visualization

This example provides real-time 3D visualization of pose graphs using the Rerun visualization framework. It's perfect for understanding the spatial structure of SLAM datasets.

**Features:**
- **3D Visualization**: SE(3) poses rendered as camera frustums with proper orientation
- **2D Visualization**: SE(2) poses displayed as colored points with trajectory paths
- **Interactive Viewer**: Pan, zoom, and rotate through the Rerun web interface
- **Customizable Display**: Adjustable scale factors, frustum sizes, and field of view
- **Real-time Updates**: Live visualization that updates as you modify parameters

**Command Line Options:**
```bash
# Visualize with default settings
cargo run --example visualize_graph_file

# Custom file and scale
cargo run --example visualize_graph_file -- data/sphere2500.g2o --scale 0.05

# Adjust visualization parameters
cargo run --example visualize_graph_file -- --frustum-size 1.0 --fov-degrees 45.0
```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd apex-solver
   ```

2. **Run the examples:**
   ```bash
   # Analyze all graph files
   cargo run --example load_graph_file
   
   # Visualize a specific graph
   cargo run --example visualize_graph_file
   ```

3. **Use in your project:**
   ```rust
   use apex_solver::{load_graph, manifold::se3::SE3};
   
   // Load a graph file
   let graph = load_graph("data/parking-garage.g2o")?;
   println!("Loaded {} vertices", graph.vertex_count());
   
   // Work with SE(3) manifolds
   let pose = SE3::identity();
   let tangent = SE3Tangent::random();
   let perturbed = pose.plus(&tangent, None, None);
   ```

## Data

The `data/` directory contains various test datasets:
- **3D Datasets**: `parking-garage.g2o`, `sphere2500.g2o`, `torus3D.g2o`
- **2D Datasets**: `M3500.g2o`, `intel.g2o`, `manhattanOlson3500.g2o`
- **Mixed Datasets**: Various real-world SLAM datasets

## Development

### Code Quality

This project maintains high code quality standards:

```bash
# Format code
cargo fmt

# Run linting
cargo clippy

# Run tests
cargo test

# Check compilation
cargo check
```

All code follows Rust best practices and is regularly checked for formatting, linting, and correctness.

## License

Licensed under the MIT License. See `LICENSE` for details.
