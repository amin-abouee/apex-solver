# apex-solver

A Rust-based non-linear least squares optimization library designed for computer vision applications including bundle adjustment, pose graph optimization, and SLAM. Built for high performance with focus on zero-cost abstractions and memory safety.

## Solver Overview

This library provides non-linear least squares optimization specifically tailored for computer vision problems. It handles pose estimation, 3D reconstruction, and visual-inertial systems by formulating problems as factor graphs where variables (poses, landmarks) are connected through measurement constraints.

## Supported Manifolds

- **SE(3)**: 3D rigid body transformations (6-DOF poses) - translation + rotation
- **SO(3)**: 3D rotations using quaternion representation  
- **SE(2)**: 2D rigid transformations (3-DOF poses) - translation + rotation
- **SO(2)**: 2D rotations

All manifolds support analytic Jacobian computations, exponential/logarithmic maps, and tangent space operations with right and left perturbation models.

## I/O Support

**Supported Formats:**
- **G2O** (`.g2o`): Industry-standard pose graph optimization format
- **TORO** (`.graph`): Tree-based network optimizer format  
- **TUM** (`.txt`, `.csv`): TUM RGB-D dataset trajectory format

Automatic format detection, robust error handling, and support for mixed 2D/3D graphs in a single data structure.

## Problem Formulation & Residual Blocks

Problems are structured as factor graphs with:
- **Vertices**: Pose variables (SE(2)/SE(3)) with unique IDs
- **Edges**: Measurement constraints between poses with information matrices
- **Unified Graph**: Single `G2oGraph` structure handling mixed vertex/edge types

The solver optimizes by minimizing residuals between predicted and observed measurements across all constraint edges.

## Optimization

Currently supports graph loading and analysis. The optimization backend is designed to integrate with existing Rust optimization libraries, providing:
- Efficient sparse matrix operations
- Manifold-aware parameter updates
- Information matrix handling for weighted constraints

## Usage

```rust
use apex_solver::{load_graph, manifold::se3::SE3};

// Load pose graph
let graph = load_graph("data/parking-garage.g2o")?;
println!("Loaded {} vertices", graph.vertex_count());

// Work with SE(3) manifolds  
let pose = SE3::identity();
let tangent = SE3Tangent::random();
let perturbed = pose.plus(&tangent, None, None);
```

## Examples

```bash
# Analyze graph files
cargo run --example load_graph_file

# Visualize pose graphs
cargo run --example visualize_graph_file -- data/sphere2500.g2o
```

## License

Licensed under the Apache Ver. 2.0 License.