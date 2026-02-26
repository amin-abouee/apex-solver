# apex-io

High-performance file I/O for pose graphs (G2O, TORO) and bundle adjustment (BAL) with SE2/SE3 support.

## Overview

This library provides parsers and writers for common SLAM (Simultaneous Localization and Mapping) and Structure-from-Motion file formats. It enables easy loading and saving of pose graph optimization problems with:

- **Memory-mapped file reading** for fast I/O on large files
- **Parallel parsing** with rayon for multi-core acceleration
- **Type-safe graph representation** with separate SE2/SE3 types
- **Comprehensive error handling** with line numbers and context
- **Optional visualization support** via Rerun integration

Target applications include:
- Pose graph optimization (2D and 3D SLAM)
- Bundle adjustment for Structure-from-Motion
- Visual odometry and localization
- Robotics trajectory optimization

## Supported Formats

| Format | Description | Vertex Types | Edge Types | Read | Write |
|--------|-------------|--------------|------------|------|-------|
| **G2O** | General Graph Optimization | SE2, SE3 | SE2, SE3 | ✓ | ✓ |
| **TORO** | Tree-based netwORk Optimizer | SE2 only | SE2 only | ✓ | ✓ |
| **BAL** | Bundle Adjustment in the Large | Cameras, Points | Observations | ✓ | - |

### G2O Format

The G2O (General Graph Optimization) format is widely used in robotics for pose graph optimization. It supports both 2D (SE2) and 3D (SE3) graphs.

**File Structure:**
```
# Comments start with #
VERTEX_SE2 id x y theta
VERTEX_SE3:QUAT id x y z qx qy qz qw
EDGE_SE2 from to dx dy dtheta info_xx info_xy info_yy info_xt info_yt info_tt
EDGE_SE3:QUAT from to dx dy dz dqx dqy dqz dqw [21 information matrix values]
```

**Example (2D):**
```
VERTEX_SE2 0 0.0 0.0 0.0
VERTEX_SE2 1 1.0 0.0 0.0
EDGE_SE2 0 1 1.0 0.0 0.0 500 0 500 0 0 500
```

### TORO Format

The TORO (Tree-based netwORk Optimizer) format is a legacy 2D SLAM format.

**File Structure:**
```
VERTEX2 id x y theta
EDGE2 from to dx dy dtheta info_xx info_xy info_yy info_xt info_yt info_tt
```

**Note:** TORO only supports SE2 (2D) graphs. Attempting to write SE3 data will result in an error.

### BAL Format

The BAL (Bundle Adjustment in the Large) format is used for large-scale bundle adjustment benchmarks in computer vision.

**File Structure:**
```
num_cameras num_points num_observations
# Observations block (one per line)
camera_idx point_idx pixel_x pixel_y
...
# Cameras block (9 parameters per camera, one per line)
rotation_x
rotation_y
rotation_z
translation_x
translation_y
translation_z
focal_length
k1
k2
...
# Points block (3 coordinates per point, one per line)
x
y
z
...
```

**Camera Model:** Snavely's 9-parameter model from Bundler:
- 3 parameters: Axis-angle rotation (rx, ry, rz)
- 3 parameters: Translation (tx, ty, tz)
- 1 parameter: Focal length (f)
- 2 parameters: Radial distortion (k1, k2)

## Installation

```toml
[dependencies]
apex-io = "0.1.0"
```

For visualization features (Rerun integration):

```toml
[dependencies]
apex-io = { version = "0.1.0", features = ["visualization"] }
```

## Data Structures

### Graph

The main container for pose graph data:

```rust
pub struct Graph {
    pub vertices_se2: HashMap<usize, VertexSE2>,
    pub vertices_se3: HashMap<usize, VertexSE3>,
    pub edges_se2: Vec<EdgeSE2>,
    pub edges_se3: Vec<EdgeSE3>,
}

impl Graph {
    pub fn new() -> Self;
    pub fn vertex_count(&self) -> usize;  // Total SE2 + SE3 vertices
    pub fn edge_count(&self) -> usize;    // Total SE2 + SE3 edges
}
```

### SE2 Types (2D)

**VertexSE2** - 2D pose (position + orientation):
```rust
pub struct VertexSE2 {
    id: usize,
    pose: SE2,  // From apex-manifolds
}

impl VertexSE2 {
    pub fn new(id: usize, pose: SE2) -> Self;
    pub fn from_vector(id: usize, data: &[f64]) -> Self;
    pub fn id(&self) -> usize;
    pub fn x(&self) -> f64;
    pub fn y(&self) -> f64;
    pub fn theta(&self) -> f64;
}
```

**EdgeSE2** - 2D constraint between vertices:
```rust
pub struct EdgeSE2 {
    pub from: usize,
    pub to: usize,
    pub measurement: SE2,
    pub information: Matrix3<f64>,  // 3x3 information matrix
}
```

### SE3 Types (3D)

**VertexSE3** - 3D pose (position + orientation):
```rust
pub struct VertexSE3 {
    id: usize,
    pose: SE3,  // From apex-manifolds
}

impl VertexSE3 {
    pub fn new(id: usize, pose: SE3) -> Self;
    pub fn from_vector(id: usize, data: &[f64]) -> Self;
    pub fn from_translation_quaternion(id: usize, t: Vector3, q: UnitQuaternion) -> Self;
    pub fn id(&self) -> usize;
    pub fn translation(&self) -> Vector3<f64>;
    pub fn rotation(&self) -> UnitQuaternion<f64>;
    pub fn x(&self) -> f64;
    pub fn y(&self) -> f64;
    pub fn z(&self) -> f64;
}
```

**EdgeSE3** - 3D constraint between vertices:
```rust
pub struct EdgeSE3 {
    pub from: usize,
    pub to: usize,
    pub measurement: SE3,
    pub information: Matrix6<f64>,  // 6x6 information matrix
}
```

### BAL Types

**BalDataset** - Complete bundle adjustment problem:
```rust
pub struct BalDataset {
    pub cameras: Vec<BalCamera>,
    pub points: Vec<BalPoint>,
    pub observations: Vec<BalObservation>,
}
```

**BalCamera** - Snavely's 9-parameter camera model:
```rust
pub struct BalCamera {
    pub rotation: Vector3<f64>,     // Axis-angle (rx, ry, rz)
    pub translation: Vector3<f64>,  // (tx, ty, tz)
    pub focal_length: f64,
    pub k1: f64,                    // Radial distortion coefficient
    pub k2: f64,
}

impl BalCamera {
    pub fn clamped_focal_length(&self, min: f64, max: f64) -> f64;
}
```

**BalPoint** - 3D landmark:
```rust
pub struct BalPoint {
    pub position: Vector3<f64>,
}
```

**BalObservation** - 2D image observation:
```rust
pub struct BalObservation {
    pub camera_index: usize,
    pub point_index: usize,
    pub x: f64,  // Pixel x-coordinate
    pub y: f64,  // Pixel y-coordinate
}
```

## API Reference

### GraphLoader Trait

All pose graph loaders implement this trait:

```rust
pub trait GraphLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Graph, IoError>;
    fn write<P: AsRef<Path>>(graph: &Graph, path: P) -> Result<(), IoError>;
}
```

### Loaders

| Loader | Formats | Description |
|--------|---------|-------------|
| `G2oLoader` | `.g2o` | G2O format with SE2/SE3 support |
| `ToroLoader` | `.graph` | TORO format (SE2 only) |
| `BalLoader` | `.txt` | BAL bundle adjustment format |

### Convenience Functions

```rust
/// Auto-detect format from file extension
pub fn load_graph<P: AsRef<Path>>(path: P) -> Result<Graph, IoError>;
```

Supported extensions:
- `.g2o` → G2oLoader
- `.graph` → ToroLoader

## Error Handling

The `IoError` enum provides detailed error information:

| Variant | Description |
|---------|-------------|
| `Io(io::Error)` | Underlying I/O error |
| `Parse { line, message }` | Parse error with line number |
| `UnsupportedVertexType(String)` | Unknown vertex type in file |
| `UnsupportedEdgeType(String)` | Unknown edge type in file |
| `InvalidNumber { line, value }` | Failed to parse number |
| `MissingFields { line }` | Insufficient fields on line |
| `DuplicateVertex { id }` | Vertex ID already exists |
| `InvalidQuaternion { line, norm }` | Quaternion not unit length |
| `UnsupportedFormat(String)` | File extension not recognized |
| `FileCreationFailed { path, reason }` | Could not create output file |

All errors include context via `.log()` and `.log_with_source()` methods for tracing integration.

## Usage Examples

### Loading a G2O File

```rust
use apex_io::{G2oLoader, GraphLoader};

let graph = G2oLoader::load("data/sphere2500.g2o")?;
println!("Loaded {} vertices and {} edges", 
    graph.vertex_count(), graph.edge_count());
```

### Auto-detect Format

```rust
use apex_io::load_graph;

// Automatically detects format from extension
let graph = load_graph("data/M3500.g2o")?;
```

### Working with SE3 Vertices and Edges

```rust
use apex_io::{G2oLoader, GraphLoader};

let graph = G2oLoader::load("sphere2500.g2o")?;

// Iterate over SE3 vertices
for (id, vertex) in &graph.vertices_se3 {
    println!("Vertex {}: position ({:.2}, {:.2}, {:.2})", 
        id, vertex.x(), vertex.y(), vertex.z());
}

// Iterate over SE3 edges
for edge in &graph.edges_se3 {
    println!("Edge {} -> {}", edge.from, edge.to);
}
```

### Working with SE2 Vertices and Edges

```rust
use apex_io::{G2oLoader, GraphLoader};

let graph = G2oLoader::load("intel.g2o")?;

// Iterate over SE2 vertices
for (id, vertex) in &graph.vertices_se2 {
    println!("Vertex {}: ({:.2}, {:.2}, {:.2} rad)", 
        id, vertex.x(), vertex.y(), vertex.theta());
}

// Iterate over SE2 edges
for edge in &graph.edges_se2 {
    println!("Edge {} -> {}: measurement = ({:.2}, {:.2}, {:.2})",
        edge.from, edge.to,
        edge.measurement.x(), edge.measurement.y(), edge.measurement.theta());
}
```

### Writing a Graph

```rust
use apex_io::{G2oLoader, GraphLoader};

let graph = G2oLoader::load("input.g2o")?;

// Write to G2O format
G2oLoader::write(&graph, "output.g2o")?;

// Write to TORO format (SE2 only)
ToroLoader::write(&graph, "output.graph")?;
```

### Loading BAL Dataset

```rust
use apex_io::BalLoader;

let dataset = BalLoader::load("problem-1778-993923-pre.txt")?;
println!("Loaded {} cameras, {} points, {} observations",
    dataset.cameras.len(),
    dataset.points.len(),
    dataset.observations.len());

// Access camera parameters
for (i, cam) in dataset.cameras.iter().enumerate() {
    println!("Camera {}: focal={:.1}, k1={:.4}, k2={:.4}",
        i, cam.focal_length, cam.k1, cam.k2);
}

// Access observations
for obs in &dataset.observations {
    println!("Camera {} sees point {} at pixel ({:.1}, {:.1})",
        obs.camera_index, obs.point_index, obs.x, obs.y);
}
```

## Performance

### Memory Mapping

All loaders use memory-mapped file I/O via `memmap2` for efficient reading of large files. This avoids loading the entire file into memory and leverages the operating system's page cache.

### Parallel Parsing

For files with more than 1000 lines, parsing is automatically parallelized using `rayon`. This significantly speeds up loading of large pose graphs and BAL datasets.

### Pre-allocation

Collections are pre-allocated based on estimated file size to minimize memory reallocations during parsing.

## Visualization Feature

Enable the `visualization` feature for Rerun integration:

```toml
apex-io = { version = "0.1.0", features = ["visualization"] }
```

This adds helper methods for converting vertices to Rerun types:

```rust
// SE2 vertices
let pos_2d: [f32; 2] = vertex_se2.to_rerun_position_2d(scale);
let pos_3d: Vec3 = vertex_se2.to_rerun_position_3d(scale, height);

// SE3 vertices
let (position, rotation): (Vec3, Quat) = vertex_se3.to_rerun_transform(scale);
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| `apex-manifolds` | Lie group types (SE2, SE3) |
| `nalgebra` | Linear algebra (vectors, matrices) |
| `memmap2` | Memory-mapped file I/O |
| `rayon` | Parallel parsing |
| `thiserror` | Error handling |
| `tracing` | Structured logging |
| `serde`, `serde_json` | Serialization support |
| `chrono` | Timestamps in file headers |
| `rerun` | Visualization (optional) |

## References

- [g2o: A General Framework for Graph Optimization](https://github.com/RainerKuemmerle/g2o)
- [TORO: Tree-based netwORk Optimizer](https://www.openslam.org/toro.html)
- [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
- [Bundler: Structure from Motion](https://www.cs.cornell.edu/~snavely/bundler/)

## License

Apache-2.0
