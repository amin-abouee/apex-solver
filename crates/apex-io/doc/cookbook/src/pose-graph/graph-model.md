# The Graph Model

All pose-graph loaders (G2O, TORO) produce the same in-memory structure: a
[`Graph`](#graph) of SE(2)/SE(3) vertices and their relative-pose edge
constraints. This page documents that shared model, the [`GraphLoader`](#graphloader-trait)
trait every format implements, the [`load_graph`](#format-dispatch) dispatcher,
and the [error model](#error-model).

## `Graph`

```rust
pub struct Graph {
    pub vertices_se2: HashMap<usize, VertexSE2>,
    pub vertices_se3: HashMap<usize, VertexSE3>,
    pub edges_se2:    Vec<EdgeSE2>,
    pub edges_se3:    Vec<EdgeSE3>,
}
```

A graph holds 2D and 3D vertices side by side, keyed by their integer id, plus
the two edge lists. Mixed 2D/3D graphs are representable but formats typically
populate one dimension.

| Method | Signature | Description |
|---|---|---|
| `new` | `fn new() -> Graph` | Empty graph (also `Default`). |
| `vertex_count` | `fn vertex_count(&self) -> usize` | `vertices_se2.len() + vertices_se3.len()`. |
| `edge_count` | `fn edge_count(&self) -> usize` | `edges_se2.len() + edges_se3.len()`. |

`Graph` implements `Clone` and `Display` (the `Display` prints every vertex and
edge with counts).

## Vertices

### `VertexSE2`

```rust
pub struct VertexSE2 { pub id: usize, pub pose: SE2 }
```

| Method | Description |
|---|---|
| `new(id, x, y, theta)` | Construct from planar pose. |
| `from_vector(id, Vector3)` | Construct from `[x, y, theta]`. |
| `id() / x() / y() / theta()` | Accessors (pose is an `apex_manifolds::se2::SE2`). |
| `to_rerun_position_2d(scale) -> [f32; 2]` | 2D point for Rerun. |
| `to_rerun_position_3d(scale, height) -> Vec3` | 3D point (feature `visualization`). |

### `VertexSE3`

```rust
pub struct VertexSE3 { pub id: usize, pub pose: SE3 }
```

| Method | Description |
|---|---|
| `new(id, translation: Vector3, rotation: UnitQuaternion)` | Construct from pose. |
| `from_vector(id, [f64; 7])` | From `[tx, ty, tz, qx, qy, qz, qw]`. |
| `from_translation_quaternion(id, Vector3, Quaternion)` | From raw quaternion. |
| `id() / translation() / rotation() / x() / y() / z()` | Accessors. |
| `to_rerun_transform(scale) -> (Vec3, Quat)` | 3D transform (feature `visualization`). |

## Edges

`EdgeSE2` and `EdgeSE3` carry a relative-pose **measurement** and an
**information matrix** (inverse covariance) — the weight used by the optimizer.

```rust
pub struct EdgeSE2 { pub from: usize, pub to: usize, pub measurement: SE2, pub information: Matrix3<f64> }
pub struct EdgeSE3 { pub from: usize, pub to: usize, pub measurement: SE3, pub information: Matrix6<f64> }
```

| Constructor | Signature |
|---|---|
| `EdgeSE2::new` | `(from, to, dx, dy, dtheta, information: Matrix3)` |
| `EdgeSE3::new` | `(from, to, translation: Vector3, rotation: UnitQuaternion, information: Matrix6)` |

The $3\times3$ (SE2) and $6\times6$ (SE3) information matrices are stored exactly
as parsed; loaders reconstruct the full symmetric matrix from the upper triangle
stored in the file.

## `GraphLoader` trait

Every file format implements one trait, so loaders are interchangeable:

```rust
pub trait GraphLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Graph, IoError>;
    fn write<P: AsRef<Path>>(graph: &Graph, path: P) -> Result<(), IoError>;
}
```

Implementors: [`G2oLoader`](./g2o.md), [`ToroLoader`](./toro.md).

## Format dispatch

`load_graph` picks the loader from the file extension:

```rust
pub fn load_graph<P: AsRef<Path>>(path: P) -> Result<Graph, IoError>;
```

| Extension | Loader |
|---|---|
| `.g2o` | `G2oLoader` |
| `.graph` | `ToroLoader` |
| _other_ | `IoError::UnsupportedFormat` |

<a id="error-model"></a>
## Error model

All graph I/O returns `Result<_, IoError>`:

```rust
pub enum IoError {
    Io(std::io::Error),
    Parse { line, message },
    UnsupportedVertexType(String),
    UnsupportedEdgeType(String),
    InvalidNumber { line, value },
    MissingFields { line },
    DuplicateVertex { id },
    InvalidQuaternion { line, norm },
    UnsupportedFormat(String),
    FileCreationFailed { path, reason },
}
```

Every variant carries context (line numbers, ids, offending values). Two helpers
attach a `tracing::error!` log and return `self` for chaining:
`err.log()` and `err.log_with_source(source)`.

## Example

```rust
use apex_io::{Graph, VertexSE2, EdgeSE2, GraphLoader, G2oLoader};
use nalgebra::Matrix3;

// Build a graph by hand …
let mut g = Graph::new();
g.vertices_se2.insert(0, VertexSE2::new(0, 0.0, 0.0, 0.0));
g.vertices_se2.insert(1, VertexSE2::new(1, 1.0, 0.0, 0.0));
g.edges_se2.push(EdgeSE2::new(0, 1, 1.0, 0.0, 0.0, Matrix3::identity()));

// … and write it back out in G2O.
G2oLoader::write(&g, "/tmp/out.g2o")?;
# Ok::<(), apex_io::IoError>(())
```

## References

- Kümmerle, R. et al. (2011). *g2o: A General Framework for Graph Optimization*. ICRA 2011.
