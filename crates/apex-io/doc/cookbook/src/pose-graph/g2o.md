# G2O

The **g2o** text format (Kümmerle et al., 2011) is the most common pose-graph
interchange format. `G2oLoader` reads and writes it, producing the shared
[`Graph`](./graph-model.md) model. It handles both 2D (`SE2`) and 3D (`SE3:QUAT`)
graphs, comments (`#`), and blank lines.

```rust
pub struct G2oLoader;
impl GraphLoader for G2oLoader { /* load, write */ }
```

## Recognised records

| Tag | Fields | Maps to |
|---|---|---|
| `VERTEX_SE2` | `id x y theta` | `VertexSE2` |
| `VERTEX_SE3:QUAT` | `id x y z qx qy qz qw` | `VertexSE3` |
| `EDGE_SE2` | `from to dx dy dθ` + 6 info values | `EdgeSE2` |
| `EDGE_SE3:QUAT` | `from to dx dy dz qx qy qz qw` + 21 info values | `EdgeSE3` |

Lines beginning with `#` and empty lines are skipped. Unknown tags raise
`IoError::UnsupportedVertexType` / `UnsupportedEdgeType`.

## Information matrices (upper triangle)

Edges store the **upper triangle** of the symmetric information matrix
(inverse covariance), row-major:

- **`EDGE_SE2`** — 6 values fill the $3\times3$ matrix
  $\begin{bmatrix} I_{11} & I_{12} & I_{13} \\ \cdot & I_{22} & I_{23} \\ \cdot & \cdot & I_{33} \end{bmatrix}$.
- **`EDGE_SE3:QUAT`** — 21 values fill the $6\times6$ matrix; the loader mirrors
  the lower triangle to reconstruct the full symmetric matrix.

## Validation

- **Duplicate vertex ids** → `IoError::DuplicateVertex { id }`.
- **Non-unit quaternions** on `VERTEX_SE3:QUAT` / `EDGE_SE3:QUAT` are checked
  against $\lVert \mathbf{q} \rVert \approx 1$; a bad norm raises
  `IoError::InvalidQuaternion { line, norm }`.
- **Malformed numbers / short lines** → `IoError::InvalidNumber` /
  `IoError::MissingFields`, both carrying the 1-based line number.

## Public parsing helpers

Beyond the trait methods, `G2oLoader` exposes the per-record parsers (useful for
custom pipelines):

| Function | Signature |
|---|---|
| `parse_vertex_se2` | `(parts: &[&str], line_num: usize) -> Result<VertexSE2, IoError>` |
| `parse_vertex_se3` | `(parts: &[&str], line_num: usize) -> Result<VertexSE3, IoError>` |

## Example

```rust
use apex_io::{G2oLoader, GraphLoader};

let graph = G2oLoader::load("data/odometry/3d/sphere2500.g2o")?;
assert!(!graph.vertices_se3.is_empty());

// Round-trip: write back to disk.
G2oLoader::write(&graph, "/tmp/sphere_copy.g2o")?;
# Ok::<(), apex_io::IoError>(())
```

A minimal G2O file:

```text
VERTEX_SE2 0 0.0 0.0 0.0
VERTEX_SE2 1 1.0 0.0 0.0
EDGE_SE2 0 1 1.0 0.0 0.0  500 0 0 500 0 500
# 3D vertex
VERTEX_SE3:QUAT 2 0 0 0 0 0 0 1
```

## References

- Kümmerle, R., Grisetti, G., Strasdat, H., Konolige, K. & Burgard, W. (2011). *g2o: A General Framework for Graph Optimization*. ICRA 2011, 3607–3613.
