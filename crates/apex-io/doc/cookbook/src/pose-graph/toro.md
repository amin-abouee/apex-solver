# TORO

**TORO** (Tree-based netwORk Optimizer) is the older 2D pose-graph format that
predates g2o. `ToroLoader` reads and writes it into the shared
[`Graph`](./graph-model.md) model. Files use the `.graph` extension, so
[`load_graph`](./graph-model.md#format-dispatch) dispatches to it automatically.

```rust
pub struct ToroLoader;
impl GraphLoader for ToroLoader { /* load, write */ }
```

## Recognised records

| Tag | Fields | Maps to |
|---|---|---|
| `VERTEX2` | `id x y theta` | `VertexSE2` |
| `EDGE2` | `from to dx dy dθ` + 6 info values | `EdgeSE2` |

TORO in this crate targets **2D** graphs. The `EDGE2` information block is the
same 6-value upper triangle of the $3\times3$ matrix used by
[`EDGE_SE2`](./g2o.md#information-matrices-upper-triangle), though TORO files
historically order the entries differently; the loader maps them into the
symmetric `Matrix3<f64>` on `EdgeSE2`.

## Errors

Identical to the [G2O error model](./graph-model.md#error-model): duplicate ids,
malformed numbers, and short lines all surface as `IoError` variants with line
context.

## Example

```rust
use apex_io::{ToroLoader, GraphLoader, load_graph};

// Direct loader:
let graph = ToroLoader::load("data/odometry/2d/intel.graph")?;
assert!(!graph.vertices_se2.is_empty());

// Or via extension dispatch (.graph → TORO):
let graph = load_graph("data/odometry/2d/intel.graph")?;
# Ok::<(), apex_io::IoError>(())
```

A minimal TORO file:

```text
VERTEX2 0 0.0 0.0 0.0
VERTEX2 1 1.0 0.0 0.0
EDGE2 0 1 1.0 0.0 0.0  500 0 0 500 0 500
```

## References

- Grisetti, G., Stachniss, C., Grzonka, S. & Burgard, W. (2007). *A Tree Parameterization for Efficiently Computing Maximum Likelihood Maps using Gradient Descent*. RSS 2007.
