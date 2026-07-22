# Introduction

`apex-io` is the input/output layer of Apex Solver. It reads and writes the file
formats, sensor datasets, and message streams that feed the optimizer, and it
converts them into the `apex-manifolds` pose types (`SE2`, `SE3`) used throughout
the solver. This book documents **every public functionality** of the crate,
organised by domain.

## What the crate covers

| Domain | Formats / Sources | Entry points |
|---|---|---|
| **Pose-graph files** | G2O (`.g2o`), TORO (`.graph`) | [`G2oLoader`](./pose-graph/g2o.md), [`ToroLoader`](./pose-graph/toro.md), [`load_graph`](./pose-graph/graph-model.md) |
| **Bundle adjustment** | BAL (`.txt`/`.bz2`) | [`BalLoader`](./pose-graph/bal.md) |
| **Sensor datasets** | ASL / EuRoC (MAV0 layout) | [`AslReader`, `AslStream`](./datasets/asl.md) |
| **ROS1 bags** | `.bag` (v2.0) | [`Ros1Reader`, `Ros1Writer`](./rosbag/ros1.md) |
| **ROS2 bags** | SQLite3 & MCAP | [`Reader`, `Writer`](./rosbag/ros2.md) |
| **Live DDS** | ROS2 over the wire (feature `dds`) | [`DdsSubscriber`, `DdsListener`](./rosbag/dds.md) |
| **Datasets** | download + registry | [`DatasetRegistry`, `ensure_*_dataset`](./utils/datasets.md) |
| **Tooling** | 7 CLI binaries | [CLI Tools](./utils/cli.md) |

## Design conventions

- **Fallible everywhere.** Public reads/writes return `Result<_, IoError>` (graph
  formats) or `Result<_, BagError>` / `Result<_, AslError>` (bags, datasets). No
  panics on malformed input — see [The Graph Model](./pose-graph/graph-model.md#error-model).
- **Pose types come from `apex-manifolds`.** Graph vertices hold `SE2` / `SE3`
  values, not raw arrays. Quaternions follow the crate-wide $w$-first convention.
- **Streaming where it matters.** Large sources (ROS2 bags, ASL images) expose
  iterators / cursors so you never have to hold a whole dataset in memory.
- **Zero-copy raw path.** Every bag reader offers a *raw* variant
  (`RawMessage`, `raw_messages`) that hands you the serialized bytes without
  deserializing into a typed message — ideal for copy/filter/convert pipelines.

## Feature flags

Most functionality is available with the default feature set. Three optional
features gate heavier dependencies (full details in
[Feature Flags](./appendix/features.md)):

| Feature | Enables | Extra deps |
|---|---|---|
| `visualization` | `to_rerun_*` conversions on graph vertices | `rerun` |
| `dds` | Live ROS2 subscription (`dds` module, `dds_multi_listener` bin) | `rustdds`, `tokio`, `futures` |
| `asl-async` | Async ASL streaming | `tokio` |

## A worked first example

```rust
use apex_io::{load_graph, G2oLoader, GraphLoader};

// Dispatch by extension (.g2o → G2O, .graph → TORO):
let graph = load_graph("data/odometry/3d/sphere2500.g2o")?;
println!("{} vertices, {} edges", graph.vertex_count(), graph.edge_count());

// Or call a specific loader directly:
let graph = G2oLoader::load("data/odometry/2d/M3500.g2o")?;
# Ok::<(), apex_io::IoError>(())
```

## Build the Book

```bash
cargo install mdbook --locked
cargo install mdbook-katex --locked

mdbook serve crates/apex-io/doc/cookbook --open
mdbook build crates/apex-io/doc/cookbook
```
