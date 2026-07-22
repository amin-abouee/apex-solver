# Logging & Visualization

## Logging

`apex-io` uses the [`tracing`](https://docs.rs/tracing) ecosystem. One helper
installs a sensible default subscriber:

```rust
pub fn init_logger();
```

Re-exported at the crate root as `apex_io::init_logger`. Call it once at program
start to get formatted, level-filtered logs (honoring `RUST_LOG`). Every
`IoError` / `BagError` also offers `.log()` and `.log_with_source(src)` to emit a
`tracing::error!` and return `self`, so errors can be logged inline while being
propagated:

```rust
use apex_io::{init_logger, load_graph};

init_logger();
let graph = load_graph("scene.g2o").map_err(|e| e.log())?;
# Ok::<(), apex_io::IoError>(())
```

The error variants carry structured context (line numbers, ids, offending
values), so logs pinpoint exactly where a parse failed.

## Visualization (feature `visualization`)

> Enabled with `--features visualization`, which pulls in `rerun`.

Graph vertices convert directly into [Rerun](https://rerun.io) primitives so a
pose graph can be streamed to the viewer:

| Method | On | Returns |
|---|---|---|
| `to_rerun_position_2d(scale) -> [f32; 2]` | `VertexSE2` | 2D point (`Points2D`) |
| `to_rerun_position_3d(scale, height) -> Vec3` | `VertexSE2` | 3D point at a fixed height |
| `to_rerun_transform(scale) -> (Vec3, Quat)` | `VertexSE3` | position + rotation (`Transform3D`) |

`to_rerun_position_2d` is always available; the `Vec3` / `Quat` variants require
the feature (they use `rerun`'s `glam` re-exports).

```rust,ignore
# #[cfg(feature = "visualization")]
# {
use apex_io::VertexSE3;
use nalgebra::{Vector3, UnitQuaternion};

let v = VertexSE3::new(0, Vector3::new(1.0, 2.0, 3.0), UnitQuaternion::identity());
let (position, rotation) = v.to_rerun_transform(0.1);   // scaled by 0.1
# }
```

The solver's own visualization observer consumes these conversions to draw
trajectories and landmarks live during optimization.
