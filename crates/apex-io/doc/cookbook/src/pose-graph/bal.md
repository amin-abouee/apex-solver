# BAL (Bundle Adjustment in the Large)

The **BAL** format (Agarwal et al., UW) ships the classic large-scale bundle
adjustment problems (Ladybug, Venice, Dubrovnik, …). Unlike the pose-graph
formats, it does *not* map onto [`Graph`](./graph-model.md); it loads into a
dedicated [`BalDataset`](#baldataset) of cameras, 3D points, and 2D observations.
`BalLoader` reads plain-text and `.bz2`-compressed files.

```rust
pub struct BalLoader;
impl BalLoader { pub fn load(path: impl AsRef<Path>) -> Result<BalDataset, IoError>; }
```

## File layout

```text
<num_cameras> <num_points> <num_observations>
<cam_idx> <pt_idx> <x> <y>          × num_observations
<camera params: 9 lines each>       × num_cameras
<point params: 3 lines each>        × num_points
```

Each camera uses **Snavely's 9-parameter model**; each point is a 3-vector.

## Data types

### `BalCamera` — Snavely 9-parameter model

```rust
pub struct BalCamera {
    pub rotation: Vector3<f64>,     // axis-angle (rx, ry, rz)
    pub translation: Vector3<f64>,  // (tx, ty, tz)
    pub focal_length: f64,
    pub k1: f64,                    // radial distortion
    pub k2: f64,
}
```

Rotation is the compact **axis-angle** representation (Bundler / `-Z` looking
camera — see the camera cookbook's [BAL Pinhole](../../../../apex-camera-models/doc/cookbook/src/bal-pinhole.md) chapter for the projection math).

### `BalPoint`

```rust
pub struct BalPoint { pub position: Vector3<f64> }
```

### `BalObservation`

```rust
pub struct BalObservation {
    pub camera_index: usize,
    pub point_index: usize,
    pub x: f64,   // pixel measurement
    pub y: f64,
}
```

### `BalDataset`

```rust
pub struct BalDataset {
    pub cameras: Vec<BalCamera>,
    pub points: Vec<BalPoint>,
    pub observations: Vec<BalObservation>,
}
```

## Focal-length normalization

Some BAL problems contain invalid (negative or non-finite) focal lengths. On
load, each is replaced by the constant

```rust
pub const DEFAULT_FOCAL_LENGTH: f64 = 500.0;
```

while all positive values are preserved unchanged, so the loaded dataset is
always ready for optimization.

## Example

```rust
use apex_io::BalLoader;

let dataset = BalLoader::load("data/bundle_adjustment/problem-21-11315-pre.txt")?;
assert_eq!(dataset.cameras.len(), 21);
assert_eq!(dataset.points.len(), 11315);
println!("{} observations", dataset.observations.len());
# Ok::<(), apex_io::IoError>(())
```

Downloading BAL problems is automated by
[`ensure_ba_dataset`](../utils/datasets.md) and the `download_datasets` CLI.

## References

- Agarwal, S., Snavely, N., Seitz, S. M. & Szeliski, R. (2010). *Bundle Adjustment in the Large*. ECCV 2010, 29–42.
- Snavely, N., Seitz, S. M. & Szeliski, R. (2006). *Photo Tourism: Exploring Photo Collections in 3D*. ACM SIGGRAPH 2006.
