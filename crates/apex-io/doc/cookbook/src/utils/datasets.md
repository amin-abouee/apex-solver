# Datasets & Downloads

`apex_io::utils` manages benchmark datasets: a compile-time **registry** of known
datasets (parsed from an embedded `datasets.toml`) and helpers that **download
and cache** them on first use. This is what makes the integration tests
(`test_load_sphere2500`, …) self-contained.

## Standard directories

```rust
pub const ODOMETRY_DATA_DIR: &str      = "data/odometry";
pub const ODOMETRY_DATA_DIR_2D: &str   = "data/odometry/2d";
pub const ODOMETRY_DATA_DIR_3D: &str   = "data/odometry/3d";
pub const BUNDLE_ADJUSTMENT_DATA_DIR: &str = "data/bundle_adjustment";
```

## The registry

```rust
pub struct DatasetRegistry {
    // odometry datasets keyed by short name ("sphere2500", "intel", …)
    // bundle-adjustment collections keyed by name ("ladybug", …)
}
impl DatasetRegistry {
    pub fn load() -> io::Result<Self>;                       // from embedded datasets.toml
    pub fn odometry_path(&self, name: &str) -> Option<PathBuf>;
    pub fn odometry_by_category(&self, category: &str) -> Vec<(&str, &OdometryEntry)>;
    pub fn ba_path(&self, name: &str, cameras: u32, points: u32) -> Option<PathBuf>;
    pub fn ba_sorted(&self) -> Vec<(&str, &BaEntry)>;
}
```

Entry types:

| Type | Purpose |
|---|---|
| `OdometryEntry` | A pose-graph dataset (URL, category, filename). |
| `BaEntry` | A BAL collection: `largest() -> Option<[u32;2]>` gives its biggest `(cameras, points)`; `problem_url(cameras, points) -> String` builds the download URL for one problem. |

`datasets.toml` is embedded with `include_str!`, so **no URLs are hardcoded in
code** — add a dataset by editing the TOML.

## Ensure-and-cache helpers

These return a local path, downloading (and decompressing) only if the file is
not already present:

```rust
pub fn ensure_odometry_dataset(name: &str) -> io::Result<PathBuf>;
pub fn ensure_ba_dataset(name: &str, cameras: u32, points: u32) -> io::Result<PathBuf>;
```

Re-exported at the crate root (`apex_io::ensure_odometry_dataset`, …).

## Low-level download utilities

| Function | Signature | Description |
|---|---|---|
| `download_file` | `(url: &str, dest: &Path) -> io::Result<()>` | HTTP GET via `ureq` to a file. |
| `decompress_bzip2` | `(src: &Path, dest: &Path) -> io::Result<()>` | Decompress a `.bz2` (used by BAL). |

## Sources

| Source | Kind | URL |
|---|---|---|
| Luca Carlone (MIT) | 2D/3D pose graphs (g2o) | <https://lucacarlone.mit.edu/datasets/> |
| UW BAL | Bundle adjustment | <https://grail.cs.washington.edu/projects/bal/> |

## Example

```rust
use apex_io::{ensure_odometry_dataset, ensure_ba_dataset, G2oLoader, GraphLoader, BalLoader};

// Downloads to data/odometry/3d/ on first call, then returns the cached path.
let path = ensure_odometry_dataset("sphere2500")?;
let graph = G2oLoader::load(&path)?;

let ba_path = ensure_ba_dataset("ladybug", 49, 7776)?;
let dataset = BalLoader::load(&ba_path)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

See the [`download_datasets`](./cli.md#download_datasets) CLI for batch downloads.
