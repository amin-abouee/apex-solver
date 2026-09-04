# Feature Flags

`apex-io` keeps heavy or optional dependencies behind Cargo features. The default
feature set (`default = ["download", "cli"]`) covers pose graphs, the dataset registry
with auto-download, ASL (synchronous), trajectory I/O, and the clap-based
binaries. Bag handling, visualization, live DDS, and async ASL are opt-in.

| Feature | Default | Enables | Extra dependencies |
|---|:---:|---|---|
| `download` | ✓ | Dataset auto-download (`download_file`, `decompress_bzip2`, `extract_tar_gz`, the `ensure_*_dataset` network fallback) and the `download_datasets` binary | `ureq`, `bzip2`, `flate2`, `tar` |
| `cli` | ✓ | CLI parsing for `download_datasets`, `bag_filter`, `bag_convert` | `clap` |
| `rosbag` | ✗ | ROS1/ROS2 bag read/write (SQLite3 & MCAP storage, CDR, message types) and the `bag_info`, `bag_filter`, `bag_convert`, `extract_topic_data`, `write_dummy_bag` binaries | `rusqlite` (bundled SQLite), `mcap`, `zstd`, `lz4_flex`, `serde_yaml`, `byteorder`, `hex`, `bzip2` |
| `visualization` | ✗ | `to_rerun_*` conversions on `VertexSE2`/`VertexSE3`; solver visualization | `rerun` |
| `dds` | ✗ | The `dds` module (`DdsSubscriber`, `DdsListener`, QoS mapping) and the `dds_multi_listener` binary | `rustdds`, `tokio`, `futures` |
| `asl-async` | ✗ | Async ASL streaming | `tokio` |

> **Migrating from ≤ 0.3:** `rosbag` used to be built unconditionally. Add
> `features = ["rosbag"]` to keep bag I/O. `image` stays unconditional (the
> ASL streaming API returns `image::DynamicImage`).

## Enabling features

```bash
# ROS1/ROS2 bag I/O + binaries
cargo build -p apex-io --features rosbag

# Offline build (no network deps)
cargo build -p apex-io --no-default-features

# Live DDS subscription
cargo build -p apex-io --features dds

# Rerun visualization
cargo build -p apex-io --features visualization

# Multiple features
cargo build -p apex-io --features "rosbag dds visualization"
```

In `Cargo.toml`:

```toml
[dependencies]
apex-io = { version = "0.3", features = ["rosbag"] }
```

## What always works (no features)

- Pose-graph formats: [G2O](../pose-graph/g2o.md), [TORO](../pose-graph/toro.md),
  the [`Graph` model](../pose-graph/graph-model.md), and [`load_graph`].
- Bundle adjustment: [BAL](../pose-graph/bal.md).
- Sensor datasets: [ASL/EuRoC](../datasets/asl.md) (synchronous).
- Utilities: dataset registry paths (downloads need `download`), logging.
- CLI tools: `download_datasets` (needs `download`); `bag_*` tools need `rosbag`;
  `dds_multi_listener` needs `dds`.

## Feature-gated symbols at a glance

| Symbol / item | Feature |
|---|---|
| `apex_io::rosbag` module and all its items | `rosbag` |
| `bag_info`, `bag_filter`, `bag_convert`, `extract_topic_data`, `write_dummy_bag` binaries | `rosbag` |
| `apex_io::utils::{download_file, decompress_bzip2, extract_tar_gz}` | `download` |
| `ensure_*_dataset` network fallback (local-file path always works) | `download` |
| `download_datasets` binary | `download` |
| `apex_io::dds` module (re-export) | `dds` |
| `DdsSubscriber`, `DdsListener`, `DdsSubscriberConfig`, `ReceivedMessage` | `dds` |
| `qos_mapping::{to_dds_reliability, to_dds_durability, to_dds_history}` | `dds` |
| `dds_multi_listener` binary | `dds` |
| `VertexSE2::to_rerun_position_3d`, `VertexSE3::to_rerun_transform` | `visualization` |
