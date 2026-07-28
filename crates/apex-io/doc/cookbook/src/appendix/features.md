# Feature Flags

`apex-io` keeps heavy or optional dependencies behind Cargo features. The default
feature set (`default = []`) covers all the pose-graph, dataset, ROS1, and ROS2
functionality; the three optional features add visualization, live DDS, and async
ASL.

| Feature | Default | Enables | Extra dependencies |
|---|:---:|---|---|
| `visualization` | ✗ | `to_rerun_*` conversions on `VertexSE2`/`VertexSE3`; solver visualization | `rerun` |
| `dds` | ✗ | The `dds` module (`DdsSubscriber`, `DdsListener`, QoS mapping) and the `dds_multi_listener` binary | `rustdds`, `tokio`, `futures` |
| `asl-async` | ✗ | Async ASL streaming | `tokio` |

## Enabling features

```bash
# Live DDS subscription
cargo build -p apex-io --features dds

# Rerun visualization
cargo build -p apex-io --features visualization

# Multiple features
cargo build -p apex-io --features "dds visualization"
```

In `Cargo.toml`:

```toml
[dependencies]
apex-io = { version = "0.2", features = ["dds"] }
```

## What always works (no features)

- Pose-graph formats: [G2O](../pose-graph/g2o.md), [TORO](../pose-graph/toro.md),
  the [`Graph` model](../pose-graph/graph-model.md), and [`load_graph`].
- Bundle adjustment: [BAL](../pose-graph/bal.md).
- Sensor datasets: [ASL/EuRoC](../datasets/asl.md) (synchronous).
- ROS bags: [ROS1](../rosbag/ros1.md) and [ROS2](../rosbag/ros2.md) read/write,
  SQLite3 & MCAP storage, CDR, [message types](../rosbag/messages.md).
- Utilities: [dataset registry & downloads](../utils/datasets.md), logging, and
  six of the seven [CLI tools](../utils/cli.md) (all but `dds_multi_listener`).

## Feature-gated symbols at a glance

| Symbol / item | Feature |
|---|---|
| `apex_io::dds` module (re-export) | `dds` |
| `DdsSubscriber`, `DdsListener`, `DdsSubscriberConfig`, `ReceivedMessage` | `dds` |
| `qos_mapping::{to_dds_reliability, to_dds_durability, to_dds_history}` | `dds` |
| `dds_multi_listener` binary | `dds` |
| `VertexSE2::to_rerun_position_3d`, `VertexSE3::to_rerun_transform` | `visualization` |
