# CLI Tools

`apex-io` ships **seven** command-line binaries (declared as `[[bin]]` targets)
covering dataset download, bag inspection, filtering, conversion, extraction,
authoring, and live DDS listening. Run any of them with `cargo run -p apex-io
--bin <name> -- <args>`.

| Binary | Purpose | Feature |
|---|---|---|
| [`download_datasets`](#download_datasets) | Download pose-graph & BAL datasets | `download` (default) |
| [`bag_info`](#bag_info) | Summarize a ROS2 bag (fast) | `rosbag` |
| [`bag_filter`](#bag_filter) | Copy/filter a ROS2 bag | `rosbag` |
| [`extract_topic_data`](#extract_topic_data) | Export one topic's data | `rosbag` |
| [`bag_convert`](#bag_convert) | Convert ROS1 ↔ ROS2 | `rosbag` |
| [`write_dummy_bag`](#write_dummy_bag) | Author a ROS2 bag from scratch | `rosbag` |
| [`dds_multi_listener`](#dds_multi_listener) | Listen to live DDS topics | `dds` |

> Feature-gated binaries are skipped when their feature is off. Examples:
> `cargo run -p apex-io --features rosbag --bin bag_info -- …`
> (default features already include `download`).

<a id="download_datasets"></a>
## `download_datasets`

Downloads pose-graph (Luca Carlone, g2o) and bundle-adjustment (UW BAL) datasets
listed in the embedded `datasets.toml` (no hardcoded URLs). Selection is by
group number:

```bash
cargo run -p apex-io --bin download_datasets -- --select 3   # all odometry g2o
cargo run -p apex-io --bin download_datasets -- --select 0   # largest BA per collection
```

Backed by [`DatasetRegistry`](./datasets.md) and `ensure_*_dataset`.

<a id="bag_info"></a>
## `bag_info`

Prints a comprehensive ROS2 bag summary — storage files & sizes, duration/timing,
per-topic message counts, human-readable timestamps — by reading only
`metadata.yaml` (via [`read_bag_metadata_fast`](../rosbag/overview.md#fast-metadata)),
so it never opens the storage files.

```bash
cargo run -p apex-io --bin bag_info -- path/to/rosbag2_dir
```

<a id="bag_filter"></a>
## `bag_filter`

Reads a ROS2 bag and writes it to a new location, optionally filtering by topic
and/or time range — using the [raw path](../rosbag/overview.md#connections-and-messages)
so payloads are copied without re-decoding.

```bash
cargo run -p apex-io --bin bag_filter -- --input in_bag --output out_bag --topics /imu,/odom
```

<a id="extract_topic_data"></a>
## `extract_topic_data`

Reads a specific topic from a ROS2 bag and exports its data (e.g. decoded
messages / images) for offline use.

```bash
cargo run -p apex-io --bin extract_topic_data -- --input in_bag --topic /cam0/image_raw
```

<a id="bag_convert"></a>
## `bag_convert`

Converts between ROS1 (`.bag`) and ROS2 (SQLite3 / MCAP). **Direction is
auto-detected from the input path.**

```bash
cargo run -p apex-io --bin bag_convert -- input.bag output_ros2_dir
cargo run -p apex-io --bin bag_convert -- input_ros2_dir output.bag
```

<a id="write_dummy_bag"></a>
## `write_dummy_bag`

Writes a ROS2 bag containing **all supported message types** — a runnable example
of authoring a bag with the [`Writer`](../rosbag/ros2.md#writer) and the
[typed messages](../rosbag/messages.md).

```bash
cargo run -p apex-io --bin write_dummy_bag -- output_dir
```

<a id="dds_multi_listener"></a>
## `dds_multi_listener`

> Requires `--features dds`.

Subscribes to multiple ROS2 DDS topics simultaneously and logs each message
(topic, type, size, timestamps). Reads `ROS_DOMAIN_ID` from the environment
(default `0`). Built on [`DdsListener`](../rosbag/dds.md#ddslistener-multi-topic).

```bash
ROS_DOMAIN_ID=0 cargo run -p apex-io --features dds --bin dds_multi_listener -- /imu /odom
```
