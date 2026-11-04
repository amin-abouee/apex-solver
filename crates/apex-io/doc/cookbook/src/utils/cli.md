# CLI Tools

`apex-io` ships **twelve** command-line binaries (declared as `[[bin]]` targets)
covering dataset download, bag inspection, filtering, conversion, merging and
verification, VRS inspection and export, authoring, and live DDS listening. Run any of
them with `cargo run -p apex-io --bin <name> -- <args>`.

| Binary | Purpose | Feature |
|---|---|---|
| [`download_datasets`](#download_datasets) | Download pose-graph & BAL datasets | `download` (default) |
| [`bag_info`](#bag_info) | Summarize a ROS2 bag or bare `.mcap` (fast) | `rosbag` |
| [`bag_filter`](#bag_filter) | Copy a bag, keeping some topics and/or a time window | `rosbag` |
| [`bag_convert`](#bag_convert) | Convert between ROS1 and ROS2 containers and encodings | `rosbag` |
| [`bag_merge`](#bag_merge) | Join consecutive bags of one recording | `rosbag` |
| [`bag_verify`](#bag_verify) | Check that two bags hold the same messages | `rosbag` |
| [`extract_topic_data`](#extract_topic_data) | Export one topic's data | `rosbag` |
| [`write_dummy_bag`](#write_dummy_bag) | Author a ROS2 bag from scratch | `rosbag` |
| [`vrs_info`](#vrs_info) | Inspect a VRS (Project Aria) file | `vrs` |
| [`vrs_copy`](#vrs_copy) | Copy, trim or re-compress a VRS file | `vrs` |
| [`vrs_dump`](#vrs_dump) | Export one VRS stream (CSV / JPEG) | `vrs` |
| [`dds_multi_listener`](#dds_multi_listener) | Listen to live DDS topics | `dds` |

> Feature-gated binaries are skipped when their feature is off. Examples:
> `cargo run -p apex-io --features rosbag --bin bag_info -- …`
> (default features already include `download`). Most of the bag tools also need `cli`
> (on by default) for argument parsing.

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

Prints a bag summary — storage files & sizes, duration/timing, per-topic message counts,
human-readable timestamps — for a ROS2 bag directory (from its `metadata.yaml`, via
[`read_bag_metadata_fast`](../rosbag/overview.md#fast-metadata)) or a bare `.mcap` file
(from its MCAP summary section). It never reads message data, so it is instant on large
bags, and it flags topics that `bag_convert` cannot convert.

```bash
cargo run -p apex-io --features rosbag --bin bag_info -- path/to/rosbag2_dir
```

<a id="bag_filter"></a>
## `bag_filter`

Copies a bag, keeping only selected topics (`--topics`) or excluding some
(`--exclude`), and/or a time window: `--start` / `--end` in absolute nanoseconds, or
`--start-offset` / `--end-offset` in seconds from the start of the bag. `--list-topics`
only lists what the input holds. It reads any supported
container (ROS1 `.bag`, ROS2 sqlite3 or MCAP directory, bare `.mcap`) and writes any
container with the **same payload encoding**: message bytes are copied verbatim, nothing
is deserialized, so speed is bounded by I/O and optional compression. To change the
encoding (ROS1 ↔ ROS2) use [`bag_convert`](#bag_convert).

```bash
cargo run -p apex-io --features rosbag --bin bag_filter -- in_bag out_bag --topics /imu,/odom --start-offset 5 --end-offset 60
```

<a id="bag_convert"></a>
## `bag_convert`

Converts bags between ROS1 and ROS2 containers and payload encodings. It reads any
supported input and writes `.bag`, a ROS2 directory (sqlite3 or MCAP) or a bare `.mcap`.
Messages are streamed: when the encoding is unchanged the bytes are copied verbatim,
otherwise they are re-encoded ROS1 ↔ CDR by a converter compiled from the message
definition (see [`transcode`](../rosbag/overview.md#converting-between-ros1-and-ros2-transcode)),
so user-defined message types convert too. It also converts `PointCloud2` and `TFMessage`,
and can read the input in time windows.

```bash
cargo run -p apex-io --features rosbag --bin bag_convert -- input.bag output_ros2_dir
cargo run -p apex-io --features rosbag --bin bag_convert -- input_ros2_dir output.bag
```

<a id="bag_merge"></a>
## `bag_merge`

Concatenates consecutive bags of one recording into a single bag. Long recordings are
often cut into fixed-duration files (the VBR dataset splits every 80 s); this joins them
back, **in the order given**, without re-encoding: payload bytes are copied verbatim, so
every input must already use the encoding the output container holds (ROS1 for `.bag`,
CDR for ROS2 directories; run `bag_convert` first if they differ). A topic declared with a
different type or encoding in a later input is an error, and so is an input whose messages
start before the end of the previous one by more than `--tolerance-ms`: the inputs are
then not consecutive and the merged file would not be time-ordered.

```bash
cargo run -p apex-io --features rosbag --bin bag_merge -- part_0 part_1 part_2 --output merged \
    --format mcap --compression zstd
```

<a id="bag_verify"></a>
## `bag_verify`

Checks that two bags hold the same messages: per topic, the message count and a digest over
every `(timestamp, canonical payload)` pair. The canonical form is CDR, so ROS1 payloads are
normalised through the converter and the comparison ignores container, payload encoding and
the ROS1 header `seq`. Use it to validate `bag_convert` and `bag_filter` output.

```bash
cargo run -p apex-io --features rosbag --bin bag_verify -- a_bag b_bag --topics /imu,/odom
```

<a id="extract_topic_data"></a>
## `extract_topic_data`

Reads one topic from a ROS2 bag and exports it: image messages (`sensor_msgs/msg/Image`,
`CompressedImage`) become PNG files, every other message type becomes a CSV with
timestamped rows. Arguments are positional: the bag, the topic, the output folder.

```bash
cargo run -p apex-io --features rosbag --bin extract_topic_data -- in_bag /cam0/image_raw out_dir
```

<a id="write_dummy_bag"></a>
## `write_dummy_bag`

Writes a ROS2 bag containing **all supported message types** — a runnable example
of authoring a bag with the [`Writer`](../rosbag/ros2.md#writer) and the
[typed messages](../rosbag/messages.md).

```bash
cargo run -p apex-io --bin write_dummy_bag -- output_dir
```

<a id="vrs_info"></a>
## `vrs_info`

Inspects a VRS file (a Project Aria recording). It reads only the header, description and
index, never the sensor data, so it is instant even for 20 GB recordings. `--tags` prints
the description tags and `--verify-index` checks the index against a full record scan.

```bash
cargo run -p apex-io --features vrs --bin vrs_info -- recording.vrs --tags
```

<a id="vrs_copy"></a>
## `vrs_copy`

Copies, trims or re-compresses a VRS file. Tags, record formats and payload bytes are
kept; `--streams` selects streams, `--start` / `--end` a time window (seconds relative to
the first data record, or absolute with `--absolute`), and `--compression zstd|lz4|none`
the record compression.

```bash
cargo run -p apex-io --features vrs --bin vrs_copy -- in.vrs out.vrs --start 10 --end 20
```

<a id="vrs_dump"></a>
## `vrs_dump`

Exports one stream of an Aria VRS file. IMU, magnetometer, barometer, GPS and time-domain
streams are written as CSV to stdout; camera streams write one `.jpg` per frame (the
original JPEG bytes) into `--images <dir>` and print a CSV index of the frames.

```bash
cargo run -p apex-io --features vrs --bin vrs_dump -- recording.vrs --stream 1202-1 > imu.csv
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
