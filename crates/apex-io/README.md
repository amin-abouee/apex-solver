# apex-io

High-performance file I/O for robotics data — pose graphs (G2O, TORO, BAL) and ROS2 bag files (SQLite3 and MCAP), with optional live DDS topic subscription.

## What's new in 0.4.0 — feature flags & trajectories

- **`rosbag` is now an opt-in feature** (was built unconditionally); `download` and `clap`
  moved behind the default-on `download` / `cli` features. See the
  [feature reference](doc/cookbook/src/appendix/features.md) and the [changelog](CHANGELOG.md).
- **`trajectory` module** — a unified timestamped-pose container with format detection and
  TUM read/write plus ASL ground-truth loading (`load_trajectory`, `TumLoader`,
  `AslTrajectoryLoader`).
- **Pose-graph modules moved into `graph::`** (`g2o`/`toro` under `graph::`; the loaders
  remain re-exported at the crate root) and `IoError` moved to `graph::IoError`.

## What's new in 0.3.0 — Cookbook

A complete reference book documenting **every public functionality** of the crate — the
pose-graph formats (G2O, TORO, BAL), the ASL/EuRoC dataset readers, ROS1/ROS2 bag read/write
(SQLite3 & MCAP storage, CDR, message types), live DDS subscription, the dataset registry, and
all seven CLI binaries — organized by domain with per-type/per-function API tables. Build it
with `mdbook build doc/cookbook`.

## What's new in 0.2.0 — `unsafe`-free I/O

`apex-io` v0.2.0 **removes all `unsafe` code** from the crate. Previous versions used `unsafe { memmap2::Mmap::map(&file) }` in four locations — `bal.rs`, `g2o.rs`, `toro.rs`, and `rosbag/storage/mcap.rs` — to memory-map input files. Memory mapping carries an inherent soundness footgun (concurrent external file mutation is undefined behaviour) and provided no measurable benefit for the access patterns used here, all of which read each file end-to-end exactly once.

The four call sites are now safe, idiomatic standard-library reads:

| Loader | Before | After |
|---|---|---|
| `BalLoader::load` | `unsafe { Mmap::map(&file) }` → `from_utf8` → `.lines()` | `std::fs::read_to_string(path)` → `.lines()` |
| `G2oLoader::load` | `unsafe { Mmap::map(&file) }` → `from_utf8` → parallel parse | `std::fs::read_to_string(path)` → parallel parse |
| `ToroLoader::load` | `unsafe { Mmap::map(&file) }` → `from_utf8` → sequential parse | `std::fs::read_to_string(path)` → sequential parse |
| `McapStorageReader::open` | `Vec<memmap2::Mmap>` field, `unsafe { Mmap::map }` per file | `Vec<Box<[u8]>>` field, `std::fs::read(path).into_boxed_slice()` per file (still feeds `mcap::MessageStream` via `&[u8]`) |

**Compatibility:** the public API of every loader is unchanged. **Performance:** equivalent or marginally better on warm caches; sequential whole-file reads avoid the page-fault overhead mmap incurs on first touch. The only theoretical regression is on > RAM MCAP files — but the previous design already required the entire bag to be addressable as a contiguous `&[u8]` for `mcap::MessageStream::new`, so this is not a practical regression. The `memmap2` dependency has been dropped from the crate manifest.

## Origin

The ROS2 bag reading and writing functionality in this crate was originally developed as a
standalone Rust library at [rosbags-rs](https://github.com/amin-abouee/rosbags-rs) and has
since been merged into `apex-io`. Bags produced and read by this crate are interoperable with
the `ros2 bag` CLI.

## Overview

- **Pose graph I/O** — G2O and TORO (2D/3D SLAM), BAL (bundle adjustment)
- **ROS2 bag reading** — SQLite3 and MCAP formats, topic + time-range filtering, raw and deserialized APIs
- **ROS2 bag writing** — SQLite3 and MCAP with optional zstd compression
- **Live DDS subscription** — async topic listener for running ROS2 systems (optional `dds` feature)
- **Every standard message, ROS1 ⇄ ROS2** — a schema-driven transcoder re-encodes ROS1 ⇄ CDR payloads for any type with a definition (embedded in the bag, or from a built-in catalog of the `std`/`geometry`/`sensor`/`nav`/`tf2`/`visualization`/… interface packages) without typed structs; typed CDR decoding is available for the common sensor messages
- **Safe, fast I/O + parallel parsing** — buffered reads (`std::fs::read[_to_string]`) and `rayon`; **no `unsafe` and no memory-mapping** as of v0.2.0
- **Format-agnostic `BagReader` / `BagWriter`** — one streaming API over ROS1 `.bag`, ROS2 sqlite3, ROS2 MCAP directories and bare `.mcap` files (ROS1- or ROS2-profile)
- **CLI binaries** — `bag_info`, `bag_filter`, `bag_convert`, `bag_verify`, `extract_topic_data`, `write_dummy_bag`, `dds_multi_listener`

## Supported Formats

| Format | Description | Read | Write |
|--------|-------------|------|-------|
| **G2O** | General Graph Optimization (SE2 + SE3) | ✓ | ✓ |
| **TORO** | Tree-based netwORk Optimizer (SE2 only) | ✓ | ✓ |
| **BAL** | Bundle Adjustment in the Large | ✓ | — |
| **ROS2 bag / SQLite3** | `.db3` storage format | ✓ | ✓ |
| **ROS2 bag / MCAP** | `.mcap` storage format | ✓ | ✓ |

## Installation

```toml
# Core: pose graphs + dataset registry (no bag I/O, no network)
[dependencies]
apex-io = "0.4.0"

# With ROS1/ROS2 bag reading & writing (SQLite3 + MCAP, CDR, message types)
apex-io = { version = "0.4.0", features = ["rosbag"] }

# Without dataset auto-download (offline builds)
apex-io = { version = "0.4.0", default-features = false }

# With Rerun visualization helpers
apex-io = { version = "0.4.0", features = ["visualization"] }

# With live DDS topic subscription (requires a DDS runtime)
apex-io = { version = "0.4.0", features = ["dds"] }
```

| Feature | Default | Enables | Extra dependencies |
|---|:---:|---|---|
| `vrs` | ✗ | Meta VRS (Project Aria) reader/writer, `vrs_*` binaries, VRS in the `bag_*` tools | `zstd`, `lz4_flex` (no new dependencies) |
| `download` | ✓ | Dataset auto-download (`download_file`, `decompress_bzip2`, `extract_tar_gz`, `ensure_*_dataset` network fallback, `download_datasets` binary) | `ureq`, `bzip2`, `flate2`, `tar` |
| `cli` | ✓ | CLI parsing for `download_datasets`, `bag_filter`, `bag_convert` | `clap` |
| `rosbag` | ✗ | ROS1/ROS2 bag read/write, `bag_*` binaries | `rusqlite` (bundled SQLite), `mcap`, `zstd`, `lz4_flex`, `serde_yaml`, `byteorder`, `hex`, `bzip2`, `md5`, `crc32fast` |
| `visualization` | ✗ | `to_rerun_*` conversions | `rerun` |
| `dds` | ✗ | Live DDS subscription, `dds_multi_listener` binary | `rustdds`, `tokio`, `futures` |

> **Note:** `apex-io` depends on `apex-manifolds` (for SE2/SE3 types). Both crates must be
> available on crates.io. If you are using the workspace they are handled automatically.

---

## ROS2 Bag Reading

### Basic — iterate all messages

```rust
use apex_io::rosbag::Reader;

let mut reader = Reader::new("path/to/my_bag")?;
reader.open()?;

// Bag-level metadata
println!("Duration:  {:.2}s", reader.duration() as f64 / 1e9);
println!("Start:     {} ns", reader.start_time());
println!("End:       {} ns", reader.end_time());
println!("Messages:  {}", reader.message_count());

// List topics
for topic in reader.topics() {
    println!("  {} [{}]  {} msgs",
        topic.name, topic.message_type, topic.message_count);
}

// Iterate every message
for msg in reader.messages()? {
    let msg = msg?;
    println!("{} @ {} ns  ({} bytes)",
        msg.connection.topic, msg.timestamp, msg.data.len());
}

reader.close()?;
```

### Filter by topic

```rust
use apex_io::rosbag::Reader;

let mut reader = Reader::new("path/to/my_bag")?;
reader.open()?;

// Select connections whose topic matches the filter
let conns = reader.connections();
let imu_conns: Vec<_> = conns.iter()
    .filter(|c| c.topic == "/imu/data")
    .cloned()
    .collect();

for msg in reader.messages_filtered(Some(&imu_conns), None, None)? {
    let msg = msg?;
    println!("IMU @ {} ns", msg.timestamp);
}
```

### Filter by time range

```rust
use apex_io::rosbag::Reader;

let mut reader = Reader::new("path/to/my_bag")?;
reader.open()?;

// Times are in nanoseconds since epoch
let start_ns: u64 = 1_700_000_000_000_000_000;
let end_ns:   u64 = 1_700_000_005_000_000_000; // 5 seconds later

for msg in reader.messages_filtered(None, Some(start_ns), Some(end_ns))? {
    let msg = msg?;
    println!("{} @ {} ns", msg.connection.topic, msg.timestamp);
}
```

### Combined: topic + time filter

```rust
use apex_io::rosbag::Reader;

let mut reader = Reader::new("path/to/my_bag")?;
reader.open()?;

let conns = reader.connections();
let cam_conns: Vec<_> = conns.iter()
    .filter(|c| c.topic.starts_with("/camera"))
    .cloned()
    .collect();

let start_ns = reader.start_time();
let end_ns   = start_ns + 10_000_000_000; // first 10 seconds

for msg in reader.messages_filtered(Some(&cam_conns), Some(start_ns), Some(end_ns))? {
    let msg = msg?;
    println!("{} @ {} ns  {} bytes", msg.connection.topic, msg.timestamp, msg.data.len());
}
```

### High-performance raw mode (no deserialization)

```rust
use apex_io::rosbag::Reader;

let mut reader = Reader::new("path/to/my_bag")?;
reader.open()?;

// raw_messages_filtered skips CDR deserialization — useful for copying or forwarding
for raw in reader.raw_messages_filtered(None, None, None)? {
    let raw = raw?;
    println!("raw {} bytes on {}", raw.raw_data.len(), raw.connection.topic);
}

// Or batch-collect everything at once (single allocation, no iterator overhead)
let batch = reader.read_raw_messages_batch(None, None, None)?;
println!("Loaded {} raw messages", batch.len());
```

### Fast metadata-only reading

```rust
use apex_io::rosbag::read_bag_metadata_fast;

// Reads only metadata.yaml — never opens the .db3 or .mcap file
let meta = read_bag_metadata_fast("path/to/my_bag")?;
println!("Duration: {:.2}s", meta.duration() as f64 / 1e9);
println!("Messages: {}", meta.message_count());
println!("Storage:  {}", meta.info().storage_identifier);

for topic in &meta.info().topics_with_message_count {
    println!("  {} [{}]  {} msgs",
        topic.topic_metadata.name,
        topic.topic_metadata.message_type,
        topic.message_count);
}
```

---

## ROS2 Bag Writing

### Write a minimal bag

```rust
use apex_io::rosbag::{Writer, StoragePlugin};

let mut writer = Writer::new(
    "output_bag",
    None,                        // latest bag format version
    Some(StoragePlugin::Sqlite3),
)?;
writer.open()?;

// Register a topic before writing to it
let conn = writer.add_connection(
    "/my_topic".to_string(),
    "std_msgs/msg/String".to_string(),
    None,   // message definition (auto-resolved if None)
    None,   // type description hash
    None,   // serialization format (defaults to "cdr")
    None,   // QoS profiles
)?;

// Write raw CDR-serialized bytes with a nanosecond timestamp
let timestamp_ns: u64 = 1_700_000_000_000_000_000;
writer.write(&conn, timestamp_ns, b"\x00\x01\x00\x00\x06\x00\x00\x00hello\x00")?;

writer.close()?;
```

### Write to MCAP format with zstd compression

```rust
use apex_io::rosbag::{Writer, StoragePlugin, CompressionMode, CompressionFormat};

let mut writer = Writer::new("output_bag", None, Some(StoragePlugin::Mcap))?;
writer.open()?;

// Enable file-level zstd compression
writer.set_compression(CompressionMode::File, CompressionFormat::Zstd)?;

let conn = writer.add_connection(
    "/lidar/points".to_string(),
    "sensor_msgs/msg/PointCloud2".to_string(),
    None, None, None, None,
)?;

for (i, payload) in payloads.iter().enumerate() {
    let ts = 1_700_000_000_000_000_000u64 + i as u64 * 100_000_000; // 100 ms apart
    writer.write(&conn, ts, payload)?;
}

writer.close()?;
```

### Write multiple topics

```rust
use apex_io::rosbag::{Writer, StoragePlugin};

let mut writer = Writer::new("multi_topic_bag", None, Some(StoragePlugin::Sqlite3))?;
writer.open()?;

let imu_conn = writer.add_connection(
    "/imu/data".to_string(),
    "sensor_msgs/msg/Imu".to_string(),
    None, None, None, None,
)?;

let odom_conn = writer.add_connection(
    "/odom".to_string(),
    "nav_msgs/msg/Odometry".to_string(),
    None, None, None, None,
)?;

// Interleave messages from different topics
writer.write(&imu_conn,  1_000_000_000, &imu_bytes)?;
writer.write(&odom_conn, 1_050_000_000, &odom_bytes)?;
writer.write(&imu_conn,  1_100_000_000, &imu_bytes2)?;

writer.close()?;
```

---

## Live DDS Subscription

Enable the `dds` feature and have a running ROS2 node on the same DDS domain.

```toml
apex-io = { version = "0.4.0", features = ["dds"] }
```

### Subscribe to a single topic

```rust
use apex_io::dds::{DdsSubscriber, DdsSubscriberConfig};
use apex_io::rosbag::{QosReliability, QosDurability};

let config = DdsSubscriberConfig {
    topic: "/imu/data".to_string(),
    message_type: "sensor_msgs/msg/Imu".to_string(),
    reliability: QosReliability::Reliable,
    durability: QosDurability::Volatile,
    history_depth: 10,
    domain_id: 0,
    channel_capacity: 128,
};

let subscriber = DdsSubscriber::new(config)?;

// `listen` spawns a background thread and returns a channel receiver
let rx = subscriber.listen()?;

// Process messages as they arrive
for raw_msg in rx {
    println!("Received {} bytes on {} @ {} ns",
        raw_msg.raw_data.len(),
        raw_msg.connection.topic,
        raw_msg.timestamp);
}
```

### Subscribe to multiple topics

Use a thread per topic and collect via a shared channel, or use the `dds_multi_listener` binary:

```rust
use apex_io::dds::{DdsSubscriber, DdsSubscriberConfig};
use std::sync::mpsc;
use std::thread;

let topics = vec![
    ("/imu/data", "sensor_msgs/msg/Imu"),
    ("/odom",     "nav_msgs/msg/Odometry"),
    ("/tf",       "tf2_msgs/msg/TFMessage"),
];

let (tx, rx) = mpsc::channel();

for (topic, msg_type) in topics {
    let tx = tx.clone();
    let config = DdsSubscriberConfig {
        topic: topic.to_string(),
        message_type: msg_type.to_string(),
        reliability: Default::default(),
        durability: Default::default(),
        history_depth: 10,
        domain_id: 0,
        channel_capacity: 64,
    };
    thread::spawn(move || {
        let sub = DdsSubscriber::new(config).unwrap();
        let sub_rx = sub.listen().unwrap();
        for msg in sub_rx {
            tx.send(msg).unwrap();
        }
    });
}

// Unified receive loop across all topics
for msg in rx {
    println!("{} @ {} ns", msg.connection.topic, msg.timestamp);
}
```

### Topic name conversion

ROS2 topic names are automatically converted to DDS topic names:

```rust
use apex_io::dds::DdsSubscriber;

// "/imu/data"  →  "rt/imu/data"
let dds_topic = DdsSubscriber::ros2_to_dds_topic("/imu/data");
```

---

## Pose Graph Formats

### G2O Format

The G2O (General Graph Optimization) format supports both 2D (SE2) and 3D (SE3) pose graphs.

**File structure:**
```
VERTEX_SE2 id x y theta
VERTEX_SE3:QUAT id x y z qx qy qz qw
EDGE_SE2 from to dx dy dtheta info_xx info_xy info_yy info_xt info_yt info_tt
EDGE_SE3:QUAT from to dx dy dz dqx dqy dqz dqw [21 info matrix values]
```

```rust
use apex_io::{G2oLoader, GraphLoader};

// Load
let graph = G2oLoader::load("data/sphere2500.g2o")?;
println!("{} vertices, {} edges", graph.vertex_count(), graph.edge_count());

// Iterate SE3 vertices
for (id, v) in &graph.vertices_se3 {
    println!("Vertex {}: ({:.2}, {:.2}, {:.2})", id, v.x(), v.y(), v.z());
}

// Iterate SE3 edges
for edge in &graph.edges_se3 {
    println!("Edge {} → {}", edge.from, edge.to);
}

// Write back
G2oLoader::write(&graph, "output.g2o")?;
```

### TORO Format

TORO supports SE2 (2D) graphs only. Writing SE3 data returns an error.

**File structure:**
```
VERTEX2 id x y theta
EDGE2 from to dx dy dtheta info_xx info_xy info_yy info_xt info_yt info_tt
```

```rust
use apex_io::{ToroLoader, GraphLoader};

let graph = ToroLoader::load("data/intel.graph")?;
for (id, v) in &graph.vertices_se2 {
    println!("Vertex {}: ({:.2}, {:.2}, {:.2} rad)", id, v.x(), v.y(), v.theta());
}
ToroLoader::write(&graph, "output.graph")?;
```

### Auto-detect format

```rust
use apex_io::load_graph;

let graph = load_graph("data/M3500.g2o")?;   // .g2o  → G2oLoader
let graph = load_graph("data/intel.graph")?; // .graph → ToroLoader
```

### BAL Format (Bundle Adjustment)

**File structure:**
```
num_cameras num_points num_observations
camera_idx point_idx pixel_x pixel_y  # one observation per line
...
# then 9 camera parameters per camera (rotation x/y/z, translation x/y/z, focal, k1, k2)
# then 3 point coordinates per point (x, y, z)
```

```rust
use apex_io::BalLoader;

let dataset = BalLoader::load("problem-1778-993923-pre.txt")?;
println!("{} cameras, {} points, {} observations",
    dataset.cameras.len(), dataset.points.len(), dataset.observations.len());

for (i, cam) in dataset.cameras.iter().enumerate() {
    println!("Camera {}: f={:.1} k1={:.4} k2={:.4}", i, cam.focal_length, cam.k1, cam.k2);
}
for obs in &dataset.observations {
    println!("Camera {} sees point {} at ({:.1}, {:.1})",
        obs.camera_index, obs.point_index, obs.x, obs.y);
}
```

---

## VRS / Project Aria (`vrs` feature)

A from-scratch, safe-Rust implementation of Meta's **VRS** container (`VisionRecordVRS2`), as used by
Project Aria recordings such as the LaMaria dataset. No FFI, no memory mapping; opening a 20 GB file reads
only the header, tags and index. Enable with `--features vrs` (it only needs `zstd`/`lz4_flex`, which `rosbag` already uses).

```rust
use apex_io::vrs::aria::{AriaRecording, AriaSensor, Sample};

let mut rec = AriaRecording::open("recording.vrs")?;

// 1) Pull samples one by one, merged in capture-time order across sensors.
let mut cursor = rec.cursor(&[AriaSensor::SlamLeft, AriaSensor::ImuRight]);
while let Some(sample) = rec.next_sample(&mut cursor)? {
    match sample {
        Sample::Imu(imu) => { /* imu.time_s, imu.accel, imu.gyro */ }
        Sample::Image(img) => { /* img.jpeg (original bytes), exposure, gain */ }
        _ => {}
    }
}

// 2) Or ask by time.
let (t0, _) = rec.time_range(AriaSensor::ImuRight).unwrap();
let imu = rec.imu_between(AriaSensor::ImuRight, t0 + 10.0, t0 + 10.5)?;
let frame = rec.image_nearest(AriaSensor::SlamLeft, t0 + 10.0)?;
```

* **Generic layer** (`apex_io::vrs`): `VrsReader` (index, tags, `DataLayout`/`RecordFormat` decoding, time-ordered
  `Cursor`, `find_nearest` / range queries, zero-allocation `read_record_into`), `VrsWriter` (streams, tags, per-record
  zstd/lz4, index written at close) and `LayoutWriter` for building typed record headers.
* **Aria layer**: typed `ImuSample`, `BaroSample`, `GpsSample`, `ImageFrame` (JPEG passed through undecoded),
  camera/IMU configs and the factory calibration JSON. ~500k IMU samples/s, so a recording pulls far faster than real time.
* **Bag tools**: VRS is a first-class container in `bag_info`, `bag_filter`, `bag_convert` and `bag_verify`.
  Reading maps Aria streams to ROS topics (`/aria/imu_right` `sensor_msgs/Imu`, `/aria/camera_slam_left/compressed`
  `sensor_msgs/CompressedImage`, magnetometer, barometer, GPS); Wi-Fi/Bluetooth/clock streams are listed and reported
  as unsupported. Writing stores each ROS topic as a VRS stream holding the raw message bytes plus topic/type/definition/encoding,
  so *any* bag survives ROS → VRS → ROS byte for byte.
* **Tools**: `vrs_info` (streams, rates, tags, `--verify-index`), `vrs_copy` (trim by stream/time, re-compress),
  `vrs_dump` (CSV for IMU/baro/GPS, JPEG extraction for cameras).

Validated against Meta's own reader (`projectaria-tools`) with `scripts/vrs_crosscheck.py`: decoded IMU/barometer
values match and Meta reads files written by `VrsWriter` identically to the originals. Format notes (derived from real
files; the upstream header comments are partly outdated): file header 80 B, record header 32 B, index entries 19 B
`{ts, size, type, stream}` with offsets implied by cumulative sizes, var-size layout pieces stored as `(offset, length)` pairs.

## CLI Binaries

All binaries ship with the crate. Run them via:

```bash
cargo run -p apex-io --bin <name> -- <args>
```

### `bag_info` — inspect a bag (index/summary only, never reads messages)

```bash
cargo run --release -p apex-io --bin bag_info -- path/to/bag_or_file   # .bag | .mcap | ROS2 dir
```

Output: container, size, duration, start/end, message count and a per-topic table (type,
payload encoding, count, rate). Topics that `bag_convert` cannot re-encode are flagged and logged
with a warning.

### `bag_filter` — copy / filter a bag (raw bytes, streaming)

Reads any container and writes any container whose payload encoding matches; message bytes are
copied verbatim (nothing is deserialized). Memory use is independent of bag size.

```bash
# Keep topics, write a bare MCAP with zstd chunks
bag_filter in.bag out.mcap --topics /imu0,/cam0/image_raw --compression zstd

# Time window: ns since epoch (--start inclusive, --end exclusive) or seconds from the first message
bag_filter in.mcap out_dir --start 1700000000000000000 --end 1700000010000000000
bag_filter in.mcap out_dir --start-offset 20 --end-offset 30 --format sqlite3

# Everything except some topics
bag_filter in.mcap out.bag --exclude /cam1/image_raw --compression lz4
```

`--format` is `bag`, `sqlite3`, `mcap` (ROS2 directory) or `mcap-file`; by default it follows the
output extension (`.bag`, `.mcap`) and otherwise writes an MCAP directory.

### `bag_convert` — ROS1 ⇄ ROS2, any container

```bash
bag_convert hilti.mcap out_ros2_dir --format mcap      # ROS1-profile MCAP -> ROS2 (CDR) bag
bag_convert ros2_dir out.bag --compression lz4         # ROS2 -> ROS1 .bag
bag_convert in.bag out.mcap --encoding cdr             # .bag -> bare ROS2-profile MCAP
bag_convert in.bag --list-topics                       # which topics are convertible
```

Payloads are re-encoded by a converter compiled from each topic's message definition (from the bag,
or the built-in catalog for ROS2 sqlite bags that store none); byte arrays such as images are
block-copied. Topics with no usable definition are skipped with a `WARN` (`--strict` makes that an
error). The ROS1 output carries real md5sums and full definitions, and ROS1 `seq` is regenerated as 0.

### `bag_verify` — prove two bags hold the same messages

```bash
bag_verify original.mcap converted_dir     # per-topic count + digest; exit 0 if identical
```

The digest covers `(timestamp, canonical CDR payload)`, so it ignores container, encoding and ROS1
`seq` and validates `bag_filter` / `bag_convert` output.

### `extract_topic_data` — export to CSV or PNG

```bash
# Extract any topic to CSV (one row per message)
cargo run -p apex-io --bin extract_topic_data -- \
    path/to/my_bag /imu/data output_folder/

# Extract image topic to PNG files (one file per frame)
cargo run -p apex-io --bin extract_topic_data -- \
    path/to/my_bag /camera/image_raw output_folder/
```

### `write_dummy_bag` — create a demo bag

```bash
# Write to ./demo_bag with 29+ message types
cargo run -p apex-io --bin write_dummy_bag

# Custom output path
cargo run -p apex-io --bin write_dummy_bag -- /tmp/my_demo_bag
```

### `download_datasets` — download public SLAM datasets

```bash
cargo run -p apex-io --bin download_datasets
```

Downloads G2O, TORO, and BAL benchmark datasets configured in `datasets.toml`.

### `dds_multi_listener` — live multi-topic DDS listener (`dds` feature required)

```bash
cargo run -p apex-io --features dds --bin dds_multi_listener -- \
    --topics /imu/data,/odom,/tf \
    --domain-id 0
```

Connects to a running ROS2 system and prints incoming messages. Requires a compatible DDS
runtime (CycloneDDS, FastDDS) and active ROS2 nodes on the same domain.

---

## Data Structures

### Graph

```rust
pub struct Graph {
    pub vertices_se2: HashMap<usize, VertexSE2>,
    pub vertices_se3: HashMap<usize, VertexSE3>,
    pub edges_se2:    Vec<EdgeSE2>,
    pub edges_se3:    Vec<EdgeSE3>,
}
impl Graph {
    pub fn vertex_count(&self) -> usize;
    pub fn edge_count(&self) -> usize;
}
```

### ROS2 bag core types

```rust
pub struct Connection {
    pub id:                   u32,
    pub topic:                String,        // "/imu/data"
    pub message_type:         String,        // "sensor_msgs/msg/Imu"
    pub serialization_format: String,        // "cdr"
    pub offered_qos_profiles: Vec<QosProfile>,
    // ...
}

pub struct Message {
    pub connection: Connection,
    pub topic:      String,
    pub timestamp:  u64,      // nanoseconds since epoch
    pub data:       Vec<u8>,  // CDR-serialized bytes
}

pub struct RawMessage {
    pub connection: Connection,
    pub timestamp:  u64,
    pub raw_data:   Vec<u8>,
}

pub enum StoragePlugin     { Sqlite3, Mcap }
pub enum CompressionMode   { None, Message, File, Storage }
pub enum CompressionFormat { None, Zstd }
```

### QoS Profile

```rust
pub struct QosProfile {
    pub history:                     QosHistory,      // KeepLast | KeepAll | SystemDefault
    pub depth:                       u32,
    pub reliability:                 QosReliability,  // Reliable | BestEffort | SystemDefault
    pub durability:                  QosDurability,   // TransientLocal | Volatile | SystemDefault
    pub deadline:                    QosTime,
    pub lifespan:                    QosTime,
    pub liveliness:                  QosLiveliness,
    pub liveliness_lease_duration:   QosTime,
    pub avoid_ros_namespace_conventions: bool,
}
```

---

## Performance

| Technique | Applied to |
|-----------|-----------|
| Single-syscall buffered read (`std::fs::read_to_string` / `std::fs::read`) | All loaders — replaced `memmap2` in v0.2.0; safe (`#![forbid(unsafe_code)]`-friendly), one allocation, no page-fault overhead, equivalent throughput on the sequential whole-file parse paths used here |
| Parallel parsing (`rayon`) | Files > 1 000 lines — multi-core acceleration |
| Raw message API | Bag reading without CDR deserialization overhead |
| Batch read | `read_raw_messages_batch` — single allocation, no iterator overhead |
| Metadata-only path | `read_bag_metadata_fast` — zero storage file I/O |
| Pre-allocated collections | Estimated capacity from file size to minimize reallocs |

---

## Error Handling

### Pose graph — `IoError`

| Variant | Description |
|---------|-------------|
| `Io(io::Error)` | Underlying I/O error |
| `Parse { line, message }` | Parse error with line number |
| `UnsupportedVertexType(String)` | Unknown vertex type in file |
| `UnsupportedEdgeType(String)` | Unknown edge type in file |
| `InvalidNumber { line, value }` | Failed to parse a number |
| `MissingFields { line }` | Insufficient fields on a line |
| `DuplicateVertex { id }` | Vertex ID collision |
| `InvalidQuaternion { line, norm }` | Non-unit quaternion |
| `UnsupportedFormat(String)` | Unrecognized file extension |
| `FileCreationFailed { path, reason }` | Output file creation failed |

### ROS2 bag — `BagError`

`BagError` covers metadata parsing, storage backend (SQLite3/MCAP), CDR deserialization,
compression, and connection mismatch failures. All variants include enough context (file path,
topic, byte offset) for actionable diagnostics.

---

## Visualization Feature

```toml
apex-io = { version = "0.4.0", features = ["visualization"] }
```

```rust
// SE2 vertex → Rerun types
let pos_2d: [f32; 2] = vertex_se2.to_rerun_position_2d(scale);
let pos_3d: Vec3     = vertex_se2.to_rerun_position_3d(scale, height);

// SE3 vertex → Rerun types
let (position, rotation): (Vec3, Quat) = vertex_se3.to_rerun_transform(scale);
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `apex-manifolds` | Lie group types (SE2, SE3) |
| `nalgebra` | Linear algebra (vectors, matrices) |
| `rayon` | Parallel parsing |
| `thiserror` | Error type derivation |
| `tracing` | Structured logging |
| `serde`, `serde_json` | Serialization |
| `serde_yaml` | YAML metadata parsing (ROS2 bags) |
| `chrono` | Timestamps in file headers |
| `byteorder` | CDR byte-order handling |
| `bytes` | Efficient byte buffers |
| `hex` | Hex encoding for diagnostics |
| `image` | PNG export for image topics |
| `rusqlite` | SQLite3 storage backend |
| `zstd` | Compression support |
| `mcap` | MCAP storage backend |
| `clap` | CLI argument parsing |
| `ureq` | Dataset download |
| `rerun` | Visualization *(optional — `visualization` feature)* |
| `rustdds`, `tokio`, `futures` | Live DDS subscription *(optional — `dds` feature)* |

---

## References

- [rosbags-rs](https://github.com/amin-abouee/rosbags-rs) — original standalone ROS2 bag library (now merged here)
- [g2o: General Framework for Graph Optimization](https://github.com/RainerKuemmerle/g2o)
- [TORO: Tree-based netwORk Optimizer](https://www.openslam.org/toro.html)
- [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
- [Bundler: Structure from Motion](https://www.cs.cornell.edu/~snavely/bundler/)
- [ROS2 bag file format](https://github.com/ros2/rosbag2)
- [MCAP format](https://mcap.dev)

## License

Apache-2.0
