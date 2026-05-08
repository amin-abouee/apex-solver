# apex-io

High-performance file I/O for robotics data — pose graphs (G2O, TORO, BAL) and ROS2 bag files (SQLite3 and MCAP), with optional live DDS topic subscription.

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
- **94+ ROS2 message types** — CDR deserialization for `geometry_msgs`, `sensor_msgs`, `nav_msgs`, `std_msgs`, `tf2_msgs`, and more
- **Safe, fast I/O + parallel parsing** — buffered reads (`std::fs::read[_to_string]`) and `rayon`; **no `unsafe` and no memory-mapping** as of v0.2.0
- **CLI binaries** — `bag_info`, `bag_filter`, `extract_topic_data`, `write_dummy_bag`, `dds_multi_listener`

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
# Core: pose graphs + ROS2 bag reader/writer
[dependencies]
apex-io = "0.2.0"

# With Rerun visualization helpers
apex-io = { version = "0.2.0", features = ["visualization"] }

# With live DDS topic subscription (requires a DDS runtime)
apex-io = { version = "0.2.0", features = ["dds"] }
```

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
apex-io = { version = "0.2.0", features = ["dds"] }
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

## CLI Binaries

All binaries ship with the crate. Run them via:

```bash
cargo run -p apex-io --bin <name> -- <args>
```

### `bag_info` — inspect bag metadata (fast, no storage file opened)

```bash
cargo run -p apex-io --bin bag_info -- path/to/my_bag
```

Reads only `metadata.yaml`. Output: version, storage format, compression, file sizes, duration,
start/end timestamps, message count, and per-topic breakdown.

### `bag_filter` — copy and filter a bag

```bash
# Filter by topic
cargo run -p apex-io --bin bag_filter -- input_bag output_bag \
    --topics /camera/image_raw,/imu/data

# Filter by time range (nanoseconds since epoch)
cargo run -p apex-io --bin bag_filter -- input_bag output_bag \
    --start 1700000000000000000 \
    --end   1700000010000000000

# Convert SQLite3 → MCAP
cargo run -p apex-io --bin bag_filter -- input_bag output_bag \
    --storage mcap

# All options combined with compression
cargo run -p apex-io --bin bag_filter -- input_bag output_bag \
    --topics /lidar/points \
    --start 1700000000000000000 \
    --end   1700000060000000000 \
    --storage mcap \
    --compression-mode file \
    --compression-format zstd
```

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
apex-io = { version = "0.2.0", features = ["visualization"] }
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
