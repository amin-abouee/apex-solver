# Changelog

All notable changes to `apex-io` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-29

First release prepared for publishing to crates.io. This version promotes the crate from an
internal workspace dependency to a standalone publishable library, and adds the full ROS2 bag
and DDS feature set that was developed since v0.1.0.

### Added

- **ROS2 bag support** — merged from [rosbags-rs](https://github.com/amin-abouee/rosbags-rs):
  - `Reader` — open SQLite3 and MCAP bags, iterate messages, inspect topics and metadata
  - `Writer` — create SQLite3 and MCAP bags with optional zstd compression (file-level and message-level)
  - Topic and time-range filtering via `messages_filtered` / `raw_messages_filtered`
  - Raw message API (`raw_messages`, `read_raw_messages_batch`) for zero-deserialization copy/forward workflows
  - `read_bag_metadata_fast` — reads only `metadata.yaml`, never opens the storage file
  - 94+ ROS2 message type definitions with CDR deserialization (`geometry_msgs`, `sensor_msgs`, `nav_msgs`, `std_msgs`, `tf2_msgs`, and more)
  - `StoragePlugin` enum to select SQLite3 or MCAP at runtime
  - `CompressionMode` / `CompressionFormat` for zstd compression control
  - `QosProfile` and associated QoS types (`QosHistory`, `QosReliability`, `QosDurability`, `QosLiveliness`)
  - `BagMetadata` — structured access to `metadata.yaml` with `duration()`, `start_time()`, `end_time()`, `message_count()`
  - Integration tests against reference SQLite3 and MCAP bags in `tests/test_bags/`
- **Live DDS subscription** (`dds` feature) — subscribe to topics on a running ROS2 system:
  - `DdsSubscriber` — async subscriber backed by a channel; `listen()` returns `mpsc::Receiver<RawMessage>`
  - `DdsSubscriberConfig` — topic, message type, QoS (reliability, durability, history depth), domain ID, channel capacity
  - `DdsListener` and `ReceivedMessage` for lower-level listener access
  - `DdsSubscriber::ros2_to_dds_topic()` — converts `/imu/data` → `rt/imu/data`
  - `DdsSubscriber::ros2_type_to_dds_type()` — ROS2 → DDS type mangling
  - `DdsError` with structured variants for connection, type, and QoS errors
  - QoS profile mapping helpers (`qos_mapping` module)
  - Raw bytes adapter for capturing untyped DDS payloads
- **CLI binaries**:
  - `bag_info` — display bag metadata (reads only `metadata.yaml`, never opens storage)
  - `bag_filter` — copy and filter bags by topic list, time range, and storage format; supports format conversion (SQLite3 ↔ MCAP) and compression
  - `extract_topic_data` — export a single topic to CSV rows or PNG image files
  - `write_dummy_bag` — create a demo bag containing 29+ ROS2 message types; useful for testing
  - `dds_multi_listener` — multi-topic live DDS listener (requires `dds` feature)
- **Dataset registry** (`DatasetRegistry`, `ensure_odometry_dataset`, `ensure_ba_dataset`) — on-demand download of public G2O, TORO, and BAL benchmark datasets with caching in `data/`
- **Unit and integration tests** — comprehensive test coverage for all loaders (G2O, TORO, BAL), rosbag reader/writer, CDR deserializer, SQLite3 and MCAP storage backends, DDS subscriber, and error types

### Changed

- ROS2 bag support is now always compiled in (was previously an optional feature flag)
- Version bumped from `0.1.0` to `0.2.0` in `Cargo.toml`
- Workspace `Cargo.toml` dependency updated to `apex-io = "0.2.0"`
- README rewritten with full API documentation, examples, and cargo publish instructions

---

## [0.1.0] - 2026-01-30

Initial creation of the `apex-io` crate as part of the `apex-solver` workspace restructuring
([apex-solver v1.1.0](../../doc/CHANGELOG.md#110---2026-02-21)). Extracted from the monolithic
`apex-solver` crate to be independently publishable and usable.

### Added

- **G2O format** (`G2oLoader`) — read and write General Graph Optimization files:
  - `VERTEX_SE2` / `VERTEX_SE3:QUAT` vertex parsing
  - `EDGE_SE2` / `EDGE_SE3:QUAT` edge parsing with full information matrices
  - Both 2D (SE2) and 3D (SE3) graphs in a single file are supported
- **TORO format** (`ToroLoader`) — read and write Tree-based netwORk Optimizer files (SE2 only):
  - `VERTEX2` / `EDGE2` parsing
  - Writing SE3 data returns an error
- **BAL format** (`BalLoader`) — read Bundle Adjustment in the Large datasets:
  - Snavely's 9-parameter camera model (axis-angle rotation, translation, focal length, k1, k2)
  - `BalDataset` with `cameras`, `points`, and `observations`
  - Invalid camera filtering and focal length normalization
- **`Graph` container** — `HashMap`-backed storage for `VertexSE2`, `VertexSE3`, `EdgeSE2`, `EdgeSE3`
- **`GraphLoader` trait** — unified `load` / `write` interface implemented by all loaders
- **`load_graph` convenience function** — auto-detects format from file extension (`.g2o`, `.graph`)
- **`IoError`** — structured error type with line numbers, field counts, and duplicate-vertex detection
- **Memory-mapped I/O** (`memmap2`) — avoids loading full files into heap memory
- **Parallel parsing** (`rayon`) — automatically parallelizes files with more than 1 000 lines
- **`visualization` feature** — Rerun helpers (`to_rerun_position_2d`, `to_rerun_position_3d`, `to_rerun_transform`) for SE2/SE3 vertices
- **Logging** (`init_logger`) — centralized `tracing-subscriber` setup with `RUST_LOG` support

---

*For the top-level apex-solver workspace changelog see [../../doc/CHANGELOG.md](../../doc/CHANGELOG.md)*
