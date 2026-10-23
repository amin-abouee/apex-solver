# Changelog

All notable changes to `apex-io` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (LaMAria)
- **`lamaria::Calibration`**: a typed reader for the dataset's `aria_calibrations/<sequence>.json` (two
  `RAD_TAN_THIN_PRISM_FISHEYE` cameras and the IMU: intrinsics, `T_body_sensor`, noise and saturation). All 86 files parse.
- **`TumNanosLoader`** and `TrajectoryFormat::TumNanos`: TUM rows with an integer-nanosecond timestamp, which is how LaMAria's
  `ground_truth/pGT/<sequence>.txt` is stored. `TumLoader` reads such a bare integer as seconds and overflows.
- `tests/lamaria_real_files.rs` (set `APEX_LAMARIA_ROOT`): parses every calibration, loads every ground truth, and checks it lies
  inside its recording's time range, which also fixes the unit as nanoseconds.
- Cookbook chapter *VRS and LaMAria*, and the `vrs` feature in the feature table.

### Fixed
- A ROS 2 MCAP bag whose `metadata.yaml` says `compression_mode: message` with an empty `compression_format` (what `bag_convert`
  wrote) is read with plain payloads. The MESSAGE-compression check introduced with the reader overhaul rejected it with
  "Unsupported compression format".

### Changed (merge)
- The VRS work and the ROS 1 / ROS 2 overhaul are merged onto the line that reads MCAP with `mcap` 0.23: the MCAP reader and writer
  are the chunked implementation of the overhaul, and the hand-written per-type converters of `bag_convert` are replaced by its
  schema-driven transcoder (which covers `PointCloud2` and `tf2_msgs/TFMessage`). The `mcap_stream` benchmark is superseded by
  `rosbag_bench`.

### Added (verified optimisation pass)
- **`BagReader::iter(&Selection)`**: a pull iterator (`BagIterator`: `next_into`, `next_message`, `Iterator`) over *every*
  container (ROS1, sqlite3, MCAP, VRS) yielding exactly what `for_each` yields, for consumers such as a VIO front end that keep
  state between samples. `BagReader::params()` returns a `ContainerParams` (ROS2 `metadata.yaml` facts; for VRS the file tags,
  factory calibration JSON and typed camera/IMU configs).
- `VrsReader::verify_record_chain()` (also in `vrs_info --verify-index`), `VrsReader::data_ordinal()`, and an opt-in device-clock
  to UTC `aria::TimeDomainMap` with exact integer interpolation.
- `lifecycle_msgs` (ROS2) and `actionlib_msgs` (ROS1) in the standard-message catalog (168 types); ROS1 md5sums verified against
  the distribution's values.
- `doc/perf_accuracy_review.md`: the claim-by-claim verification behind this pass.

### Changed (performance)
- **ROS1 reader** opens from the tail only and indexes lazily per query: open 2.8 ms -> 0.05 ms, 2 s window read -92 %, full read
  -45 % on the synthetic fixture. Index metadata is validated (`IndexData` `ver` = 1 per the bag 2.0 spec, header counts, per-chunk
  message counts against the chunk infos).
- **MCAP writer** is an own chunked encoder with order-preserving parallel chunk compression: zstd writes ~2.5x faster, and the
  uncompressed path no longer copies chunks (peak memory 335 -> 68 MiB on a 1.3 GB bag). Output validated with the `mcap` crate
  and Python `mcap` (CRCs, index-driven reads).
- **ROS1 writer** compresses bz2/lz4 chunks on all cores, in order: bz2 bag output 25 -> 163 MiB/s.
- **sqlite writer** uses 32 KiB pages: camera/lidar-heavy writes ~2x faster, reads unchanged within noise.
- `Reader` decodes rosbag2 **MESSAGE**-compressed payloads (sqlite and MCAP-directory bags) with a reused zstd context;
  previously such bags read back as zstd garbage.

### Fixed
- VRS-as-bag message `seq` is the record's ordinal within its stream (was a per-reader counter, so repeating a query changed the bytes).
- `CdrDeserializer::read_i64/read_u64` added; `read_sequence` no longer pre-allocates from a corrupt length.

### Added (VRS)
- **`vrs` feature: Meta VRS / Project Aria reader and writer**, implemented from scratch in safe Rust (no crate
  existed). `VrsReader` (index with rebuild fallback, description tags, `DataLayout`/`RecordFormat`, time-ordered
  `Cursor`, nearest/range lookups, zero-allocation reads), `VrsWriter` + `LayoutWriter`, `copy_filtered`, and typed
  `aria::AriaRecording` (IMU, magnetometer, barometer, GPS, JPEG camera frames, configs, calibration) with one-by-one
  and by-timestamp access suited to embedding in a VIO pipeline.
- `BagKind::Vrs`: VRS in `BagReader`/`BagWriter` and every `bag_*` tool. Aria streams read as standard ROS topics;
  ROS topics write losslessly (raw bytes + type/definition/encoding tags).
- Tools `vrs_info`, `vrs_copy`, `vrs_dump`; `scripts/vrs_sweep.sh` and `scripts/vrs_crosscheck.py` (Meta reader).

### Breaking
- `rosbag::Message::connection` and `RawMessage::connection` are now
  `Arc<Connection>` (was an owned `Connection` cloned per message, including its
  message definition). Field access is unchanged through `Deref`.
- `Ros1Reader::raw_messages_filtered` keeps returning a `Vec`, but streaming
  callers should use the new `for_each_raw_filtered` (borrowed slices, no
  per-message allocation).
- `bag_filter`: `--standard-copy`, `--buffer-size-mb`, `--batch-size`,
  `--compression-mode`/`--compression-format` were removed (writes are always
  streamed); use `--compression none|zstd|lz4|bz2` and `--format`.
- Time windows are consistently `[start, end)` for every backend (the MCAP
  reader previously included `end`).
- `Writer::configure_buffer`'s `buffer_size_mb` is ignored; `Writer::write*`
  write straight through to the storage backend.

### Added
- **`BagReader` / `BagWriter`**: one streaming API over ROS1 `.bag`, ROS2
  sqlite3, ROS2 MCAP directories and bare `.mcap` files, with `detect_kind`.
- **`rosbag::transcode`**: schema-driven ROS1 <-> CDR converter compiled from
  the message definition. Handles every type with a definition (including
  user-defined ones), block-copies byte arrays, and ships a catalog of 161
  standard interface types (std/geometry/sensor/nav/shape/stereo/trajectory/
  visualization/diagnostic msgs, tf2_msgs, builtin_interfaces, rcl_interfaces,
  rosgraph_msgs, ...) for ROS2 sqlite bags that carry no schema. Includes
  genmsg-compatible md5sum computation and definition rendering for both
  dialects. Unsupported topics are reported with `tracing::warn!`.
- `bag_convert` rewritten on the transcoder (any container to any container,
  streaming, `--strict`, `--list-topics`); new `bag_verify` tool;
  `scripts/dataset_sweep.sh` runs info/filter/convert/verify over a dataset tree.
- `bag_info` reads ROS1 `.bag`, sqlite3 and MCAP; adds per-topic encoding, rate
  and a "not convertible" marker.
- Real `McapWriter` (chunked, optional zstd, summary + statistics).

### Fixed
- **`McapWriter` was a stub** that wrote only the file magic, so
  `--format mcap` outputs were empty.
- **ROS1 `lz4` chunks used the wrong format** (size-prefixed LZ4 block) and were
  unreadable by ROS tools; they are now LZ4 frames like `rosbag` writes. Older
  block-format bags still read.
- SQLite reader materialised the whole result set in memory; it now streams with
  keyset pagination. Messages on topics missing from `metadata.yaml` were dropped
  silently; they are now read.
- `Ros1Reader::open` read every chunk payload just to index the bag; it now
  seeks past them.
- `Reader::topics()` order is deterministic (sorted by name).

### Performance
- Allocation-free ROS1 message path (reader and writer); sqlite writer uses one
  rolling transaction, no journal, and builds the timestamp index at close.
- SIMD CRC-32 for MCAP chunk validation (reads ~2.4x faster, CPU-bound case).
- MCAP chunks are decompressed and CRC-checked on a small worker pool
  (order-preserving, bounded memory): a further ~2.7x on a 4 GB zstd bag, and
  `bag_convert` of a ~1 GB bag drops from ~4 s to ~0.6 s.

### Fixed
- **MCAP reading is bounded-memory and profile-agnostic.** The MCAP storage
  backend no longer loads whole files into the heap (`fs::read` of a 13 GB bag
  allocated 13 GB) nor buffers every message into a `Vec` before iterating.
  It now seeks through the container's summary section (footer → summary) and
  streams messages chunk-by-chunk, decompressing and holding one chunk at a
  time; chunks outside a requested time window are skipped without being
  read. Multi-file iteration is a k-way merge in log-time order.
  Implementation: the new `ros2::storage::mcap_format` module — a small,
  seek-based MCAP container parser (summary, chunk indexes, chunk records,
  zstd/lz4 payloads, chunk CRC validation) that does not need the `mcap`
  crate's whole-buffer read API. The workspace's `unsafe_code = "forbid"`
  ruled out memory-mapping.
- **Topic metadata on MCAP bags is now correct.** Connections built from
  MCAP summaries report the schema name (e.g. `sensor_msgs/Image`) as
  `message_type` — previously they reported the channel's *message encoding*
  (e.g. `cdr`) there — and report the channel encoding as
  `serialization_format`, instead of a hardcoded `"cdr"`. Per-topic message
  counts come from the Statistics record rather than a full message scan.
- **`bag_info` handles bare `.mcap` files** (and reports storage id
  accordingly) instead of requiring a bag directory.
- `MCAP` summary discovery no longer walks and decompresses the entire file
  just to list topics (milliseconds vs. minutes on multi-GB bags).

### Added
- **`rosbag::Reader` accepts a bare `.mcap` file** (single-file bags such as
  the HILTI SLAM Challenge recordings) in addition to a bag directory.
  Topics, message counts, and the time range are read from the MCAP summary
  on `open()`; `metadata()` returns `None` for such readers, and
  `duration()`/`start_time()`/`end_time()`/`message_count()` come from the
  summary. Bare files with an extension other than `.mcap` are rejected
  (ROS1 `.bag` files belong to `rosbag::ros1::Ros1Reader`).
- **ROS1-profile MCAP support end to end.** ROS1-serialized payloads inside
  MCAP containers (`profile: ros1`, `message_encoding: ros1`, `ros1msg`
  schemas) now flow through `Reader`: connections carry
  `serialization_format = "ros1"`, so payload decoding dispatches on the
  encoding — `ros1::deserialize::Ros1Deserializer` + `ros1::messages` for
  `ros1`, `cdr::CdrDeserializer` + `messages` for `cdr`. Documented with an
  example in the `rosbag` module docs.
- Chunk-index time-skip filtering and per-chunk CRC validation (IEEE CRC-32,
  validated when a chunk declares one).
- `tests/hilti_mcap.rs`: env-gated (`APEX_HILTI_MCAP`) integration test
  decoding a real ROS1-profile dataset bag end to end.

## [0.4.0] - 2026-09-29

### Breaking
- **`rosbag` is now an opt-in feature** (was built unconditionally). Depending on `apex-io`
  no longer compiles `rusqlite` (bundled SQLite), `mcap`, `zstd`, `lz4_flex`, `serde_yaml`,
  `byteorder`, `hex`. Bag I/O users must opt in:
  ```toml
  apex-io = { version = "0.4", features = ["rosbag"] }
  ```
  The `bag_*` binaries now require `--features rosbag`.
- **`download` and `clap` are feature-gated** behind the default-on `download` and `cli`
  features. `--no-default-features` leaves a lean library; the `ensure_*_dataset` helpers
  then only serve already-downloaded files. `download_datasets` requires `--features
  download`.
- **Pose-graph modules moved into a dedicated `graph` module**: `g2o`/`toro` become
  `graph::g2o` / `graph::toro` (`G2oLoader` / `ToroLoader` remain re-exported at the crate
  root), and the crate-root `IoError` moves to `graph::IoError` alongside the `Graph`
  model (`Graph`, `VertexSE2/3`, `EdgeSE2/3`, `GraphLoader`, `load_graph`).

### Added
- **`trajectory` module** — a unified timestamped-pose container (`Trajectory`,
  `TrajectoryPose`, `InertialState`) with format detection (`TrajectoryFormat`),
  `load_trajectory` / `load_trajectory_as`, time-span and interpolation helpers
  (`pose_at`, `position_at_seconds`, `se3_at`, …), and TUM read/write
  (`TumLoader`) plus ASL ground-truth loading (`AslTrajectoryLoader`,
  `load_mav0_trajectory`, `AslLayout`).
- **Multi-sensor dataset support**: `ensure_sensor_dataset`, `SENSOR_DATA_DIR`, and CSV
  utilities backing the NCLT-style odometry + GNSS fusion workflows.

### Fixed
- **ROS 2 CDR deserialization alignment** in the reader.
- **ROS1 chunk cache** is bounded to two payloads; reader `close`/`Drop` symmetry.
- Reader diagnostics now route through `tracing` (no direct prints).

## [0.3.0] - 2026-07-30

### Added
- **Cookbook** (`doc/cookbook`, mdBook + KaTeX) documenting every public functionality of the
  crate, organized by domain with per-type / per-function API tables:
  - Pose-graph formats — the `Graph` model, G2O, TORO, BAL.
  - Sensor datasets — ASL / EuRoC (`AslReader`, `AslStream`).
  - ROS bags — shared types, ROS1 (`Ros1Reader`/`Ros1Writer` + wire codec), ROS2
    (`Reader`/`Writer`, SQLite3 & MCAP storage, CDR), DDS live subscription, and the catalog
    of supported ROS1/ROS2 message types.
  - Utilities & tools — dataset registry & downloads, all seven CLI binaries, logging &
    visualization, and a feature-flag reference.

### Notes
- No public API changes; this release adds documentation only.

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
([apex-solver v1.1.0](../../CHANGELOG.md#110---2026-02-21)). Extracted from the monolithic
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

*For the top-level apex-solver workspace changelog see [../../CHANGELOG.md](../../CHANGELOG.md)*
