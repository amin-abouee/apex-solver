# ROS Bags — Overview & Shared Types

The `rosbag` module reads and writes both **ROS1** (`.bag`) and **ROS2**
(SQLite3 / MCAP) bags, and can subscribe to a live **DDS** stream. All three
share one vocabulary of types (`Connection`, `Message`, `RawMessage`, QoS, …)
defined in `rosbag::types`, so a topic read from a ROS1 bag can be written to a
ROS2 bag without translation. This page is the shared reference; the format
specifics live in [ROS1](./ros1.md), [ROS2](./ros2.md), and [DDS](./dds.md).

## Module map

| Path | Contents |
|---|---|
| `rosbag::types` | Shared `Connection`, `Message`, `RawMessage`, QoS, compression & storage enums |
| `rosbag::error` | `BagError` + `Result`/`ReaderResult`/`WriterResult` aliases |
| `rosbag::ros1` | ROS1 `.bag` reader/writer + CDR-free ROS1 (de)serializer |
| `rosbag::ros2` | ROS2 reader/writer, CDR, metadata, storage plugins, message types |
| `rosbag::ros2::dds` | Live DDS subscription (feature `dds`) |
| `read_bag_metadata_fast` | Fast ROS2 `metadata.yaml`-only summary (no storage open) |

Top-level re-exports: `Reader`, `Writer` (ROS2), `Ros1Reader`, `Ros1Writer`,
`Ros1Compression`, `BagMetadata`, `TopicMetadata`, `BagError`, and the shared
`types::*`.

## Connections and messages

A **`Connection`** describes a topic (its name, ROS message type, and message
definition/schema); readers and writers are keyed by it.

```rust
pub struct Connection {
    pub id: u32,
    pub topic: String,
    pub message_type: String,
    pub message_definition: MessageDefinition,
    pub type_description_hash: String,
    // … qos, etc.
}
```

Two message representations flow through every reader/writer:

| Type | Payload | Use |
|---|---|---|
| **`Message`** | topic, timestamp, **typed/decoded** contents | Typed consumption |
| **`RawMessage`** | topic, timestamp, **serialized bytes** | Zero-copy copy/filter/convert |

The raw path (`raw_messages`, `write_raw_message`) never touches CDR
(de)serialization, which is what makes format conversion and topic filtering
cheap.

`TopicInfo` summarizes a topic: `msgtype()` and `msgcount()` accessors.

## Quality of Service (QoS)

ROS2 QoS is modelled faithfully so it round-trips through bags and DDS:

```rust
pub struct QosProfile {
    pub history: QosHistory,          // KeepLast | KeepAll
    pub depth: i32,
    pub reliability: QosReliability,  // Reliable | BestEffort
    pub durability: QosDurability,    // Volatile | TransientLocal
    pub liveliness: QosLiveliness,    // Automatic | ManualByTopic
    pub deadline: QosTime,
    pub lifespan: QosTime,
    // … lease durations
}
```

Each enum exposes `as_str()`; `QosTime` is a `(sec, nsec)` pair. The
[DDS chapter](./dds.md#qos-mapping) documents how these map to the wire.

## Compression & storage enums

| Enum | Variants |
|---|---|
| `CompressionMode` | `None`, `Message`, `File`, `Storage` |
| `CompressionFormat` | `None`, `Zstd` |
| `StoragePlugin` | `Sqlite3`, `Mcap` |
| `MessageDefinitionFormat` | `None`, … (schema encodings) |

ROS1 additionally supports `bz2` and `lz4` chunk compression via
[`Ros1Compression`](./ros1.md#compression).

## Time types

`Duration`, `StartingTime`, and `QosTime` carry nanosecond-resolution ROS time.
Readers expose `start_time()`, `end_time()`, `duration()`, and `message_count()`.

## Fast metadata

```rust
pub fn read_bag_metadata_fast<P: AsRef<Path>>(bag_path: P) -> Result<BagMetadata>;
```

Reads only a ROS2 bag's `metadata.yaml` — topics, counts, timing, compression —
**without** opening any storage file. This backs the `bag_info` CLI and is the
fastest way to inspect a bag.

## Error model

Everything returns `Result<_, BagError>`. `BagError` has constructor helpers for
each failure class:

`generic`, `writer`, `compression`, `invalid_message_data`,
`cdr_deserialization`, `message_type_not_found`, `schema_validation`,
`connection_not_found`, `connection_already_exists`, `invalid_qos_profile`.

Aliases: `Result<T>`, `ReaderResult<T>`, `WriterResult<T>` (all
`= Result<T, BagError>`); `ReaderError = BagError`.
