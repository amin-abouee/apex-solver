# ROS2 Bags

`rosbag::ros2` reads and writes ROS2 bags backed by either **SQLite3** or
**MCAP** storage, with a `metadata.yaml` sidecar. A single `Reader` / `Writer`
API works across both storage plugins; the plugin is chosen from the bag
(reading) or by `StoragePlugin` (writing).

Re-exports: `Reader`, `Writer`, `BagMetadata`, `TopicMetadata`, plus submodules
`cdr`, `messages`, `metadata`, `storage`.

## `Reader`

```rust
pub struct Reader;
impl Reader { pub fn new<P: AsRef<Path>>(bag_path: P) -> Result<Self>; }
```

| Method | Signature | Description |
|---|---|---|
| `open` / `close` | `(&mut self) -> Result<()>` | Open/close the storage file(s). |
| `is_open` | `-> bool` | Reader state. |
| `metadata` | `-> Option<&BagMetadata>` | Parsed `metadata.yaml`. |
| `topics` | `-> Vec<TopicInfo>` | Topic summaries. |
| `connections` | `-> &[Connection]` | Connections (topic + type + QoS). |
| `start_time` / `end_time` / `duration` / `message_count` | `-> u64` | Timing & counts. |
| `messages` | `-> Result<Box<dyn Iterator<Item = Result<Message>>>>` | Stream decoded messages. |
| `messages_filtered` | `(topics, start, end) -> …` | Decoded, filtered. |
| `raw_messages` | `-> Result<Box<dyn Iterator<Item = Result<RawMessage>>>>` | Stream raw bytes. |
| `raw_messages_filtered` | `(topics, start, end) -> …` | Raw, filtered. |
| `read_raw_messages_batch` | `(…) -> …` | Batch raw read. |

Messages are exposed as **iterators**, so a multi-GB bag streams through without
being held in memory.

## `Writer`

```rust
pub struct Writer;
impl Writer {
    pub fn new<P: AsRef<Path>>(bag_path: P, storage: StoragePlugin, …) -> Result<Self>;
}
```

| Method | Signature | Description |
|---|---|---|
| `set_compression` | `(mode, format) -> Result<()>` | e.g. per-message / file, Zstd. |
| `set_custom_data` | `(key, value) -> Result<()>` | Extra `metadata.yaml` fields. |
| `configure_buffer` | `(…) -> Result<()>` | Write-buffer sizing. |
| `flush_buffer` | `(&mut self) -> Result<()>` | Force a flush. |
| `open` / `close` | `(&mut self) -> Result<()>` | Open storage / finalize + write metadata. |
| `add_connection` | `(topic, type, …) -> Result<Connection>` | Register a topic. |
| `write` | `(&Connection, timestamp: u64, data: &[u8]) -> Result<()>` | Append a serialized message. |
| `write_raw_message` | `(…)` | Append a `RawMessage`. |
| `write_raw_messages_batch` | `(…)` | Batch write. |
| `connections` / `is_open` | | State accessors. |

## Storage plugins

```rust
pub trait StorageReader { /* open, read, topics, … */ }
pub trait StorageWriter: std::any::Any { /* open, write, close, … */ }

pub fn create_storage_reader(path, plugin) -> Result<Box<dyn StorageReader>>;
pub fn create_storage_writer(path, plugin) -> Result<Box<dyn StorageWriter>>;
```

| Plugin | Backend | Notes |
|---|---|---|
| `StoragePlugin::Sqlite3` | `rusqlite` | The classic `rosbag2` `.db3` format. |
| `StoragePlugin::Mcap` | `mcap` | The newer self-describing container. |

The `Reader`/`Writer` delegate to these; you rarely construct them directly, but
`create_storage_reader/writer` let you target a specific backend.

## Metadata

```rust
pub struct BagMetadata; // parsed metadata.yaml
impl BagMetadata {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn validate(&self) -> Result<()>;
    pub fn info(&self) -> &BagFileInformation;
    pub fn start_time(&self) / end_time(&self) / duration(&self) / message_count(&self) -> u64;
    pub fn is_compressed(&self) -> bool;
    pub fn compression_mode(&self) -> Option<&str>;
}
```

Supporting structs: `BagFileInformation`, `TopicWithMessageCount`,
`TopicMetadata`, `FileInformation`, and the `QosProfilesField` enum (QoS may be
stored inline or as a YAML string). For a summary without opening storage, use
[`read_bag_metadata_fast`](./overview.md#fast-metadata).

## CDR (de)serialization

`rosbag::ros2::cdr` implements the **Common Data Representation** used on the ROS2
wire: it decodes `RawMessage` bytes into the [typed messages](./messages.md) and
encodes them back, honoring CDR alignment and endianness. Decode failures surface
as `BagError::cdr_deserialization { message_type, … }`.

## Example

```rust
use apex_io::rosbag::{Reader, Writer};
use apex_io::rosbag::types::StoragePlugin;

// Stream a bag, filtered to two topics.
let mut reader = Reader::new("rosbag2_2024")?;
reader.open()?;
for msg in reader.messages_filtered(Some(&["/imu", "/odom"]), None, None)? {
    let msg = msg?;
    // … handle msg …
}

// Convert it to MCAP with Zstd file compression (raw path — no re-decode).
let mut writer = Writer::new("out_mcap", StoragePlugin::Mcap /*, … */)?;
writer.open()?;
// … copy connections + raw_messages …
writer.close()?;
# Ok::<(), apex_io::rosbag::BagError>(())
```

## References

- ROS2 `rosbag2` design — https://github.com/ros2/rosbag2
- MCAP specification — https://mcap.dev
- OMG *Common Data Representation (CDR)*, part of the DDS-RTPS specification.
