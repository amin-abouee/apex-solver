# ROS1 Bags

`rosbag::ros1` reads and writes ROS1 **`.bag` v2.0** files natively (no `rosbag`
C++ dependency), using the [shared types](./overview.md). It also exposes the
low-level ROS1 wire (de)serializer so you can decode message payloads yourself.

Re-exports: `Ros1Reader`, `Ros1Writer`, `Ros1Compression`, `OpCode`,
`Ros1Serializer`, `Ros1Deserializer`.

## `Ros1Reader`

```rust
pub struct Ros1Reader;
impl Ros1Reader { pub fn new(path: impl Into<PathBuf>) -> Result<Self>; }
```

| Method | Signature | Description |
|---|---|---|
| `open` | `(&mut self) -> Result<()>` | Parse the index and connection records. |
| `connections` | `-> &[Connection]` | All topics/connections. |
| `topics` | `-> &HashMap<String, TopicInfo>` | Topic summaries. |
| `start_time` / `end_time` / `duration` | `-> u64` | Timing (ns). |
| `message_count` | `-> u64` | Total messages. |
| `messages` | `(&mut self) -> Result<Vec<Message>>` | All decoded messages. |
| `raw_messages` | `(&mut self) -> Result<Vec<RawMessage>>` | All raw (serialized) messages. |
| `messages_filtered` | `(topics, start, end) -> …` | Decoded, filtered by topic/time. |
| `raw_messages_filtered` | `(topics, start, end) -> …` | Raw, filtered by topic/time. |

## `Ros1Writer`

```rust
pub struct Ros1Writer;
impl Ros1Writer { pub fn new(path: impl Into<PathBuf>) -> Result<Self>; }
```

| Method | Signature | Description |
|---|---|---|
| `set_compression` | `(Ros1Compression) -> Result<()>` | Chunk compression (before `open`). |
| `set_chunk_threshold` | `(bytes: usize) -> Result<()>` | Bytes per chunk before flush. |
| `open` | `(&mut self) -> Result<()>` | Write the bag header. |
| `add_connection` | `(topic, message_type, …) -> Result<Connection>` | Register a topic. |
| `write` | `(&Connection, timestamp_ns: u64, data: &[u8]) -> Result<()>` | Append a serialized message. |
| `close` | `(&mut self) -> Result<()>` | Flush chunks + write the index. |
| `is_open` | `-> bool` | Writer state. |

<a id="compression"></a>
## Compression

```rust
pub enum Ros1Compression { None, Bz2, Lz4 }
```

`as_str()` and `parse(&str)` convert to/from the on-disk name. Set it with
`Ros1Writer::set_compression` before `open()`.

## Wire format & low-level codec

The ROS1 record layer is exposed for advanced use:

- **`OpCode`** — bag record op codes, with `from_u8` / `as_u8`.
- **`Ros1Deserializer<'a>`** — little-endian reader over a `&[u8]`:
  `read_u8/i8/bool/u16/i16/u32/i32/u64/i64/f32/f64`, `read_string`,
  `read_time_nanos`, `read_bytes`, `read_seq`, `read_array`, plus `position()`
  and `remaining()`.
- **`Ros1Serializer`** — the matching writer: `write_u8/…/f64`, `into_bytes`,
  `as_slice`.

These implement the ROS1 message serialization (little-endian, length-prefixed
strings/arrays) used to decode the [ROS1 message types](./messages.md).

## Example

```rust
use apex_io::rosbag::{Ros1Reader, Ros1Writer, Ros1Compression};

// Read every message, filtered to one topic.
let mut reader = Ros1Reader::new("input.bag")?;
reader.open()?;
let imu = reader.messages_filtered(Some(&["/imu"]), None, None)?;

// Write a compressed copy.
let mut writer = Ros1Writer::new("copy.bag")?;
writer.set_compression(Ros1Compression::Lz4)?;
writer.open()?;
// … add_connection + write per message …
writer.close()?;
# Ok::<(), apex_io::rosbag::BagError>(())
```

## References

- ROS Wiki — *Bags/Format/2.0*. http://wiki.ros.org/Bags/Format/2.0
