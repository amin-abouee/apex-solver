//! ROS1 bag v2.0 writer.
//!
//! Buffers messages into chunks; each chunk is written when its uncompressed
//! payload exceeds [`Ros1Writer::chunk_threshold`].  Index data records follow
//! each chunk; connection and chunk-info records are written at close.
//!
//! ```text
//! magic
//! bag-header record           (rewritten on close with final index offset)
//! ( chunk record + index-data records )*
//! connection record*          (one per declared topic)
//! chunk-info record*          (one per chunk written)
//! ```

use crate::rosbag::error::{BagError, Result};
use crate::rosbag::types::{Connection, MessageDefinition, MessageDefinitionFormat};
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

use super::format::{OpCode, Ros1Compression};
use super::record::{RecordHeader, write_record};

/// Total size (in bytes) reserved on disk for the bag-header record so that
/// it can be rewritten in place at close time without shifting the rest of
/// the file.  Matches the rosbag tool's `BAG_HEADER_LENGTH` constant.
const BAG_HEADER_RESERVED: usize = 4096;

/// Default chunk threshold (768 KiB) before flushing.
const DEFAULT_CHUNK_THRESHOLD: usize = 768 * 1024;

#[derive(Debug, Clone)]
struct ConnDescriptor {
    id: u32,
    topic: String,
    msg_type: String,
    md5sum: String,
    message_definition: String,
    callerid: String,
    latching: bool,
}

#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    /// Timestamp in nanoseconds since epoch.
    time_ns: u64,
    /// Offset of the message-data record inside the uncompressed chunk stream.
    offset: u32,
}

struct ChunkInfo {
    chunk_pos: u64,
    start_time_ns: u64,
    end_time_ns: u64,
    /// Message counts per connection id.
    counts: HashMap<u32, u32>,
}

/// Writer for ROS1 bag v2.0 files.
pub struct Ros1Writer {
    path: PathBuf,
    file: Option<BufWriter<File>>,
    compression: Ros1Compression,
    chunk_threshold: usize,
    is_open: bool,

    next_conn_id: u32,
    connections: Vec<ConnDescriptor>,

    // Current chunk state.
    chunk_buf: Vec<u8>,
    chunk_start_ns: Option<u64>,
    chunk_end_ns: u64,
    chunk_indices: HashMap<u32, Vec<IndexEntry>>,
    /// Connections already emitted inside the current chunk.
    chunk_emitted_conns: HashMap<u32, ()>,

    chunks: Vec<ChunkInfo>,
    bag_header_pos: u64,
}

impl Ros1Writer {
    /// Create a new writer for `path`. The file is not created until [`Self::open`].
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if path.exists() {
            return Err(BagError::BagAlreadyExists { path });
        }
        Ok(Self {
            path,
            file: None,
            compression: Ros1Compression::None,
            chunk_threshold: DEFAULT_CHUNK_THRESHOLD,
            is_open: false,
            next_conn_id: 0,
            connections: Vec::new(),
            chunk_buf: Vec::new(),
            chunk_start_ns: None,
            chunk_end_ns: 0,
            chunk_indices: HashMap::new(),
            chunk_emitted_conns: HashMap::new(),
            chunks: Vec::new(),
            bag_header_pos: 0,
        })
    }

    /// Set the compression to use for chunk payloads. Must be called before
    /// [`Self::open`].
    pub fn set_compression(&mut self, compression: Ros1Compression) -> Result<()> {
        if self.is_open {
            return Err(BagError::writer("set_compression called after open"));
        }
        self.compression = compression;
        Ok(())
    }

    /// Override the per-chunk size threshold (default 768 KiB).
    pub fn set_chunk_threshold(&mut self, bytes: usize) -> Result<()> {
        if self.is_open {
            return Err(BagError::writer("set_chunk_threshold called after open"));
        }
        self.chunk_threshold = bytes.max(1);
        Ok(())
    }

    /// Open the underlying file and write the magic + placeholder bag header.
    pub fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .read(true)
            .open(&self.path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(super::format::MAGIC)?;

        self.bag_header_pos = super::format::MAGIC.len() as u64;
        write_placeholder_bag_header(&mut writer)?;

        self.file = Some(writer);
        self.is_open = true;
        Ok(())
    }

    /// Declare a topic.  Returns the resulting [`Connection`] handle which
    /// must be passed back to [`Self::write`].
    pub fn add_connection(
        &mut self,
        topic: impl Into<String>,
        msg_type: impl Into<String>,
        md5sum: impl Into<String>,
        message_definition: impl Into<String>,
        latching: bool,
    ) -> Result<Connection> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }
        let topic = topic.into();
        let msg_type = msg_type.into();
        let md5sum = md5sum.into();
        let message_definition = message_definition.into();

        if self.connections.iter().any(|c| c.topic == topic) {
            return Err(BagError::connection_already_exists(topic));
        }

        let id = self.next_conn_id;
        self.next_conn_id += 1;
        let descriptor = ConnDescriptor {
            id,
            topic: topic.clone(),
            msg_type: msg_type.clone(),
            md5sum: md5sum.clone(),
            message_definition: message_definition.clone(),
            callerid: String::new(),
            latching,
        };
        self.connections.push(descriptor);

        Ok(Connection {
            id,
            topic,
            message_type: msg_type,
            message_definition: MessageDefinition {
                format: if message_definition.is_empty() {
                    MessageDefinitionFormat::None
                } else {
                    MessageDefinitionFormat::Msg
                },
                data: message_definition,
            },
            type_description_hash: md5sum,
            message_count: 0,
            serialization_format: "ros1".to_string(),
            offered_qos_profiles: Vec::new(),
        })
    }

    /// Write a single message.  `timestamp_ns` is nanoseconds since epoch.
    pub fn write(&mut self, connection: &Connection, timestamp_ns: u64, data: &[u8]) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }
        let descriptor = self
            .connections
            .iter()
            .find(|c| c.id == connection.id)
            .cloned()
            .ok_or_else(|| BagError::connection_not_found(&connection.topic))?;

        // Emit the connection record into the chunk on first use within this chunk.
        if !self.chunk_emitted_conns.contains_key(&descriptor.id) {
            let offset = encode_connection_record(&mut self.chunk_buf, &descriptor)?;
            self.chunk_emitted_conns.insert(descriptor.id, ());
            let _ = offset; // connection records are not indexed
        }

        let offset = self.chunk_buf.len();
        if offset > u32::MAX as usize {
            return Err(BagError::writer("chunk exceeds 4 GiB uncompressed"));
        }
        encode_message_data_record(&mut self.chunk_buf, descriptor.id, timestamp_ns, data)?;

        self.chunk_indices
            .entry(descriptor.id)
            .or_default()
            .push(IndexEntry {
                time_ns: timestamp_ns,
                offset: offset as u32,
            });

        self.chunk_start_ns = Some(match self.chunk_start_ns {
            Some(s) => s.min(timestamp_ns),
            None => timestamp_ns,
        });
        self.chunk_end_ns = self.chunk_end_ns.max(timestamp_ns);

        if self.chunk_buf.len() >= self.chunk_threshold {
            self.flush_chunk()?;
        }
        Ok(())
    }

    /// Flush the current chunk (if non-empty), write the tail records, then
    /// rewrite the bag header with the correct `index_pos`.
    pub fn close(&mut self) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }
        if !self.chunk_buf.is_empty() {
            self.flush_chunk()?;
        }

        let file = self.file.as_mut().ok_or(BagError::BagNotOpen)?;
        let index_pos = file.stream_position()?;

        // Connection records (re-declared in the tail for fast indexing).
        for c in &self.connections {
            let mut header = RecordHeader::new();
            header
                .put_op(OpCode::Connection)
                .put_u32("conn", c.id)
                .put_str("topic", &c.topic);
            let data = encode_connection_data(c);
            write_record(file, &header, &data)?;
        }

        // Chunk-info records.
        for ci in &self.chunks {
            let mut header = RecordHeader::new();
            header
                .put_op(OpCode::ChunkInfo)
                .put_u32("ver", 1)
                .put_u64("chunk_pos", ci.chunk_pos)
                .put_time(
                    "start_time",
                    (ci.start_time_ns / 1_000_000_000) as u32,
                    (ci.start_time_ns % 1_000_000_000) as u32,
                )
                .put_time(
                    "end_time",
                    (ci.end_time_ns / 1_000_000_000) as u32,
                    (ci.end_time_ns % 1_000_000_000) as u32,
                )
                .put_u32("count", ci.counts.len() as u32);

            let mut data = Vec::with_capacity(ci.counts.len() * 8);
            // Deterministic order.
            let mut keys: Vec<&u32> = ci.counts.keys().collect();
            keys.sort();
            for k in keys {
                data.write_u32::<LittleEndian>(*k)?;
                data.write_u32::<LittleEndian>(ci.counts[k])?;
            }
            write_record(file, &header, &data)?;
        }

        // Rewrite the bag header in place.
        file.flush()?;
        file.seek(SeekFrom::Start(self.bag_header_pos))?;
        write_bag_header(
            file,
            index_pos,
            self.connections.len() as u32,
            self.chunks.len() as u32,
        )?;
        file.flush()?;

        self.file = None;
        self.is_open = false;
        Ok(())
    }

    /// Whether the writer is currently open.
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    fn flush_chunk(&mut self) -> Result<()> {
        let file = self.file.as_mut().ok_or(BagError::BagNotOpen)?;
        let chunk_pos = file.stream_position()?;
        let uncompressed = std::mem::take(&mut self.chunk_buf);
        let uncompressed_size = uncompressed.len() as u32;

        let payload = match self.compression {
            Ros1Compression::None => uncompressed.clone(),
            Ros1Compression::Bz2 => {
                let mut out = Vec::new();
                let mut enc = bzip2::write::BzEncoder::new(&mut out, bzip2::Compression::default());
                enc.write_all(&uncompressed)?;
                enc.finish()?;
                out
            }
            Ros1Compression::Lz4 => lz4_flex::block::compress_prepend_size(&uncompressed),
        };

        let mut chunk_header = RecordHeader::new();
        chunk_header
            .put_op(OpCode::Chunk)
            .put_str("compression", self.compression.as_str())
            .put_u32("size", uncompressed_size);
        write_record(file, &chunk_header, &payload)?;

        // IndexData records, one per connection in this chunk.
        let mut conn_ids: Vec<u32> = self.chunk_indices.keys().copied().collect();
        conn_ids.sort();
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for cid in &conn_ids {
            let entries = &self.chunk_indices[cid];
            counts.insert(*cid, entries.len() as u32);

            let mut header = RecordHeader::new();
            header
                .put_op(OpCode::IndexData)
                .put_u32("ver", 1)
                .put_u32("conn", *cid)
                .put_u32("count", entries.len() as u32);

            let mut data = Vec::with_capacity(entries.len() * 12);
            for e in entries {
                let secs = (e.time_ns / 1_000_000_000) as u32;
                let nsecs = (e.time_ns % 1_000_000_000) as u32;
                data.write_u32::<LittleEndian>(secs)?;
                data.write_u32::<LittleEndian>(nsecs)?;
                data.write_u32::<LittleEndian>(e.offset)?;
            }
            write_record(file, &header, &data)?;
        }

        self.chunks.push(ChunkInfo {
            chunk_pos,
            start_time_ns: self.chunk_start_ns.unwrap_or(0),
            end_time_ns: self.chunk_end_ns,
            counts,
        });

        // Reset chunk state.
        self.chunk_indices.clear();
        self.chunk_emitted_conns.clear();
        self.chunk_start_ns = None;
        self.chunk_end_ns = 0;
        Ok(())
    }
}

impl Drop for Ros1Writer {
    fn drop(&mut self) {
        if self.is_open {
            let _ = self.close();
        }
    }
}

fn write_bag_header<W: Write + Seek>(
    w: &mut W,
    index_pos: u64,
    conn_count: u32,
    chunk_count: u32,
) -> Result<()> {
    let mut header = RecordHeader::new();
    header
        .put_op(OpCode::BagHeader)
        .put_u64("index_pos", index_pos)
        .put_u32("conn_count", conn_count)
        .put_u32("chunk_count", chunk_count);

    let header_bytes = header.encode();
    // Pad the data section so the whole record occupies BAG_HEADER_RESERVED bytes.
    let overhead = 4 + header_bytes.len() + 4;
    if overhead > BAG_HEADER_RESERVED {
        return Err(BagError::writer("bag header exceeds reserved size"));
    }
    let pad_len = BAG_HEADER_RESERVED - overhead;
    let padding = vec![b' '; pad_len];

    w.write_u32::<LittleEndian>(header_bytes.len() as u32)?;
    w.write_all(&header_bytes)?;
    w.write_u32::<LittleEndian>(pad_len as u32)?;
    w.write_all(&padding)?;
    Ok(())
}

fn write_placeholder_bag_header<W: Write + Seek>(w: &mut W) -> Result<()> {
    write_bag_header(w, 0, 0, 0)
}

fn encode_connection_data(c: &ConnDescriptor) -> Vec<u8> {
    let mut sub = RecordHeader::new();
    sub.put_str("topic", &c.topic)
        .put_str("type", &c.msg_type)
        .put_str("md5sum", &c.md5sum)
        .put_str("message_definition", &c.message_definition)
        .put_str("callerid", &c.callerid)
        .put_str("latching", if c.latching { "1" } else { "0" });
    sub.encode()
}

fn encode_connection_record(buf: &mut Vec<u8>, c: &ConnDescriptor) -> Result<usize> {
    let start = buf.len();
    let mut header = RecordHeader::new();
    header
        .put_op(OpCode::Connection)
        .put_u32("conn", c.id)
        .put_str("topic", &c.topic);
    let data = encode_connection_data(c);
    write_record(buf, &header, &data)?;
    Ok(start)
}

fn encode_message_data_record(
    buf: &mut Vec<u8>,
    conn_id: u32,
    timestamp_ns: u64,
    data: &[u8],
) -> Result<()> {
    let mut header = RecordHeader::new();
    let secs = (timestamp_ns / 1_000_000_000) as u32;
    let nsecs = (timestamp_ns % 1_000_000_000) as u32;
    header
        .put_op(OpCode::MessageData)
        .put_u32("conn", conn_id)
        .put_time("time", secs, nsecs);
    write_record(buf, &header, data)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn open_writes_magic_and_placeholder() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("a.bag");
        let mut w = Ros1Writer::new(&path)?;
        w.open()?;
        w.close()?;

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(super::super::format::MAGIC));
        // Magic + 4096-byte reserved bag header + no chunks/conns.
        assert_eq!(
            bytes.len(),
            super::super::format::MAGIC.len() + BAG_HEADER_RESERVED
        );
        Ok(())
    }

    #[test]
    fn add_connection_rejects_duplicates() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dup.bag");
        let mut w = Ros1Writer::new(&path)?;
        w.open()?;
        let _ = w.add_connection("/imu", "sensor_msgs/Imu", "abc", "", false)?;
        let dup = w.add_connection("/imu", "sensor_msgs/Imu", "abc", "", false);
        assert!(matches!(dup, Err(BagError::ConnectionAlreadyExists { .. })));
        Ok(())
    }

    #[test]
    fn cannot_open_existing_file() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("exists.bag");
        std::fs::write(&path, b"x").unwrap();
        let r = Ros1Writer::new(&path);
        assert!(matches!(r, Err(BagError::BagAlreadyExists { .. })));
        Ok(())
    }

    #[test]
    fn is_open_accessor() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("accessor.bag");
        let mut w = Ros1Writer::new(&path)?;
        assert!(!w.is_open());
        w.open()?;
        assert!(w.is_open());
        w.close()?;
        assert!(!w.is_open());
        Ok(())
    }

    #[test]
    fn set_compression_after_open_errors() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("comp.bag");
        let mut w = Ros1Writer::new(&path)?;
        w.open()?;
        let result = w.set_compression(Ros1Compression::Bz2);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn set_chunk_threshold_after_open_errors() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("thresh.bag");
        let mut w = Ros1Writer::new(&path)?;
        w.open()?;
        let result = w.set_chunk_threshold(1024);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn write_before_open_errors() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("noopen.bag");
        let mut w = Ros1Writer::new(&path)?;
        let conn = Connection {
            id: 0,
            topic: "/test".into(),
            message_type: "std_msgs/String".into(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "ros1".into(),
            offered_qos_profiles: Vec::new(),
        };
        let result = w.write(&conn, 0, &[]);
        assert!(matches!(result, Err(BagError::BagNotOpen)));
        Ok(())
    }

    #[test]
    fn add_connection_before_open_errors() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("noopen2.bag");
        let mut w = Ros1Writer::new(&path)?;
        let result = w.add_connection("/t", "std_msgs/String", "md5", "", false);
        assert!(matches!(result, Err(BagError::BagNotOpen)));
        Ok(())
    }

    #[test]
    fn close_when_not_open_is_noop() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("noop.bag");
        let mut w = Ros1Writer::new(&path)?;
        assert!(w.close().is_ok());
        Ok(())
    }

    #[test]
    fn write_unknown_connection_errors() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("unknown.bag");
        let mut w = Ros1Writer::new(&path)?;
        w.open()?;
        let fake_conn = Connection {
            id: 999,
            topic: "/fake".into(),
            message_type: "std_msgs/String".into(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "ros1".into(),
            offered_qos_profiles: Vec::new(),
        };
        let result = w.write(&fake_conn, 0, &[]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn full_write_read_cycle() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cycle.bag");
        let mut w = Ros1Writer::new(&path)?;
        w.open()?;
        let conn = w.add_connection("/imu", "sensor_msgs/Imu", "md5sum", "def", false)?;
        w.write(&conn, 1_000_000_000, &[1, 2, 3])?;
        w.write(&conn, 2_000_000_000, &[4, 5, 6])?;
        w.close()?;

        let mut r = crate::rosbag::ros1::Ros1Reader::new(&path)?;
        r.open()?;
        assert_eq!(r.message_count(), 2);
        assert_eq!(r.connections().len(), 1);
        assert!(!r.topics().is_empty());
        assert!(r.start_time() > 0);
        assert!(r.end_time() >= r.start_time());
        assert!(r.duration() > 0);
        Ok(())
    }
}
