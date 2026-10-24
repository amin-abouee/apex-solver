//! MCAP storage backend for writing bag files.
//!
//! One schema per message type, one channel per [`Connection`], and a summary
//! section (statistics, chunk indexes) on [`close`](StorageWriter::close) so
//! the file is seekable by [`McapStorageReader`](super::McapStorageReader) and
//! by third-party tools. Chunks are compressed on several cores, in order
//! (see [`chunked`](super::chunked)).
//!
//! Message payloads are written verbatim, so the channel's
//! `message_encoding` follows the connection's `serialization_format`
//! (`cdr` for ROS2 payloads, `ros1` for ROS1 payloads).

use super::chunked::{ChunkCompression, ChunkedWriter};
use crate::rosbag::error::{BagError, Result};
use crate::rosbag::storage::StorageWriter;
use crate::rosbag::types::{CompressionMode, Connection};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Uncompressed chunk target. Larger than a single camera frame so per-chunk
/// index and compression overhead amortise; small enough that readers (which
/// hold a few decompressed chunks per worker) stay light.
const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Per-connection write state.
struct ChannelState {
    id: u16,
    sequence: u32,
}

/// A schema declared by `add_msgtype`, registered in the file on first use.
struct PendingSchema {
    encoding: &'static str,
    data: Vec<u8>,
}

/// Writer producing a single `.mcap` file.
pub struct McapWriter {
    mcap_path: PathBuf,
    profile: &'static str,
    compression: ChunkCompression,
    writer: Option<ChunkedWriter>,
    /// Schemas declared by `add_msgtype`, keyed by message type name.
    pending_schemas: HashMap<String, PendingSchema>,
    /// Ids of schemas already written to the file.
    schema_ids: HashMap<String, u16>,
    /// Channels registered by `add_connection`, keyed by connection id.
    channels: HashMap<u32, ChannelState>,
}

impl McapWriter {
    /// Create a writer for a ROS2 bag directory: the file is
    /// `<bag_dir>/<bag_dir_name>.mcap`, matching `metadata.yaml`'s
    /// `relative_file_paths`.
    pub fn new(path: &Path, compression_mode: CompressionMode) -> Result<Self> {
        let file_name = path
            .file_name()
            .ok_or_else(|| BagError::writer("bag path has no file name component"))?
            .to_string_lossy();
        let mcap_path = path.join(format!("{file_name}.mcap"));
        Ok(Self::with_file(mcap_path, "ros2", compression_mode))
    }

    /// Create a writer for a bare `.mcap` file. `profile` is the MCAP header
    /// profile (`"ros2"` or `"ros1"`).
    ///
    /// Chunks are zstd-compressed only for [`CompressionMode::Storage`]; the
    /// other modes are applied by the caller (per message / whole file).
    pub fn with_file(
        mcap_path: PathBuf,
        profile: &'static str,
        compression_mode: CompressionMode,
    ) -> Self {
        let compression = match compression_mode {
            CompressionMode::Storage => ChunkCompression::Zstd,
            _ => ChunkCompression::None,
        };
        Self {
            mcap_path,
            profile,
            compression,
            writer: None,
            pending_schemas: HashMap::new(),
            schema_ids: HashMap::new(),
            channels: HashMap::new(),
        }
    }
}

impl StorageWriter for McapWriter {
    fn open(&mut self) -> Result<()> {
        if self.writer.is_some() {
            return Err(BagError::BagAlreadyOpen);
        }
        self.writer = Some(ChunkedWriter::create(
            &self.mcap_path,
            self.profile,
            self.compression,
            CHUNK_SIZE,
        )?);
        Ok(())
    }

    fn close(&mut self, _version: u32, _metadata: &str) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.finish()?;
        }
        self.pending_schemas.clear();
        self.schema_ids.clear();
        self.channels.clear();
        Ok(())
    }

    fn add_msgtype(&mut self, connection: &Connection) -> Result<()> {
        if self.writer.is_none() {
            return Err(BagError::BagNotOpen);
        }
        let encoding = if connection.serialization_format == "ros1" {
            "ros1msg"
        } else {
            "ros2msg"
        };
        self.pending_schemas.insert(
            connection.message_type.clone(),
            PendingSchema {
                encoding,
                data: connection.message_definition.data.clone().into_bytes(),
            },
        );
        Ok(())
    }

    fn add_connection(
        &mut self,
        connection: &Connection,
        offered_qos_profiles: &str,
    ) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(BagError::BagNotOpen)?;

        // Register the schema the first time a channel needs it.
        let schema_id = match self.schema_ids.get(&connection.message_type) {
            Some(&id) => id,
            None => match self.pending_schemas.get(&connection.message_type) {
                Some(s) => {
                    let id = writer.add_schema(&connection.message_type, s.encoding, &s.data)?;
                    self.schema_ids.insert(connection.message_type.clone(), id);
                    id
                }
                None => 0,
            },
        };

        let mut metadata = BTreeMap::new();
        if !offered_qos_profiles.is_empty() {
            metadata.insert(
                "offered_qos_profiles".to_string(),
                offered_qos_profiles.to_string(),
            );
        }
        if !connection.type_description_hash.is_empty() {
            metadata.insert(
                "topic_type_hash".to_string(),
                connection.type_description_hash.clone(),
            );
        }
        let encoding = if connection.serialization_format.is_empty() {
            "cdr"
        } else {
            connection.serialization_format.as_str()
        };
        let id = writer.add_channel(schema_id, &connection.topic, encoding, metadata)?;
        self.channels
            .insert(connection.id, ChannelState { id, sequence: 0 });
        Ok(())
    }

    fn write(&mut self, connection: &Connection, timestamp: u64, data: &[u8]) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(BagError::BagNotOpen)?;
        let channel = self
            .channels
            .get_mut(&connection.id)
            .ok_or_else(|| BagError::connection_not_found(&connection.topic))?;
        let sequence = channel.sequence;
        channel.sequence = channel.sequence.wrapping_add(1);
        writer.write_message(channel.id, sequence, timestamp, timestamp, data)
    }

    fn is_open(&self) -> bool {
        self.writer.is_some()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for McapWriter {
    fn drop(&mut self) {
        // Best effort: an unclosed writer would otherwise leave a file
        // without a summary section.
        let _ = self.close(0, "");
    }
}

#[cfg(test)]
mod tests {
    use super::super::McapStorageReader;
    use super::*;
    use crate::rosbag::storage::StorageReader;
    use crate::rosbag::types::MessageDefinition;
    use tempfile::tempdir;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn conn(id: u32, topic: &str, ty: &str, fmt: &str) -> Connection {
        Connection {
            id,
            topic: topic.to_string(),
            message_type: ty.to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: fmt.to_string(),
            offered_qos_profiles: Vec::new(),
        }
    }

    fn open_writer(dir: &Path, mode: CompressionMode) -> Result<McapWriter> {
        let bag = dir.join("bag");
        std::fs::create_dir_all(&bag)?;
        let mut w = McapWriter::new(&bag, mode)?;
        w.open()?;
        Ok(w)
    }

    #[test]
    fn lifecycle_flags() -> TestResult {
        let dir = tempdir()?;
        let bag = dir.path().join("bag");
        std::fs::create_dir_all(&bag)?;
        let mut w = McapWriter::new(&bag, CompressionMode::None)?;
        assert!(!w.is_open());
        w.open()?;
        assert!(w.is_open());
        assert!(matches!(w.open(), Err(BagError::BagAlreadyOpen)));
        w.close(9, "")?;
        assert!(!w.is_open());
        w.close(9, "")?; // idempotent
        Ok(())
    }

    #[test]
    fn operations_require_open() -> TestResult {
        let dir = tempdir()?;
        let mut w = McapWriter::new(&dir.path().join("bag"), CompressionMode::None)?;
        let c = conn(1, "/t", "std_msgs/msg/String", "cdr");
        assert!(w.add_msgtype(&c).is_err());
        assert!(w.add_connection(&c, "").is_err());
        assert!(w.write(&c, 0, b"x").is_err());
        Ok(())
    }

    #[test]
    fn unknown_connection_is_rejected() -> TestResult {
        let dir = tempdir()?;
        let mut w = open_writer(dir.path(), CompressionMode::None)?;
        let c = conn(7, "/nope", "std_msgs/msg/String", "cdr");
        assert!(matches!(
            w.write(&c, 0, b"x"),
            Err(BagError::ConnectionNotFound { .. })
        ));
        Ok(())
    }

    fn round_trip(mode: CompressionMode) -> TestResult {
        let dir = tempdir()?;
        let mut w = open_writer(dir.path(), mode)?;
        let a = conn(1, "/a", "sensor_msgs/msg/Imu", "cdr");
        let b = conn(2, "/b", "sensor_msgs/Image", "ros1");
        for c in [&a, &b] {
            w.add_msgtype(c)?;
            w.add_connection(c, "")?;
        }
        // Interleaved, in log-time order.
        for i in 0..50u64 {
            w.write(&a, 1_000 + 2 * i, &i.to_le_bytes())?;
            w.write(&b, 1_001 + 2 * i, &[i as u8; 3])?;
        }
        w.close(9, "")?;

        let path = dir.path().join("bag").join("bag.mcap");
        let mut r = McapStorageReader::new(vec![path.as_path()], vec![])?;
        r.open()?;

        let conns = r.get_topics_from_mcap()?;
        assert_eq!(conns.len(), 2);
        let by_topic: HashMap<_, _> = conns.iter().map(|c| (c.topic.as_str(), c)).collect();
        assert_eq!(by_topic["/a"].message_type, "sensor_msgs/msg/Imu");
        assert_eq!(by_topic["/a"].serialization_format, "cdr");
        assert_eq!(by_topic["/a"].message_count, 50);
        assert_eq!(by_topic["/b"].serialization_format, "ros1");

        let msgs = r
            .raw_messages()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(msgs.len(), 100);
        assert!(msgs.windows(2).all(|p| p[0].timestamp <= p[1].timestamp));
        assert_eq!(msgs[0].connection.topic, "/a");
        assert_eq!(msgs[0].raw_data, 0u64.to_le_bytes());
        assert_eq!(msgs[99].raw_data, vec![49u8; 3]);
        Ok(())
    }

    #[test]
    fn round_trip_uncompressed() -> TestResult {
        round_trip(CompressionMode::None)
    }

    #[test]
    fn round_trip_zstd_chunks() -> TestResult {
        round_trip(CompressionMode::Storage)
    }

    #[test]
    fn output_is_readable_by_mcap_crate() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("plain.mcap");
        let mut w = McapWriter::with_file(path.clone(), "ros1", CompressionMode::Storage);
        w.open()?;
        let c = conn(1, "/t", "std_msgs/String", "ros1");
        w.add_msgtype(&c)?;
        w.add_connection(&c, "")?;
        for i in 0..10u64 {
            w.write(&c, i, &[i as u8])?;
        }
        w.close(0, "")?;

        let bytes = std::fs::read(&path)?;
        let msgs = mcap::MessageStream::new(&bytes)?.collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(msgs.len(), 10);
        assert_eq!(msgs[3].sequence, 3);
        assert_eq!(msgs[3].channel.message_encoding, "ros1");
        assert_eq!(msgs[3].channel.topic, "/t");
        let schema = msgs[3].channel.schema.as_ref().ok_or("schema")?;
        assert_eq!(
            (schema.name.as_str(), schema.encoding.as_str()),
            ("std_msgs/String", "ros1msg")
        );
        let summary = mcap::Summary::read(&bytes)?.ok_or("no summary")?;
        assert_eq!(summary.stats.ok_or("no stats")?.message_count, 10);
        Ok(())
    }

    /// Enough chunks to engage the parallel decompression pool: order must be
    /// exact, windows honoured, and abandoning the iterator must not hang.
    #[test]
    fn many_chunk_file_reads_in_order_through_parallel_decompression() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("big.mcap");
        let mut w = McapWriter::with_file(path.clone(), "ros2", CompressionMode::Storage);
        w.open()?;
        let c = conn(1, "/big", "t/msg/Big", "cdr");
        w.add_msgtype(&c)?;
        w.add_connection(&c, "")?;
        // ~48 MiB over 4 MiB chunks => >10 chunks.
        let n = 240u64;
        for i in 0..n {
            let mut payload = vec![(i % 251) as u8; 200 * 1024];
            payload[..8].copy_from_slice(&i.to_le_bytes());
            w.write(&c, 1_000 + i, &payload)?;
        }
        w.close(0, "")?;

        let mut r = McapStorageReader::new(vec![path.as_path()], vec![])?;
        r.open()?;
        let all = r
            .raw_messages()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(all.len() as u64, n);
        for (i, m) in all.iter().enumerate() {
            assert_eq!(m.timestamp, 1_000 + i as u64);
            assert_eq!(&m.raw_data[..8], &(i as u64).to_le_bytes());
        }

        let windowed = r
            .raw_messages_filtered(None, Some(1_100), Some(1_150))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(windowed.len(), 50);
        assert_eq!(windowed[0].timestamp, 1_100);

        // Drop the iterator after one message: workers must shut down.
        let mut it = r.raw_messages()?;
        assert!(it.next().is_some());
        drop(it);
        Ok(())
    }
}
