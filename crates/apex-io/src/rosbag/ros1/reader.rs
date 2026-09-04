//! ROS1 bag v2.0 reader.
//!
//! The reader parses the index-data records at the tail of the bag so that
//! messages can be iterated in timestamp order with optional topic/time-range
//! filtering, without re-parsing every chunk.  Chunks are decompressed lazily
//! on demand and only the few most recently used payloads are retained.

use crate::rosbag::error::{BagError, Result};
use crate::rosbag::types::{
    Connection, Message, MessageDefinition, MessageDefinitionFormat, RawMessage, TopicInfo,
};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::PathBuf;

use super::format::{MAGIC, OpCode, Ros1Compression};
use super::record::{RecordHeader, parse_header, read_record};

/// One indexed message: which connection, when, where to find it.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    conn_id: u32,
    time_ns: u64,
    chunk_pos: u64,
    /// Offset into the *uncompressed* chunk record stream.
    offset: u32,
}

/// Decompressed chunk payloads retained by [`Ros1Reader`].
///
/// Two is enough to absorb the alternation that interleaved topics can produce
/// while iterating in timestamp order, and bounds the reader's memory at a
/// couple of chunks rather than the whole inflated bag.
const CHUNK_CACHE_CAPACITY: usize = 2;

/// Reader for ROS1 bag v2.0 files.
pub struct Ros1Reader {
    path: PathBuf,
    file: Option<BufReader<File>>,
    is_open: bool,

    connections: Vec<Connection>,
    topics: HashMap<String, TopicInfo>,
    index: Vec<IndexEntry>,
    /// Most-recently-used decompressed chunk payloads, as `(chunk_pos, data)`.
    ///
    /// Bounded to [`CHUNK_CACHE_CAPACITY`]: index iteration is timestamp
    /// ordered, so consecutive reads land in the same chunk and a tiny cache
    /// captures all of the available locality. An unbounded map would hold the
    /// entire *decompressed* bag — gigabytes — for the reader's lifetime.
    chunk_cache: Vec<(u64, Vec<u8>)>,

    start_time_ns: u64,
    end_time_ns: u64,
}

impl Ros1Reader {
    /// Create a reader for `path`. Call [`Self::open`] to actually parse.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            return Err(BagError::BagNotFound { path });
        }
        Ok(Self {
            path,
            file: None,
            is_open: false,
            connections: Vec::new(),
            topics: HashMap::new(),
            index: Vec::new(),
            chunk_cache: Vec::new(),
            start_time_ns: u64::MAX,
            end_time_ns: 0,
        })
    }

    /// Open the bag and parse the connection + index tail.
    pub fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        // Magic line.
        let mut magic_buf = [0u8; 13];
        reader.read_exact(&mut magic_buf)?;
        if magic_buf != MAGIC {
            let found = String::from_utf8_lossy(&magic_buf).to_string();
            // Accept a few format variants by checking only the version part.
            if !magic_buf.starts_with(b"#ROSBAG ") {
                return Err(BagError::Ros1InvalidMagic { found });
            }
            if &magic_buf[8..] != b"V2.0\n" {
                return Err(BagError::Ros1UnsupportedVersion {
                    found: found.trim_end().to_string(),
                });
            }
        }

        // Bag header record.
        let bag_header =
            read_record(&mut reader)?.ok_or_else(|| BagError::Ros1MalformedRecord {
                reason: "missing bag header record".into(),
            })?;
        if bag_header.header.op()? != OpCode::BagHeader {
            return Err(BagError::Ros1MalformedRecord {
                reason: "expected bag header record".into(),
            });
        }
        let index_pos = bag_header.header.get_u64("index_pos")?;
        if index_pos == 0 {
            return Err(BagError::Ros1IndexCorrupt {
                reason: "bag header reports index_pos = 0 (writer may not have called close())"
                    .into(),
            });
        }

        // Parse the tail: connections + chunk-info records starting at index_pos.
        // We also need IndexData records, which appear *after each chunk* in the
        // body, not in the tail.  We parse them by scanning between chunk
        // records and the index_pos.
        let mut connections: HashMap<u32, Connection> = HashMap::new();
        let mut chunk_positions: Vec<u64> = Vec::new();

        reader.seek(SeekFrom::Start(index_pos))?;
        while let Some(rec) = read_record(&mut reader)? {
            match rec.header.op()? {
                OpCode::Connection => {
                    let conn = parse_connection_record(&rec.header, &rec.data)?;
                    connections.insert(conn.id, conn);
                }
                OpCode::ChunkInfo => {
                    let chunk_pos = rec.header.get_u64("chunk_pos")?;
                    chunk_positions.push(chunk_pos);
                }
                _ => {
                    // Other record types may appear in the tail of some
                    // writers; ignore them.
                }
            }
        }

        // For each chunk, jump past its chunk record and read the following
        // IndexData records (until we hit either another Chunk or index_pos).
        chunk_positions.sort();
        let mut index = Vec::new();
        for chunk_pos in &chunk_positions {
            reader.seek(SeekFrom::Start(*chunk_pos))?;
            let chunk_rec =
                read_record(&mut reader)?.ok_or_else(|| BagError::Ros1IndexCorrupt {
                    reason: format!("chunk at {chunk_pos} truncated"),
                })?;
            if chunk_rec.header.op()? != OpCode::Chunk {
                return Err(BagError::Ros1IndexCorrupt {
                    reason: format!("expected chunk record at {chunk_pos}"),
                });
            }

            // Read IndexData records until the next non-IndexData op or EOF.
            loop {
                let pos = reader.stream_position()?;
                if pos >= index_pos {
                    break;
                }
                let Some(rec) = read_record(&mut reader)? else {
                    break;
                };
                let op = rec.header.op()?;
                if op != OpCode::IndexData {
                    // Rewind: this record belongs to the next chunk or tail.
                    reader.seek(SeekFrom::Start(pos))?;
                    break;
                }
                let conn_id = rec.header.get_u32("conn")?;
                let count = rec.header.get_u32("count")? as usize;
                if rec.data.len() < count * 12 {
                    return Err(BagError::Ros1IndexCorrupt {
                        reason: format!(
                            "IndexData conn={conn_id} truncated: expected {} bytes, got {}",
                            count * 12,
                            rec.data.len()
                        ),
                    });
                }
                let mut cur = Cursor::new(&rec.data);
                for _ in 0..count {
                    let secs = cur.read_u32::<LittleEndian>()? as u64;
                    let nsecs = cur.read_u32::<LittleEndian>()? as u64;
                    let offset = cur.read_u32::<LittleEndian>()?;
                    let time_ns = secs.saturating_mul(1_000_000_000).saturating_add(nsecs);
                    index.push(IndexEntry {
                        conn_id,
                        time_ns,
                        chunk_pos: *chunk_pos,
                        offset,
                    });
                }
            }
        }

        // Sort the index by time (stable so equal-time messages keep file
        // order).  This matches the ROS2 reader iteration contract.
        index.sort_by_key(|e| e.time_ns);

        let connections_vec: Vec<Connection> = {
            let mut v: Vec<Connection> = connections.into_values().collect();
            v.sort_by_key(|c| c.id);
            v
        };

        // Build topic info + message counts.
        let mut counts: HashMap<u32, u64> = HashMap::new();
        let (mut start_ns, mut end_ns) = (u64::MAX, 0u64);
        for e in &index {
            *counts.entry(e.conn_id).or_default() += 1;
            start_ns = start_ns.min(e.time_ns);
            end_ns = end_ns.max(e.time_ns);
        }
        let connections_vec: Vec<Connection> = connections_vec
            .into_iter()
            .map(|mut c| {
                c.message_count = *counts.get(&c.id).unwrap_or(&0);
                c
            })
            .collect();
        let mut topics: HashMap<String, TopicInfo> = HashMap::new();
        for c in &connections_vec {
            let entry = topics.entry(c.topic.clone()).or_insert_with(|| TopicInfo {
                name: c.topic.clone(),
                message_type: c.message_type.clone(),
                message_definition: c.message_definition.clone(),
                message_count: 0,
                connections: Vec::new(),
            });
            entry.message_count += c.message_count;
            entry.connections.push(c.clone());
        }

        self.file = Some(reader);
        self.connections = connections_vec;
        self.topics = topics;
        self.index = index;
        self.start_time_ns = if start_ns == u64::MAX { 0 } else { start_ns };
        self.end_time_ns = end_ns;
        self.is_open = true;
        Ok(())
    }

    /// Connections discovered while opening.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Topic table, keyed by topic name.
    pub fn topics(&self) -> &HashMap<String, TopicInfo> {
        &self.topics
    }

    /// Duration of the bag in nanoseconds (`end - start`).
    pub fn duration(&self) -> u64 {
        self.end_time_ns.saturating_sub(self.start_time_ns)
    }

    /// Earliest message timestamp in nanoseconds since epoch.
    pub fn start_time(&self) -> u64 {
        self.start_time_ns
    }

    /// Latest message timestamp in nanoseconds since epoch.
    pub fn end_time(&self) -> u64 {
        self.end_time_ns
    }

    /// Total number of messages in the bag.
    pub fn message_count(&self) -> u64 {
        self.index.len() as u64
    }

    /// Iterate over all messages in timestamp order.
    pub fn messages(&mut self) -> Result<Vec<Message>> {
        self.messages_filtered(None, None, None)
    }

    /// Iterate over raw (undeserialized) messages in timestamp order.
    pub fn raw_messages(&mut self) -> Result<Vec<RawMessage>> {
        self.raw_messages_filtered(None, None, None)
    }

    /// Filtered iteration.  `connections`/`start_ns`/`stop_ns` are inclusive
    /// on `start_ns` and exclusive on `stop_ns`.
    pub fn messages_filtered(
        &mut self,
        connections: Option<&[Connection]>,
        start_ns: Option<u64>,
        stop_ns: Option<u64>,
    ) -> Result<Vec<Message>> {
        let raws = self.raw_messages_filtered(connections, start_ns, stop_ns)?;
        let conn_by_id: HashMap<u32, Connection> =
            self.connections.iter().map(|c| (c.id, c.clone())).collect();
        Ok(raws
            .into_iter()
            .map(|r| {
                let connection = conn_by_id
                    .get(&r.connection.id)
                    .cloned()
                    .unwrap_or(r.connection.clone());
                let topic = connection.topic.clone();
                Message {
                    connection,
                    topic,
                    timestamp: r.timestamp,
                    data: r.raw_data,
                }
            })
            .collect())
    }

    /// Filtered raw-message iteration (no deserialization).
    pub fn raw_messages_filtered(
        &mut self,
        connections: Option<&[Connection]>,
        start_ns: Option<u64>,
        stop_ns: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }
        let id_filter: Option<std::collections::HashSet<u32>> =
            connections.map(|cs| cs.iter().map(|c| c.id).collect());
        let conn_by_id: HashMap<u32, Connection> =
            self.connections.iter().map(|c| (c.id, c.clone())).collect();

        let mut out = Vec::new();
        // Take the entries we need into a local Vec so the borrow of self.index
        // ends before we reborrow self mutably for read_chunk.
        let selected: Vec<IndexEntry> = self
            .index
            .iter()
            .copied()
            .filter(|e| {
                id_filter.as_ref().is_none_or(|s| s.contains(&e.conn_id))
                    && start_ns.is_none_or(|s| e.time_ns >= s)
                    && stop_ns.is_none_or(|s| e.time_ns < s)
            })
            .collect();

        for e in selected {
            let payload = self.read_message_payload(e.chunk_pos, e.offset)?;
            let conn =
                conn_by_id
                    .get(&e.conn_id)
                    .cloned()
                    .ok_or_else(|| BagError::Ros1IndexCorrupt {
                        reason: format!("index references unknown conn id {}", e.conn_id),
                    })?;
            out.push(RawMessage {
                connection: conn,
                timestamp: e.timestamp_ns_or_self(e.time_ns),
                raw_data: payload,
            });
        }
        Ok(out)
    }

    /// Decompress the chunk at `chunk_pos` (cached) and return the payload of
    /// the message-data record starting at `offset`.
    fn read_message_payload(&mut self, chunk_pos: u64, offset: u32) -> Result<Vec<u8>> {
        // Move a hit to the back (most recent); on a miss, inflate and evict
        // the least recently used entry once at capacity.
        if let Some(idx) = self
            .chunk_cache
            .iter()
            .position(|(pos, _)| *pos == chunk_pos)
        {
            let entry = self.chunk_cache.remove(idx);
            self.chunk_cache.push(entry);
        } else {
            let inflated = self.inflate_chunk(chunk_pos)?;
            if self.chunk_cache.len() >= CHUNK_CACHE_CAPACITY {
                self.chunk_cache.remove(0);
            }
            self.chunk_cache.push((chunk_pos, inflated));
        }
        // The requested chunk is now the most recent entry.
        let buf = &self.chunk_cache[self.chunk_cache.len() - 1].1;
        if offset as usize >= buf.len() {
            return Err(BagError::Ros1IndexCorrupt {
                reason: format!("message offset {offset} out of bounds in chunk at {chunk_pos}"),
            });
        }
        let mut cur = Cursor::new(&buf[offset as usize..]);
        let rec = read_record(&mut cur)?.ok_or_else(|| BagError::Ros1IndexCorrupt {
            reason: "truncated message record".into(),
        })?;
        if rec.header.op()? != OpCode::MessageData {
            return Err(BagError::Ros1IndexCorrupt {
                reason: "expected message-data record at index offset".into(),
            });
        }
        Ok(rec.data)
    }

    fn inflate_chunk(&mut self, chunk_pos: u64) -> Result<Vec<u8>> {
        let file = self.file.as_mut().ok_or(BagError::BagNotOpen)?;
        file.seek(SeekFrom::Start(chunk_pos))?;
        let rec = read_record(file)?.ok_or_else(|| BagError::Ros1IndexCorrupt {
            reason: format!("chunk at {chunk_pos} missing"),
        })?;
        if rec.header.op()? != OpCode::Chunk {
            return Err(BagError::Ros1IndexCorrupt {
                reason: format!("expected chunk record at {chunk_pos}"),
            });
        }
        let compression_str = rec.header.get_str("compression")?;
        let compression = Ros1Compression::parse(compression_str).ok_or_else(|| {
            BagError::UnsupportedCompressionFormat {
                format: compression_str.to_string(),
            }
        })?;
        let uncompressed_size = rec.header.get_u32("size")? as usize;

        let inflated = match compression {
            Ros1Compression::None => rec.data,
            Ros1Compression::Bz2 => {
                let mut out = Vec::with_capacity(uncompressed_size);
                let mut dec = bzip2::read::BzDecoder::new(&rec.data[..]);
                dec.read_to_end(&mut out)?;
                out
            }
            Ros1Compression::Lz4 => lz4_flex::block::decompress_size_prepended(&rec.data)
                .map_err(|e| BagError::compression(format!("lz4 decode: {e}")))?,
        };
        if inflated.len() != uncompressed_size {
            return Err(BagError::Ros1IndexCorrupt {
                reason: format!(
                    "chunk uncompressed size mismatch: expected {uncompressed_size}, got {}",
                    inflated.len()
                ),
            });
        }
        Ok(inflated)
    }

    /// Release the file handle and cached chunk payloads.
    ///
    /// Idempotent, and mirrored by [`Drop`], matching `Ros1Writer` and the
    /// ros2 `Reader`/`Writer`. The OS would reclaim the descriptor on drop
    /// anyway; this also drops the decompressed chunks and leaves `is_open`
    /// reporting the truth.
    pub fn close(&mut self) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }
        self.file = None;
        self.chunk_cache.clear();
        self.chunk_cache.shrink_to_fit();
        self.is_open = false;
        Ok(())
    }
}

impl Drop for Ros1Reader {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

impl IndexEntry {
    fn timestamp_ns_or_self(&self, t: u64) -> u64 {
        t
    }
}

fn parse_connection_record(header: &RecordHeader, data: &[u8]) -> Result<Connection> {
    let id = header.get_u32("conn")?;
    let topic = header.get_str("topic")?.to_string();

    let sub = parse_header(data)?;
    let msg_type = sub.get_str("type")?.to_string();
    let md5sum = sub.get_str("md5sum")?.to_string();
    let message_definition = sub
        .get("message_definition")
        .map(|b| String::from_utf8_lossy(b).to_string())
        .unwrap_or_default();

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The chunk cache must stay bounded while iterating a multi-chunk bag.
    ///
    /// Regression for the unbounded `HashMap<u64, Vec<u8>>` that retained every
    /// decompressed chunk for the reader's lifetime — the whole inflated bag.
    #[test]
    fn chunk_cache_stays_bounded_across_many_chunks() -> Result<()> {
        use crate::rosbag::ros1::format::Ros1Compression;
        use crate::rosbag::ros1::writer::Ros1Writer;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("many_chunks.bag");

        // A tiny chunk threshold forces many chunks for a small payload.
        let mut w = Ros1Writer::new(&path)?;
        w.set_compression(Ros1Compression::Lz4)?;
        w.set_chunk_threshold(1024)?;
        w.open()?;
        let conn = w.add_connection("/t", "std_msgs/String", "hash", "", false)?;
        let payload = vec![7u8; 512];
        for i in 0..200u64 {
            w.write(&conn, 1_000 + i, &payload)?;
        }
        w.close()?;

        let mut r = Ros1Reader::new(&path)?;
        r.open()?;
        let msgs = r.raw_messages()?;
        assert_eq!(msgs.len(), 200, "all messages should be read back");

        // Guard against the test going vacuous: it only proves anything if the
        // read actually spanned more chunks than the cache can hold.
        let distinct_chunks = r
            .index
            .iter()
            .map(|e| e.chunk_pos)
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct_chunks > CHUNK_CACHE_CAPACITY,
            "bag has only {distinct_chunks} chunks; test cannot detect an unbounded cache"
        );

        assert!(
            r.chunk_cache.len() <= CHUNK_CACHE_CAPACITY,
            "chunk cache grew to {} entries, cap is {CHUNK_CACHE_CAPACITY}",
            r.chunk_cache.len()
        );

        r.close()?;
        assert!(
            r.chunk_cache.is_empty(),
            "close() should drop cached chunks"
        );
        assert!(!r.is_open);
        Ok(())
    }

    #[test]
    fn open_missing_file_errors() {
        let r = Ros1Reader::new("/nonexistent/path.bag");
        assert!(matches!(r, Err(BagError::BagNotFound { .. })));
    }

    #[test]
    fn accessors_before_open_return_defaults() -> Result<()> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bag");
        std::fs::write(&path, b"dummy").unwrap();
        let r = Ros1Reader::new(&path)?;
        assert!(r.connections().is_empty());
        assert!(r.topics().is_empty());
        assert_eq!(r.duration(), 0);
        assert_eq!(r.start_time(), u64::MAX);
        assert_eq!(r.end_time(), 0);
        assert_eq!(r.message_count(), 0);
        Ok(())
    }

    #[test]
    fn raw_messages_filtered_before_open_errors() -> Result<()> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bag");
        std::fs::write(&path, b"dummy").unwrap();
        let mut r = Ros1Reader::new(&path)?;
        let result = r.raw_messages_filtered(None, None, None);
        assert!(matches!(result, Err(BagError::BagNotOpen)));
        Ok(())
    }

    #[test]
    fn messages_filtered_before_open_errors() -> Result<()> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bag");
        std::fs::write(&path, b"dummy").unwrap();
        let mut r = Ros1Reader::new(&path)?;
        let result = r.messages_filtered(None, None, None);
        assert!(matches!(result, Err(BagError::BagNotOpen)));
        Ok(())
    }

    #[test]
    fn index_entry_timestamp_ns_or_self() {
        let entry = IndexEntry {
            conn_id: 0,
            time_ns: 12345,
            chunk_pos: 0,
            offset: 0,
        };
        assert_eq!(entry.timestamp_ns_or_self(12345), 12345);
    }
}
