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
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;

use super::format::{MAGIC, OpCode, Ros1Compression};
use super::record::{
    RecordHeader, parse_header, read_record, read_record_prefix, read_record_skip_data,
    record_data_slice, scan_header_fields,
};

/// One indexed message: which connection, when, where to find it.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    conn_id: u32,
    time_ns: u64,
    chunk_pos: u64,
    /// Offset into the *uncompressed* chunk record stream.
    offset: u32,
}

/// What the bag's tail says about one chunk (a `ChunkInfo` record).
#[derive(Debug, Clone)]
struct ChunkMeta {
    pos: u64,
    start_ns: u64,
    end_ns: u64,
    /// `(connection id, message count)` for the connections present.
    counts: Vec<(u32, u32)>,
}

/// Buffer for the open-time tail walk and per-chunk index reads. Each chunk
/// visit seeks (which discards the buffer), so this stays small.
const READ_BUFFER: usize = 16 * 1024;

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
    /// Chunk metadata from the tail; the per-message index is built lazily
    /// per query (see [`Self::build_index`]).
    chunks: Vec<ChunkMeta>,
    /// Total message count, summed from the chunk infos.
    total_messages: u64,
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
            chunks: Vec::new(),
            total_messages: 0,
            chunk_cache: Vec::new(),
            start_time_ns: u64::MAX,
            end_time_ns: 0,
        })
    }

    /// Open the bag: read the bag header and the tail (connection and
    /// chunk-info records). No chunk and no per-message index is touched, so
    /// this costs the same for a 20 GB bag as for a 20 MB one; the
    /// per-message index is built on demand by the read calls.
    pub fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }
        let file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(READ_BUFFER, file);

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
        let header_conn_count = bag_header.header.get_u32("conn_count")?;
        let header_chunk_count = bag_header.header.get_u32("chunk_count")?;

        // The tail: connection records and one ChunkInfo per chunk.
        let mut connections: HashMap<u32, Connection> = HashMap::new();
        let mut chunks: Vec<ChunkMeta> = Vec::new();
        reader.seek(SeekFrom::Start(index_pos))?;
        while let Some(rec) = read_record(&mut reader)? {
            match rec.header.op()? {
                OpCode::Connection => {
                    let conn = parse_connection_record(&rec.header, &rec.data)?;
                    connections.insert(conn.id, conn);
                }
                OpCode::ChunkInfo => chunks.push(parse_chunk_info(&rec.header, &rec.data)?),
                // Other record types may appear in the tail of some writers.
                _ => {}
            }
        }
        chunks.sort_by_key(|c| c.pos);

        if header_chunk_count != 0 && chunks.len() as u32 != header_chunk_count {
            return Err(BagError::Ros1IndexCorrupt {
                reason: format!(
                    "bag header claims {header_chunk_count} chunks but the tail holds {}",
                    chunks.len()
                ),
            });
        }
        if header_conn_count != 0 && connections.len() as u32 != header_conn_count {
            return Err(BagError::Ros1IndexCorrupt {
                reason: format!(
                    "bag header claims {header_conn_count} connections but the tail holds {}",
                    connections.len()
                ),
            });
        }

        // Per-connection counts and the time range come straight from the
        // chunk infos -- no message index needed.
        let mut counts: HashMap<u32, u64> = HashMap::new();
        let (mut start_ns, mut end_ns) = (u64::MAX, 0u64);
        for chunk in &chunks {
            for &(conn, n) in &chunk.counts {
                *counts.entry(conn).or_default() += u64::from(n);
            }
            if !chunk.counts.is_empty() {
                start_ns = start_ns.min(chunk.start_ns);
                end_ns = end_ns.max(chunk.end_ns);
            }
        }
        let mut connections_vec: Vec<Connection> = connections.into_values().collect();
        connections_vec.sort_by_key(|c| c.id);
        for c in &mut connections_vec {
            c.message_count = counts.get(&c.id).copied().unwrap_or(0);
        }

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

        self.total_messages = counts.values().sum();
        self.file = Some(reader);
        self.connections = connections_vec;
        self.topics = topics;
        self.chunks = chunks;
        self.start_time_ns = if start_ns == u64::MAX { 0 } else { start_ns };
        self.end_time_ns = end_ns;
        self.is_open = true;
        Ok(())
    }

    /// Build the time-sorted per-message index for one query.
    ///
    /// Only chunks that overlap `[start_ns, stop_ns)` and hold at least one
    /// of the wanted connections are visited, and only those connections'
    /// `IndexData` records are decoded. Entry counts are cross-checked
    /// against the chunk infos, so a damaged index fails loudly.
    fn build_index(
        &mut self,
        wanted: Option<&HashSet<u32>>,
        start_ns: Option<u64>,
        stop_ns: Option<u64>,
    ) -> Result<Vec<IndexEntry>> {
        let file = self.file.as_mut().ok_or(BagError::BagNotOpen)?;
        let mut entries = Vec::new();
        let mut header = Vec::new();
        let mut data = Vec::new();

        for chunk in &self.chunks {
            let overlaps = start_ns.is_none_or(|s| chunk.end_ns >= s)
                && stop_ns.is_none_or(|e| chunk.start_ns < e);
            let relevant = chunk
                .counts
                .iter()
                .any(|(c, n)| *n > 0 && wanted.is_none_or(|w| w.contains(c)));
            if !overlaps || !relevant {
                continue;
            }

            file.seek(SeekFrom::Start(chunk.pos))?;
            let (chunk_header, _) =
                read_record_skip_data(file)?.ok_or_else(|| BagError::Ros1IndexCorrupt {
                    reason: format!("chunk at {} truncated", chunk.pos),
                })?;
            if chunk_header.op()? != OpCode::Chunk {
                return Err(BagError::Ros1IndexCorrupt {
                    reason: format!("expected chunk record at {}", chunk.pos),
                });
            }

            // One IndexData record per connection follows the chunk.
            for _ in 0..chunk.counts.len() {
                let data_len = read_record_prefix(file, &mut header)?.ok_or_else(|| {
                    BagError::Ros1IndexCorrupt {
                        reason: format!("index of chunk at {} truncated", chunk.pos),
                    }
                })? as usize;
                let (mut op, mut conn, mut ver, mut count) = (None, None, None, None);
                scan_header_fields(&header, |k, v| match (k, v.len()) {
                    (b"op", 1) => op = Some(v[0]),
                    (b"conn", 4) => conn = Some(u32::from_le_bytes([v[0], v[1], v[2], v[3]])),
                    (b"ver", 4) => ver = Some(u32::from_le_bytes([v[0], v[1], v[2], v[3]])),
                    (b"count", 4) => count = Some(u32::from_le_bytes([v[0], v[1], v[2], v[3]])),
                    _ => {}
                })?;
                if op != Some(OpCode::IndexData.as_u8()) {
                    return Err(BagError::Ros1IndexCorrupt {
                        reason: format!("expected IndexData after chunk at {}", chunk.pos),
                    });
                }
                let (Some(conn), Some(ver), Some(count)) = (conn, ver, count) else {
                    return Err(BagError::Ros1IndexCorrupt {
                        reason: "IndexData record missing conn/ver/count".into(),
                    });
                };
                // The bag 2.0 spec fixes IndexData `ver` at 1.
                if ver != 1 {
                    return Err(BagError::Ros1IndexCorrupt {
                        reason: format!("unsupported IndexData version {ver} (want 1)"),
                    });
                }
                let count = count as usize;
                if count.checked_mul(12).is_none_or(|n| n > data_len) {
                    return Err(BagError::Ros1IndexCorrupt {
                        reason: format!(
                            "IndexData conn={conn} truncated: {count} entries in {data_len} bytes"
                        ),
                    });
                }
                if !wanted.is_none_or(|w| w.contains(&conn)) {
                    file.seek_relative(data_len as i64)?;
                    continue;
                }
                // Cross-check against the chunk info.
                if chunk
                    .counts
                    .iter()
                    .find(|(c, _)| *c == conn)
                    .is_none_or(|(_, n)| *n as usize != count)
                {
                    return Err(BagError::Ros1IndexCorrupt {
                        reason: format!(
                            "chunk at {} lists a different message count for conn {conn}",
                            chunk.pos
                        ),
                    });
                }
                data.resize(data_len, 0);
                file.read_exact(&mut data)?;
                for e in data[..count * 12].as_chunks::<12>().0 {
                    let secs = u64::from(u32::from_le_bytes([e[0], e[1], e[2], e[3]]));
                    let nsecs = u64::from(u32::from_le_bytes([e[4], e[5], e[6], e[7]]));
                    let time_ns = secs.saturating_mul(1_000_000_000).saturating_add(nsecs);
                    if start_ns.is_some_and(|s| time_ns < s)
                        || stop_ns.is_some_and(|e| time_ns >= e)
                    {
                        continue;
                    }
                    entries.push(IndexEntry {
                        conn_id: conn,
                        time_ns,
                        chunk_pos: chunk.pos,
                        offset: u32::from_le_bytes([e[8], e[9], e[10], e[11]]),
                    });
                }
            }
        }

        // Stable sort: equal stamps keep file order. Chunks arrive in file
        // order and are mostly time-ordered, which the sort exploits.
        entries.sort_by_key(|e| e.time_ns);
        Ok(entries)
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
        self.total_messages
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
        Ok(raws
            .into_iter()
            .map(|r| Message {
                topic: r.connection.topic.clone(),
                connection: r.connection,
                timestamp: r.timestamp,
                data: r.raw_data,
            })
            .collect())
    }

    /// Filtered raw-message collection (no deserialization).
    ///
    /// Materialises every selected message; prefer
    /// [`Self::for_each_raw_filtered`] for large bags.
    pub fn raw_messages_filtered(
        &mut self,
        connections: Option<&[Connection]>,
        start_ns: Option<u64>,
        stop_ns: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        let shared: HashMap<u32, std::sync::Arc<Connection>> = self
            .connections
            .iter()
            .map(|c| (c.id, std::sync::Arc::new(c.clone())))
            .collect();
        let mut out = Vec::new();
        self.for_each_raw_filtered(connections, start_ns, stop_ns, |conn, ts, data| {
            let connection = shared
                .get(&conn.id)
                .cloned()
                .unwrap_or_else(|| conn.clone().into());
            out.push(RawMessage {
                connection,
                timestamp: ts,
                raw_data: data.to_vec(),
            });
            Ok(())
        })?;
        Ok(out)
    }

    /// Stream selected messages in timestamp order to `f` as borrowed slices.
    ///
    /// No per-message allocation or `Connection` clone: `data` points into the
    /// cached decompressed chunk. `start_ns` is inclusive, `stop_ns` exclusive.
    pub fn for_each_raw_filtered<F>(
        &mut self,
        connections: Option<&[Connection]>,
        start_ns: Option<u64>,
        stop_ns: Option<u64>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&Connection, u64, &[u8]) -> Result<()>,
    {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }
        let id_filter: Option<HashSet<u32>> =
            connections.map(|cs| cs.iter().map(|c| c.id).collect());
        let conn_index: HashMap<u32, usize> = self
            .connections
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();

        // Index only what this query needs (topics, time window); entries
        // come back time-sorted and already window-filtered.
        let entries = self.build_index(id_filter.as_ref(), start_ns, stop_ns)?;

        // Move the cache out so the closure can borrow `self.connections`.
        let mut cache = std::mem::take(&mut self.chunk_cache);
        let result = (|| -> Result<()> {
            for e in &entries {
                let ci = *conn_index
                    .get(&e.conn_id)
                    .ok_or_else(|| BagError::Ros1IndexCorrupt {
                        reason: format!("index references unknown conn id {}", e.conn_id),
                    })?;
                let chunk = Self::cached_chunk(&mut cache, &mut self.file, e.chunk_pos)?;
                let tail =
                    chunk
                        .get(e.offset as usize..)
                        .ok_or_else(|| BagError::Ros1IndexCorrupt {
                            reason: format!(
                                "message offset {} out of bounds in chunk at {}",
                                e.offset, e.chunk_pos
                            ),
                        })?;
                f(&self.connections[ci], e.time_ns, record_data_slice(tail)?)?;
            }
            Ok(())
        })();
        self.chunk_cache = cache;
        result
    }

    /// The inflated payload of the chunk at `chunk_pos`, via a small
    /// most-recently-used cache (an LRU of [`CHUNK_CACHE_CAPACITY`] chunks).
    fn cached_chunk<'c>(
        cache: &'c mut Vec<(u64, Vec<u8>)>,
        file: &mut Option<BufReader<File>>,
        chunk_pos: u64,
    ) -> Result<&'c [u8]> {
        let idx = match cache.iter().position(|(p, _)| *p == chunk_pos) {
            Some(i) => {
                let entry = cache.remove(i);
                cache.push(entry);
                cache.len() - 1
            }
            None => {
                let inflated = Self::inflate_chunk_at(file, chunk_pos)?;
                if cache.len() >= CHUNK_CACHE_CAPACITY {
                    cache.remove(0);
                }
                cache.push((chunk_pos, inflated));
                cache.len() - 1
            }
        };
        Ok(&cache[idx].1)
    }

    /// Pull-style iteration: the same messages, in the same order, as
    /// [`Self::for_each_raw_filtered`], one `next_into` call at a time.
    pub fn raw_iter(
        &mut self,
        connections: Option<&[Connection]>,
        start_ns: Option<u64>,
        stop_ns: Option<u64>,
    ) -> Result<Ros1MessageIter<'_>> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }
        let id_filter: Option<HashSet<u32>> =
            connections.map(|cs| cs.iter().map(|c| c.id).collect());
        let entries = self.build_index(id_filter.as_ref(), start_ns, stop_ns)?;
        Ok(Ros1MessageIter {
            reader: self,
            entries,
            pos: 0,
        })
    }

    fn inflate_chunk_at(file: &mut Option<BufReader<File>>, chunk_pos: u64) -> Result<Vec<u8>> {
        let file = file.as_mut().ok_or(BagError::BagNotOpen)?;
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
            Ros1Compression::Lz4 => {
                const LZ4_FRAME_MAGIC: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];
                if rec.data.starts_with(&LZ4_FRAME_MAGIC) {
                    // What `rosbag`/roslz4 write: an LZ4 frame stream.
                    let mut out = Vec::with_capacity(uncompressed_size);
                    lz4_flex::frame::FrameDecoder::new(&rec.data[..])
                        .read_to_end(&mut out)
                        .map_err(|e| BagError::compression(format!("lz4 frame decode: {e}")))?;
                    out
                } else {
                    // Bags written by earlier versions of this crate used a
                    // size-prefixed raw block.
                    lz4_flex::block::decompress_size_prepended(&rec.data)
                        .map_err(|e| BagError::compression(format!("lz4 decode: {e}")))?
                }
            }
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

/// A pull iterator over one query of a [`Ros1Reader`]; see
/// [`Ros1Reader::raw_iter`].
pub struct Ros1MessageIter<'a> {
    reader: &'a mut Ros1Reader,
    entries: Vec<IndexEntry>,
    pos: usize,
}

impl Ros1MessageIter<'_> {
    /// Messages still to come.
    pub fn remaining(&self) -> usize {
        self.entries.len() - self.pos
    }

    /// Copy the next message's payload into `buf` (cleared first) and return
    /// `(connection id, timestamp in ns)`; `None` at the end.
    pub fn next_into(&mut self, buf: &mut Vec<u8>) -> Result<Option<(u32, u64)>> {
        let Some(e) = self.entries.get(self.pos).copied() else {
            return Ok(None);
        };
        self.pos += 1;
        let chunk = Ros1Reader::cached_chunk(
            &mut self.reader.chunk_cache,
            &mut self.reader.file,
            e.chunk_pos,
        )?;
        let tail = chunk
            .get(e.offset as usize..)
            .ok_or_else(|| BagError::Ros1IndexCorrupt {
                reason: format!(
                    "message offset {} out of bounds in chunk at {}",
                    e.offset, e.chunk_pos
                ),
            })?;
        buf.clear();
        buf.extend_from_slice(record_data_slice(tail)?);
        Ok(Some((e.conn_id, e.time_ns)))
    }
}

impl Drop for Ros1Reader {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn parse_chunk_info(header: &RecordHeader, data: &[u8]) -> Result<ChunkMeta> {
    let count = header.get_u32("count")? as usize;
    if count.checked_mul(8).is_none_or(|n| data.len() < n) {
        return Err(BagError::Ros1IndexCorrupt {
            reason: format!(
                "ChunkInfo truncated: {count} entries in {} bytes",
                data.len()
            ),
        });
    }
    let counts = data[..count * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|e| {
            (
                u32::from_le_bytes([e[0], e[1], e[2], e[3]]),
                u32::from_le_bytes([e[4], e[5], e[6], e[7]]),
            )
        })
        .collect();
    Ok(ChunkMeta {
        pos: header.get_u64("chunk_pos")?,
        start_ns: header.get_time_nanos("start_time")?,
        end_ns: header.get_time_nanos("end_time")?,
        counts,
    })
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
        let distinct_chunks = r.chunks.len();
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
}
