//! A chunked MCAP encoder that compresses chunks on several cores.
//!
//! MCAP chunks are independent, so while the caller keeps appending messages
//! to the current chunk, finished chunks are compressed (and CRC-checked) on
//! the `rayon` pool and written back **in order**. Unlike the `mcap` crate's
//! writer, which compresses each chunk inline (and spins up a fresh zstd
//! thread pool per chunk), this scales with the core count.
//!
//! The produced file follows the MCAP 0.x specification:
//!
//! ```text
//! magic | Header | (Schema | Channel)* | (Chunk MessageIndex*)* | DataEnd
//!       | summary: Schema* Channel* ChunkIndex* Statistics | SummaryOffset* | Footer | magic
//! ```
//!
//! Records are `op: u8, len: u64, content`; strings are `u32 len + UTF-8`.

use crate::rosbag::error::{BagError, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc;

const MAGIC: &[u8; 8] = b"\x89MCAP0\r\n";

const OP_HEADER: u8 = 0x01;
const OP_FOOTER: u8 = 0x02;
const OP_SCHEMA: u8 = 0x03;
const OP_CHANNEL: u8 = 0x04;
const OP_MESSAGE: u8 = 0x05;
const OP_CHUNK: u8 = 0x06;
const OP_MESSAGE_INDEX: u8 = 0x07;
const OP_CHUNK_INDEX: u8 = 0x08;
const OP_STATISTICS: u8 = 0x0B;
const OP_SUMMARY_OFFSET: u8 = 0x0E;
const OP_DATA_END: u8 = 0x0F;

/// zstd level for chunk compression (the library default).
const ZSTD_LEVEL: i32 = 3;
/// Fixed bytes of a message record before its payload
/// (`channel u16, sequence u32, log_time u64, publish_time u64`).
const MESSAGE_FIXED: usize = 2 + 4 + 8 + 8;

/// Chunk compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkCompression {
    None,
    Zstd,
}

impl ChunkCompression {
    fn name(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Zstd => "zstd",
        }
    }
}

fn put_u16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len() as u32);
    b.extend_from_slice(s.as_bytes());
}

/// `op, u64 content length, content`.
fn record(op: u8, content: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(9 + content.len());
    out.push(op);
    put_u64(&mut out, content.len() as u64);
    out.extend_from_slice(content);
    out
}

#[derive(Clone)]
struct SchemaRec {
    id: u16,
    name: String,
    encoding: String,
    data: Vec<u8>,
}

#[derive(Clone)]
struct ChannelRec {
    id: u16,
    schema_id: u16,
    topic: String,
    message_encoding: String,
    metadata: BTreeMap<String, String>,
}

/// Messages accumulating for the next chunk.
#[derive(Default)]
struct ChunkBuilder {
    /// Concatenated message records (the chunk's uncompressed payload).
    records: Vec<u8>,
    /// Per channel: `(log_time, offset of the record in `records`)`.
    index: BTreeMap<u16, Vec<(u64, u64)>>,
    start: u64,
    end: u64,
    count: u64,
}

/// A compressed chunk ready to be written, plus what the summary needs.
struct DoneChunk {
    /// Chunk record followed by its message-index records, as written.
    chunk_record: Vec<u8>,
    /// `(channel, message index record bytes)`, in channel order.
    message_indexes: Vec<(u16, Vec<u8>)>,
    start: u64,
    end: u64,
    uncompressed_size: u64,
    compressed_size: u64,
}

struct ChunkIndexRec {
    start: u64,
    end: u64,
    chunk_start_offset: u64,
    chunk_length: u64,
    message_index_offsets: BTreeMap<u16, u64>,
    message_index_length: u64,
    compression: &'static str,
    compressed_size: u64,
    uncompressed_size: u64,
}

fn build_chunk(b: ChunkBuilder, compression: ChunkCompression) -> Result<DoneChunk> {
    let uncompressed_size = b.records.len() as u64;
    let crc = crc32fast::hash(&b.records);
    let compressed: Vec<u8> = match compression {
        ChunkCompression::None => b.records,
        ChunkCompression::Zstd => zstd::bulk::compress(&b.records, ZSTD_LEVEL)
            .map_err(|e| BagError::writer(format!("zstd: {e}")))?,
    };

    let mut content = Vec::with_capacity(compressed.len() + 64);
    put_u64(&mut content, b.start);
    put_u64(&mut content, b.end);
    put_u64(&mut content, uncompressed_size);
    put_u32(&mut content, crc);
    put_str(&mut content, compression.name());
    put_u64(&mut content, compressed.len() as u64);
    content.extend_from_slice(&compressed);
    let chunk_record = record(OP_CHUNK, &content);

    let message_indexes = b
        .index
        .into_iter()
        .map(|(channel, entries)| {
            let mut c = Vec::with_capacity(2 + 4 + entries.len() * 16);
            put_u16(&mut c, channel);
            put_u32(&mut c, (entries.len() * 16) as u32);
            for (t, off) in entries {
                put_u64(&mut c, t);
                put_u64(&mut c, off);
            }
            (channel, record(OP_MESSAGE_INDEX, &c))
        })
        .collect();

    Ok(DoneChunk {
        chunk_record,
        message_indexes,
        start: b.start,
        end: b.end,
        uncompressed_size,
        compressed_size: compressed.len() as u64,
    })
}

/// Streams an MCAP file, compressing chunks in parallel.
pub(super) struct ChunkedWriter {
    out: Option<BufWriter<File>>,
    pos: u64,
    compression: ChunkCompression,
    chunk_size: usize,
    schemas: Vec<SchemaRec>,
    channels: Vec<ChannelRec>,
    current: ChunkBuilder,
    chunk_indexes: Vec<ChunkIndexRec>,

    // Parallel pipeline: finished chunks come back tagged with their sequence
    // number and are written strictly in order.
    tx: mpsc::Sender<(u64, std::result::Result<DoneChunk, String>)>,
    rx: mpsc::Receiver<(u64, std::result::Result<DoneChunk, String>)>,
    next_seq: u64,
    next_to_write: u64,
    ready: BTreeMap<u64, DoneChunk>,
    inflight: usize,
    max_inflight: usize,

    // Statistics.
    message_count: u64,
    channel_counts: BTreeMap<u16, u64>,
    time_range: Option<(u64, u64)>,
}

impl ChunkedWriter {
    /// Create `path` and write the magic and Header record.
    pub(super) fn create(
        path: &Path,
        profile: &str,
        compression: ChunkCompression,
        chunk_size: usize,
    ) -> Result<Self> {
        let file = File::create(path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(MAGIC)?;
        let mut header = Vec::new();
        put_str(&mut header, profile);
        put_str(&mut header, "apex-io");
        let header = record(OP_HEADER, &header);
        out.write_all(&header)?;

        let workers = std::thread::available_parallelism().map_or(2, |n| n.get());
        let (tx, rx) = mpsc::channel();
        Ok(Self {
            out: Some(out),
            pos: (MAGIC.len() + header.len()) as u64,
            compression,
            chunk_size: chunk_size.max(1),
            schemas: Vec::new(),
            channels: Vec::new(),
            current: ChunkBuilder::default(),
            chunk_indexes: Vec::new(),
            tx,
            rx,
            next_seq: 0,
            next_to_write: 0,
            ready: BTreeMap::new(),
            inflight: 0,
            // Enough queued work to keep every core busy, bounded memory:
            // at most `max_inflight` chunks wait uncompressed.
            max_inflight: (workers * 2).max(2),
            message_count: 0,
            channel_counts: BTreeMap::new(),
            time_range: None,
        })
    }

    fn out(&mut self) -> Result<&mut BufWriter<File>> {
        self.out
            .as_mut()
            .ok_or_else(|| BagError::writer("MCAP writer already finished"))
    }

    fn write_raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.out()?.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// Register a schema; returns its id.
    pub(super) fn add_schema(&mut self, name: &str, encoding: &str, data: &[u8]) -> Result<u16> {
        self.barrier()?;
        let id = u16::try_from(self.schemas.len() + 1)
            .map_err(|_| BagError::writer("too many MCAP schemas"))?;
        let rec = SchemaRec {
            id,
            name: name.to_string(),
            encoding: encoding.to_string(),
            data: data.to_vec(),
        };
        self.write_raw(&schema_record(&rec))?;
        self.schemas.push(rec);
        Ok(id)
    }

    /// Register a channel; returns its id. `schema_id` 0 means "no schema".
    pub(super) fn add_channel(
        &mut self,
        schema_id: u16,
        topic: &str,
        message_encoding: &str,
        metadata: BTreeMap<String, String>,
    ) -> Result<u16> {
        self.barrier()?;
        let id = u16::try_from(self.channels.len())
            .map_err(|_| BagError::writer("too many MCAP channels"))?;
        let rec = ChannelRec {
            id,
            schema_id,
            topic: topic.to_string(),
            message_encoding: message_encoding.to_string(),
            metadata,
        };
        self.write_raw(&channel_record(&rec))?;
        self.channels.push(rec);
        Ok(id)
    }

    /// Append one message.
    pub(super) fn write_message(
        &mut self,
        channel: u16,
        sequence: u32,
        log_time: u64,
        publish_time: u64,
        data: &[u8],
    ) -> Result<()> {
        if usize::from(channel) >= self.channels.len() {
            return Err(BagError::writer(format!("unknown MCAP channel {channel}")));
        }
        let c = &mut self.current;
        let offset = c.records.len() as u64;
        c.records.push(OP_MESSAGE);
        put_u64(&mut c.records, (MESSAGE_FIXED + data.len()) as u64);
        put_u16(&mut c.records, channel);
        put_u32(&mut c.records, sequence);
        put_u64(&mut c.records, log_time);
        put_u64(&mut c.records, publish_time);
        c.records.extend_from_slice(data);
        c.index.entry(channel).or_default().push((log_time, offset));
        if c.count == 0 {
            c.start = log_time;
            c.end = log_time;
        } else {
            c.start = c.start.min(log_time);
            c.end = c.end.max(log_time);
        }
        c.count += 1;

        self.message_count += 1;
        *self.channel_counts.entry(channel).or_default() += 1;
        self.time_range = Some(match self.time_range {
            Some((a, b)) => (a.min(log_time), b.max(log_time)),
            None => (log_time, log_time),
        });

        if self.current.records.len() >= self.chunk_size {
            self.submit_current()?;
        }
        Ok(())
    }

    /// Hand the current chunk to the pool (or compress inline when there is
    /// nothing to compress).
    fn submit_current(&mut self) -> Result<()> {
        if self.current.count == 0 {
            return Ok(());
        }
        let builder = std::mem::take(&mut self.current);
        let seq = self.next_seq;
        self.next_seq += 1;

        if self.compression == ChunkCompression::None {
            // Nothing worth parallelising: build and write right away.
            let done = build_chunk(builder, self.compression)?;
            self.ready.insert(seq, done);
            return self.write_ready();
        }

        let (tx, compression) = (self.tx.clone(), self.compression);
        self.inflight += 1;
        rayon::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                build_chunk(builder, compression)
            }))
            .map_err(|_| "chunk compression panicked".to_string())
            .and_then(|r| r.map_err(|e| e.to_string()));
            let _ = tx.send((seq, result));
        });

        // Keep memory bounded: when too many chunks wait, take results.
        while self.inflight >= self.max_inflight {
            self.receive(true)?;
        }
        // Opportunistically write whatever already finished.
        self.receive(false)?;
        Ok(())
    }

    /// Collect finished chunks. With `block`, first wait for one to arrive
    /// (only when something is actually in flight, so this cannot hang).
    fn receive(&mut self, block: bool) -> Result<()> {
        if block && self.inflight > 0 {
            let item = self
                .rx
                .recv()
                .map_err(|_| BagError::writer("MCAP compression workers disconnected"))?;
            self.accept(item)?;
        }
        while let Ok(item) = self.rx.try_recv() {
            self.accept(item)?;
        }
        self.write_ready()
    }

    fn accept(&mut self, item: (u64, std::result::Result<DoneChunk, String>)) -> Result<()> {
        let (seq, result) = item;
        self.inflight -= 1;
        let done = result.map_err(|e| BagError::writer(format!("MCAP chunk: {e}")))?;
        self.ready.insert(seq, done);
        Ok(())
    }

    /// Write every chunk whose predecessors are all written.
    fn write_ready(&mut self) -> Result<()> {
        while let Some(done) = self.ready.remove(&self.next_to_write) {
            self.next_to_write += 1;
            let chunk_start_offset = self.pos;
            self.write_raw(&done.chunk_record)?;
            let chunk_length = done.chunk_record.len() as u64;

            let mut offsets = BTreeMap::new();
            let mut index_len = 0u64;
            for (channel, rec) in &done.message_indexes {
                offsets.insert(*channel, self.pos);
                self.write_raw(rec)?;
                index_len += rec.len() as u64;
            }
            self.chunk_indexes.push(ChunkIndexRec {
                start: done.start,
                end: done.end,
                chunk_start_offset,
                chunk_length,
                message_index_offsets: offsets,
                message_index_length: index_len,
                compression: self.compression.name(),
                compressed_size: done.compressed_size,
                uncompressed_size: done.uncompressed_size,
            });
        }
        Ok(())
    }

    /// Wait until every submitted chunk is on disk (before writing a
    /// top-level record, to keep the file's record order well-defined).
    fn barrier(&mut self) -> Result<()> {
        self.submit_current()?;
        while self.inflight > 0 {
            self.receive(true)?;
        }
        self.write_ready()
    }

    /// Finish the data section, write the summary, footer and closing magic.
    pub(super) fn finish(&mut self) -> Result<()> {
        if self.out.is_none() {
            return Ok(());
        }
        self.barrier()?;
        if self.next_to_write != self.next_seq {
            return Err(BagError::writer("MCAP chunks missing at finish"));
        }

        self.write_raw(&record(OP_DATA_END, &0u32.to_le_bytes()))?;

        // ---- summary ----------------------------------------------------
        let summary_start = self.pos;
        let mut groups: Vec<(u8, u64, u64)> = Vec::new();

        let start = self.pos;
        let bytes: Vec<u8> = self.schemas.iter().flat_map(schema_record).collect();
        if !bytes.is_empty() {
            self.write_raw(&bytes)?;
            groups.push((OP_SCHEMA, start, bytes.len() as u64));
        }

        let start = self.pos;
        let bytes: Vec<u8> = self.channels.iter().flat_map(channel_record).collect();
        if !bytes.is_empty() {
            self.write_raw(&bytes)?;
            groups.push((OP_CHANNEL, start, bytes.len() as u64));
        }

        let start = self.pos;
        let bytes: Vec<u8> = self
            .chunk_indexes
            .iter()
            .flat_map(chunk_index_record)
            .collect();
        if !bytes.is_empty() {
            self.write_raw(&bytes)?;
            groups.push((OP_CHUNK_INDEX, start, bytes.len() as u64));
        }

        let start = self.pos;
        let stats = self.statistics_record();
        self.write_raw(&stats)?;
        groups.push((OP_STATISTICS, start, stats.len() as u64));

        let summary_offset_start = self.pos;
        for (op, group_start, group_len) in groups {
            let mut c = vec![op];
            put_u64(&mut c, group_start);
            put_u64(&mut c, group_len);
            self.write_raw(&record(OP_SUMMARY_OFFSET, &c))?;
        }

        let mut footer = Vec::new();
        put_u64(&mut footer, summary_start);
        put_u64(&mut footer, summary_offset_start);
        put_u32(&mut footer, 0); // summary CRC: 0 = not computed
        self.write_raw(&record(OP_FOOTER, &footer))?;
        self.write_raw(MAGIC)?;

        let mut out = self
            .out
            .take()
            .ok_or_else(|| BagError::writer("MCAP writer already finished"))?;
        out.flush()?;
        Ok(())
    }

    fn statistics_record(&self) -> Vec<u8> {
        let mut c = Vec::new();
        put_u64(&mut c, self.message_count);
        put_u16(&mut c, self.schemas.len() as u16);
        put_u32(&mut c, self.channels.len() as u32);
        put_u32(&mut c, 0); // attachments
        put_u32(&mut c, 0); // metadata records
        put_u32(&mut c, self.chunk_indexes.len() as u32);
        let (a, b) = self.time_range.unwrap_or((0, 0));
        put_u64(&mut c, a);
        put_u64(&mut c, b);
        put_u32(&mut c, (self.channel_counts.len() * 10) as u32);
        for (ch, n) in &self.channel_counts {
            put_u16(&mut c, *ch);
            put_u64(&mut c, *n);
        }
        record(OP_STATISTICS, &c)
    }
}

impl Drop for ChunkedWriter {
    fn drop(&mut self) {
        if self.out.is_some()
            && let Err(e) = self.finish()
        {
            tracing::warn!("MCAP file not finalised cleanly: {e}");
        }
    }
}

fn schema_record(s: &SchemaRec) -> Vec<u8> {
    let mut c = Vec::with_capacity(16 + s.name.len() + s.encoding.len() + s.data.len());
    put_u16(&mut c, s.id);
    put_str(&mut c, &s.name);
    put_str(&mut c, &s.encoding);
    put_u32(&mut c, s.data.len() as u32);
    c.extend_from_slice(&s.data);
    record(OP_SCHEMA, &c)
}

fn channel_record(ch: &ChannelRec) -> Vec<u8> {
    let mut c = Vec::new();
    put_u16(&mut c, ch.id);
    put_u16(&mut c, ch.schema_id);
    put_str(&mut c, &ch.topic);
    put_str(&mut c, &ch.message_encoding);
    let mut meta = Vec::new();
    for (k, v) in &ch.metadata {
        put_str(&mut meta, k);
        put_str(&mut meta, v);
    }
    put_u32(&mut c, meta.len() as u32);
    c.extend_from_slice(&meta);
    record(OP_CHANNEL, &c)
}

fn chunk_index_record(ci: &ChunkIndexRec) -> Vec<u8> {
    let mut c = Vec::new();
    put_u64(&mut c, ci.start);
    put_u64(&mut c, ci.end);
    put_u64(&mut c, ci.chunk_start_offset);
    put_u64(&mut c, ci.chunk_length);
    put_u32(&mut c, (ci.message_index_offsets.len() * 10) as u32);
    for (ch, off) in &ci.message_index_offsets {
        put_u16(&mut c, *ch);
        put_u64(&mut c, *off);
    }
    put_u64(&mut c, ci.message_index_length);
    put_str(&mut c, ci.compression);
    put_u64(&mut c, ci.compressed_size);
    put_u64(&mut c, ci.uncompressed_size);
    record(OP_CHUNK_INDEX, &c)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;
    /// `(channel, sequence, log time, payload)` of one written message.
    type Written = (u16, u32, u64, Vec<u8>);

    fn payload(i: u64) -> Vec<u8> {
        // varied sizes and compressibility
        let n = 50 + (i as usize * 37) % 4000;
        (0..n).map(|k| ((k as u64 * 31 + i) % 251) as u8).collect()
    }

    /// 3 channels, `n` messages, timestamps deliberately not monotonic.
    fn write_file(
        path: &Path,
        compression: ChunkCompression,
        chunk_size: usize,
        n: u64,
    ) -> Result<Vec<Written>> {
        let mut w = ChunkedWriter::create(path, "ros2", compression, chunk_size)?;
        let schema = w.add_schema("pkg/msg/T", "ros2msg", b"int32 x")?;
        let ch = [
            w.add_channel(schema, "/a", "cdr", BTreeMap::new())?,
            w.add_channel(
                schema,
                "/b",
                "cdr",
                BTreeMap::from([("k".into(), "v".into())]),
            )?,
            w.add_channel(0, "/c", "ros1", BTreeMap::new())?,
        ];
        let mut written = Vec::new();
        let mut seq = [0u32; 3];
        for i in 0..n {
            let c = (i % 3) as usize;
            // jitter backwards now and then
            let t = 1_000 + i * 10 - if i % 7 == 0 { 5 } else { 0 };
            let data = payload(i);
            w.write_message(ch[c], seq[c], t, t + 1, &data)?;
            written.push((ch[c], seq[c], t, data));
            seq[c] += 1;
        }
        w.finish()?;
        Ok(written)
    }

    #[test]
    fn many_tiny_chunks_are_written_in_order_whatever_finishes_first() -> TestResult {
        for compression in [ChunkCompression::None, ChunkCompression::Zstd] {
            let dir = tempdir()?;
            let path = dir.path().join("t.mcap");
            // chunk_size 1 => every message is its own chunk, hundreds in flight.
            let written = write_file(&path, compression, 1, 600)?;

            let bytes = std::fs::read(&path)?;
            let got =
                mcap::MessageStream::new(&bytes)?.collect::<std::result::Result<Vec<_>, _>>()?;
            assert_eq!(got.len(), written.len(), "{compression:?}");
            for (g, (ch, seq, t, data)) in got.iter().zip(&written) {
                assert_eq!(g.channel.topic, ["/a", "/b", "/c"][usize::from(*ch)]);
                assert_eq!((g.sequence, g.log_time, g.publish_time), (*seq, *t, *t + 1));
                assert_eq!(g.data.as_ref(), data.as_slice());
            }
        }
        Ok(())
    }

    #[test]
    fn summary_indexes_and_statistics_are_self_consistent() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("s.mcap");
        let written = write_file(&path, ChunkCompression::Zstd, 20_000, 400)?;
        let bytes = std::fs::read(&path)?;
        let summary = mcap::Summary::read(&bytes)?.ok_or("no summary")?;

        let stats = summary.stats.ok_or("no statistics")?;
        assert_eq!(stats.message_count, 400);
        assert_eq!((stats.schema_count, stats.channel_count), (1, 3));
        assert_eq!(stats.chunk_count as usize, summary.chunk_indexes.len());
        assert_eq!(stats.channel_message_counts.values().sum::<u64>(), 400);
        let (lo, hi) = written
            .iter()
            .fold((u64::MAX, 0), |(a, b), m| (a.min(m.2), b.max(m.2)));
        assert_eq!((stats.message_start_time, stats.message_end_time), (lo, hi));

        assert!(
            summary.chunk_indexes.len() > 3,
            "test must span several chunks"
        );
        let mut total_uncompressed = 0u64;
        for ci in &summary.chunk_indexes {
            // chunk_start_offset points at a Chunk record of exactly chunk_length
            let at = ci.chunk_start_offset as usize;
            assert_eq!(bytes[at], OP_CHUNK);
            let len = u64::from_le_bytes(bytes[at + 1..at + 9].try_into()?);
            assert_eq!(len + 9, ci.chunk_length);
            assert_eq!(ci.compression, "zstd");
            // message index records sit right after the chunk, one per channel
            let mut expect = at as u64 + ci.chunk_length;
            for off in ci.message_index_offsets.values() {
                assert_eq!(*off, expect);
                assert_eq!(bytes[*off as usize], OP_MESSAGE_INDEX);
                let l = u64::from_le_bytes(bytes[*off as usize + 1..*off as usize + 9].try_into()?);
                expect += l + 9;
            }
            assert_eq!(
                expect - (at as u64 + ci.chunk_length),
                ci.message_index_length
            );
            assert!(ci.compressed_size < ci.uncompressed_size);
            total_uncompressed += ci.uncompressed_size;
        }
        assert!(total_uncompressed > 0);
        Ok(())
    }

    #[test]
    fn channel_metadata_schema_links_and_empty_files_survive() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("m.mcap");
        write_file(&path, ChunkCompression::None, 8_192, 30)?;
        let bytes = std::fs::read(&path)?;
        let summary = mcap::Summary::read(&bytes)?.ok_or("no summary")?;
        let by_topic: BTreeMap<_, _> = summary
            .channels
            .values()
            .map(|c| (c.topic.as_str(), c.clone()))
            .collect();
        assert_eq!(
            by_topic["/b"].metadata.get("k").map(String::as_str),
            Some("v")
        );
        assert_eq!(
            by_topic["/a"].schema.as_ref().map(|s| s.name.as_str()),
            Some("pkg/msg/T")
        );
        assert!(by_topic["/c"].schema.is_none());
        assert_eq!(by_topic["/c"].message_encoding, "ros1");

        // A file with no messages is still a valid MCAP.
        let empty = dir.path().join("e.mcap");
        let mut w = ChunkedWriter::create(&empty, "ros2", ChunkCompression::Zstd, 1 << 20)?;
        w.finish()?;
        let bytes = std::fs::read(&empty)?;
        let summary = mcap::Summary::read(&bytes)?.ok_or("no summary")?;
        assert_eq!(summary.stats.ok_or("stats")?.message_count, 0);
        assert_eq!(mcap::MessageStream::new(&bytes)?.count(), 0);
        Ok(())
    }

    #[test]
    fn unknown_channels_are_rejected_and_finish_is_idempotent() -> TestResult {
        let dir = tempdir()?;
        let mut w = ChunkedWriter::create(
            &dir.path().join("u.mcap"),
            "ros2",
            ChunkCompression::None,
            1024,
        )?;
        assert!(w.write_message(5, 0, 1, 1, b"x").is_err());
        w.finish()?;
        w.finish()?;
        Ok(())
    }
}
