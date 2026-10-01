//! Read-ahead for sequential VRS scans.
//!
//! [`VrsReader::sequential`] wraps a time-ordered scan in a background
//! thread that reads and decompresses records ahead of the consumer, with a
//! bounded number of record payloads in flight. This overlaps I/O and
//! decompression with whatever the caller does with each record — the shape
//! of a VIO loop pulling camera + IMU data one record at a time. Results are
//! delivered in exactly the order (and with exactly the payloads) the
//! synchronous [`VrsReader::next_record_into`] cursor produces.

use super::codec::Decompressor;
use super::error::{Result, VrsError};
use super::format::{RECORD_HEADER_SIZE, RecordHeader, RecordType, StreamId};
use super::reader::{Cursor, RecordFilter, RecordMeta, VrsReader};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

/// How far ahead to read: at most this many records and this many stored
/// bytes in flight (a file of tiny IMU records batches hundreds of records,
/// a file of large camera frames only a handful — both without unbounded
/// memory). A large record budget also keeps the per-batch channel round
/// trip negligible.
const DEFAULT_MAX_RECORDS: usize = 256;
const DEFAULT_MAX_BYTES: usize = 32 << 20;
/// Hard cap on one batch, so a single dispatch does not monopolize the scan.
const MAX_BATCH_RECORDS: usize = 1024;

/// One record handed to the worker: everything needed to read it with a
/// private file handle, so the consumer's reader is never touched.
struct Job {
    index: usize,
    offset: u64,
    size: u32,
    stream: StreamId,
    record_type: RecordType,
    timestamp: f64,
}

/// One decompressed record on its way back, in job order. A read or
/// decompression failure travels with the record and is surfaced by
/// [`SequentialReader::next_into`] when the record's turn comes.
struct Delivered {
    meta: RecordMeta,
    /// Stored size, for the read-ahead byte budget.
    size: u32,
    payload: Result<Vec<u8>>,
}

struct Worker {
    handle: Option<JoinHandle<()>>,
    job_tx: Option<SyncSender<Vec<Job>>>,
    res_rx: Option<Receiver<Vec<Delivered>>>,
}

impl Worker {
    fn send(&mut self, batch: Vec<Job>) -> Result<()> {
        let Some(tx) = self.job_tx.as_ref() else {
            return Err(VrsError::Usage("read-ahead worker closed".into()));
        };
        tx.send(batch)
            .map_err(|_| VrsError::Usage("read-ahead worker exited".into()))
    }

    fn spawn(file: File) -> Result<Self> {
        // Jobs flow through a bounded channel (backpressure). Results are
        // unbounded so the worker never blocks on its way back to reading
        // more jobs — the amount of buffered payload stays bounded by the
        // depth gate in `dispatch`, which decides what is sent at all.
        let (job_tx, job_rx) = sync_channel::<Vec<Job>>(1);
        let (res_tx, res_rx) = std::sync::mpsc::channel::<Vec<Delivered>>();
        let handle = std::thread::Builder::new()
            .name("vrs-read-ahead".into())
            .spawn(move || {
                let mut codec = Decompressor::new();
                let mut raw = Vec::new();
                while let Ok(batch) = job_rx.recv() {
                    let mut out = Vec::with_capacity(batch.len());
                    for job in batch {
                        out.push(read_job(&file, &mut codec, job, &mut raw));
                    }
                    if res_tx.send(out).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| VrsError::Usage(format!("cannot spawn read-ahead thread: {e}")))?;
        Ok(Self {
            handle: Some(handle),
            job_tx: Some(job_tx),
            res_rx: Some(res_rx),
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Close the job channel so the worker's recv loop ends, then wait
        // for it. The worker never blocks on sending results (unbounded
        // channel), so the join cannot deadlock.
        self.job_tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Read and decompress one record with the worker's own file handle,
/// validating it against the job's identity.
fn read_job(file: &File, codec: &mut Decompressor, job: Job, raw: &mut Vec<u8>) -> Delivered {
    let read = (|| -> Result<(u32, Vec<u8>)> {
        let mut file = file;
        file.seek(SeekFrom::Start(job.offset))?;
        raw.resize(job.size as usize, 0);
        file.read_exact(raw)?;
        let header = RecordHeader::parse(raw)?;
        if header.record_size != job.size || header.stream_id() != job.stream {
            return Err(VrsError::corrupt(format!(
                "record {} does not match the index",
                job.index
            )));
        }
        let mut payload = Vec::new();
        codec.decompress(
            header.compression,
            &raw[RECORD_HEADER_SIZE..],
            header.uncompressed_size as usize,
            &mut payload,
        )?;
        Ok((header.format_version, payload))
    })();
    let (format_version, payload) = match read {
        Ok((v, p)) => (v, Ok(p)),
        Err(e) => (0, Err(e)),
    };
    Delivered {
        meta: RecordMeta {
            index: job.index,
            stream: job.stream,
            record_type: job.record_type,
            timestamp: job.timestamp,
            format_version,
        },
        size: job.size,
        payload,
    }
}

/// Time-ordered scan of a [`VrsReader`] with a background thread reading
/// ahead. Yields exactly the records (and payload bytes) of the equivalent
/// synchronous cursor loop.
pub struct SequentialReader<'a> {
    reader: &'a mut VrsReader,
    cursor: Cursor,
    worker: Option<Worker>,
    pending: VecDeque<Delivered>,
    pending_bytes: usize,
    /// Jobs sent to the worker but not yet received back.
    inflight: usize,
    inflight_bytes: usize,
    max_records: usize,
    max_bytes: usize,
    done: bool,
}

impl<'a> SequentialReader<'a> {
    pub(super) fn new(reader: &'a mut VrsReader, filter: &RecordFilter) -> Result<Self> {
        let cursor = reader.cursor(filter);
        Ok(Self {
            reader,
            cursor,
            worker: None,
            pending: VecDeque::new(),
            pending_bytes: 0,
            inflight: 0,
            inflight_bytes: 0,
            max_records: DEFAULT_MAX_RECORDS,
            max_bytes: DEFAULT_MAX_BYTES,
            done: false,
        })
    }

    /// Cap on records in flight (and thus on buffered payloads).
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.max_records = depth.max(1);
        self
    }

    /// Queue the next batch of matching records to the worker while there is
    /// capacity. Idle until the scan is exhausted or capacity is full.
    fn dispatch(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        let mut batch = Vec::new();
        let mut batch_bytes = 0usize;
        while batch.len() < MAX_BATCH_RECORDS
            && self.inflight + self.pending.len() + batch.len() < self.max_records
            && self.inflight_bytes + self.pending_bytes + batch_bytes < self.max_bytes
        {
            let Some(index) = self.reader.next_index(&mut self.cursor) else {
                self.done = true;
                break;
            };
            let info = self.reader.records()[index];
            batch_bytes += info.size as usize;
            batch.push(Job {
                index,
                offset: info.offset,
                size: info.size,
                stream: info.stream,
                record_type: info.record_type,
                timestamp: info.timestamp,
            });
        }
        if batch.is_empty() {
            return Ok(());
        }
        let worker = match &mut self.worker {
            Some(w) => w,
            None => {
                let file = self.reader.file_handle().try_clone()?;
                self.worker.insert(Worker::spawn(file)?)
            }
        };
        self.inflight += batch.len();
        self.inflight_bytes += batch_bytes;
        worker.send(batch)?;
        Ok(())
    }

    /// Read the next record into `buf` (replacing its contents); `None` at
    /// the end of the scan.
    pub fn next_into(&mut self, buf: &mut Vec<u8>) -> Result<Option<RecordMeta>> {
        loop {
            if let Some(first) = self.pending.pop_front() {
                self.pending_bytes -= first.size as usize;
                // Refill only below the halfway watermark: dispatching on
                // every pop degrades to one-record batches and the thread
                // round trip (~µs) would dwarf the work per record.
                if self.pending.len() + self.inflight < self.max_records / 2 {
                    self.dispatch()?;
                }
                let payload = first.payload?;
                let meta = first.meta;
                *buf = payload;
                return Ok(Some(meta));
            }
            if self.done && self.inflight == 0 {
                return Ok(None);
            }
            self.dispatch()?;
            // `dispatch` may have just exhausted the scan (done=true) with
            // nothing in flight: re-check before blocking on the worker.
            if self.done && self.inflight == 0 {
                return Ok(None);
            }
            // Wait for the next batch of results (the worker keeps scanning).
            let Some(worker) = self.worker.as_mut() else {
                return Ok(None);
            };
            let Some(batch) = worker.res_rx.as_ref().and_then(|rx| rx.recv().ok()) else {
                self.done = true;
                continue;
            };
            self.inflight -= batch.len();
            self.inflight_bytes -= batch.iter().map(|d| d.size as usize).sum::<usize>();
            self.pending_bytes += batch.iter().map(|d| d.size as usize).sum::<usize>();
            self.pending.extend(batch);
        }
    }
}

impl Drop for SequentialReader<'_> {
    fn drop(&mut self) {
        // Worker::drop unblocks and joins the thread.
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use super::super::writer::{VrsWriteOptions, VrsWriter, stream_tags};
    use super::super::{DataLayout, LayoutWriter};
    use std::collections::BTreeMap;

    /// 300 small records + 30 large ones across two streams.
    fn make(path: &std::path::Path) {
        let layout = DataLayout::from_json(
            r#"{"data_layout":[{"name":"seq","type":"DataPieceValue<uint32_t>","offset":0}]}"#,
        )
        .unwrap();
        let a = StreamId::new(600, 1);
        let b = StreamId::new(601, 1);
        let mut w = VrsWriter::create(path, VrsWriteOptions::default()).unwrap();
        for (id, name) in [(a, "A"), (b, "B")] {
            w.add_stream(id, stream_tags(name, BTreeMap::new())).unwrap();
            w.define_record_format(id, RecordType::Data, 1, "data_layout+custom", &[&layout])
                .unwrap();
        }
        for i in 0..300i64 {
            for (id, t0) in [(a, 0.0), (b, 0.0005)] {
                let mut lw = LayoutWriter::new(&layout);
                lw.set_u32("seq", i as u32).unwrap();
                let mut payload = lw.finish();
                // Large every tenth record (compressed), small otherwise.
                if i % 10 == 0 {
                    payload.extend(std::iter::repeat_n(i as u8, 4000));
                } else {
                    payload.extend(std::iter::repeat_n(0u8, 40));
                }
                w.write_record(id, RecordType::Data, 1, i as f64 * 0.001 + t0, &payload)
                    .unwrap();
            }
        }
        w.finish().unwrap();
    }

    fn scan(reader: &mut VrsReader, filter: &RecordFilter) -> Vec<(RecordMeta, Vec<u8>)> {
        let mut out = Vec::new();
        let mut cur = reader.cursor(filter);
        let mut buf = Vec::new();
        while let Some(meta) = reader.next_record_into(&mut cur, &mut buf).unwrap() {
            out.push((meta, buf.clone()));
        }
        out
    }

    #[test]
    fn sequential_read_ahead_matches_the_cursor_scan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.vrs");
        make(&path);

        let mut r = VrsReader::open(&path).unwrap();
        let all = RecordFilter::all();
        let reference = scan(&mut r, &all);

        for depth in [1, 3, 16, 64] {
            let mut r = VrsReader::open(&path).unwrap();
            let mut seq = r.sequential(&RecordFilter::all()).unwrap().with_depth(depth);
            let mut buf = Vec::new();
            let mut got = Vec::new();
            while let Some(meta) = seq.next_into(&mut buf).unwrap() {
                got.push((meta, buf.clone()));
            }
            assert_eq!(got, reference, "depth {depth}");
        }

        // A filtered scan yields exactly the subset the cursor yields.
        let only_b = RecordFilter::streams([StreamId::new(601, 1)]);
        let mut r = VrsReader::open(&path).unwrap();
        let reference = scan(&mut r, &only_b);
        let mut r = VrsReader::open(&path).unwrap();
        let mut seq = r.sequential(&only_b).unwrap();
        let mut buf = Vec::new();
        let mut got = Vec::new();
        while let Some(meta) = seq.next_into(&mut buf).unwrap() {
            got.push((meta, buf.clone()));
        }
        assert_eq!(got, reference);
    }
}
