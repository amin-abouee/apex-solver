//! Copy (and optionally trim / re-compress) a VRS file.
//!
//! Tags, record formats and payload bytes are preserved; only the container
//! framing and compression are rewritten. Configuration and state records of
//! the selected streams are always kept; data records are filtered by
//! stream and time window and written in capture-time order.

use super::codec;
use super::error::Result;
use super::format::{Compression, RecordType, StreamId};
use super::reader::{RecordFilter, VrsReader};
use super::writer::{VrsWriteOptions, VrsWriter, WriteSummary};
use std::path::Path;

/// Which part of the source to copy.
#[derive(Debug, Clone, Default)]
pub struct CopyOptions {
    /// Streams to keep; empty means all.
    pub streams: Vec<StreamId>,
    /// First data timestamp to keep (seconds, inclusive).
    pub start: Option<f64>,
    /// Data timestamp to stop at (seconds, exclusive).
    pub end: Option<f64>,
    /// Compression of the output records.
    pub compression: Option<Compression>,
}

/// Copy `src` to `dst` (which must not exist).
pub fn copy_filtered(
    src: &mut VrsReader,
    dst: impl AsRef<Path>,
    options: &CopyOptions,
) -> Result<WriteSummary> {
    let wanted: Vec<StreamId> = if options.streams.is_empty() {
        src.streams()
    } else {
        options
            .streams
            .iter()
            .copied()
            .filter(|s| src.streams().contains(s))
            .collect()
    };

    let mut write_opts = VrsWriteOptions::default();
    if let Some(c) = options.compression {
        write_opts.compression = c;
    }
    let target = write_opts.compression;
    let mut out = VrsWriter::create(dst, write_opts)?;
    for (k, v) in src.file_tags() {
        out.set_file_tag(k.clone(), v.clone())?;
    }
    for &id in &wanted {
        let tags = src.stream_tags(id).cloned().unwrap_or_default();
        out.add_stream(id, tags)?;
    }

    let mut buf = Vec::new();
    // Configuration and state records first, as a recorder would write them:
    // globally ordered by timestamp (then original file order), because
    // readers expect the record sequence to be time-sorted.
    let mut special: Vec<(f64, u64, usize)> = Vec::new();
    for &id in &wanted {
        for index in [src.configuration_record(id), src.state_record(id)]
            .into_iter()
            .flatten()
        {
            let info = src.records()[index];
            special.push((info.timestamp, info.offset, index));
        }
    }
    special.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for (_, _, index) in special {
        let meta = src.read_record_into(index, &mut buf)?;
        out.write_record(
            meta.stream,
            meta.record_type,
            meta.format_version,
            meta.timestamp,
            &buf,
        )?;
    }

    let mut filter = RecordFilter::streams(wanted.iter().copied());
    filter.start = options.start;
    filter.end = options.end;
    let mut cursor = src.cursor(&filter);
    let mut de = codec::Decompressor::new();
    let mut raw = Vec::new();
    let mut payload = Vec::new();
    while let Some(index) = src.next_index(&mut cursor) {
        // Read the record as stored, then take the cheapest route to the
        // target compression: verbatim passthrough when it already matches,
        // direct compression of the stored bytes when the source is stored,
        // and a decompress round-trip only when both sides differ.
        let stored = src.read_stored_record_into(index, &mut raw)?;
        debug_assert_eq!(stored.meta.record_type, RecordType::Data);
        if stored.compression == target {
            out.write_record_raw(stored.meta, &stored.body, target, stored.uncompressed_size)?;
        } else if stored.compression == Compression::None {
            out.write_record(
                stored.meta.stream,
                stored.meta.record_type,
                stored.meta.format_version,
                stored.meta.timestamp,
                &stored.body,
            )?;
        } else {
            de.decompress(
                stored.compression,
                &stored.body,
                stored.uncompressed_size as usize,
                &mut payload,
            )?;
            out.write_record(
                stored.meta.stream,
                stored.meta.record_type,
                stored.meta.format_version,
                stored.meta.timestamp,
                &payload,
            )?;
        }
    }
    out.finish()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vrs::description::StreamTags;

    fn make(path: &Path) {
        let mut w = VrsWriter::create(path, VrsWriteOptions::default()).unwrap();
        w.set_file_tag("session_id", "s1").unwrap();
        for t in [600, 601] {
            let mut tags = StreamTags::default();
            tags.user.insert("n".into(), t.to_string());
            w.add_stream(StreamId::new(t, 1), tags).unwrap();
        }
        for t in [600, 601] {
            let id = StreamId::new(t, 1);
            w.write_record(id, RecordType::Configuration, 1, 0.0, b"config")
                .unwrap();
            for i in 0..100 {
                w.write_record(
                    id,
                    RecordType::Data,
                    1,
                    f64::from(i) * 0.01,
                    &[i as u8; 200],
                )
                .unwrap();
            }
        }
        w.finish().unwrap();
    }

    #[test]
    fn passthrough_and_transcodes_preserve_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let src_path = dir.path().join("src.vrs");
        {
            let id = StreamId::new(700, 1);
            let mut w = VrsWriter::create(&src_path, VrsWriteOptions::default()).unwrap();
            w.add_stream(id, StreamTags::default()).unwrap();
            // Small (stored) and large (compressed) records, both kinds.
            for i in 0..40u32 {
                let payload: Vec<u8> = if i % 2 == 0 {
                    vec![i as u8; 16]
                } else {
                    (0..4000u32).map(|k| (k % 251) as u8).collect()
                };
                w.write_record(id, RecordType::Data, 1, f64::from(i) * 0.1, &payload)
                    .unwrap();
            }
            w.finish().unwrap();
        }
        let payloads = |path: &Path| -> Vec<(f64, Vec<u8>)> {
            let mut r = VrsReader::open(path).unwrap();
            let mut cur = r.cursor(&RecordFilter::all());
            let mut buf = Vec::new();
            let mut out = Vec::new();
            while let Some(meta) = r.next_record_into(&mut cur, &mut buf).unwrap() {
                out.push((meta.timestamp, buf.clone()));
            }
            out
        };
        let reference = payloads(&src_path);

        for target in [
            Compression::Zstd, // same as source: exercises verbatim passthrough
            Compression::None, // decompress and store
            Compression::Lz4,  // decompress and recompress
        ] {
            let dst = dir.path().join(format!("{target:?}.vrs"));
            let mut src = VrsReader::open(&src_path).unwrap();
            let opts = CopyOptions {
                compression: Some(target),
                ..CopyOptions::default()
            };
            copy_filtered(&mut src, &dst, &opts).unwrap();
            assert_eq!(payloads(&dst), reference, "{target:?}");
        }
    }

    #[test]
    fn trims_streams_and_time_and_keeps_tags() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.vrs"), dir.path().join("b.vrs"));
        make(&a);
        let mut src = VrsReader::open(&a).unwrap();
        let opts = CopyOptions {
            streams: vec![StreamId::new(601, 1)],
            start: Some(0.20),
            end: Some(0.50),
            compression: Some(Compression::Lz4),
        };
        let summary = copy_filtered(&mut src, &b, &opts).unwrap();
        assert_eq!(summary.records, 31, "config + 30 data");

        let mut dst = VrsReader::open(&b).unwrap();
        assert_eq!(dst.streams(), [StreamId::new(601, 1)]);
        assert_eq!(dst.file_tags()["session_id"], "s1");
        assert_eq!(
            dst.stream_tags(StreamId::new(601, 1)).unwrap().user["n"],
            "601"
        );
        let d = dst.data_records(StreamId::new(601, 1));
        assert_eq!(d.len(), 30);
        let first = dst.read_record(d[0] as usize).unwrap();
        assert!((first.meta.timestamp - 0.20).abs() < 1e-9);
        assert_eq!(first.payload, vec![20u8; 200]);
        let cfg = dst.configuration_record(StreamId::new(601, 1)).unwrap();
        assert_eq!(dst.read_record(cfg).unwrap().payload, b"config");
    }
}
