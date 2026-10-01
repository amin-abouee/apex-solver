//! Record index: reading it from the file, or rebuilding it by walking records.
//!
//! The index record's payload is `u32 nStreams`, `nStreams` stream ids,
//! `u32 nRecords`, then (compressed per the record header) `nRecords` 19-byte
//! entries in file order. Entries carry sizes, not offsets: a record's offset
//! is `firstUserRecordOffset` plus the sizes of all records before it.

use super::codec;
use super::error::{Result, VrsError};
use super::format::{
    Compression, DESCRIPTION_TYPE_ID, FileHeader, INDEX_ENTRY_SIZE, INDEX_TYPE_ID, IndexEntry,
    RECORD_HEADER_SIZE, RecordHeader, RecordType, StreamId,
};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

/// Where one record lives and what it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordInfo {
    /// Capture time in seconds.
    pub timestamp: f64,
    /// Absolute file offset of the record header.
    pub offset: u64,
    /// Whole-record size (header + payload).
    pub size: u32,
    /// Kind of record.
    pub record_type: RecordType,
    /// Owning stream.
    pub stream: StreamId,
}

/// How the in-memory index was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexSource {
    /// Read from the file's index record.
    File,
    /// The file had no usable index; it was rebuilt by scanning all records.
    Rebuilt,
}

/// Upper bound on index entries we are willing to allocate for (≈ 38 GB of index data).
const MAX_RECORDS: u64 = 1 << 31;

fn read_exact_at(file: &mut File, offset: u64, len: usize) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read the description/index-style special record at `offset`: header plus
/// decompressed payload.
pub fn read_special_record(
    file: &mut File,
    offset: u64,
    file_len: u64,
) -> Result<(RecordHeader, Vec<u8>)> {
    if offset
        .checked_add(RECORD_HEADER_SIZE as u64)
        .is_none_or(|e| e > file_len)
    {
        return Err(VrsError::corrupt(format!(
            "record offset {offset} is outside the file"
        )));
    }
    let header = RecordHeader::parse(&read_exact_at(file, offset, RECORD_HEADER_SIZE)?)?;
    if offset + u64::from(header.record_size) > file_len {
        return Err(VrsError::corrupt("record extends past end of file"));
    }
    let raw = read_exact_at(
        file,
        offset + RECORD_HEADER_SIZE as u64,
        header.payload_size(),
    )?;
    let mut payload = Vec::new();
    codec::decompress(
        header.compression,
        &raw,
        header.uncompressed_size as usize,
        &mut payload,
    )?;
    Ok((header, payload))
}

/// Turn index entries into [`RecordInfo`]s with absolute offsets.
fn with_offsets(entries: &[IndexEntry], first_offset: u64) -> Vec<RecordInfo> {
    let mut offset = first_offset;
    entries
        .iter()
        .map(|e| {
            let info = RecordInfo {
                timestamp: e.timestamp,
                offset,
                size: e.record_size,
                record_type: e.record_type,
                stream: e.stream,
            };
            offset += u64::from(e.record_size);
            info
        })
        .collect()
}

/// Decode an index record into entries. `None` if the record is unusable.
fn decode_index(header: &RecordHeader, raw: &[u8]) -> Option<Vec<IndexEntry>> {
    let n_streams = u32::from_le_bytes(raw.get(..4)?.try_into().ok()?) as usize;
    let prefix = 4usize.checked_add(n_streams.checked_mul(6)?)?;
    let n_records = u32::from_le_bytes(raw.get(prefix..prefix + 4)?.try_into().ok()?) as usize;
    let body = raw.get(prefix + 4..)?;
    if n_records as u64 > MAX_RECORDS {
        return None;
    }
    let expected = n_records.checked_mul(INDEX_ENTRY_SIZE)?;
    // A compressed body cannot plausibly expand more than ~4096x; refuse
    // counts that would make us allocate far more than the file can hold.
    if header.compression != Compression::None
        && expected > body.len().saturating_mul(4096).max(1 << 20)
    {
        return None;
    }

    let mut decoded = Vec::new();
    let entries_bytes: &[u8] = if header.compression == Compression::None {
        body.get(..expected)?
    } else {
        codec::decompress(header.compression, body, expected, &mut decoded).ok()?;
        &decoded
    };
    let (chunks, _) = entries_bytes.as_chunks::<INDEX_ENTRY_SIZE>();
    chunks.iter().map(|c| IndexEntry::parse(c).ok()).collect()
}

/// Try to load the index recorded in the file. `Ok(None)` means "absent or
/// inconsistent -- rebuild".
pub fn read_index(
    file: &mut File,
    header: &FileHeader,
    file_len: u64,
) -> Result<Option<Vec<RecordInfo>>> {
    let (Ok(index_offset), Ok(first_user)) = (
        u64::try_from(header.index_record_offset),
        u64::try_from(header.first_user_record_offset),
    ) else {
        return Ok(None);
    };
    if index_offset == 0 || first_user == 0 {
        return Ok(None);
    }
    let Ok((rec_header, payload)) = read_index_record(file, index_offset, file_len) else {
        return Ok(None);
    };
    if rec_header.type_id != INDEX_TYPE_ID {
        return Ok(None);
    }
    let Some(entries) = decode_index_payload(&rec_header, &payload) else {
        return Ok(None);
    };
    let infos = with_offsets(&entries, first_user);
    let end = infos
        .last()
        .map_or(first_user, |l| l.offset + u64::from(l.size));
    // An index that runs past the end describes a truncated file.
    if end > file_len {
        return Ok(None);
    }
    Ok(Some(infos))
}

/// The index payload is only partially compressed, so it cannot go through
/// the generic whole-payload decompression in [`read_special_record`].
fn read_index_record(
    file: &mut File,
    offset: u64,
    file_len: u64,
) -> Result<(RecordHeader, Vec<u8>)> {
    if offset
        .checked_add(RECORD_HEADER_SIZE as u64)
        .is_none_or(|e| e > file_len)
    {
        return Err(VrsError::corrupt("index offset outside the file"));
    }
    let header = RecordHeader::parse(&read_exact_at(file, offset, RECORD_HEADER_SIZE)?)?;
    if offset + u64::from(header.record_size) > file_len {
        return Err(VrsError::corrupt("index record extends past end of file"));
    }
    let raw = read_exact_at(
        file,
        offset + RECORD_HEADER_SIZE as u64,
        header.payload_size(),
    )?;
    Ok((header, raw))
}

fn decode_index_payload(header: &RecordHeader, raw: &[u8]) -> Option<Vec<IndexEntry>> {
    decode_index(header, raw)
}

/// Offset where user records start when the header does not say.
fn first_user_offset(file: &mut File, header: &FileHeader, file_len: u64) -> Result<u64> {
    if let Ok(first) = u64::try_from(header.first_user_record_offset)
        && first > 0
    {
        return Ok(first);
    }
    // Header was never finalised: skip description (and index, if present).
    let desc = u64::try_from(header.description_record_offset)
        .map_err(|_| VrsError::corrupt("negative description offset"))?;
    let (h, _) = read_special_record(file, desc, file_len)?;
    let mut pos = desc + u64::from(h.record_size);
    if pos + RECORD_HEADER_SIZE as u64 <= file_len {
        let next = RecordHeader::parse(&read_exact_at(file, pos, RECORD_HEADER_SIZE)?)?;
        if next.type_id == INDEX_TYPE_ID {
            pos += u64::from(next.record_size);
        }
    }
    Ok(pos)
}

/// Rebuild the index by walking record headers from the first user record.
/// A partially written final record ends the walk.
pub fn rebuild_index(
    file: &mut File,
    header: &FileHeader,
    file_len: u64,
) -> Result<Vec<RecordInfo>> {
    let mut pos = first_user_offset(file, header, file_len)?;
    file.seek(SeekFrom::Start(pos))?;
    let mut reader = BufReader::with_capacity(64 * 1024, &mut *file);
    let mut infos = Vec::new();
    let mut head = [0u8; RECORD_HEADER_SIZE];

    while pos + RECORD_HEADER_SIZE as u64 <= file_len {
        reader.read_exact(&mut head)?;
        let h = RecordHeader::parse(&head)?;
        let end = pos + u64::from(h.record_size);
        if end > file_len {
            break; // truncated tail
        }
        // Special records (description, and an index written at the end of
        // the file) are not stream records.
        let special = h.type_id == INDEX_TYPE_ID || h.type_id == DESCRIPTION_TYPE_ID;
        if !special {
            infos.push(RecordInfo {
                timestamp: h.timestamp,
                offset: pos,
                size: h.record_size,
                record_type: h.record_type,
                stream: h.stream_id(),
            });
        }
        reader.seek_relative(i64::from(h.record_size) - RECORD_HEADER_SIZE as i64)?;
        pos = end;
    }
    Ok(infos)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn entry(ts: f64, size: u32, t: RecordType, s: StreamId) -> IndexEntry {
        IndexEntry {
            timestamp: ts,
            record_size: size,
            record_type: t,
            stream: s,
        }
    }

    #[test]
    fn offsets_accumulate_from_the_first_record() {
        let s = StreamId::new(1, 1);
        let e = [
            entry(0.0, 100, RecordType::Data, s),
            entry(1.0, 50, RecordType::Data, s),
            entry(2.0, 70, RecordType::Data, s),
        ];
        let infos = with_offsets(&e, 1000);
        assert_eq!(
            infos.iter().map(|i| i.offset).collect::<Vec<_>>(),
            [1000, 1100, 1150]
        );
    }

    #[test]
    fn decodes_compressed_and_stored_index_payloads() {
        let s = StreamId::new(214, 1);
        let entries: Vec<IndexEntry> = (0..100)
            .map(|i| entry(f64::from(i), 40 + i as u32, RecordType::Data, s))
            .collect();
        let body: Vec<u8> = entries.iter().flat_map(|e| e.to_bytes()).collect();

        for comp in [Compression::None, Compression::Zstd] {
            let mut raw = Vec::new();
            raw.extend_from_slice(&1u32.to_le_bytes());
            raw.extend_from_slice(&s.to_bytes());
            raw.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            raw.extend_from_slice(&codec::compress(comp, &body).unwrap());
            let header = RecordHeader {
                record_size: 0,
                previous_record_size: 0,
                type_id: INDEX_TYPE_ID,
                format_version: 2,
                timestamp: f64::MAX,
                instance: 0,
                record_type: RecordType::Data,
                compression: comp,
                uncompressed_size: 0,
            };
            assert_eq!(decode_index(&header, &raw).unwrap(), entries, "{comp:?}");
        }
    }

    #[test]
    fn hostile_index_payloads_yield_none_not_panics() {
        let header = RecordHeader {
            record_size: 32,
            previous_record_size: 0,
            type_id: INDEX_TYPE_ID,
            format_version: 2,
            timestamp: 0.0,
            instance: 0,
            record_type: RecordType::Data,
            compression: Compression::Zstd,
            uncompressed_size: 0,
        };
        assert!(decode_index(&header, &[]).is_none());
        assert!(decode_index(&header, &u32::MAX.to_le_bytes()).is_none());
        // plausible prefix, absurd record count
        let mut raw = 0u32.to_le_bytes().to_vec();
        raw.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_index(&header, &raw).is_none());
    }
}
