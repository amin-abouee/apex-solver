//! On-disk structures of the VRS container (file format version `VRS2`).
//!
//! Everything is little-endian and tightly packed. Layout of a file:
//!
//! ```text
//! FileHeader (80 B)
//! description record      @ descriptionRecordOffset (= 80)
//! index record            @ indexRecordOffset (immediately after the description)
//! user records ...        @ firstUserRecordOffset
//! ```
//!
//! Every record starts with a 32-byte [`RecordHeader`] whose `record_size`
//! covers header and payload; records are chained backwards through
//! `previous_record_size`.

use super::error::{Result, VrsError};
use std::fmt;

/// Size of the file header on disk.
pub const FILE_HEADER_SIZE: usize = 80;
/// Size of a record header on disk.
pub const RECORD_HEADER_SIZE: usize = 32;
/// Size of one index entry on disk.
pub const INDEX_ENTRY_SIZE: usize = 19;

/// `"Visi"` as a little-endian `u32`.
pub const MAGIC_1: u32 = u32::from_le_bytes(*b"Visi");
/// `"onRe"` as a little-endian `u32`.
pub const MAGIC_2: u32 = u32::from_le_bytes(*b"onRe");
/// `"cord"` as a little-endian `u32`.
pub const MAGIC_3: u32 = u32::from_le_bytes(*b"cord");
/// `"VRS2"` as a little-endian `u32`: the supported file format version.
pub const FORMAT_VERSION: u32 = u32::from_le_bytes(*b"VRS2");

/// Record type id of the description record.
pub const DESCRIPTION_TYPE_ID: i32 = 2;
/// Record type id of the index record.
pub const INDEX_TYPE_ID: i32 = 1;

fn le_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn le_i32(b: &[u8], at: usize) -> i32 {
    le_u32(b, at) as i32
}
fn le_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}
fn le_f64(b: &[u8], at: usize) -> f64 {
    f64::from_bits(le_u64(b, at))
}

/// Identifies one stream: a recordable class plus an instance number
/// (`214-1` is the Aria RGB camera).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId {
    /// Recordable type id (sensor class).
    pub type_id: i32,
    /// Instance of that class.
    pub instance: u16,
}

impl StreamId {
    /// Build a stream id.
    pub fn new(type_id: i32, instance: u16) -> Self {
        Self { type_id, instance }
    }

    /// Parse from the 6-byte on-disk form.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < 6 {
            return Err(VrsError::corrupt("truncated stream id"));
        }
        Ok(Self::new(le_i32(b, 0), le_u16(b, 4)))
    }

    /// Serialise to the 6-byte on-disk form.
    pub fn to_bytes(self) -> [u8; 6] {
        let mut out = [0u8; 6];
        out[..4].copy_from_slice(&self.type_id.to_le_bytes());
        out[4..].copy_from_slice(&self.instance.to_le_bytes());
        out
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.type_id, self.instance)
    }
}

/// Kind of a record within a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RecordType {
    /// Stream state (rarely used by Aria).
    State,
    /// Stream configuration: sensor parameters, once per stream.
    Configuration,
    /// A sample (image, IMU measurement, ...).
    Data,
    /// Tags record.
    Tags,
    /// Any other value found on disk.
    Other(u8),
}

impl RecordType {
    /// Decode the on-disk byte.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::State,
            2 => Self::Configuration,
            3 => Self::Data,
            4 => Self::Tags,
            other => Self::Other(other),
        }
    }

    /// Encode to the on-disk byte.
    pub fn to_u8(self) -> u8 {
        match self {
            Self::State => 1,
            Self::Configuration => 2,
            Self::Data => 3,
            Self::Tags => 4,
            Self::Other(v) => v,
        }
    }

    /// Name as used in `RF:`/`DL:` tag keys.
    pub fn name(self) -> &'static str {
        match self {
            Self::State => "State",
            Self::Configuration => "Configuration",
            Self::Data => "Data",
            Self::Tags => "Tags",
            Self::Other(_) => "Undefined",
        }
    }
}

/// Per-record payload compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// Stored as is.
    None,
    /// LZ4 (frame).
    Lz4,
    /// Zstandard.
    Zstd,
}

impl Compression {
    /// Decode the on-disk byte.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::Zstd),
            other => Err(VrsError::unsupported(format!("compression type {other}"))),
        }
    }

    /// Encode to the on-disk byte.
    pub fn to_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Zstd => 2,
        }
    }
}

/// The 80-byte header at the start of every VRS file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    /// Unique id assigned at creation.
    pub creation_id: u64,
    /// Size of this header on disk (80 in all known files).
    pub file_header_size: u32,
    /// Size of each record header (32).
    pub record_header_size: u32,
    /// Offset of the index record.
    pub index_record_offset: i64,
    /// Offset of the description record.
    pub description_record_offset: i64,
    /// Offset of the first stream record.
    pub first_user_record_offset: i64,
}

impl FileHeader {
    /// Parse and validate the magic values and basic geometry.
    pub fn parse(b: &[u8]) -> Result<Option<Self>> {
        if b.len() < FILE_HEADER_SIZE {
            return Ok(None);
        }
        if le_u32(b, 0) != MAGIC_1 || le_u32(b, 4) != MAGIC_2 || le_u32(b, 72) != MAGIC_3 {
            return Ok(None);
        }
        if le_u32(b, 76) != FORMAT_VERSION {
            return Err(VrsError::unsupported(format!(
                "VRS format version {:?}",
                String::from_utf8_lossy(&b[76..80])
            )));
        }
        let header = Self {
            creation_id: le_u64(b, 8),
            file_header_size: le_u32(b, 16),
            record_header_size: le_u32(b, 20),
            index_record_offset: le_u64(b, 24) as i64,
            description_record_offset: le_u64(b, 32) as i64,
            first_user_record_offset: le_u64(b, 40) as i64,
        };
        if header.file_header_size as usize != FILE_HEADER_SIZE {
            return Err(VrsError::unsupported(format!(
                "file header size {}",
                header.file_header_size
            )));
        }
        if header.record_header_size as usize != RECORD_HEADER_SIZE {
            return Err(VrsError::unsupported(format!(
                "record header size {}",
                header.record_header_size
            )));
        }
        Ok(Some(header))
    }

    /// Serialise to the 80-byte on-disk form.
    pub fn to_bytes(&self) -> [u8; FILE_HEADER_SIZE] {
        let mut out = [0u8; FILE_HEADER_SIZE];
        out[0..4].copy_from_slice(&MAGIC_1.to_le_bytes());
        out[4..8].copy_from_slice(&MAGIC_2.to_le_bytes());
        out[8..16].copy_from_slice(&self.creation_id.to_le_bytes());
        out[16..20].copy_from_slice(&self.file_header_size.to_le_bytes());
        out[20..24].copy_from_slice(&self.record_header_size.to_le_bytes());
        out[24..32].copy_from_slice(&self.index_record_offset.to_le_bytes());
        out[32..40].copy_from_slice(&self.description_record_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.first_user_record_offset.to_le_bytes());
        // 48..72: reserved (zero)
        out[72..76].copy_from_slice(&MAGIC_3.to_le_bytes());
        out[76..80].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out
    }
}

/// The 32-byte header preceding every record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordHeader {
    /// Header plus payload size in bytes.
    pub record_size: u32,
    /// Size of the preceding record (0 for the first).
    pub previous_record_size: u32,
    /// Stream (or special record) type id.
    pub type_id: i32,
    /// Version of the record's `RecordFormat`.
    pub format_version: u32,
    /// Capture time in seconds.
    pub timestamp: f64,
    /// Stream instance.
    pub instance: u16,
    /// Kind of record.
    pub record_type: RecordType,
    /// Payload compression.
    pub compression: Compression,
    /// Payload size once decompressed (0 when stored uncompressed).
    pub uncompressed_size: u32,
}

impl RecordHeader {
    /// Parse from at least 32 bytes.
    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < RECORD_HEADER_SIZE {
            return Err(VrsError::corrupt("truncated record header"));
        }
        let header = Self {
            record_size: le_u32(b, 0),
            previous_record_size: le_u32(b, 4),
            type_id: le_i32(b, 8),
            format_version: le_u32(b, 12),
            timestamp: le_f64(b, 16),
            instance: le_u16(b, 24),
            record_type: RecordType::from_u8(b[26]),
            compression: Compression::from_u8(b[27])?,
            uncompressed_size: le_u32(b, 28),
        };
        if (header.record_size as usize) < RECORD_HEADER_SIZE {
            return Err(VrsError::corrupt(format!(
                "record size {} smaller than its header",
                header.record_size
            )));
        }
        Ok(header)
    }

    /// Serialise to the 32-byte on-disk form.
    pub fn to_bytes(&self) -> [u8; RECORD_HEADER_SIZE] {
        let mut out = [0u8; RECORD_HEADER_SIZE];
        out[0..4].copy_from_slice(&self.record_size.to_le_bytes());
        out[4..8].copy_from_slice(&self.previous_record_size.to_le_bytes());
        out[8..12].copy_from_slice(&self.type_id.to_le_bytes());
        out[12..16].copy_from_slice(&self.format_version.to_le_bytes());
        out[16..24].copy_from_slice(&self.timestamp.to_le_bytes());
        out[24..26].copy_from_slice(&self.instance.to_le_bytes());
        out[26] = self.record_type.to_u8();
        out[27] = self.compression.to_u8();
        out[28..32].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        out
    }

    /// Stream this record belongs to.
    pub fn stream_id(&self) -> StreamId {
        StreamId::new(self.type_id, self.instance)
    }

    /// Payload size on disk.
    pub fn payload_size(&self) -> usize {
        self.record_size as usize - RECORD_HEADER_SIZE
    }
}

/// One entry of the index: where (implicitly) and when a record is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexEntry {
    /// Capture time in seconds.
    pub timestamp: f64,
    /// Whole-record size (header + payload).
    pub record_size: u32,
    /// Kind of record.
    pub record_type: RecordType,
    /// Owning stream.
    pub stream: StreamId,
}

impl IndexEntry {
    /// Parse one 19-byte entry.
    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < INDEX_ENTRY_SIZE {
            return Err(VrsError::corrupt("truncated index entry"));
        }
        Ok(Self {
            timestamp: le_f64(b, 0),
            record_size: le_u32(b, 8),
            record_type: RecordType::from_u8(b[12]),
            stream: StreamId::new(le_i32(b, 13), le_u16(b, 17)),
        })
    }

    /// Serialise to 19 bytes.
    pub fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut out = [0u8; INDEX_ENTRY_SIZE];
        out[0..8].copy_from_slice(&self.timestamp.to_le_bytes());
        out[8..12].copy_from_slice(&self.record_size.to_le_bytes());
        out[12] = self.record_type.to_u8();
        out[13..17].copy_from_slice(&self.stream.type_id.to_le_bytes());
        out[17..19].copy_from_slice(&self.stream.instance.to_le_bytes());
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// First 80 bytes of `R_06_medium.vrs` (LaMaria).
    const REAL_HEADER: &str = concat!(
        "566973696f6e52657580fe3bb9c01518500000002000000006910000000000005000000000000000",
        "2bff480000000000000000000000000000000000000000000000000000000000636f726456525332",
    );

    fn from_hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap_or(0))
            .collect()
    }

    #[test]
    fn magic_constants_spell_the_file_signature() {
        let mut sig = Vec::new();
        sig.extend_from_slice(&MAGIC_1.to_le_bytes());
        sig.extend_from_slice(&MAGIC_2.to_le_bytes());
        sig.extend_from_slice(&MAGIC_3.to_le_bytes());
        sig.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        assert_eq!(sig, b"VisionRecordVRS2");
    }

    #[test]
    fn parses_a_real_file_header() {
        let bytes = from_hex(REAL_HEADER);
        assert_eq!(bytes.len(), FILE_HEADER_SIZE);
        let h = FileHeader::parse(&bytes).unwrap().unwrap();
        assert_eq!(h.file_header_size, 80);
        assert_eq!(h.record_header_size, 32);
        assert_eq!(h.description_record_offset, 80);
        assert_eq!(h.index_record_offset, 37126);
        assert_eq!(h.first_user_record_offset, 4_783_915);
        assert_eq!(h.to_bytes().to_vec(), bytes);
    }

    #[test]
    fn rejects_non_vrs_and_unknown_versions() {
        assert!(FileHeader::parse(&[0u8; 80]).unwrap().is_none());
        assert!(FileHeader::parse(&[0u8; 10]).unwrap().is_none());
        let mut bytes = from_hex(REAL_HEADER);
        bytes[76..80].copy_from_slice(b"VRS9");
        assert!(FileHeader::parse(&bytes).is_err());
    }

    #[test]
    fn record_header_round_trips() {
        let h = RecordHeader {
            record_size: 4138,
            previous_record_size: 4_746_789,
            type_id: 214,
            format_version: 2,
            timestamp: 529.400368975,
            instance: 1,
            record_type: RecordType::Data,
            compression: Compression::Zstd,
            uncompressed_size: 321_737,
        };
        let parsed = RecordHeader::parse(&h.to_bytes()).unwrap();
        assert_eq!(parsed, h);
        assert_eq!(parsed.stream_id().to_string(), "214-1");
        assert_eq!(parsed.payload_size(), 4138 - 32);
    }

    #[test]
    fn record_header_rejects_garbage() {
        assert!(RecordHeader::parse(&[0u8; 10]).is_err());
        let mut b = [0u8; 32];
        b[0] = 8; // record_size < header size
        assert!(RecordHeader::parse(&b).is_err());
        let mut b = RecordHeader::parse(&{
            let mut ok = [0u8; 32];
            ok[0] = 40;
            ok
        })
        .unwrap()
        .to_bytes();
        b[27] = 9; // unknown compression
        assert!(RecordHeader::parse(&b).is_err());
    }

    #[test]
    fn index_entry_round_trips_and_matches_a_real_one() {
        // First real entry: config record of 214-1, 4138 bytes, ts 5e-324.
        let e = IndexEntry {
            timestamp: f64::from_bits(1),
            record_size: 4138,
            record_type: RecordType::Configuration,
            stream: StreamId::new(214, 1),
        };
        assert_eq!(IndexEntry::parse(&e.to_bytes()).unwrap(), e);
        assert_eq!(e.to_bytes().len(), INDEX_ENTRY_SIZE);
    }

    #[test]
    fn stream_id_bytes_round_trip() {
        let s = StreamId::new(1202, 2);
        assert_eq!(StreamId::from_bytes(&s.to_bytes()).unwrap(), s);
        assert!(StreamId::from_bytes(&[0; 3]).is_err());
    }
}
