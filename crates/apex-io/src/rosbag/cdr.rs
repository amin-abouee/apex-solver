//! CDR (Common Data Representation) deserialization for ROS2 messages
//!
//! This module implements CDR deserialization according to the OMG CDR specification
//! used by ROS2 for message serialization.

use crate::rosbag::error::{ReaderError, Result};
use std::convert::TryInto;

/// CDR header information
#[derive(Debug, Clone, Copy)]
pub struct CdrHeader {
    pub endianness: Endianness,
    pub encapsulation_kind: u8,
}

/// Byte order endianness
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Endianness {
    LittleEndian,
    BigEndian,
}

/// CDR deserializer for reading binary message data
pub struct CdrDeserializer<'a> {
    data: &'a [u8],
    pos: usize,
    endianness: Endianness,
}

impl<'a> CdrDeserializer<'a> {
    /// Create a new CDR deserializer from raw message data
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(ReaderError::generic("CDR data too short for header"));
        }

        // Parse CDR header (4 bytes)
        let header = CdrHeader::parse(&data[0..4])?;

        Ok(Self {
            data,
            pos: 4, // Skip the 4-byte header
            endianness: header.endianness,
        })
    }

    /// Get current position in the data
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get the total length of the data
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Check if there are enough bytes remaining from current position
    pub fn has_remaining(&self, bytes: usize) -> bool {
        self.pos + bytes <= self.data.len()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Align position to the specified boundary
    fn align(&mut self, alignment: usize) {
        self.pos = (self.pos + alignment - 1) & !(alignment - 1);
    }

    /// Read a primitive value with proper alignment and endianness
    fn read_primitive<T>(&mut self, size: usize) -> Result<T>
    where
        T: FromBytes,
    {
        self.align(size);

        if self.pos + size > self.data.len() {
            return Err(ReaderError::generic(format!(
                "CDR data truncated: need {} bytes at pos {}, but only {} bytes available",
                size,
                self.pos,
                self.data.len()
            )));
        }

        let bytes = &self.data[self.pos..self.pos + size];
        self.pos += size;

        T::from_bytes(bytes, self.endianness)
    }

    /// Read an i8 value
    pub fn read_i8(&mut self) -> Result<i8> {
        self.read_primitive(1)
    }

    /// Read a u8 value
    pub fn read_u8(&mut self) -> Result<u8> {
        self.read_primitive(1)
    }

    /// Read a u16 value
    pub fn read_u16(&mut self) -> Result<u16> {
        self.read_primitive(2)
    }

    /// Read an i32 value
    pub fn read_i32(&mut self) -> Result<i32> {
        self.read_primitive(4)
    }

    /// Read a u32 value
    pub fn read_u32(&mut self) -> Result<u32> {
        self.read_primitive(4)
    }

    /// Read an f64 value
    pub fn read_f64(&mut self) -> Result<f64> {
        // In CDR, f64 values are aligned to 8-byte boundaries
        self.align(8);

        if self.pos + 8 > self.data.len() {
            return Err(ReaderError::generic(format!(
                "CDR data truncated: need 8 bytes at pos {}, but only {} bytes available",
                self.pos,
                self.data.len()
            )));
        }

        let bytes = &self.data[self.pos..self.pos + 8];
        self.pos += 8;

        f64::from_bytes(bytes, self.endianness)
    }

    /// Read a string value
    pub fn read_string(&mut self) -> Result<String> {
        let length = self.read_u32()? as usize;

        if length == 0 {
            return Ok(String::new());
        }

        if self.pos + length > self.data.len() {
            return Err(ReaderError::generic("CDR string data truncated"));
        }

        // String includes null terminator, but we need to handle the case where it might not
        let string_bytes = if length > 0 && self.data[self.pos + length - 1] == 0 {
            // Has null terminator
            &self.data[self.pos..self.pos + length - 1]
        } else {
            // No null terminator
            &self.data[self.pos..self.pos + length]
        };

        self.pos += length;

        String::from_utf8(string_bytes.to_vec())
            .map_err(|_| ReaderError::generic("Invalid UTF-8 in CDR string"))
    }

    /// Read a fixed-size array of f64 values
    pub fn read_f64_array<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut array = [0.0; N];
        for item in array.iter_mut().take(N) {
            *item = self.read_f64()?;
        }
        Ok(array)
    }

    /// Read a sequence (variable-length array) of elements
    pub fn read_sequence<T, F>(&mut self, read_element: F) -> Result<Vec<T>>
    where
        F: Fn(&mut Self) -> Result<T>,
    {
        let length = self.read_u32()? as usize;
        let mut vec = Vec::with_capacity(length);

        for _ in 0..length {
            vec.push(read_element(self)?);
        }

        Ok(vec)
    }

    /// Read a sequence of bytes (for data fields)
    pub fn read_byte_sequence(&mut self) -> Result<Vec<u8>> {
        let length = self.read_u32()? as usize;

        if self.pos + length > self.data.len() {
            return Err(ReaderError::generic(format!(
                "CDR data truncated: need {} bytes at pos {}, but only {} bytes available",
                length,
                self.pos,
                self.data.len()
            )));
        }

        let bytes = self.data[self.pos..self.pos + length].to_vec();
        self.pos += length;

        Ok(bytes)
    }

    /// Read a boolean value
    pub fn read_bool(&mut self) -> Result<bool> {
        let byte = self.read_u8()?;
        Ok(byte != 0)
    }

    /// Read an f32 value
    pub fn read_f32(&mut self) -> Result<f32> {
        self.align(4);

        if self.pos + 4 > self.data.len() {
            return Err(ReaderError::generic(format!(
                "CDR data truncated: need 4 bytes at pos {}, but only {} bytes available",
                self.pos,
                self.data.len()
            )));
        }

        let bytes = &self.data[self.pos..self.pos + 4];
        self.pos += 4;

        f32::from_bytes(bytes, self.endianness)
    }
}

impl CdrHeader {
    /// Parse CDR header from the first 4 bytes
    pub fn parse(header_bytes: &[u8]) -> Result<Self> {
        if header_bytes.len() != 4 {
            return Err(ReaderError::generic("CDR header must be exactly 4 bytes"));
        }

        let endianness = match header_bytes[1] {
            0 => Endianness::BigEndian,
            1 => Endianness::LittleEndian,
            _ => return Err(ReaderError::generic("Invalid CDR endianness flag")),
        };

        Ok(Self {
            endianness,
            encapsulation_kind: header_bytes[2],
        })
    }
}

/// Trait for converting bytes to primitive types with endianness handling
trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8], endianness: Endianness) -> Result<Self>;
}

impl FromBytes for i8 {
    fn from_bytes(bytes: &[u8], _endianness: Endianness) -> Result<Self> {
        if bytes.len() != 1 {
            return Err(ReaderError::generic("Invalid i8 bytes"));
        }
        Ok(bytes[0] as i8)
    }
}

impl FromBytes for u8 {
    fn from_bytes(bytes: &[u8], _endianness: Endianness) -> Result<Self> {
        if bytes.len() != 1 {
            return Err(ReaderError::generic("Invalid u8 bytes"));
        }
        Ok(bytes[0])
    }
}

impl FromBytes for u16 {
    fn from_bytes(bytes: &[u8], endianness: Endianness) -> Result<Self> {
        let array: [u8; 2] = bytes
            .try_into()
            .map_err(|_| ReaderError::generic("Invalid u16 bytes"))?;

        Ok(match endianness {
            Endianness::LittleEndian => u16::from_le_bytes(array),
            Endianness::BigEndian => u16::from_be_bytes(array),
        })
    }
}

impl FromBytes for i32 {
    fn from_bytes(bytes: &[u8], endianness: Endianness) -> Result<Self> {
        let array: [u8; 4] = bytes
            .try_into()
            .map_err(|_| ReaderError::generic("Invalid i32 bytes"))?;

        Ok(match endianness {
            Endianness::LittleEndian => i32::from_le_bytes(array),
            Endianness::BigEndian => i32::from_be_bytes(array),
        })
    }
}

impl FromBytes for u32 {
    fn from_bytes(bytes: &[u8], endianness: Endianness) -> Result<Self> {
        let array: [u8; 4] = bytes
            .try_into()
            .map_err(|_| ReaderError::generic("Invalid u32 bytes"))?;

        Ok(match endianness {
            Endianness::LittleEndian => u32::from_le_bytes(array),
            Endianness::BigEndian => u32::from_be_bytes(array),
        })
    }
}

impl FromBytes for f32 {
    fn from_bytes(bytes: &[u8], endianness: Endianness) -> Result<Self> {
        let array: [u8; 4] = bytes
            .try_into()
            .map_err(|_| ReaderError::generic("Invalid f32 bytes"))?;

        Ok(match endianness {
            Endianness::LittleEndian => f32::from_le_bytes(array),
            Endianness::BigEndian => f32::from_be_bytes(array),
        })
    }
}

impl FromBytes for f64 {
    fn from_bytes(bytes: &[u8], endianness: Endianness) -> Result<Self> {
        let array: [u8; 8] = bytes
            .try_into()
            .map_err(|_| ReaderError::generic("Invalid f64 bytes"))?;

        Ok(match endianness {
            Endianness::LittleEndian => f64::from_le_bytes(array),
            Endianness::BigEndian => f64::from_be_bytes(array),
        })
    }
}

#[cfg(test)]
mod tests {

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    use super::*;

    fn le_header() -> Vec<u8> {
        vec![0x00, 0x01, 0x00, 0x00]
    }

    fn be_header() -> Vec<u8> {
        vec![0x00, 0x00, 0x00, 0x00]
    }

    // ── CdrHeader::parse ────────────────────────────────────────────────────

    #[test]
    fn test_cdr_header_parsing() -> TestResult {
        let header_le = CdrHeader::parse(&[0x00, 0x01, 0x00, 0x00])?;
        assert_eq!(header_le.endianness, Endianness::LittleEndian);
        assert_eq!(header_le.encapsulation_kind, 0);

        let header_be = CdrHeader::parse(&[0x00, 0x00, 0x00, 0x00])?;
        assert_eq!(header_be.endianness, Endianness::BigEndian);
        Ok(())
    }

    #[test]
    fn cdr_header_wrong_length_returns_err() -> TestResult {
        assert!(CdrHeader::parse(&[0x00, 0x01, 0x00]).is_err());
        assert!(CdrHeader::parse(&[]).is_err());
        assert!(CdrHeader::parse(&[0x00, 0x01, 0x00, 0x00, 0x00]).is_err());
        Ok(())
    }

    #[test]
    fn cdr_header_invalid_endian_flag_returns_err() -> TestResult {
        assert!(CdrHeader::parse(&[0x00, 0x02, 0x00, 0x00]).is_err());
        Ok(())
    }

    // ── CdrDeserializer::new ────────────────────────────────────────────────

    #[test]
    fn deserializer_new_too_short_returns_err() -> TestResult {
        assert!(CdrDeserializer::new(&[0x00, 0x01, 0x00]).is_err());
        assert!(CdrDeserializer::new(&[]).is_err());
        Ok(())
    }

    #[test]
    fn deserializer_new_le_header_ok() -> TestResult {
        let data = le_header();
        let d = CdrDeserializer::new(&data)?;
        assert_eq!(d.position(), 4);
        assert_eq!(d.data_len(), 4);
        Ok(())
    }

    #[test]
    fn deserializer_new_be_header_ok() -> TestResult {
        let data = be_header();
        let d = CdrDeserializer::new(&data)?;
        assert_eq!(d.position(), 4);
        Ok(())
    }

    // ── position / data_len / has_remaining ────────────────────────────────

    #[test]
    fn has_remaining_true_and_false() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&[0x01, 0x02]);
        let d = CdrDeserializer::new(&data)?;
        assert!(d.has_remaining(2));
        assert!(d.has_remaining(1));
        assert!(!d.has_remaining(3));
        Ok(())
    }

    #[test]
    fn data_len_matches_slice() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&[0xAA; 8]);
        let d = CdrDeserializer::new(&data)?;
        assert_eq!(d.data_len(), 12);
        Ok(())
    }

    #[test]
    fn data_accessor_returns_full_slice() -> TestResult {
        let data = le_header();
        let d = CdrDeserializer::new(&data)?;
        assert_eq!(d.data(), &data[..]);
        Ok(())
    }

    // ── read_primitive tests ────────────────────────────────────────────────

    #[test]
    fn test_primitive_deserialization() -> TestResult {
        let data = [
            0x00, 0x01, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
        ];
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_i32()?, 42);
        assert_eq!(d.read_u32()?, 0x04030201);
        Ok(())
    }

    #[test]
    fn read_u8_le() -> TestResult {
        let mut data = le_header();
        data.push(0xAB);
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_u8()?, 0xAB);
        Ok(())
    }

    #[test]
    fn read_i8_le() -> TestResult {
        let mut data = le_header();
        data.push(0xFF); // -1 as i8
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_i8()?, -1i8);
        Ok(())
    }

    #[test]
    fn read_u16_le() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&300u16.to_le_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_u16()?, 300);
        Ok(())
    }

    #[test]
    fn read_u16_be() -> TestResult {
        let mut data = be_header();
        data.extend_from_slice(&300u16.to_be_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_u16()?, 300);
        Ok(())
    }

    #[test]
    fn read_i32_be() -> TestResult {
        let mut data = be_header();
        data.extend_from_slice(&(-99i32).to_be_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_i32()?, -99);
        Ok(())
    }

    #[test]
    fn read_u32_be() -> TestResult {
        let mut data = be_header();
        data.extend_from_slice(&0xDEADBEEFu32.to_be_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_u32()?, 0xDEADBEEF);
        Ok(())
    }

    #[test]
    fn read_f64_le() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&[0u8; 4]); // 4 pad bytes to align f64 to 8
        data.extend_from_slice(&1.5f64.to_le_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        // pos starts at 4, align(8) moves to 8
        assert!((d.read_f64()? - 1.5).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn read_f32_le() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&2.5f32.to_le_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        assert!((d.read_f32()? - 2.5f32).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn read_bool_true_and_false() -> TestResult {
        let mut data = le_header();
        data.push(0x01);
        data.push(0x00);
        let mut d = CdrDeserializer::new(&data)?;
        assert!(d.read_bool()?);
        assert!(!d.read_bool()?);
        Ok(())
    }

    // ── read_string ────────────────────────────────────────────────────────

    #[test]
    fn read_string_empty() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&0u32.to_le_bytes()); // length = 0
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_string()?, "");
        Ok(())
    }

    #[test]
    fn read_string_with_null_terminator() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&5u32.to_le_bytes()); // length = 5 (includes \0)
        data.extend_from_slice(b"hello\0"); // but only 5 bytes
        // Actually length=5 and we have "hell\0" = 5 bytes
        let mut data2 = le_header();
        data2.extend_from_slice(&5u32.to_le_bytes());
        data2.extend_from_slice(b"hell\0");
        let mut d = CdrDeserializer::new(&data2)?;
        assert_eq!(d.read_string()?, "hell");
        Ok(())
    }

    #[test]
    fn read_string_without_null_terminator() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&3u32.to_le_bytes()); // length = 3, no null
        data.extend_from_slice(b"abc"); // not null-terminated → keep all bytes
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_string()?, "abc");
        Ok(())
    }

    // ── read_f64_array ─────────────────────────────────────────────────────

    #[test]
    fn read_f64_array_3() -> TestResult {
        // f64 uses align(8), starting from pos 4. align to 8 → pos 8 (4 pad bytes needed).
        let mut data = le_header();
        data.extend_from_slice(&[0u8; 4]); // padding to reach 8-byte alignment
        for &v in &[1.0f64, 2.0, 3.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut d = CdrDeserializer::new(&data)?;
        let arr = d.read_f64_array::<3>()?;
        assert!((arr[0] - 1.0).abs() < 1e-12);
        assert!((arr[1] - 2.0).abs() < 1e-12);
        assert!((arr[2] - 3.0).abs() < 1e-12);
        Ok(())
    }

    // ── read_sequence ──────────────────────────────────────────────────────

    #[test]
    fn read_sequence_of_u32() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&3u32.to_le_bytes()); // count = 3
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(&30u32.to_le_bytes());
        let mut d = CdrDeserializer::new(&data)?;
        let seq = d.read_sequence(|d| d.read_u32())?;
        assert_eq!(seq, vec![10, 20, 30]);
        Ok(())
    }

    #[test]
    fn read_sequence_empty() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&0u32.to_le_bytes()); // count = 0
        let mut d = CdrDeserializer::new(&data)?;
        let seq = d.read_sequence(|d| d.read_u32())?;
        assert!(seq.is_empty());
        Ok(())
    }

    // ── read_byte_sequence ────────────────────────────────────────────────

    #[test]
    fn read_byte_sequence_basic() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&4u32.to_le_bytes()); // length = 4
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let mut d = CdrDeserializer::new(&data)?;
        assert_eq!(d.read_byte_sequence()?, &[0xAA, 0xBB, 0xCC, 0xDD]);
        Ok(())
    }

    // ── truncation errors ─────────────────────────────────────────────────

    #[test]
    fn read_i32_truncated_returns_err() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&[0x01, 0x02]); // only 2 bytes, need 4
        let mut d = CdrDeserializer::new(&data)?;
        assert!(d.read_i32().is_err());
        Ok(())
    }

    #[test]
    fn read_f64_truncated_returns_err() -> TestResult {
        let mut data = le_header();
        data.extend_from_slice(&[0u8; 4]); // padding
        data.extend_from_slice(&[0x01, 0x02, 0x03]); // only 3 bytes, need 8
        let mut d = CdrDeserializer::new(&data)?;
        assert!(d.read_f64().is_err());
        Ok(())
    }
}
