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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_cdr_header_parsing() {
        // Little endian header
        let header_le = CdrHeader::parse(&[0x00, 0x01, 0x00, 0x00]).unwrap();
        assert_eq!(header_le.endianness, Endianness::LittleEndian);
        assert_eq!(header_le.encapsulation_kind, 0);

        // Big endian header
        let header_be = CdrHeader::parse(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        assert_eq!(header_be.endianness, Endianness::BigEndian);
        assert_eq!(header_be.encapsulation_kind, 0);
    }

    #[test]
    fn test_primitive_deserialization() {
        // Test data with little endian header + some values
        let data = [
            0x00, 0x01, 0x00, 0x00, // CDR header (little endian)
            0x2A, 0x00, 0x00, 0x00, // i32: 42
            0x01, 0x02, 0x03, 0x04, // u32: 0x04030201
        ];

        let mut deserializer = CdrDeserializer::new(&data).unwrap();

        let int_val = deserializer.read_i32().unwrap();
        assert_eq!(int_val, 42);

        let uint_val = deserializer.read_u32().unwrap();
        assert_eq!(uint_val, 0x04030201);
    }
}
