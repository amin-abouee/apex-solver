//! ROS1 bag v2.0 format constants.
//!
//! Reference: <http://wiki.ros.org/Bags/Format/2.0>

/// First line of every v2.0 ROS1 bag, including trailing newline.
pub const MAGIC: &[u8] = b"#ROSBAG V2.0\n";

/// Record op codes as defined by the ROS1 bag v2.0 spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    /// Bag file header.
    BagHeader = 0x03,
    /// Chunk container.
    Chunk = 0x05,
    /// Connection (topic + message type) declaration.
    Connection = 0x07,
    /// Message data record.
    MessageData = 0x02,
    /// Index entries for a chunk's messages.
    IndexData = 0x04,
    /// Chunk summary record.
    ChunkInfo = 0x06,
}

impl OpCode {
    /// Parse an op-code byte.
    pub fn from_u8(op: u8) -> Option<Self> {
        Some(match op {
            0x03 => Self::BagHeader,
            0x05 => Self::Chunk,
            0x07 => Self::Connection,
            0x02 => Self::MessageData,
            0x04 => Self::IndexData,
            0x06 => Self::ChunkInfo,
            _ => return None,
        })
    }

    /// Raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Chunk-payload compression as encoded in the `compression=` header field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ros1Compression {
    /// Stored uncompressed.
    #[default]
    None,
    /// Compressed with bzip2.
    Bz2,
    /// Compressed with LZ4 (block format).
    Lz4,
}

impl Ros1Compression {
    /// On-wire identifier in the chunk record header.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bz2 => "bz2",
            Self::Lz4 => "lz4",
        }
    }

    /// Parse the on-wire identifier.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "none" => Self::None,
            "bz2" => Self::Bz2,
            "lz4" => Self::Lz4,
            _ => return None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn op_code_round_trip() {
        for op in [
            OpCode::BagHeader,
            OpCode::Chunk,
            OpCode::Connection,
            OpCode::MessageData,
            OpCode::IndexData,
            OpCode::ChunkInfo,
        ] {
            assert_eq!(OpCode::from_u8(op.as_u8()), Some(op));
        }
    }

    #[test]
    fn op_code_unknown_returns_none() {
        assert_eq!(OpCode::from_u8(0xff), None);
    }

    #[test]
    fn compression_round_trip() {
        for c in [
            Ros1Compression::None,
            Ros1Compression::Bz2,
            Ros1Compression::Lz4,
        ] {
            assert_eq!(Ros1Compression::parse(c.as_str()), Some(c));
        }
    }

    #[test]
    fn magic_is_v2() {
        assert_eq!(MAGIC, b"#ROSBAG V2.0\n");
    }
}
