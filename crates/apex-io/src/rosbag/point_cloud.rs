//! Reading named fields out of a `sensor_msgs/PointCloud2` data blob — the
//! same for ROS1 and ROS2, which share the layout description.
//!
//! A cloud is `point_count` records of `point_step` bytes; each
//! [`PointField`] names a scalar at a byte `offset` with a ROS `datatype`
//! code. [`field_values`] widens any numeric field to `f64`, so a LiDAR
//! driver's `timestamp` (often `FLOAT64`) and `x`/`y`/`z` (`FLOAT32`) read
//! through one call.

use super::error::{BagError, Result};
use super::messages::PointField;

/// `sensor_msgs/PointField` datatype codes.
pub mod datatype {
    pub const INT8: u8 = 1;
    pub const UINT8: u8 = 2;
    pub const INT16: u8 = 3;
    pub const UINT16: u8 = 4;
    pub const INT32: u8 = 5;
    pub const UINT32: u8 = 6;
    pub const FLOAT32: u8 = 7;
    pub const FLOAT64: u8 = 8;
}

/// Size in bytes of a field of `datatype`, `None` for an unknown code.
pub fn datatype_size(code: u8) -> Option<usize> {
    match code {
        datatype::INT8 | datatype::UINT8 => Some(1),
        datatype::INT16 | datatype::UINT16 => Some(2),
        datatype::INT32 | datatype::UINT32 | datatype::FLOAT32 => Some(4),
        datatype::FLOAT64 => Some(8),
        _ => None,
    }
}

fn invalid(reason: String) -> BagError {
    BagError::InvalidMessageData { reason }
}

/// The values of field `name` for every point, widened to `f64`.
///
/// # Errors
///
/// [`BagError::InvalidMessageData`] when the field is missing, has an unknown
/// datatype, does not fit in `point_step`, or `data` is shorter than
/// `point_count · point_step`.
pub fn field_values(
    fields: &[PointField],
    name: &str,
    data: &[u8],
    point_step: usize,
    point_count: usize,
    big_endian: bool,
) -> Result<Vec<f64>> {
    let field = fields
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| invalid(format!("point cloud has no field `{name}`")))?;
    let size = datatype_size(field.datatype).ok_or_else(|| {
        invalid(format!(
            "field `{name}` has unknown datatype {}",
            field.datatype
        ))
    })?;
    let offset = field.offset as usize;
    if offset + size > point_step {
        return Err(invalid(format!(
            "field `{name}` at offset {offset} does not fit a {point_step}-byte point"
        )));
    }
    let needed = point_count
        .checked_mul(point_step)
        .ok_or_else(|| invalid("point cloud size overflows".to_string()))?;
    if data.len() < needed {
        return Err(invalid(format!(
            "point cloud data has {} bytes, {point_count} points of {point_step} need {needed}",
            data.len()
        )));
    }
    let decode = |bytes: &[u8]| -> f64 {
        let mut buffer = [0u8; 8];
        buffer[..bytes.len()].copy_from_slice(bytes);
        if big_endian {
            buffer[..bytes.len()].reverse();
        }
        match field.datatype {
            datatype::INT8 => f64::from(buffer[0] as i8),
            datatype::UINT8 => f64::from(buffer[0]),
            datatype::INT16 => f64::from(i16::from_le_bytes([buffer[0], buffer[1]])),
            datatype::UINT16 => f64::from(u16::from_le_bytes([buffer[0], buffer[1]])),
            datatype::INT32 => f64::from(i32::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3],
            ])),
            datatype::UINT32 => f64::from(u32::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3],
            ])),
            datatype::FLOAT32 => f64::from(f32::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3],
            ])),
            _ => f64::from_le_bytes(buffer),
        }
    };
    Ok((0..point_count)
        .map(|index| {
            let start = index * point_step + offset;
            decode(&data[start..start + size])
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, offset: u32, datatype: u8) -> PointField {
        PointField {
            name: name.to_string(),
            offset,
            datatype,
            count: 1,
        }
    }

    /// Two points laid out like a Hesai driver: x, y, z f32, time f64, ring u16.
    fn cloud(big_endian: bool) -> (Vec<PointField>, Vec<u8>) {
        let fields = vec![
            field("x", 0, datatype::FLOAT32),
            field("y", 4, datatype::FLOAT32),
            field("z", 8, datatype::FLOAT32),
            field("timestamp", 16, datatype::FLOAT64),
            field("ring", 24, datatype::UINT16),
            field("tag", 26, datatype::INT8),
        ];
        let mut data = Vec::new();
        for (i, (x, t, ring, tag)) in [(1.5f32, 100.25f64, 7u16, -3i8), (-2.0, 100.35, 31, 4)]
            .into_iter()
            .enumerate()
        {
            let mut record = vec![0u8; 32];
            let put = |record: &mut Vec<u8>, at: usize, bytes: &[u8]| {
                let mut b = bytes.to_vec();
                if big_endian {
                    b.reverse();
                }
                record[at..at + b.len()].copy_from_slice(&b);
            };
            put(&mut record, 0, &x.to_le_bytes());
            put(&mut record, 4, &(i as f32).to_le_bytes());
            put(&mut record, 8, &0.5f32.to_le_bytes());
            put(&mut record, 16, &t.to_le_bytes());
            put(&mut record, 24, &ring.to_le_bytes());
            put(&mut record, 26, &tag.to_le_bytes());
            data.extend(record);
        }
        (fields, data)
    }

    #[test]
    fn every_numeric_type_reads_in_both_byte_orders() -> Result<()> {
        for big in [false, true] {
            let (fields, data) = cloud(big);
            assert_eq!(
                field_values(&fields, "x", &data, 32, 2, big)?,
                vec![1.5, -2.0]
            );
            assert_eq!(
                field_values(&fields, "y", &data, 32, 2, big)?,
                vec![0.0, 1.0]
            );
            assert_eq!(
                field_values(&fields, "timestamp", &data, 32, 2, big)?,
                vec![100.25, 100.35]
            );
            assert_eq!(
                field_values(&fields, "ring", &data, 32, 2, big)?,
                vec![7.0, 31.0]
            );
            assert_eq!(
                field_values(&fields, "tag", &data, 32, 2, big)?,
                vec![-3.0, 4.0]
            );
        }
        Ok(())
    }

    #[test]
    fn malformed_clouds_are_errors() {
        let (fields, data) = cloud(false);
        assert!(field_values(&fields, "intensity", &data, 32, 2, false).is_err());
        assert!(field_values(&fields, "x", &data[..40], 32, 2, false).is_err());
        assert!(field_values(&fields, "timestamp", &data, 20, 2, false).is_err());
        assert!(field_values(&[field("x", 0, 9)], "x", &data, 32, 2, false).is_err());
        assert!(field_values(&fields, "x", &data, usize::MAX, 2, false).is_err());
        assert_eq!(datatype_size(datatype::FLOAT64), Some(8));
        assert_eq!(datatype_size(0), None);
    }
}
