//! ROS2 message type definitions
//!
//! This module contains Rust definitions for common ROS2 message types
//! that match the official ROS2 API specifications.

use super::cdr::{CdrDeserializer, CdrResult as Result};
use super::error::{Ros2BagError, Ros2BagResult};
use crate::rosbag::point_cloud::PointCloudResult;

/// builtin_interfaces/msg/Time
#[derive(Debug, Clone, PartialEq)]
pub struct Time {
    pub sec: i32,
    pub nanosec: u32,
}

/// std_msgs/msg/Header
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub stamp: Time,
    pub frame_id: String,
}

/// geometry_msgs/msg/Vector3
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// geometry_msgs/msg/Quaternion
#[derive(Debug, Clone, PartialEq)]
pub struct Quaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

/// geometry_msgs/msg/Point
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// geometry_msgs/msg/Pose
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Pose {
    pub position: Point,
    pub orientation: Quaternion,
}

/// geometry_msgs/msg/PoseWithCovariance
#[derive(Debug, Clone, PartialEq)]
pub struct PoseWithCovariance {
    pub pose: Pose,
    pub covariance: [f64; 36],
}

impl Default for PoseWithCovariance {
    fn default() -> Self {
        Self {
            pose: Default::default(),
            covariance: [0.0; 36],
        }
    }
}

/// geometry_msgs/msg/PoseWithCovarianceStamped
#[derive(Debug, Clone, PartialEq)]
pub struct PoseWithCovarianceStamped {
    pub header: Header,
    pub pose: PoseWithCovariance,
}

/// geometry_msgs/msg/Transform
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub translation: Vector3,
    pub rotation: Quaternion,
}

/// geometry_msgs/msg/TransformStamped
#[derive(Debug, Clone, PartialEq)]
pub struct TransformStamped {
    pub header: Header,
    pub child_frame_id: String,
    pub transform: Transform,
}

/// geometry_msgs/msg/Twist
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Twist {
    pub linear: Vector3,
    pub angular: Vector3,
}

/// geometry_msgs/msg/TwistWithCovariance
#[derive(Debug, Clone, PartialEq)]
pub struct TwistWithCovariance {
    pub twist: Twist,
    pub covariance: [f64; 36],
}

impl Default for TwistWithCovariance {
    fn default() -> Self {
        Self {
            twist: Default::default(),
            covariance: [0.0; 36],
        }
    }
}

/// sensor_msgs/msg/Imu
#[derive(Debug, Clone, PartialEq)]
pub struct Imu {
    pub header: Header,
    pub orientation: Quaternion,
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: Vector3,
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: Vector3,
    pub linear_acceleration_covariance: [f64; 9],
}

/// nav_msgs/msg/Odometry
#[derive(Debug, Clone, PartialEq)]
pub struct Odometry {
    pub header: Header,
    pub child_frame_id: String,
    pub pose: PoseWithCovariance,
    pub twist: TwistWithCovariance,
}

/// geometry_msgs/msg/PoseStamped
#[derive(Debug, Clone, PartialEq)]
pub struct PoseStamped {
    pub header: Header,
    pub pose: Pose,
}

/// geometry_msgs/msg/PointStamped
#[derive(Debug, Clone, PartialEq)]
pub struct PointStamped {
    pub header: Header,
    pub point: Point,
}

/// nav_msgs/msg/Path
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub header: Header,
    pub poses: Vec<PoseStamped>,
}

/// sensor_msgs/msg/NavSatStatus
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NavSatStatus {
    pub status: i8,
    pub service: u16,
}

/// sensor_msgs/msg/NavSatFix
#[derive(Debug, Clone, PartialEq)]
pub struct NavSatFix {
    pub header: Header,
    pub status: NavSatStatus,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub position_covariance: [f64; 9],
    pub position_covariance_type: u8,
}

/// std_msgs/msg/String
#[derive(Debug, Clone, PartialEq)]
pub struct StdString {
    pub data: String,
}

/// sensor_msgs/msg/PointField
#[derive(Debug, Clone, PartialEq)]
pub struct PointField {
    pub name: String,
    pub offset: u32,
    pub datatype: u8,
    pub count: u32,
}

/// sensor_msgs/msg/PointCloud2
#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud2 {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub fields: Vec<PointField>,
    pub is_bigendian: bool,
    pub point_step: u32,
    pub row_step: u32,
    pub data: Vec<u8>,
    pub is_dense: bool,
}

/// sensor_msgs/msg/Image
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub encoding: String,
    pub is_bigendian: u8,
    pub step: u32,
    pub data: Vec<u8>,
}

/// geometry_msgs/msg/Point32
#[derive(Debug, Clone, PartialEq)]
pub struct Point32 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// std_msgs/msg/ColorRGBA
#[derive(Debug, Clone, PartialEq)]
pub struct ColorRGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// builtin_interfaces/msg/Duration
#[derive(Debug, Clone, PartialEq)]
pub struct Duration {
    pub sec: i32,
    pub nanosec: u32,
}

/// Read an `f64` with CDR's alignment applied.
///
/// This used to read eight raw bytes from the current position, documented as
/// "without automatic alignment" — so every `geometry_msgs` type built on it
/// (`Vector3`, `Quaternion`, `Point`, `Twist`, `Imu`, ...) skipped the padding
/// CDR requires before an 8-byte field. After a variable-length field such as
/// `header.frame_id` that reads from inside the padding gap, which is how
/// coordinates decoded as ~1e200 while the 4-byte timestamps beside them
/// survived.
///
/// It also was not faster: it went byte at a time through `read_u8`, where
/// [`CdrDeserializer::read_f64`] copies the slice once.
///
/// Kept as a named function so the call sites stay readable and so this note
/// has somewhere to live. See `tests/ros2_cdr_alignment_tests.rs`.
fn read_f64_manual(deserializer: &mut CdrDeserializer) -> Result<f64> {
    deserializer.read_f64()
}

/// Helper function to manually read f64 array without automatic alignment
fn read_f64_array_manual<const N: usize>(deserializer: &mut CdrDeserializer) -> Result<[f64; N]> {
    let mut array = [0.0; N];
    for item in array.iter_mut().take(N) {
        *item = read_f64_manual(deserializer)?;
    }
    Ok(array)
}

/// Trait for deserializing ROS2 messages from CDR data
pub trait FromCdr: Sized {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self>;
}

impl FromCdr for Time {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            sec: deserializer.read_i32()?,
            nanosec: deserializer.read_u32()?,
        })
    }
}

impl FromCdr for Header {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            stamp: Time::from_cdr(deserializer)?,
            frame_id: deserializer.read_string()?,
        })
    }
}

impl FromCdr for Vector3 {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            x: read_f64_manual(deserializer)?,
            y: read_f64_manual(deserializer)?,
            z: read_f64_manual(deserializer)?,
        })
    }
}

impl FromCdr for Quaternion {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            x: read_f64_manual(deserializer)?,
            y: read_f64_manual(deserializer)?,
            z: read_f64_manual(deserializer)?,
            w: read_f64_manual(deserializer)?,
        })
    }
}

impl FromCdr for Point {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            x: read_f64_manual(deserializer)?,
            y: read_f64_manual(deserializer)?,
            z: read_f64_manual(deserializer)?,
        })
    }
}

impl FromCdr for Pose {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            position: Point::from_cdr(deserializer)?,
            orientation: Quaternion::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for Transform {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            translation: Vector3::from_cdr(deserializer)?,
            rotation: Quaternion::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for TransformStamped {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            child_frame_id: deserializer.read_string()?,
            transform: Transform::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for Imu {
    /// `sensor_msgs/msg/Imu` in spec order; a short message is an error, never
    /// a message padded with zeros.
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            orientation: Quaternion::from_cdr(deserializer)?,
            orientation_covariance: deserializer.read_f64_array()?,
            angular_velocity: Vector3::from_cdr(deserializer)?,
            angular_velocity_covariance: deserializer.read_f64_array()?,
            linear_acceleration: Vector3::from_cdr(deserializer)?,
            linear_acceleration_covariance: deserializer.read_f64_array()?,
        })
    }
}

impl FromCdr for PoseWithCovariance {
    /// `geometry_msgs/msg/PoseWithCovariance`: the pose, then all 36
    /// covariance entries; a truncated covariance is an error.
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            pose: Pose::from_cdr(deserializer)?,
            covariance: read_f64_array_manual(deserializer)?,
        })
    }
}

impl FromCdr for PoseWithCovarianceStamped {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            pose: PoseWithCovariance::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for PointStamped {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            point: Point::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for NavSatStatus {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            status: deserializer.read_i8()?,
            service: deserializer.read_u16()?,
        })
    }
}

impl FromCdr for NavSatFix {
    /// `sensor_msgs/msg/NavSatFix` in spec order: the status and covariance
    /// are read, not invented.
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            status: NavSatStatus::from_cdr(deserializer)?,
            latitude: deserializer.read_f64()?,
            longitude: deserializer.read_f64()?,
            altitude: deserializer.read_f64()?,
            position_covariance: deserializer.read_f64_array()?,
            position_covariance_type: deserializer.read_u8()?,
        })
    }
}

impl FromCdr for Twist {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            linear: Vector3::from_cdr(deserializer)?,
            angular: Vector3::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for TwistWithCovariance {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            twist: Twist::from_cdr(deserializer)?,
            covariance: read_f64_array_manual(deserializer)?,
        })
    }
}

impl FromCdr for Odometry {
    /// `nav_msgs/msg/Odometry` in spec order. A message that does not decode
    /// is an error: no stamp, frame or pose is invented for it.
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            child_frame_id: deserializer.read_string()?,
            pose: PoseWithCovariance::from_cdr(deserializer)?,
            twist: TwistWithCovariance::from_cdr(deserializer)?,
        })
    }
}

impl FromCdr for StdString {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            data: deserializer.read_string()?,
        })
    }
}

impl FromCdr for PointField {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            name: deserializer.read_string()?,
            offset: deserializer.read_u32()?,
            datatype: deserializer.read_u8()?,
            count: deserializer.read_u32()?,
        })
    }
}

impl PointCloud2 {
    /// Number of points, `height · width`.
    pub fn point_count(&self) -> usize {
        self.height as usize * self.width as usize
    }

    /// The values of field `name` for every point, widened to `f64` — see
    /// [`crate::rosbag::point_cloud::field_values`].
    pub fn field_values(&self, name: &str) -> PointCloudResult<Vec<f64>> {
        crate::rosbag::point_cloud::field_values(
            &self.fields,
            name,
            &self.data,
            self.point_step as usize,
            self.point_count(),
            self.is_bigendian,
        )
    }
}

impl FromCdr for PointCloud2 {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            height: deserializer.read_u32()?,
            width: deserializer.read_u32()?,
            fields: deserializer.read_sequence(|d| PointField::from_cdr(d))?,
            is_bigendian: deserializer.read_bool()?,
            point_step: deserializer.read_u32()?,
            row_step: deserializer.read_u32()?,
            data: deserializer.read_byte_sequence()?,
            is_dense: deserializer.read_bool()?,
        })
    }
}

impl FromCdr for Image {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            header: Header::from_cdr(deserializer)?,
            height: deserializer.read_u32()?,
            width: deserializer.read_u32()?,
            encoding: deserializer.read_string()?,
            is_bigendian: deserializer.read_u8()?,
            step: deserializer.read_u32()?,
            data: deserializer.read_byte_sequence()?,
        })
    }
}

impl FromCdr for Point32 {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            x: deserializer.read_f32()?,
            y: deserializer.read_f32()?,
            z: deserializer.read_f32()?,
        })
    }
}

impl FromCdr for ColorRGBA {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            r: deserializer.read_f32()?,
            g: deserializer.read_f32()?,
            b: deserializer.read_f32()?,
            a: deserializer.read_f32()?,
        })
    }
}

impl FromCdr for Duration {
    fn from_cdr(deserializer: &mut CdrDeserializer) -> Result<Self> {
        Ok(Self {
            sec: deserializer.read_i32()?,
            nanosec: deserializer.read_u32()?,
        })
    }
}

/// A message decoded by [`deserialize_message`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DecodedMessage {
    Imu(Box<Imu>),
    TransformStamped(Box<TransformStamped>),
    PoseWithCovarianceStamped(Box<PoseWithCovarianceStamped>),
    PointStamped(Box<PointStamped>),
    NavSatFix(Box<NavSatFix>),
    Odometry(Box<Odometry>),
}

/// Deserialize a message from CDR data based on its type name.
///
/// # Errors
///
/// [`Ros2BagError::UnsupportedMessageType`] for a type this function does not
/// decode, or the [`CdrError`](super::cdr::CdrError) of a payload that does
/// not decode as its type.
pub fn deserialize_message(data: &[u8], message_type: &str) -> Ros2BagResult<DecodedMessage> {
    let mut deserializer = CdrDeserializer::new(data)?;
    let d = &mut deserializer;
    Ok(match message_type {
        "sensor_msgs/msg/Imu" => DecodedMessage::Imu(Box::new(Imu::from_cdr(d)?)),
        "geometry_msgs/msg/TransformStamped" => {
            DecodedMessage::TransformStamped(Box::new(TransformStamped::from_cdr(d)?))
        }
        "geometry_msgs/msg/PoseWithCovarianceStamped" => DecodedMessage::PoseWithCovarianceStamped(
            Box::new(PoseWithCovarianceStamped::from_cdr(d)?),
        ),
        "geometry_msgs/msg/PointStamped" => {
            DecodedMessage::PointStamped(Box::new(PointStamped::from_cdr(d)?))
        }
        "sensor_msgs/msg/NavSatFix" => DecodedMessage::NavSatFix(Box::new(NavSatFix::from_cdr(d)?)),
        "nav_msgs/msg/Odometry" => DecodedMessage::Odometry(Box::new(Odometry::from_cdr(d)?)),
        _ => {
            return Err(Ros2BagError::UnsupportedMessageType {
                message_type: message_type.to_string(),
            });
        }
    })
}

#[cfg(test)]
mod tests {

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    use super::*;

    // ── CDR byte-sequence builders ────────────────────────────────────────

    fn le_header() -> Vec<u8> {
        vec![0x00, 0x01, 0x00, 0x00]
    }

    /// Pad to `n`-byte alignment measured from the **body**, i.e. after the
    /// 4-byte encapsulation header — the rule CDR actually specifies.
    ///
    /// These helpers are a miniature spec-conformant encoder. They previously
    /// wrote every primitive at the current offset with no padding at all,
    /// which matched the decoder's own defect and so made these fixtures agree
    /// with the implementation rather than with the standard.
    fn pad_to(buf: &mut Vec<u8>, n: usize) {
        while !(buf.len() - 4).is_multiple_of(n) {
            buf.push(0x00);
        }
    }

    fn push_u32_le(buf: &mut Vec<u8>, v: u32) {
        pad_to(buf, 4);
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_i32_le(buf: &mut Vec<u8>, v: i32) {
        pad_to(buf, 4);
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_f64_le(buf: &mut Vec<u8>, v: f64) {
        pad_to(buf, 8);
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_f32_le(buf: &mut Vec<u8>, v: f32) {
        pad_to(buf, 4);
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, s: &str) {
        let with_null = format!("{s}\0");
        push_u32_le(buf, with_null.len() as u32);
        buf.extend_from_slice(with_null.as_bytes());
    }

    fn push_empty_string(buf: &mut Vec<u8>) {
        push_u32_le(buf, 0);
    }

    /// Align the buffer, measured from the body — see [`pad_to`].
    fn align_to(buf: &mut Vec<u8>, align: usize) {
        while !(buf.len() - 4).is_multiple_of(align) {
            buf.push(0x00);
        }
    }

    // ── Time ─────────────────────────────────────────────────────────────

    #[test]
    fn time_from_cdr() -> TestResult {
        let mut data = le_header();
        push_i32_le(&mut data, 1000);
        push_u32_le(&mut data, 500_000_000);
        let mut d = CdrDeserializer::new(&data)?;
        let t = Time::from_cdr(&mut d)?;
        assert_eq!(t.sec, 1000);
        assert_eq!(t.nanosec, 500_000_000);
        Ok(())
    }

    #[test]
    fn time_from_cdr_zero() -> TestResult {
        let mut data = le_header();
        push_i32_le(&mut data, 0);
        push_u32_le(&mut data, 0);
        let mut d = CdrDeserializer::new(&data)?;
        let t = Time::from_cdr(&mut d)?;
        assert_eq!(t.sec, 0);
        assert_eq!(t.nanosec, 0);
        Ok(())
    }

    // ── Header ───────────────────────────────────────────────────────────

    #[test]
    fn header_from_cdr_empty_frame_id() -> TestResult {
        let mut data = le_header();
        push_i32_le(&mut data, 10); // stamp.sec
        push_u32_le(&mut data, 0); // stamp.nanosec
        push_empty_string(&mut data); // frame_id = ""
        let mut d = CdrDeserializer::new(&data)?;
        let h = Header::from_cdr(&mut d)?;
        assert_eq!(h.stamp.sec, 10);
        assert_eq!(h.stamp.nanosec, 0);
        assert_eq!(h.frame_id, "");
        Ok(())
    }

    #[test]
    fn header_from_cdr_with_frame_id() -> TestResult {
        let mut data = le_header();
        push_i32_le(&mut data, 5); // sec
        push_u32_le(&mut data, 100); // nanosec
        push_string(&mut data, "map");
        let mut d = CdrDeserializer::new(&data)?;
        let h = Header::from_cdr(&mut d)?;
        assert_eq!(h.stamp.sec, 5);
        assert_eq!(h.frame_id, "map");
        Ok(())
    }

    // ── Vector3 ──────────────────────────────────────────────────────────

    #[test]
    fn vector3_from_cdr() -> TestResult {
        let mut data = le_header();
        push_f64_le(&mut data, 1.0);
        push_f64_le(&mut data, 2.0);
        push_f64_le(&mut data, 3.0);
        let mut d = CdrDeserializer::new(&data)?;
        let v = Vector3::from_cdr(&mut d)?;
        assert!((v.x - 1.0).abs() < 1e-12);
        assert!((v.y - 2.0).abs() < 1e-12);
        assert!((v.z - 3.0).abs() < 1e-12);
        Ok(())
    }

    // ── Quaternion ───────────────────────────────────────────────────────

    #[test]
    fn quaternion_from_cdr_identity() -> TestResult {
        let mut data = le_header();
        push_f64_le(&mut data, 0.0); // x
        push_f64_le(&mut data, 0.0); // y
        push_f64_le(&mut data, 0.0); // z
        push_f64_le(&mut data, 1.0); // w
        let mut d = CdrDeserializer::new(&data)?;
        let q = Quaternion::from_cdr(&mut d)?;
        assert!((q.w - 1.0).abs() < 1e-12);
        assert!((q.x - 0.0).abs() < 1e-12);
        Ok(())
    }

    // ── Point ────────────────────────────────────────────────────────────

    #[test]
    fn point_from_cdr() -> TestResult {
        let mut data = le_header();
        push_f64_le(&mut data, -1.5);
        push_f64_le(&mut data, 2.5);
        push_f64_le(&mut data, 0.0);
        let mut d = CdrDeserializer::new(&data)?;
        let p = Point::from_cdr(&mut d)?;
        assert!((p.x - (-1.5)).abs() < 1e-12);
        assert!((p.y - 2.5).abs() < 1e-12);
        Ok(())
    }

    // ── Pose ─────────────────────────────────────────────────────────────

    #[test]
    fn pose_from_cdr() -> TestResult {
        let mut data = le_header();
        // position
        push_f64_le(&mut data, 1.0);
        push_f64_le(&mut data, 2.0);
        push_f64_le(&mut data, 3.0);
        // orientation
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 1.0);
        let mut d = CdrDeserializer::new(&data)?;
        let pose = Pose::from_cdr(&mut d)?;
        assert!((pose.position.x - 1.0).abs() < 1e-12);
        assert!((pose.orientation.w - 1.0).abs() < 1e-12);
        Ok(())
    }

    // ── Transform ────────────────────────────────────────────────────────

    #[test]
    fn transform_from_cdr() -> TestResult {
        let mut data = le_header();
        // translation
        push_f64_le(&mut data, 5.0);
        push_f64_le(&mut data, 6.0);
        push_f64_le(&mut data, 7.0);
        // rotation
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 1.0);
        let mut d = CdrDeserializer::new(&data)?;
        let t = Transform::from_cdr(&mut d)?;
        assert!((t.translation.x - 5.0).abs() < 1e-12);
        assert!((t.rotation.w - 1.0).abs() < 1e-12);
        Ok(())
    }

    // ── NavSatStatus ─────────────────────────────────────────────────────

    #[test]
    fn nav_sat_status_from_cdr() -> TestResult {
        let mut data = le_header();
        data.push(2u8); // status = 2 (i8)
        data.push(0x00); // align u16 to 2
        data.extend_from_slice(&3u16.to_le_bytes()); // service = 3
        let mut d = CdrDeserializer::new(&data)?;
        let s = NavSatStatus::from_cdr(&mut d)?;
        assert_eq!(s.status, 2);
        assert_eq!(s.service, 3);
        Ok(())
    }

    // ── StdString ────────────────────────────────────────────────────────

    #[test]
    fn std_string_from_cdr() -> TestResult {
        let mut data = le_header();
        push_string(&mut data, "hello world");
        let mut d = CdrDeserializer::new(&data)?;
        let s = StdString::from_cdr(&mut d)?;
        assert_eq!(s.data, "hello world");
        Ok(())
    }

    // ── Duration ─────────────────────────────────────────────────────────

    #[test]
    fn duration_from_cdr() -> TestResult {
        let mut data = le_header();
        push_i32_le(&mut data, 10);
        push_u32_le(&mut data, 999);
        let mut d = CdrDeserializer::new(&data)?;
        let dur = Duration::from_cdr(&mut d)?;
        assert_eq!(dur.sec, 10);
        assert_eq!(dur.nanosec, 999);
        Ok(())
    }

    // ── Twist ────────────────────────────────────────────────────────────

    #[test]
    fn twist_from_cdr() -> TestResult {
        let mut data = le_header();
        // linear
        push_f64_le(&mut data, 1.0);
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.0);
        // angular
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.0);
        push_f64_le(&mut data, 0.5);
        let mut d = CdrDeserializer::new(&data)?;
        let t = Twist::from_cdr(&mut d)?;
        assert!((t.linear.x - 1.0).abs() < 1e-12);
        assert!((t.angular.z - 0.5).abs() < 1e-12);
        Ok(())
    }

    // ── Point32 / ColorRGBA ──────────────────────────────────────────────

    #[test]
    fn point32_from_cdr() -> TestResult {
        let mut data = le_header();
        push_f32_le(&mut data, 1.0f32);
        push_f32_le(&mut data, 2.0f32);
        push_f32_le(&mut data, 3.0f32);
        let mut d = CdrDeserializer::new(&data)?;
        let p = Point32::from_cdr(&mut d)?;
        assert!((p.x - 1.0f32).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn color_rgba_from_cdr() -> TestResult {
        let mut data = le_header();
        push_f32_le(&mut data, 1.0f32);
        push_f32_le(&mut data, 0.5f32);
        push_f32_le(&mut data, 0.0f32);
        push_f32_le(&mut data, 0.8f32);
        let mut d = CdrDeserializer::new(&data)?;
        let c = ColorRGBA::from_cdr(&mut d)?;
        assert!((c.r - 1.0f32).abs() < 1e-6);
        assert!((c.a - 0.8f32).abs() < 1e-6);
        Ok(())
    }

    // ── Image ────────────────────────────────────────────────────────────

    #[test]
    fn image_from_cdr() -> TestResult {
        let mut data = le_header();
        // header: sec=1, nanosec=0, frame_id=""
        push_i32_le(&mut data, 1);
        push_u32_le(&mut data, 0);
        push_empty_string(&mut data); // frame_id
        // height=2, width=3
        push_u32_le(&mut data, 2);
        push_u32_le(&mut data, 3);
        // encoding="rgb8\0" (5 bytes)
        push_string(&mut data, "rgb8");
        // is_bigendian=0
        data.push(0u8);
        // step: align to 4
        align_to(&mut data, 4);
        push_u32_le(&mut data, 9); // step = width * 3
        // data: 6 bytes
        push_u32_le(&mut data, 6);
        data.extend_from_slice(&[1u8, 2, 3, 4, 5, 6]);

        let mut d = CdrDeserializer::new(&data)?;
        let img = Image::from_cdr(&mut d)?;
        assert_eq!(img.height, 2);
        assert_eq!(img.width, 3);
        assert_eq!(img.encoding, "rgb8");
        assert_eq!(img.is_bigendian, 0);
        assert_eq!(img.step, 9);
        assert_eq!(img.data, vec![1, 2, 3, 4, 5, 6]);
        Ok(())
    }

    // ── PointCloud2 ──────────────────────────────────────────────────────

    #[test]
    fn point_cloud2_from_cdr_empty_fields() -> TestResult {
        let mut data = le_header();
        // header: sec=0, nanosec=0, frame_id=""
        push_i32_le(&mut data, 0);
        push_u32_le(&mut data, 0);
        push_empty_string(&mut data);
        // height=1, width=0
        push_u32_le(&mut data, 1);
        push_u32_le(&mut data, 0);
        // fields: sequence length=0
        push_u32_le(&mut data, 0);
        // is_bigendian
        data.push(0u8);
        // point_step: align to 4
        align_to(&mut data, 4);
        push_u32_le(&mut data, 0);
        // row_step
        push_u32_le(&mut data, 0);
        // data: empty
        push_u32_le(&mut data, 0);
        // is_dense
        data.push(0u8);

        let mut d = CdrDeserializer::new(&data)?;
        let pc = PointCloud2::from_cdr(&mut d)?;
        assert_eq!(pc.height, 1);
        assert_eq!(pc.width, 0);
        assert!(pc.fields.is_empty());
        assert!(pc.data.is_empty());
        Ok(())
    }

    // ── PointField ───────────────────────────────────────────────────────

    #[test]
    fn point_field_from_cdr() -> TestResult {
        let mut data = le_header();
        push_string(&mut data, "x");
        // offset: align to 4 from current position
        align_to(&mut data, 4);
        push_u32_le(&mut data, 0); // offset
        data.push(7u8); // datatype = FLOAT32
        // count: align to 4
        align_to(&mut data, 4);
        push_u32_le(&mut data, 1); // count
        let mut d = CdrDeserializer::new(&data)?;
        let pf = PointField::from_cdr(&mut d)?;
        assert_eq!(pf.name, "x");
        assert_eq!(pf.offset, 0);
        assert_eq!(pf.datatype, 7);
        assert_eq!(pf.count, 1);
        Ok(())
    }

    // ── TwistWithCovariance ──────────────────────────────────────────────

    #[test]
    fn twist_with_covariance_from_cdr() -> TestResult {
        let mut data = le_header();
        // Twist: 6 f64 = 48 bytes
        for _ in 0..6 {
            push_f64_le(&mut data, 0.0);
        }
        // Covariance: 36 f64 = 288 bytes
        for _ in 0..36 {
            push_f64_le(&mut data, 0.0);
        }
        let mut d = CdrDeserializer::new(&data)?;
        let twc = TwistWithCovariance::from_cdr(&mut d)?;
        assert!((twc.covariance[0] - 0.0).abs() < 1e-12);
        Ok(())
    }

    // ── TransformStamped ─────────────────────────────────────────────────

    #[test]
    fn transform_stamped_from_cdr() -> TestResult {
        let mut data = le_header();
        // header: sec, nanosec, empty frame_id
        push_i32_le(&mut data, 1);
        push_u32_le(&mut data, 0);
        push_empty_string(&mut data);
        // child_frame_id
        push_string(&mut data, "base_link");
        // transform: translation (3 f64) + rotation (4 f64)
        for _ in 0..3 {
            push_f64_le(&mut data, 0.0);
        }
        push_f64_le(&mut data, 0.0); // x
        push_f64_le(&mut data, 0.0); // y
        push_f64_le(&mut data, 0.0); // z
        push_f64_le(&mut data, 1.0); // w
        let mut d = CdrDeserializer::new(&data)?;
        let ts = TransformStamped::from_cdr(&mut d)?;
        assert_eq!(ts.child_frame_id, "base_link");
        assert!((ts.transform.rotation.w - 1.0).abs() < 1e-12);
        Ok(())
    }

    // ── Imu ──────────────────────────────────────────────────────────────

    fn push_header(buf: &mut Vec<u8>, sec: i32, frame: &str) {
        push_i32_le(buf, sec);
        push_u32_le(buf, 7);
        push_string(buf, frame);
    }

    fn push_f64s(buf: &mut Vec<u8>, values: &[f64]) {
        for v in values {
            push_f64_le(buf, *v);
        }
    }

    fn imu_bytes(frame: &str) -> Vec<u8> {
        let mut data = le_header();
        push_header(&mut data, 42, frame);
        push_f64s(&mut data, &[0.1, 0.2, 0.3, 0.9]); // orientation
        push_f64s(&mut data, &[1.0; 9]);
        push_f64s(&mut data, &[4.0, 5.0, 6.0]); // angular velocity
        push_f64s(&mut data, &[2.0; 9]);
        push_f64s(&mut data, &[7.0, 8.0, 9.81]); // linear acceleration
        push_f64s(&mut data, &[3.0; 9]);
        data
    }

    #[test]
    fn imu_decodes_every_field_in_spec_order() -> TestResult {
        // A one-character frame id leaves the quaternion at byte 20, where
        // the old decoder skipped to byte 28 and read the wrong values.
        for frame in ["i", "imu4", "a_much_longer_imu_frame"] {
            let data = imu_bytes(frame);
            let imu = Imu::from_cdr(&mut CdrDeserializer::new(&data)?)?;
            assert_eq!(imu.header.frame_id, frame);
            assert_eq!(imu.header.stamp.sec, 42);
            assert_eq!(imu.orientation.w, 0.9);
            assert_eq!(imu.orientation_covariance, [1.0; 9]);
            assert_eq!(imu.angular_velocity.z, 6.0);
            assert_eq!(imu.angular_velocity_covariance, [2.0; 9]);
            assert_eq!(imu.linear_acceleration.z, 9.81);
            assert_eq!(imu.linear_acceleration_covariance, [3.0; 9]);
        }
        Ok(())
    }

    #[test]
    fn a_short_imu_is_an_error_not_zero_padded() -> TestResult {
        let data = imu_bytes("imu4");
        for cut in [data.len() - 1, data.len() - 72, 40] {
            let result = Imu::from_cdr(&mut CdrDeserializer::new(&data[..cut])?);
            assert!(
                matches!(
                    result,
                    Err(crate::rosbag::cdr::CdrError {
                        kind: crate::rosbag::cdr::CdrErrorKind::Truncated { .. },
                        ..
                    })
                ),
                "cut at {cut}: {result:?}"
            );
        }
        Ok(())
    }

    // ── PointStamped ─────────────────────────────────────────────────────

    #[test]
    fn point_stamped_from_cdr() -> TestResult {
        let mut data = le_header();
        push_header(&mut data, 0, "");
        push_f64s(&mut data, &[3.0, 4.0, 5.0]);
        let mut d = CdrDeserializer::new(&data)?;
        let ps = PointStamped::from_cdr(&mut d)?;
        assert!((ps.point.x - 3.0).abs() < 1e-12);
        assert!((ps.point.y - 4.0).abs() < 1e-12);
        assert!((ps.point.z - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn point_stamped_after_a_long_frame_id_does_not_underflow() -> TestResult {
        // The old decoder skipped `28 - position` bytes, which underflowed
        // once the header ran past byte 28.
        let mut data = le_header();
        push_header(&mut data, 0, "a_frame_id_longer_than_the_old_fixed_offset");
        push_f64s(&mut data, &[1.0, 2.0, 3.0]);
        let ps = PointStamped::from_cdr(&mut CdrDeserializer::new(&data)?)?;
        assert_eq!((ps.point.x, ps.point.y, ps.point.z), (1.0, 2.0, 3.0));
        Ok(())
    }

    // ── NavSatFix ────────────────────────────────────────────────────────

    fn nav_sat_fix_bytes() -> Vec<u8> {
        let mut data = le_header();
        push_header(&mut data, 0, "gps");
        data.push(2u8); // status.status (i8)
        pad_to(&mut data, 2);
        data.extend_from_slice(&5u16.to_le_bytes()); // status.service
        push_f64s(&mut data, &[37.5, -122.0, 100.0]);
        push_f64s(&mut data, &[0.5; 9]);
        data.push(2u8); // position_covariance_type
        data
    }

    #[test]
    fn nav_sat_fix_reads_status_and_covariance() -> TestResult {
        let data = nav_sat_fix_bytes();
        let fix = NavSatFix::from_cdr(&mut CdrDeserializer::new(&data)?)?;
        assert_eq!(fix.status.status, 2);
        assert_eq!(fix.status.service, 5);
        assert!((fix.latitude - 37.5).abs() < 1e-9);
        assert!((fix.longitude - (-122.0)).abs() < 1e-9);
        assert_eq!(fix.position_covariance, [0.5; 9]);
        assert_eq!(fix.position_covariance_type, 2);
        Ok(())
    }

    // ── PoseWithCovariance / PoseWithCovarianceStamped ───────────────────

    fn pose_with_covariance_bytes(data: &mut Vec<u8>) {
        push_f64s(data, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        let covariance: Vec<f64> = (0..36).map(f64::from).collect();
        push_f64s(data, &covariance);
    }

    #[test]
    fn pose_with_covariance_from_cdr() -> TestResult {
        let mut data = le_header();
        pose_with_covariance_bytes(&mut data);
        let pose = PoseWithCovariance::from_cdr(&mut CdrDeserializer::new(&data)?)?;
        assert_eq!(pose.pose.position.z, 3.0);
        assert_eq!(pose.pose.orientation.w, 1.0);
        assert_eq!(pose.covariance[35], 35.0);
        Ok(())
    }

    #[test]
    fn a_truncated_covariance_is_an_error_not_zeros() -> TestResult {
        let mut data = le_header();
        pose_with_covariance_bytes(&mut data);
        data.truncate(data.len() - 8);
        assert!(PoseWithCovariance::from_cdr(&mut CdrDeserializer::new(&data)?).is_err());
        Ok(())
    }

    #[test]
    fn pose_with_covariance_stamped_from_cdr() -> TestResult {
        let mut data = le_header();
        push_header(&mut data, 3, "map");
        pose_with_covariance_bytes(&mut data);
        let stamped = PoseWithCovarianceStamped::from_cdr(&mut CdrDeserializer::new(&data)?)?;
        assert_eq!(stamped.header.frame_id, "map");
        assert_eq!(stamped.pose.pose.position.x, 1.0);
        Ok(())
    }

    // ── Odometry ─────────────────────────────────────────────────────────

    fn odometry_bytes() -> Vec<u8> {
        let mut data = le_header();
        push_header(&mut data, 9, "odom");
        push_string(&mut data, "base_link");
        pose_with_covariance_bytes(&mut data);
        push_f64s(&mut data, &[0.5, 0.0, 0.0, 0.0, 0.0, 0.1]); // twist
        push_f64s(&mut data, &[0.0; 36]);
        data
    }

    #[test]
    fn odometry_decodes_header_frames_pose_and_twist() -> TestResult {
        let data = odometry_bytes();
        let odom = Odometry::from_cdr(&mut CdrDeserializer::new(&data)?)?;
        assert_eq!(odom.header.stamp.sec, 9);
        assert_eq!(odom.header.frame_id, "odom");
        assert_eq!(odom.child_frame_id, "base_link");
        assert_eq!(odom.pose.pose.position.y, 2.0);
        assert_eq!(odom.twist.twist.linear.x, 0.5);
        assert_eq!(odom.twist.twist.angular.z, 0.1);
        Ok(())
    }

    #[test]
    fn a_broken_odometry_is_an_error_not_an_invented_message() -> TestResult {
        // The old decoder answered these with stamp 0, frame "odom", child
        // "base_link", an identity pose and a zero twist.
        let data = odometry_bytes();
        for cut in [10, 30, data.len() - 1] {
            assert!(
                Odometry::from_cdr(&mut CdrDeserializer::new(&data[..cut])?).is_err(),
                "cut at {cut}"
            );
        }
        let zeros = {
            let mut z = vec![0u8; 800];
            z[1] = 0x01;
            z
        };
        // All zeros is a valid (if empty) message: it decodes as zeros
        // because those are the bytes, not because anything was invented.
        let odom = Odometry::from_cdr(&mut CdrDeserializer::new(&zeros)?)?;
        assert_eq!(odom.header.frame_id, "");
        assert_eq!(odom.child_frame_id, "");
        Ok(())
    }

    // ── deserialize_message dispatch ─────────────────────────────────────

    #[test]
    fn deserialize_message_unknown_type_returns_err() -> TestResult {
        let data = vec![0x00u8, 0x01, 0x00, 0x00];
        let result = deserialize_message(&data, "unknown_pkg/msg/Unknown");
        assert!(matches!(
            result,
            Err(Ros2BagError::UnsupportedMessageType { .. })
        ));
        Ok(())
    }

    #[test]
    fn deserialize_message_too_short_returns_err() -> TestResult {
        let result = deserialize_message(&[0x00, 0x01], "sensor_msgs/msg/Imu");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn deserialize_message_dispatches_point_stamped() -> TestResult {
        let mut data = le_header();
        push_header(&mut data, 0, "");
        push_f64s(&mut data, &[1.0, 2.0, 3.0]);
        let result = deserialize_message(&data, "geometry_msgs/msg/PointStamped")?;
        assert!(matches!(result, DecodedMessage::PointStamped(_)));
        Ok(())
    }

    #[test]
    fn deserialize_message_dispatches_nav_sat_fix() -> TestResult {
        let result = deserialize_message(&nav_sat_fix_bytes(), "sensor_msgs/msg/NavSatFix")?;
        assert!(matches!(result, DecodedMessage::NavSatFix(_)));
        Ok(())
    }

    #[test]
    fn deserialize_message_dispatches_transform_stamped() -> TestResult {
        let mut data = le_header();
        push_i32_le(&mut data, 0);
        push_u32_le(&mut data, 0);
        push_empty_string(&mut data);
        push_empty_string(&mut data); // child_frame_id
        for _ in 0..7 {
            push_f64_le(&mut data, 0.0);
        }
        let result = deserialize_message(&data, "geometry_msgs/msg/TransformStamped")?;
        assert!(matches!(result, DecodedMessage::TransformStamped(_)));
        Ok(())
    }

    #[test]
    fn deserialize_message_dispatches_odometry() -> TestResult {
        let result = deserialize_message(&odometry_bytes(), "nav_msgs/msg/Odometry")?;
        assert!(matches!(result, DecodedMessage::Odometry(_)));
        Ok(())
    }

    #[test]
    fn deserialize_message_dispatches_pose_with_covariance_stamped() -> TestResult {
        let mut data = le_header();
        push_header(&mut data, 0, "map");
        pose_with_covariance_bytes(&mut data);
        let result = deserialize_message(&data, "geometry_msgs/msg/PoseWithCovarianceStamped")?;
        assert!(matches!(
            result,
            DecodedMessage::PoseWithCovarianceStamped(_)
        ));
        Ok(())
    }

    #[test]
    fn deserialize_message_dispatches_imu() -> TestResult {
        let result = deserialize_message(&imu_bytes("imu4"), "sensor_msgs/msg/Imu")?;
        assert!(matches!(result, DecodedMessage::Imu(_)));
        Ok(())
    }
}
