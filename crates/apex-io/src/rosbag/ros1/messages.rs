//! Standard ROS1 message catalog (subset).
//!
//! Each struct exposes `from_ros1` and `to_ros1` for round-tripping against a
//! [`Ros1Deserializer`]/[`Ros1Serializer`].  Only the standard messages most
//! commonly seen in pose-graph/SLAM datasets are included; consumers can
//! always fall back to `Ros1Reader::raw_messages` for unknown types.

use crate::rosbag::error::Result;

use super::deserialize::{Ros1Deserializer, Ros1Serializer};

/// `std_msgs/Header`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    pub seq: u32,
    pub stamp_ns: u64,
    pub frame_id: String,
}

impl Header {
    /// Decode from ROS1 bytes.
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            seq: d.read_u32()?,
            stamp_ns: d.read_time_nanos()?,
            frame_id: d.read_string()?,
        })
    }

    /// Encode to ROS1 bytes.
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        s.write_u32(self.seq);
        s.write_time_nanos(self.stamp_ns);
        s.write_string(&self.frame_id);
    }
}

/// `geometry_msgs/Vector3`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            x: d.read_f64()?,
            y: d.read_f64()?,
            z: d.read_f64()?,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        s.write_f64(self.x);
        s.write_f64(self.y);
        s.write_f64(self.z);
    }
}

/// `geometry_msgs/Point`. Same wire shape as `Vector3`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            x: d.read_f64()?,
            y: d.read_f64()?,
            z: d.read_f64()?,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        s.write_f64(self.x);
        s.write_f64(self.y);
        s.write_f64(self.z);
    }
}

/// `geometry_msgs/Quaternion`.
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Quaternion {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            x: d.read_f64()?,
            y: d.read_f64()?,
            z: d.read_f64()?,
            w: d.read_f64()?,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        s.write_f64(self.x);
        s.write_f64(self.y);
        s.write_f64(self.z);
        s.write_f64(self.w);
    }
}

/// `geometry_msgs/Pose`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pose {
    pub position: Point,
    pub orientation: Quaternion,
}

impl Pose {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            position: Point::from_ros1(d)?,
            orientation: Quaternion::from_ros1(d)?,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        self.position.to_ros1(s);
        self.orientation.to_ros1(s);
    }
}

/// `geometry_msgs/PoseStamped`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseStamped {
    pub header: Header,
    pub pose: Pose,
}

impl PoseStamped {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::from_ros1(d)?,
            pose: Pose::from_ros1(d)?,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        self.header.to_ros1(s);
        self.pose.to_ros1(s);
    }
}

/// `sensor_msgs/Imu`.
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

impl Default for Imu {
    fn default() -> Self {
        Self {
            header: Header::default(),
            orientation: Quaternion::default(),
            orientation_covariance: [0.0; 9],
            angular_velocity: Vector3::default(),
            angular_velocity_covariance: [0.0; 9],
            linear_acceleration: Vector3::default(),
            linear_acceleration_covariance: [0.0; 9],
        }
    }
}

impl Imu {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        let header = Header::from_ros1(d)?;
        let orientation = Quaternion::from_ros1(d)?;
        let orientation_covariance = read_f64_9(d)?;
        let angular_velocity = Vector3::from_ros1(d)?;
        let angular_velocity_covariance = read_f64_9(d)?;
        let linear_acceleration = Vector3::from_ros1(d)?;
        let linear_acceleration_covariance = read_f64_9(d)?;
        Ok(Self {
            header,
            orientation,
            orientation_covariance,
            angular_velocity,
            angular_velocity_covariance,
            linear_acceleration,
            linear_acceleration_covariance,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        self.header.to_ros1(s);
        self.orientation.to_ros1(s);
        write_f64_9(s, &self.orientation_covariance);
        self.angular_velocity.to_ros1(s);
        write_f64_9(s, &self.angular_velocity_covariance);
        self.linear_acceleration.to_ros1(s);
        write_f64_9(s, &self.linear_acceleration_covariance);
    }
}

/// `std_msgs/String`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StringMsg {
    pub data: String,
}

impl StringMsg {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            data: d.read_string()?,
        })
    }
    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        s.write_string(&self.data);
    }
}

/// `sensor_msgs/Image`.
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

impl Image {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::from_ros1(d)?,
            height: d.read_u32()?,
            width: d.read_u32()?,
            encoding: d.read_string()?,
            is_bigendian: d.read_u8()?,
            step: d.read_u32()?,
            data: d.read_bytes()?,
        })
    }

    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        self.header.to_ros1(s);
        s.write_u32(self.height);
        s.write_u32(self.width);
        s.write_string(&self.encoding);
        s.write_u8(self.is_bigendian);
        s.write_u32(self.step);
        s.write_bytes(&self.data);
    }
}

/// `sensor_msgs/CompressedImage`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedImage {
    pub header: Header,
    pub format: String,
    pub data: Vec<u8>,
}

impl CompressedImage {
    pub fn from_ros1(d: &mut Ros1Deserializer<'_>) -> Result<Self> {
        Ok(Self {
            header: Header::from_ros1(d)?,
            format: d.read_string()?,
            data: d.read_bytes()?,
        })
    }

    pub fn to_ros1(&self, s: &mut Ros1Serializer) {
        self.header.to_ros1(s);
        s.write_string(&self.format);
        s.write_bytes(&self.data);
    }
}

fn read_f64_9(d: &mut Ros1Deserializer<'_>) -> Result<[f64; 9]> {
    let mut out = [0.0; 9];
    for slot in &mut out {
        *slot = d.read_f64()?;
    }
    Ok(out)
}

fn write_f64_9(s: &mut Ros1Serializer, arr: &[f64; 9]) {
    for v in arr {
        s.write_f64(*v);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn imu_round_trip() -> Result<()> {
        let imu = Imu {
            header: Header {
                seq: 42,
                stamp_ns: 1_234_567_890_000_000_000,
                frame_id: "imu_link".into(),
            },
            orientation: Quaternion {
                x: 0.1,
                y: 0.2,
                z: 0.3,
                w: 0.9273618,
            },
            orientation_covariance: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            angular_velocity: Vector3 {
                x: 0.01,
                y: 0.02,
                z: 0.03,
            },
            angular_velocity_covariance: [0.0; 9],
            linear_acceleration: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 9.81,
            },
            linear_acceleration_covariance: [0.0; 9],
        };

        let mut s = Ros1Serializer::new();
        imu.to_ros1(&mut s);
        let bytes = s.into_bytes();
        let mut d = Ros1Deserializer::new(&bytes);
        let back = Imu::from_ros1(&mut d)?;
        assert_eq!(back, imu);
        assert_eq!(d.remaining(), 0);
        Ok(())
    }

    #[test]
    fn pose_stamped_round_trip() -> Result<()> {
        let ps = PoseStamped {
            header: Header {
                seq: 1,
                stamp_ns: 1_700_000_000_500_000_000,
                frame_id: "map".into(),
            },
            pose: Pose {
                position: Point {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
                orientation: Quaternion::default(),
            },
        };
        let mut s = Ros1Serializer::new();
        ps.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(PoseStamped::from_ros1(&mut d)?, ps);
        Ok(())
    }

    #[test]
    fn header_round_trip() -> Result<()> {
        let h = Header {
            seq: 42,
            stamp_ns: 1_000_000_000,
            frame_id: "base_link".into(),
        };
        let mut s = Ros1Serializer::new();
        h.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Header::from_ros1(&mut d)?, h);
        Ok(())
    }

    #[test]
    fn vector3_round_trip() -> Result<()> {
        let v = Vector3 {
            x: 1.5,
            y: -2.5,
            z: 3.0,
        };
        let mut s = Ros1Serializer::new();
        v.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Vector3::from_ros1(&mut d)?, v);
        Ok(())
    }

    #[test]
    fn point_round_trip() -> Result<()> {
        let p = Point {
            x: -1.0,
            y: 0.0,
            z: 100.0,
        };
        let mut s = Ros1Serializer::new();
        p.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Point::from_ros1(&mut d)?, p);
        Ok(())
    }

    #[test]
    fn quaternion_round_trip() -> Result<()> {
        let q = Quaternion {
            x: 0.0,
            y: 0.707,
            z: 0.0,
            w: 0.707,
        };
        let mut s = Ros1Serializer::new();
        q.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Quaternion::from_ros1(&mut d)?, q);
        Ok(())
    }

    #[test]
    fn quaternion_default_is_identity() {
        let q = Quaternion::default();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn pose_round_trip() -> Result<()> {
        let p = Pose {
            position: Point {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            orientation: Quaternion::default(),
        };
        let mut s = Ros1Serializer::new();
        p.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Pose::from_ros1(&mut d)?, p);
        Ok(())
    }

    #[test]
    fn string_msg_round_trip() -> Result<()> {
        let msg = StringMsg {
            data: "hello world".into(),
        };
        let mut s = Ros1Serializer::new();
        msg.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(StringMsg::from_ros1(&mut d)?, msg);
        Ok(())
    }

    #[test]
    fn string_msg_empty_round_trip() -> Result<()> {
        let msg = StringMsg { data: "".into() };
        let mut s = Ros1Serializer::new();
        msg.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(StringMsg::from_ros1(&mut d)?, msg);
        Ok(())
    }

    #[test]
    fn image_round_trip() -> Result<()> {
        let img = Image {
            header: Header {
                seq: 1,
                stamp_ns: 1_000_000_000,
                frame_id: "camera".into(),
            },
            height: 480,
            width: 640,
            encoding: "rgb8".into(),
            is_bigendian: 0,
            step: 1920,
            data: vec![0xFF; 100],
        };
        let mut s = Ros1Serializer::new();
        img.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Image::from_ros1(&mut d)?, img);
        Ok(())
    }

    #[test]
    fn image_empty_data_round_trip() -> Result<()> {
        let img = Image {
            header: Header::default(),
            height: 0,
            width: 0,
            encoding: "".into(),
            is_bigendian: 0,
            step: 0,
            data: vec![],
        };
        let mut s = Ros1Serializer::new();
        img.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(Image::from_ros1(&mut d)?, img);
        Ok(())
    }

    #[test]
    fn compressed_image_round_trip() -> Result<()> {
        let img = CompressedImage {
            header: Header {
                seq: 5,
                stamp_ns: 2_000_000_000,
                frame_id: "cam".into(),
            },
            format: "jpeg".into(),
            data: vec![0xFF, 0xD8, 0xFF, 0xE0],
        };
        let mut s = Ros1Serializer::new();
        img.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(CompressedImage::from_ros1(&mut d)?, img);
        Ok(())
    }

    #[test]
    fn compressed_image_empty_data_round_trip() -> Result<()> {
        let img = CompressedImage {
            header: Header::default(),
            format: "".into(),
            data: vec![],
        };
        let mut s = Ros1Serializer::new();
        img.to_ros1(&mut s);
        let mut d = Ros1Deserializer::new(s.as_slice());
        assert_eq!(CompressedImage::from_ros1(&mut d)?, img);
        Ok(())
    }

    #[test]
    fn imu_default_has_identity_orientation() {
        let imu = Imu::default();
        assert_eq!(imu.orientation.w, 1.0);
        assert_eq!(imu.orientation.x, 0.0);
    }
}
