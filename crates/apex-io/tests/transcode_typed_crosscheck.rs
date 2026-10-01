//! Cross-check the schema-driven transcoder against the crate's typed ROS1
//! and CDR codecs (written and validated independently of it).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use apex_io::rosbag::cdr::CdrDeserializer;
use apex_io::rosbag::messages::{FromCdr, Image as CdrImage, Imu as CdrImu};
use apex_io::rosbag::ros1::deserialize::Ros1Serializer;
use apex_io::rosbag::ros1::messages as r1;
use apex_io::rosbag::transcode::{Dialect, Transcoder, catalog};

fn to_cdr(name: &str, ros1: &[u8]) -> Vec<u8> {
    let reg = catalog().expect("catalog");
    let t = Transcoder::compile(reg, name, Dialect::Ros1, Dialect::Ros2).expect("compile");
    let mut out = Vec::new();
    t.convert(ros1, &mut out).expect("convert");
    out
}

#[test]
fn imu_matches_typed_decoders() {
    let imu = r1::Imu {
        header: r1::Header {
            seq: 0,
            stamp_ns: 1_700_000_123_456_789_012,
            frame_id: "imu_link".into(),
        },
        orientation: r1::Quaternion {
            x: 0.1,
            y: 0.2,
            z: 0.3,
            w: 0.9,
        },
        orientation_covariance: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        angular_velocity: r1::Vector3 {
            x: -1.5,
            y: 2.5,
            z: 3.5,
        },
        angular_velocity_covariance: [0.5; 9],
        linear_acceleration: r1::Vector3 {
            x: 9.81,
            y: 0.0,
            z: -0.25,
        },
        linear_acceleration_covariance: [0.25; 9],
    };
    let mut s = Ros1Serializer::new();
    imu.to_ros1(&mut s);
    let cdr = to_cdr("sensor_msgs/Imu", &s.into_bytes());

    let mut d = CdrDeserializer::new(&cdr).unwrap();
    let got = CdrImu::from_cdr(&mut d).unwrap();
    assert_eq!(got.header.frame_id, "imu_link");
    assert_eq!(got.header.stamp.sec, 1_700_000_123);
    assert_eq!(got.header.stamp.nanosec, 456_789_012);
    assert_eq!(got.orientation.w, 0.9);
    assert_eq!(got.orientation_covariance, imu.orientation_covariance);
    assert_eq!(got.angular_velocity.y, 2.5);
    assert_eq!(got.linear_acceleration.x, 9.81);
    assert_eq!(got.linear_acceleration_covariance, [0.25; 9]);
}

#[test]
fn image_matches_typed_decoders() {
    let img = r1::Image {
        header: r1::Header {
            seq: 0,
            stamp_ns: 42_000_000_007,
            frame_id: "cam".into(),
        },
        height: 3,
        width: 5,
        encoding: "mono8".into(),
        is_bigendian: 0,
        step: 5,
        data: (0..15u8).collect(),
    };
    let mut s = Ros1Serializer::new();
    img.to_ros1(&mut s);
    let cdr = to_cdr("sensor_msgs/Image", &s.into_bytes());

    let mut d = CdrDeserializer::new(&cdr).unwrap();
    let got = CdrImage::from_cdr(&mut d).unwrap();
    assert_eq!((got.height, got.width, got.step), (3, 5, 5));
    assert_eq!(got.encoding, "mono8");
    assert_eq!(got.header.stamp.sec, 42);
    assert_eq!(got.header.stamp.nanosec, 7);
    assert_eq!(got.data, img.data);
}
