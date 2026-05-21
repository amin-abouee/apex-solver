//! ROS1 bag round-trip + filter + compression matrix.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use apex_io::rosbag::ros1::{
    Ros1Compression, Ros1Reader, Ros1Writer,
    deserialize::{Ros1Deserializer, Ros1Serializer},
    messages::{Header, Imu, PoseStamped, Quaternion, Vector3},
};
use apex_io::rosbag::types::Connection;
use std::path::PathBuf;
use tempfile::tempdir;

const IMU_TYPE: &str = "sensor_msgs/Imu";
const POSE_TYPE: &str = "geometry_msgs/PoseStamped";

fn make_imu(i: u64) -> Imu {
    Imu {
        header: Header {
            seq: i as u32,
            stamp_ns: 1_700_000_000_000_000_000 + i * 10_000_000,
            frame_id: "imu_link".into(),
        },
        orientation: Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
        orientation_covariance: [0.0; 9],
        angular_velocity: Vector3 {
            x: 0.001 * i as f64,
            y: 0.0,
            z: 0.0,
        },
        angular_velocity_covariance: [0.0; 9],
        linear_acceleration: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 9.81,
        },
        linear_acceleration_covariance: [0.0; 9],
    }
}

fn make_pose(i: u64) -> PoseStamped {
    let mut ps = PoseStamped::default();
    ps.header.seq = i as u32;
    ps.header.stamp_ns = 1_700_000_000_000_000_000 + i * 10_000_000;
    ps.header.frame_id = "map".into();
    ps.pose.position.x = i as f64;
    ps.pose.position.y = 2.0 * i as f64;
    ps.pose.position.z = 0.0;
    ps.pose.orientation = Quaternion::default();
    ps
}

fn write_demo_bag(
    path: &PathBuf,
    compression: Ros1Compression,
    n: u64,
) -> (Connection, Connection) {
    let mut w = Ros1Writer::new(path).expect("new writer");
    w.set_compression(compression).expect("set compression");
    // Small chunk threshold so we exercise multiple chunks during the test.
    w.set_chunk_threshold(8 * 1024).expect("chunk threshold");
    w.open().expect("open");

    let imu_conn = w
        .add_connection("/imu", IMU_TYPE, "abc123", "", false)
        .expect("imu conn");
    let pose_conn = w
        .add_connection("/pose", POSE_TYPE, "def456", "", false)
        .expect("pose conn");

    for i in 0..n {
        let imu = make_imu(i);
        let mut s = Ros1Serializer::new();
        imu.to_ros1(&mut s);
        w.write(&imu_conn, imu.header.stamp_ns, s.as_slice())
            .unwrap();

        let pose = make_pose(i);
        let mut s = Ros1Serializer::new();
        pose.to_ros1(&mut s);
        w.write(&pose_conn, pose.header.stamp_ns, s.as_slice())
            .unwrap();
    }
    w.close().expect("close");
    (imu_conn, pose_conn)
}

#[test]
fn round_trip_uncompressed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("uncompressed.bag");
    let n = 200u64;
    let (imu_conn, _) = write_demo_bag(&path, Ros1Compression::None, n);

    let mut r = Ros1Reader::new(&path).expect("new reader");
    r.open().expect("open reader");

    assert_eq!(r.message_count(), 2 * n);
    assert_eq!(r.connections().len(), 2);
    assert_eq!(r.topics().len(), 2);
    let imu_topic = &r.topics()["/imu"];
    assert_eq!(imu_topic.message_count, n);
    assert_eq!(imu_topic.message_type, IMU_TYPE);

    let msgs = r.messages().expect("messages");
    // Monotonic timestamps.
    for w in msgs.windows(2) {
        assert!(w[0].timestamp <= w[1].timestamp);
    }

    // Decode one Imu and assert field round-trip.
    let imu_msg = msgs
        .iter()
        .find(|m| m.connection.id == imu_conn.id)
        .expect("imu message");
    let mut d = Ros1Deserializer::new(&imu_msg.data);
    let imu = Imu::from_ros1(&mut d).expect("decode imu");
    assert_eq!(imu.linear_acceleration.z, 9.81);
    assert_eq!(imu.header.frame_id, "imu_link");
    assert_eq!(d.remaining(), 0);
}

#[test]
fn round_trip_bz2() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bz2.bag");
    let n = 100u64;
    write_demo_bag(&path, Ros1Compression::Bz2, n);

    let mut r = Ros1Reader::new(&path).unwrap();
    r.open().unwrap();
    assert_eq!(r.message_count(), 2 * n);
    let raw = r.raw_messages().unwrap();
    assert_eq!(raw.len() as u64, 2 * n);
}

#[test]
fn round_trip_lz4() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("lz4.bag");
    let n = 100u64;
    write_demo_bag(&path, Ros1Compression::Lz4, n);

    let mut r = Ros1Reader::new(&path).unwrap();
    r.open().unwrap();
    assert_eq!(r.message_count(), 2 * n);
    let raw = r.raw_messages().unwrap();
    assert_eq!(raw.len() as u64, 2 * n);
}

#[test]
fn filter_by_topic_and_time() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filter.bag");
    let n = 200u64;
    let (imu_conn, _pose_conn) = write_demo_bag(&path, Ros1Compression::None, n);

    let mut r = Ros1Reader::new(&path).unwrap();
    r.open().unwrap();

    // 2 s window starting at i=50 (50 * 10 ms = 0.5 s offset).  Window is
    // 0.5s..2.5s so 200 imu samples is too many — clamp to whatever fits.
    let start = 1_700_000_000_000_000_000 + 50 * 10_000_000;
    let stop = 1_700_000_000_000_000_000 + 150 * 10_000_000;

    let imu_only: Vec<Connection> = r
        .connections()
        .iter()
        .filter(|c| c.id == imu_conn.id)
        .cloned()
        .collect();

    let msgs = r
        .messages_filtered(Some(&imu_only), Some(start), Some(stop))
        .unwrap();
    assert_eq!(msgs.len(), 100);
    for m in &msgs {
        assert_eq!(m.connection.id, imu_conn.id);
        assert!(m.timestamp >= start && m.timestamp < stop);
    }
}

#[test]
fn read_without_open_errors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noopen.bag");
    write_demo_bag(&path, Ros1Compression::None, 5);

    let mut r = Ros1Reader::new(&path).unwrap();
    // Did not call open().
    assert!(r.messages().is_err());
}
