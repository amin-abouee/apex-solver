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

mod lazy_index {
    use apex_io::rosbag::ros1::{Ros1Compression, Ros1Reader, Ros1Writer};
    use std::collections::HashMap;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    /// 3 topics, many tiny chunks, interleaved in time.
    fn make(path: &std::path::Path, comp: Ros1Compression) -> TestResult {
        let mut w = Ros1Writer::new(path)?;
        w.set_compression(comp)?;
        w.set_chunk_threshold(2_000)?;
        w.open()?;
        let a = w.add_connection("/a", "t/A", "x", "", false)?;
        let b = w.add_connection("/b", "t/B", "x", "", false)?;
        let c = w.add_connection("/c", "t/C", "x", "", false)?;
        for i in 0..600u64 {
            let t = 1_000_000_000 + i * 1_000_000; // 1 kHz
            w.write(&a, t, &[1u8; 100])?;
            if i % 10 == 0 {
                w.write(&b, t + 1, &[2u8; 150])?;
            }
            if i % 100 == 0 {
                w.write(&c, t + 2, &i.to_le_bytes())?;
            }
        }
        w.close()?;
        Ok(())
    }

    #[test]
    fn counts_and_time_range_need_no_message_index() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("l.bag");
        make(&path, Ros1Compression::None)?;
        let mut r = Ros1Reader::new(&path)?;
        r.open()?;
        assert_eq!(r.message_count(), 600 + 60 + 6);
        let by: HashMap<_, _> = r
            .connections()
            .iter()
            .map(|c| (c.topic.as_str(), c.message_count))
            .collect();
        assert_eq!((by["/a"], by["/b"], by["/c"]), (600, 60, 6));
        assert_eq!(r.start_time(), 1_000_000_000);
        assert_eq!(r.end_time(), 1_000_000_000 + 599 * 1_000_000);
        Ok(())
    }

    #[test]
    fn topic_and_window_queries_return_exactly_the_right_messages() -> TestResult {
        for comp in [
            Ros1Compression::None,
            Ros1Compression::Lz4,
            Ros1Compression::Bz2,
        ] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("q.bag");
            make(&path, comp)?;
            let mut r = Ros1Reader::new(&path)?;
            r.open()?;

            let b: Vec<_> = r
                .connections()
                .iter()
                .filter(|c| c.topic == "/b")
                .cloned()
                .collect();
            let all_b = r.raw_messages_filtered(Some(&b), None, None)?;
            assert_eq!(all_b.len(), 60, "{comp:?}");
            assert!(all_b.windows(2).all(|w| w[0].timestamp <= w[1].timestamp));
            assert!(all_b.iter().all(|m| m.raw_data == vec![2u8; 150]));

            // window [1.2 s, 1.3 s): /a has 100 messages in it
            let a: Vec<_> = r
                .connections()
                .iter()
                .filter(|c| c.topic == "/a")
                .cloned()
                .collect();
            let win =
                r.raw_messages_filtered(Some(&a), Some(1_200_000_000), Some(1_300_000_000))?;
            assert_eq!(win.len(), 100, "{comp:?}");
            assert_eq!(win[0].timestamp, 1_200_000_000);
            assert_eq!(win[99].timestamp, 1_299_000_000);

            // everything, merged in time order
            let everything = r.raw_messages()?;
            assert_eq!(everything.len(), 666);
            assert!(
                everything
                    .windows(2)
                    .all(|w| w[0].timestamp <= w[1].timestamp)
            );

            // a window outside the bag is empty, not an error
            assert!(
                r.raw_messages_filtered(None, Some(5_000_000_000), None)?
                    .is_empty()
            );
        }
        Ok(())
    }

    /// Patch the first ChunkInfo's first message count in the tail.
    fn tamper_chunk_info_count(path: &std::path::Path) -> TestResult {
        let mut bytes = std::fs::read(path)?;
        // bag header record starts after the 13-byte magic; its index_pos
        // field tells where the tail begins.
        let rd = |b: &[u8], at: usize| {
            u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
        };
        let header_len = rd(&bytes, 13);
        let hdr = &bytes[17..17 + header_len];
        let key = b"index_pos=";
        let at = hdr
            .windows(key.len())
            .position(|w| w == key)
            .ok_or("index_pos")?
            + key.len();
        let index_pos = u64::from_le_bytes(hdr[at..at + 8].try_into()?) as usize;

        let mut pos = index_pos;
        while pos + 8 <= bytes.len() {
            let hl = rd(&bytes, pos);
            let header = bytes[pos + 4..pos + 4 + hl].to_vec();
            let dl = rd(&bytes, pos + 4 + hl);
            let data_at = pos + 4 + hl + 4;
            // op=6 is ChunkInfo; the header field is "op=" followed by the byte.
            if header.windows(4).any(|w| w == [b'o', b'p', b'=', 6]) {
                bytes[data_at + 4] = bytes[data_at + 4].wrapping_add(1); // first entry's count
                std::fs::write(path, bytes)?;
                return Ok(());
            }
            pos = data_at + dl;
        }
        Err("no ChunkInfo found".into())
    }

    #[test]
    fn a_tampered_chunk_info_is_detected_when_the_index_is_built() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("t.bag");
        make(&path, Ros1Compression::None)?;
        tamper_chunk_info_count(&path)?;

        let mut r = Ros1Reader::new(&path)?;
        r.open()?; // the tail itself is well-formed
        let result = r.raw_messages();
        assert!(result.is_err(), "inconsistent counts must not be trusted");
        Ok(())
    }

    /// Hundreds of tiny compressed chunks finish out of order on the pool;
    /// the file must still hold them in write order with a correct index.
    #[test]
    fn parallel_chunk_compression_preserves_order_and_content() -> TestResult {
        for comp in [Ros1Compression::Bz2, Ros1Compression::Lz4] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("p.bag");
            let mut w = Ros1Writer::new(&path)?;
            w.set_compression(comp)?;
            w.set_chunk_threshold(400)?; // ~3 messages per chunk
            w.open()?;
            let a = w.add_connection("/a", "t/A", "x", "", false)?;
            let b = w.add_connection("/b", "t/B", "x", "", false)?;
            let n = 1_500u64;
            for i in 0..n {
                // varied sizes/compressibility so chunk jobs take uneven time
                let len = 40 + (i as usize * 53) % 300;
                let payload: Vec<u8> = (0..len).map(|k| ((k as u64 * 7 + i) % 253) as u8).collect();
                w.write(
                    if i.is_multiple_of(2) { &a } else { &b },
                    1_000 + i,
                    &payload,
                )?;
            }
            w.close()?;

            let mut r = Ros1Reader::new(&path)?;
            r.open()?;
            assert_eq!(r.message_count(), n, "{comp:?}");
            let msgs = r.raw_messages()?;
            assert_eq!(msgs.len() as u64, n);
            for (i, m) in msgs.iter().enumerate() {
                let i = i as u64;
                assert_eq!(m.timestamp, 1_000 + i, "{comp:?}");
                let len = 40 + (i as usize * 53) % 300;
                let want: Vec<u8> = (0..len).map(|k| ((k as u64 * 7 + i) % 253) as u8).collect();
                assert_eq!(m.raw_data, want, "{comp:?} message {i}");
                assert_eq!(
                    m.connection.topic,
                    if i.is_multiple_of(2) { "/a" } else { "/b" }
                );
            }
        }
        Ok(())
    }
}
