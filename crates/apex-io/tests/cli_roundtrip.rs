//! End-to-end checks of the `bag_*` command-line tools on small synthetic bags.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use apex_io::rosbag::ros1::deserialize::Ros1Serializer;
use apex_io::rosbag::ros1::messages as r1;
use apex_io::rosbag::transcode::{Dialect, catalog_definition};
use apex_io::rosbag::{BagKind, BagWriter, NewConnection, OutputCompression};
use std::path::Path;
use std::process::{Command, Output};

fn run(bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).output().expect("spawn")
}

fn ros1_conn(topic: &str, ty: &str, definition: String) -> NewConnection {
    NewConnection {
        topic: topic.into(),
        message_type: ty.into(),
        definition,
        type_hash: "*".into(),
        serialization_format: "ros1".into(),
        qos: Vec::new(),
    }
}

/// /imu (50 msgs, standard type), /str (10 msgs), /odd (unknown type, no definition).
fn make_ros1_bag(path: &Path) {
    let mut w = BagWriter::create(path, BagKind::Ros1Bag, OutputCompression::None).unwrap();
    let imu = w
        .add_connection(&ros1_conn(
            "/imu",
            "sensor_msgs/Imu",
            catalog_definition("sensor_msgs/Imu", Dialect::Ros1).unwrap(),
        ))
        .unwrap();
    let string = w
        .add_connection(&ros1_conn(
            "/str",
            "std_msgs/String",
            "string data\n".into(),
        ))
        .unwrap();
    let odd = w
        .add_connection(&ros1_conn("/odd", "vendor_msgs/Mystery", String::new()))
        .unwrap();

    for i in 0..50u64 {
        let mut m = r1::Imu::default();
        m.header.frame_id = "imu".into();
        m.header.stamp_ns = 1_000_000_000 + i * 10_000_000;
        m.linear_acceleration.x = i as f64;
        let mut s = Ros1Serializer::new();
        m.to_ros1(&mut s);
        w.write(imu, m.header.stamp_ns, &s.into_bytes()).unwrap();
    }
    for i in 0..10u64 {
        let mut s = Ros1Serializer::new();
        r1::StringMsg {
            data: format!("hello {i}"),
        }
        .to_ros1(&mut s);
        w.write(string, 1_000_000_000 + i * 50_000_000, &s.into_bytes())
            .unwrap();
    }
    w.write(odd, 1_000_000_000, b"opaque").unwrap();
    w.close().unwrap();
}

#[test]
fn convert_filter_info_verify_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n).to_string_lossy().into_owned();
    let (src, ros2, back, sub) = (p("src.bag"), p("ros2"), p("back.bag"), p("sub.mcap"));
    make_ros1_bag(Path::new(&src));

    // bag_info sees all topics and flags the unconvertible one.
    let info = run(env!("CARGO_BIN_EXE_bag_info"), &[&src]);
    let text = String::from_utf8_lossy(&info.stdout);
    assert!(info.status.success(), "{text}");
    assert!(
        text.contains("/imu (sensor_msgs/Imu, ros1, 50 msgs"),
        "{text}"
    );
    assert!(
        text.contains("/odd") && text.contains("[not convertible]"),
        "{text}"
    );

    // Default conversion skips the unsupported topic with a warning...
    let conv = run(
        env!("CARGO_BIN_EXE_bag_convert"),
        &[&src, &ros2, "--format", "mcap"],
    );
    let err = String::from_utf8_lossy(&conv.stderr);
    assert!(conv.status.success(), "{err}");
    assert!(err.contains("/odd") && err.contains("WARN"), "{err}");

    // ...and the result matches the source on the shared topics.
    let verify = run(
        env!("CARGO_BIN_EXE_bag_verify"),
        &[&src, &ros2, "-t", "/imu,/str"],
    );
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stdout)
    );

    // ROS2 -> ROS1 and back to the same canonical content.
    let conv = run(
        env!("CARGO_BIN_EXE_bag_convert"),
        &[&ros2, &back, "--compression", "lz4"],
    );
    assert!(
        conv.status.success(),
        "{}",
        String::from_utf8_lossy(&conv.stderr)
    );
    let verify = run(
        env!("CARGO_BIN_EXE_bag_verify"),
        &[&src, &back, "-t", "/imu,/str"],
    );
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stdout)
    );

    // Filter: one topic, half-open time window.
    let filt = run(
        env!("CARGO_BIN_EXE_bag_filter"),
        &[
            &ros2,
            &sub,
            "-t",
            "/imu",
            "-s",
            "1100000000",
            "-e",
            "1300000000",
        ],
    );
    assert!(
        filt.status.success(),
        "{}",
        String::from_utf8_lossy(&filt.stderr)
    );
    assert!(String::from_utf8_lossy(&filt.stdout).contains("Copied 20 msgs"));
    let verify = run(
        env!("CARGO_BIN_EXE_bag_verify"),
        &[
            &ros2,
            &sub,
            "-t",
            "/imu",
            "-s",
            "1100000000",
            "-e",
            "1300000000",
        ],
    );
    assert!(verify.status.success());

    // --strict refuses to drop the unsupported topic.
    let strict = run(
        env!("CARGO_BIN_EXE_bag_convert"),
        &[&src, &p("strict_out"), "--format", "mcap", "--strict"],
    );
    assert!(!strict.status.success());

    // A raw filter copy cannot silently change the encoding.
    let bad = run(env!("CARGO_BIN_EXE_bag_filter"), &[&ros2, &p("x.bag")]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("bag_convert"));
}
