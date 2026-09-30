//! Env-gated integration test against a real HILTI SLAM Challenge 2023 bag
//! (ROS1-profile MCAP: `message_encoding: ros1`, `ros1msg` schemas).
//!
//! Run with:
//!
//! ```text
//! APEX_HILTI_MCAP=/path/to/siteX_handheld_Y.mcap \
//!   cargo test -p apex-io --features rosbag --test hilti_mcap -- --nocapture
//! ```
//!
//! Skipped silently when the variable is unset, so the regular suite never
//! depends on a multi-gigabyte dataset being present.

#![allow(clippy::print_stdout, clippy::print_stderr)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use apex_io::rosbag::ros1::deserialize::Ros1Deserializer;
use apex_io::rosbag::ros1::messages::{Image as Ros1Image, Imu as Ros1Imu};
use apex_io::rosbag::{Connection, Reader};

fn hilti_bag() -> Option<PathBuf> {
    std::env::var("APEX_HILTI_MCAP").ok().map(PathBuf::from)
}

#[test]
fn hilti_ros1_mcap_summary_and_decode() {
    let Some(path) = hilti_bag() else {
        eprintln!("APEX_HILTI_MCAP not set; skipping");
        return;
    };

    let mut reader = Reader::new(&path).expect("open bare mcap");
    reader.open().expect("open storage");

    // The bag-level view: six topics, five Image + one Imu, all ROS1.
    let connections = reader.connections().to_vec();
    assert_eq!(connections.len(), 6, "5 cameras + imu");
    for connection in &connections {
        assert_eq!(connection.serialization_format, "ros1");
        assert!(
            connection.message_type == "sensor_msgs/Image"
                || connection.message_type == "sensor_msgs/Imu"
        );
    }
    assert!(reader.message_count() > 0);
    assert!(reader.end_time() > reader.start_time());

    // Filtered streaming: the first cam0 message must decode as a valid
    // 720x540 mono8 sensor_msgs/Image (HILTI Alphasense format), with a
    // header stamp matching the MCAP log time.
    let cam0: Vec<Connection> = connections
        .iter()
        .filter(|c| c.topic == "/alphasense/cam0/image_raw")
        .cloned()
        .collect();
    assert_eq!(cam0.len(), 1);

    let first = reader
        .raw_messages_filtered(Some(&cam0), None, None)
        .expect("iterate")
        .next()
        .expect("at least one cam0 message")
        .expect("read message");

    let mut de = Ros1Deserializer::new(&first.raw_data);
    let image = Ros1Image::from_ros1(&mut de).expect("decode ros1 Image");
    assert_eq!((image.width, image.height), (720, 540));
    assert_eq!(image.encoding, "mono8");
    assert_eq!(image.step, 720);
    assert_eq!(image.data.len(), 720 * 540);
    let stamp_ns = image.header.stamp_ns;
    assert_eq!(
        stamp_ns, first.timestamp,
        "header stamp should match MCAP log_time (writer sets them equal)"
    );

    // IMU decode: angular velocity/linear acceleration are f64 triples in
    // sensor_msgs/Imu; just check the message parses and has a sane stamp.
    let imu: Vec<Connection> = connections
        .iter()
        .filter(|c| c.topic == "/alphasense/imu")
        .cloned()
        .collect();
    let first_imu = reader
        .raw_messages_filtered(Some(&imu), None, None)
        .expect("iterate imu")
        .next()
        .expect("at least one imu message")
        .expect("read message");
    let mut de = Ros1Deserializer::new(&first_imu.raw_data);
    let imu_msg = Ros1Imu::from_ros1(&mut de).expect("decode ros1 Imu");
    assert!(
        imu_msg.linear_acceleration.x.abs() < 100.0,
        "accelerometer sample should be physical, got {}",
        imu_msg.linear_acceleration.x
    );

    // Time-bounded read must respect the window.
    let mid = (reader.start_time() + reader.end_time()) / 2;
    let early_count = reader
        .raw_messages_filtered(Some(&cam0), None, Some(mid))
        .expect("iterate window")
        .filter_map(Result::ok)
        .count();
    let late_count = reader
        .raw_messages_filtered(Some(&cam0), Some(mid), None)
        .expect("iterate window")
        .filter_map(Result::ok)
        .count();
    let total = reader.connections()[0]
        .message_count
        .max(cam0[0].message_count);
    assert_eq!(
        early_count + late_count,
        total as usize,
        "splitting the bag at its midpoint must cover every cam0 message"
    );

    println!(
        "HILTI {}: {} topics, {} messages, cam0 first frame {}x{} mono8",
        path.display(),
        connections.len(),
        reader.message_count(),
        image.width,
        image.height
    );
}
