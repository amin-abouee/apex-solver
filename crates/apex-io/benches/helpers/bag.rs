//! A VIO-like ROS recording: two cameras, one IMU, one lidar — the same
//! messages encoded both ROS1 and CDR so every bench reads real payloads.

use super::Rng;
use apex_io::rosbag::bag::{BagKind, BagWriter, NewConnection, OutputCompression};
use apex_io::rosbag::ros1::deserialize::Ros1Serializer;
use apex_io::rosbag::ros1::messages as r1;
use apex_io::rosbag::transcode::{Dialect, Transcoder, catalog, catalog_definition};
use std::path::Path;

/// Bench recording length.
pub const DURATION_S: u64 = 20;
const CAMERA_HZ: u64 = 10;
const IMU_HZ: u64 = 200;
const LIDAR_HZ: u64 = 5;
const CAM_WIDTH: u32 = 320;
const CAM_HEIGHT: u32 = 240;
const POINTS_PER_CLOUD: u32 = 10_000;
/// Fixed epoch so timestamps look realistic and stay reproducible.
const T0_NS: u64 = 1_700_000_000_000_000_000;
const NS_PER_S: u64 = 1_000_000_000;

pub struct TopicSpec {
    pub topic: &'static str,
    /// `pkg/Type`, as used by the ROS1 dialect.
    pub ros1_type: &'static str,
}

pub struct Message {
    pub topic_idx: usize,
    pub timestamp_ns: u64,
    /// ROS1-encoded payload.
    pub ros1: Vec<u8>,
    /// CDR-encoded payload (same message, ROS2 dialect).
    pub cdr: Vec<u8>,
}

pub struct Scenario {
    pub topics: Vec<TopicSpec>,
    /// Sorted by timestamp, globally interleaved.
    pub messages: Vec<Message>,
}

impl Scenario {
    /// 400 images (2 x 10 Hz, 320x240 mono8), 4000 IMU samples (200 Hz), 100
    /// clouds (5 Hz, 10k points) — about 48 MB per encoding.
    pub fn build() -> Self {
        let topics = vec![
            TopicSpec {
                topic: "/cam0/image_raw",
                ros1_type: "sensor_msgs/Image",
            },
            TopicSpec {
                topic: "/cam1/image_raw",
                ros1_type: "sensor_msgs/Image",
            },
            TopicSpec {
                topic: "/imu0",
                ros1_type: "sensor_msgs/Imu",
            },
            TopicSpec {
                topic: "/lidar/points",
                ros1_type: "sensor_msgs/PointCloud2",
            },
        ];
        let mut messages = Vec::new();

        let cam_bytes = (CAM_WIDTH * CAM_HEIGHT) as usize;
        for cam in 0..2u32 {
            let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15 ^ (u64::from(cam + 1) * 0x0100_0001));
            let mut frame = vec![0u8; cam_bytes];
            for i in 0..(DURATION_S * CAMERA_HZ) as u32 {
                rng.fill_bytes(&mut frame);
                let ts = T0_NS + u64::from(i) * NS_PER_S / CAMERA_HZ;
                messages.push(Message {
                    topic_idx: cam as usize,
                    timestamp_ns: ts,
                    ros1: image_ros1(i, ts, &frame),
                    cdr: Vec::new(),
                });
            }
        }

        for i in 0..(DURATION_S * IMU_HZ) as u32 {
            let ts = T0_NS + u64::from(i) * NS_PER_S / IMU_HZ;
            messages.push(Message {
                topic_idx: 2,
                timestamp_ns: ts,
                ros1: imu_ros1(i, ts),
                cdr: Vec::new(),
            });
        }

        let mut rng = Rng::new(0xDEAD_BEEF_CAFE_F00D);
        for i in 0..(DURATION_S * LIDAR_HZ) as u32 {
            let ts = T0_NS + u64::from(i) * NS_PER_S / LIDAR_HZ;
            messages.push(Message {
                topic_idx: 3,
                timestamp_ns: ts,
                ros1: point_cloud2_ros1(i, ts, POINTS_PER_CLOUD, &mut rng),
                cdr: Vec::new(),
            });
        }

        // One transcoder per type, reused across every message of that type.
        let reg = catalog().expect("standard message catalog");
        let mut scratch = Vec::new();
        for spec in &topics {
            let t = Transcoder::compile(reg, spec.ros1_type, Dialect::Ros1, Dialect::Ros2)
                .expect("compile transcoder");
            for m in &mut messages {
                if topics[m.topic_idx].ros1_type != spec.ros1_type {
                    continue;
                }
                t.convert(&m.ros1, &mut scratch)
                    .unwrap_or_else(|e| panic!("convert {} to cdr: {e}", spec.ros1_type));
                m.cdr = scratch.clone();
            }
        }

        messages.sort_by_key(|m| m.timestamp_ns);
        Self { topics, messages }
    }

    /// Topic names passing a two-topic VIO-style selection.
    pub fn vio_topics() -> [&'static str; 2] {
        ["/cam0/image_raw", "/imu0"]
    }

    /// Start of a two-second window 5 s into the recording.
    pub fn window() -> (u64, u64) {
        (T0_NS + 5 * NS_PER_S, T0_NS + 7 * NS_PER_S)
    }
}

/// `pkg/Type` -> `pkg/msg/Type`.
pub fn ros2_type(ros1_type: &str) -> String {
    let (pkg, ty) = ros1_type.split_once('/').expect("pkg/Type");
    format!("{pkg}/msg/{ty}")
}

/// Write the scenario as `kind`, payloads encoded `ros1` (ROS1 bag) or
/// `cdr` (everything else).
pub fn write_bag(scenario: &Scenario, kind: BagKind, path: &Path) {
    let fmt = if kind == BagKind::Ros1Bag {
        "ros1"
    } else {
        "cdr"
    };
    let dialect = if fmt == "ros1" {
        Dialect::Ros1
    } else {
        Dialect::Ros2
    };
    let mut w = BagWriter::create(path, kind, OutputCompression::None).expect("create output bag");
    let handles: Vec<_> = scenario
        .topics
        .iter()
        .map(|t| {
            w.add_connection(&NewConnection {
                topic: t.topic.to_string(),
                message_type: if fmt == "ros1" {
                    t.ros1_type.to_string()
                } else {
                    ros2_type(t.ros1_type)
                },
                definition: catalog_definition(t.ros1_type, dialect).unwrap_or_default(),
                type_hash: String::new(),
                serialization_format: fmt.to_string(),
                qos: Vec::new(),
            })
            .expect("add connection")
        })
        .collect();
    for m in &scenario.messages {
        let data = if fmt == "ros1" { &m.ros1 } else { &m.cdr };
        w.write(handles[m.topic_idx], m.timestamp_ns, data)
            .expect("write message");
    }
    w.close().expect("close output bag");
}

fn header(seq: u32, ts_ns: u64, frame_id: &str) -> r1::Header {
    r1::Header {
        seq,
        stamp_ns: ts_ns,
        frame_id: frame_id.to_string(),
    }
}

fn image_ros1(seq: u32, ts_ns: u64, data: &[u8]) -> Vec<u8> {
    let img = r1::Image {
        header: header(seq, ts_ns, "cam"),
        height: CAM_HEIGHT,
        width: CAM_WIDTH,
        encoding: "mono8".to_string(),
        is_bigendian: 0,
        step: CAM_WIDTH,
        data: data.to_vec(),
    };
    let mut s = Ros1Serializer::new();
    img.to_ros1(&mut s);
    s.into_bytes()
}

fn imu_ros1(seq: u32, ts_ns: u64) -> Vec<u8> {
    let imu = r1::Imu {
        header: header(seq, ts_ns, "imu_link"),
        orientation: r1::Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
        orientation_covariance: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        angular_velocity: r1::Vector3 {
            x: 0.01,
            y: -0.02,
            z: 0.03,
        },
        angular_velocity_covariance: [0.0; 9],
        linear_acceleration: r1::Vector3 {
            x: 9.81,
            y: 0.0,
            z: -0.1,
        },
        linear_acceleration_covariance: [0.01; 9],
    };
    let mut s = Ros1Serializer::new();
    imu.to_ros1(&mut s);
    s.into_bytes()
}

/// Hand-rolled ROS1 `sensor_msgs/PointCloud2` (xyz + intensity, f32).
fn point_cloud2_ros1(seq: u32, ts_ns: u64, n: u32, rng: &mut Rng) -> Vec<u8> {
    const POINT_STEP: u32 = 16;
    let mut s = Ros1Serializer::new();
    s.write_u32(seq);
    s.write_u32(u32::try_from(ts_ns / NS_PER_S).expect("stamp sec"));
    s.write_u32(u32::try_from(ts_ns % NS_PER_S).expect("stamp nsec"));
    s.write_string("lidar");
    s.write_u32(1); // height (unstructured)
    s.write_u32(n); // width
    s.write_u32(4); // fields: array length
    for (name, offset) in [("x", 0), ("y", 4), ("z", 8), ("intensity", 12)] {
        s.write_string(name);
        s.write_u32(offset);
        s.write_u8(7); // FLOAT32
        s.write_u32(1); // count
    }
    s.write_u8(0); // is_bigendian
    s.write_u32(POINT_STEP); // point_step
    s.write_u32(n * POINT_STEP); // row_step
    s.write_u32(n * POINT_STEP); // data length
    for _ in 0..n {
        let (x, y, z, i) = (
            (rng.next_u64() % 40_000) as f32 / 100.0 - 200.0,
            (rng.next_u64() % 40_000) as f32 / 100.0 - 200.0,
            (rng.next_u64() % 1_000) as f32 / 100.0,
            (rng.next_u64() % 1_000) as f32 / 10.0,
        );
        s.write_f32(x);
        s.write_f32(y);
        s.write_f32(z);
        s.write_f32(i);
    }
    s.write_u8(1); // is_dense
    s.into_bytes()
}
