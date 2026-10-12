#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (progress, warnings).

//! Convert between ROS1 (.bag) and ROS2 (SQLite3 / MCAP) bag files.
//!
//! Direction is auto-detected from the input path:
//!   - `.bag` file  →  ROS1 → ROS2
//!   - directory with `metadata.yaml`  →  ROS2 → ROS1
//!
//! Unsupported message types are skipped with a warning; all other messages
//! are re-serialised (ROS1 wire format ↔ CDR) in a single batch pass.
//!
//! Usage:
//!   `cargo run -p apex-io --bin bag_convert -- <INPUT> <OUTPUT> [OPTIONS]`

use apex_io::rosbag::Reader as Ros2Reader;
use apex_io::rosbag::Writer as Ros2Writer;
use apex_io::rosbag::ros1::deserialize::Ros1Serializer;
use apex_io::rosbag::ros1::messages as ros1_msg;
use apex_io::rosbag::ros1::{Ros1Deserializer, Ros1Reader, Ros1Writer};
use apex_io::rosbag::ros2::cdr::{CdrDeserializer, CdrSerializer};
use apex_io::rosbag::types::{CompressionFormat, CompressionMode, Connection, StoragePlugin};
use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    about = "Convert between ROS1 (.bag) and ROS2 (SQLite3/MCAP) bag files",
    long_about = "Auto-detects direction: .bag file → ROS1→ROS2, directory → ROS2→ROS1.\n\
                  Unsupported message types are skipped with a warning."
)]
struct Args {
    /// Input bag: ROS1 `.bag` file or ROS2 bag directory
    input: PathBuf,

    /// Output path (ROS2 directory or ROS1 `.bag` file)
    output: PathBuf,

    /// Output format when writing ROS2: `sqlite3` or `mcap`
    #[arg(long, default_value = "sqlite3")]
    format: String,

    /// Compression mode: `none`, `file`, or `message`
    #[arg(long, default_value = "none")]
    compression: String,

    /// Only convert these topics (comma-separated); default: all
    #[arg(long, value_delimiter = ',')]
    topics: Vec<String>,

    /// Print topics in the source bag and exit
    #[arg(long)]
    list_topics: bool,

    /// Verbose timing output
    #[arg(short, long)]
    verbose: bool,

    /// ROS1 → ROS2: read the input this many seconds at a time, so memory
    /// holds one window of messages rather than the whole bag
    #[arg(long, default_value_t = 2.0)]
    window_seconds: f64,
}

// ── Supported type registry ───────────────────────────────────────────────────

type ConvertFn = fn(&[u8]) -> Option<Vec<u8>>;

fn ros1_to_cdr_fn(ros1_type: &str) -> Option<ConvertFn> {
    match ros1_type {
        "std_msgs/String" => Some(conv_string_ros1_cdr),
        "geometry_msgs/Vector3" => Some(conv_vector3_ros1_cdr),
        "geometry_msgs/Point" => Some(conv_point_ros1_cdr),
        "geometry_msgs/Quaternion" => Some(conv_quaternion_ros1_cdr),
        "geometry_msgs/Pose" => Some(conv_pose_ros1_cdr),
        "geometry_msgs/PoseStamped" => Some(conv_pose_stamped_ros1_cdr),
        "sensor_msgs/Imu" => Some(conv_imu_ros1_cdr),
        "sensor_msgs/Image" => Some(conv_image_ros1_cdr),
        "sensor_msgs/CompressedImage" => Some(conv_compressed_image_ros1_cdr),
        "sensor_msgs/PointCloud2" => Some(conv_point_cloud2_ros1_cdr),
        "tf2_msgs/TFMessage" => Some(conv_tf_message_ros1_cdr),
        _ => None,
    }
}

fn cdr_to_ros1_fn(ros1_type: &str) -> Option<ConvertFn> {
    match ros1_type {
        "std_msgs/String" => Some(conv_string_cdr_ros1),
        "geometry_msgs/Vector3" => Some(conv_vector3_cdr_ros1),
        "geometry_msgs/Point" => Some(conv_point_cdr_ros1),
        "geometry_msgs/Quaternion" => Some(conv_quaternion_cdr_ros1),
        "geometry_msgs/Pose" => Some(conv_pose_cdr_ros1),
        "geometry_msgs/PoseStamped" => Some(conv_pose_stamped_cdr_ros1),
        "sensor_msgs/Imu" => Some(conv_imu_cdr_ros1),
        "sensor_msgs/Image" => Some(conv_image_cdr_ros1),
        "sensor_msgs/CompressedImage" => Some(conv_compressed_image_cdr_ros1),
        "sensor_msgs/PointCloud2" => Some(conv_point_cloud2_cdr_ros1),
        "tf2_msgs/TFMessage" => Some(conv_tf_message_cdr_ros1),
        _ => None,
    }
}

// ── Type name helpers ─────────────────────────────────────────────────────────

/// `sensor_msgs/Imu` → `sensor_msgs/msg/Imu`
fn ros1_type_to_ros2(t: &str) -> String {
    if let Some(slash) = t.find('/') {
        let pkg = &t[..slash];
        let ty = &t[slash + 1..];
        if !ty.contains('/') {
            return format!("{pkg}/msg/{ty}");
        }
    }
    t.to_string()
}

/// `sensor_msgs/msg/Imu` → `sensor_msgs/Imu`
fn ros2_type_to_ros1(t: &str) -> String {
    // Remove the `/msg/` segment if present
    let parts: Vec<&str> = t.splitn(3, '/').collect();
    if parts.len() == 3 && parts[1] == "msg" {
        return format!("{}/{}", parts[0], parts[2]);
    }
    t.to_string()
}

fn topics_match(filter: &[String], topic: &str) -> bool {
    filter.is_empty() || filter.iter().any(|t| t == topic)
}

fn parse_storage(s: &str) -> Result<StoragePlugin> {
    match s {
        "sqlite3" => Ok(StoragePlugin::Sqlite3),
        "mcap" => Ok(StoragePlugin::Mcap),
        other => Err(format!("Unknown storage format '{other}'. Use 'sqlite3' or 'mcap'").into()),
    }
}

fn parse_compression(s: &str) -> Result<CompressionMode> {
    match s {
        "none" => Ok(CompressionMode::None),
        "file" => Ok(CompressionMode::File),
        "message" => Ok(CompressionMode::Message),
        other => {
            Err(format!("Unknown compression '{other}'. Use 'none', 'file', or 'message'").into())
        }
    }
}

// ── Direction detection ───────────────────────────────────────────────────────

fn input_is_ros1(path: &std::path::Path) -> bool {
    path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("bag"))
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    if input_is_ros1(&args.input) {
        convert_ros1_to_ros2(&args)
    } else if args.input.is_dir() {
        convert_ros2_to_ros1(&args)
    } else {
        Err(format!(
            "Input '{}' must be a .bag file (ROS1) or a bag directory (ROS2)",
            args.input.display()
        )
        .into())
    }
}

// ── ROS1 → ROS2 ──────────────────────────────────────────────────────────────

fn convert_ros1_to_ros2(args: &Args) -> Result<()> {
    let t0 = std::time::Instant::now();

    let mut ros1 =
        Ros1Reader::new(&args.input).map_err(|e| format!("Cannot open input bag: {e}"))?;
    ros1.open()
        .map_err(|e| format!("Failed to open ROS1 bag: {e}"))?;

    let connections = ros1.connections().to_vec();

    if args.list_topics {
        println!("Topics in ROS1 bag:");
        for c in &connections {
            let support = if ros1_to_cdr_fn(&c.message_type).is_some() {
                "✓"
            } else {
                "✗ (unsupported)"
            };
            let count = ros1.topics().get(&c.topic).map_or(0, |t| t.message_count);
            println!(
                "  {} ({}) {count} messages {}",
                c.topic, c.message_type, support
            );
        }
        return Ok(());
    }

    // Filter to requested topics, then warn once per unsupported type
    let mut warned: HashSet<String> = HashSet::new();
    let supported: Vec<Connection> = connections
        .into_iter()
        .filter(|c| topics_match(&args.topics, &c.topic))
        .filter(|c| {
            if ros1_to_cdr_fn(&c.message_type).is_some() {
                true
            } else {
                if warned.insert(c.message_type.clone()) {
                    eprintln!("WARN: skipping unsupported ROS1 type '{}'", c.message_type);
                }
                false
            }
        })
        .collect();

    if supported.is_empty() {
        eprintln!("No convertible topics found. Exiting.");
        return Ok(());
    }

    let storage = parse_storage(&args.format)?;
    let compression = parse_compression(&args.compression)?;

    let mut ros2 = Ros2Writer::new(&args.output, None, Some(storage))
        .map_err(|e| format!("Cannot create output bag: {e}"))?;
    ros2.set_compression(compression, CompressionFormat::None)?;
    ros2.configure_buffer(50, 500)?;
    ros2.open()
        .map_err(|e| format!("Failed to open ROS2 output: {e}"))?;

    // Register one writer connection per supported topic
    let mut conn_map: HashMap<u32, Connection> = HashMap::new();
    for c in &supported {
        let ros2_type = ros1_type_to_ros2(&c.message_type);
        let w_conn = ros2
            .add_connection(c.topic.clone(), ros2_type, None, None, None, None)
            .map_err(|e| format!("add_connection failed: {e}"))?;
        conn_map.insert(c.id, w_conn);
    }

    // Read in time windows ([start, stop) each, so none is read twice):
    // memory holds one window, not the whole bag.
    if !(args.window_seconds.is_finite() && args.window_seconds > 0.0) {
        return Err(format!("--window-seconds must be > 0, got {}", args.window_seconds).into());
    }
    let window_ns = (args.window_seconds * 1e9) as u64;
    let (first_ns, last_ns) = (ros1.start_time(), ros1.end_time());

    let t1 = std::time::Instant::now();
    let mut batch: Vec<(Connection, u64, Vec<u8>)> = Vec::with_capacity(500);
    let mut converted = 0u64;
    let mut skipped = 0u64;

    let mut window_start = first_ns;
    while window_start <= last_ns {
        let window_stop = window_start.saturating_add(window_ns.max(1));
        let raw = ros1
            .raw_messages_filtered(Some(&supported), Some(window_start), Some(window_stop))
            .map_err(|e| format!("Failed to read ROS1 messages: {e}"))?;
        for msg in raw {
            let convert = match ros1_to_cdr_fn(&msg.connection.message_type) {
                Some(f) => f,
                None => continue,
            };
            match convert(&msg.raw_data) {
                Some(cdr) => {
                    let w_conn = conn_map[&msg.connection.id].clone();
                    batch.push((w_conn, msg.timestamp, cdr));
                    converted += 1;
                    if batch.len() >= 500 {
                        ros2.write_raw_messages_batch(&batch)
                            .map_err(|e| format!("Batch write failed: {e}"))?;
                        batch.clear();
                    }
                }
                None => skipped += 1,
            }
        }
        if args.verbose {
            println!(
                "  {:.0} / {:.0} s, {converted} messages",
                (window_stop.min(last_ns) - first_ns) as f64 * 1e-9,
                (last_ns - first_ns) as f64 * 1e-9
            );
        }
        window_start = window_stop;
    }
    if !batch.is_empty() {
        ros2.write_raw_messages_batch(&batch)
            .map_err(|e| format!("Final batch write failed: {e}"))?;
    }

    ros2.close()
        .map_err(|e| format!("Failed to close output: {e}"))?;
    drop(ros1); // Ros1Reader has no explicit close()

    if skipped > 0 {
        eprintln!("WARN: {skipped} messages failed conversion (malformed data?)");
    }
    if args.verbose {
        println!(
            "Converted {converted} messages in {:?} (total {:?})",
            t1.elapsed(),
            t0.elapsed()
        );
    } else {
        println!(
            "Done: {converted} messages → {} ({})",
            args.output.display(),
            args.format
        );
    }
    Ok(())
}

// ── ROS2 → ROS1 ──────────────────────────────────────────────────────────────

fn convert_ros2_to_ros1(args: &Args) -> Result<()> {
    let t0 = std::time::Instant::now();

    let mut ros2 =
        Ros2Reader::new(&args.input).map_err(|e| format!("Cannot open input bag: {e}"))?;
    ros2.open()
        .map_err(|e| format!("Failed to open ROS2 bag: {e}"))?;

    let connections = ros2.connections().to_vec();

    if args.list_topics {
        println!("Topics in ROS2 bag:");
        for c in &connections {
            let ros1_type = ros2_type_to_ros1(&c.message_type);
            let support = if cdr_to_ros1_fn(&ros1_type).is_some() {
                "✓"
            } else {
                "✗ (unsupported)"
            };
            println!("  {} ({}) {}", c.topic, c.message_type, support);
        }
        return Ok(());
    }

    let mut warned: HashSet<String> = HashSet::new();
    let supported: Vec<Connection> = connections
        .into_iter()
        .filter(|c| topics_match(&args.topics, &c.topic))
        .filter(|c| {
            let ros1_type = ros2_type_to_ros1(&c.message_type);
            if cdr_to_ros1_fn(&ros1_type).is_some() {
                true
            } else {
                if warned.insert(c.message_type.clone()) {
                    eprintln!("WARN: skipping unsupported ROS2 type '{}'", c.message_type);
                }
                false
            }
        })
        .collect();

    if supported.is_empty() {
        eprintln!("No convertible topics found. Exiting.");
        return Ok(());
    }

    let mut ros1 =
        Ros1Writer::new(&args.output).map_err(|e| format!("Cannot create output .bag: {e}"))?;
    ros1.open()
        .map_err(|e| format!("Failed to open ROS1 output: {e}"))?;

    // Register one writer connection per supported topic
    let mut conn_map: HashMap<u32, Connection> = HashMap::new();
    for c in &supported {
        let ros1_type = ros2_type_to_ros1(&c.message_type);
        // Use the type hash as md5sum placeholder when available
        let md5 = if c.type_description_hash.is_empty() {
            "*".to_string()
        } else {
            c.type_description_hash.clone()
        };
        let w_conn = ros1
            .add_connection(c.topic.clone(), ros1_type, md5, String::new(), false)
            .map_err(|e| format!("add_connection failed: {e}"))?;
        conn_map.insert(c.id, w_conn);
    }

    // Single I/O read pass (batch)
    let raw = ros2
        .read_raw_messages_batch(Some(&supported), None, None)
        .map_err(|e| format!("Failed to read ROS2 messages: {e}"))?;

    if args.verbose {
        println!("Read {} messages in {:?}", raw.len(), t0.elapsed());
    }

    let t1 = std::time::Instant::now();
    let mut converted = 0u64;
    let mut skipped = 0u64;

    for msg in raw {
        let ros1_type = ros2_type_to_ros1(&msg.connection.message_type);
        let convert = match cdr_to_ros1_fn(&ros1_type) {
            Some(f) => f,
            None => continue,
        };
        match convert(&msg.raw_data) {
            Some(ros1_bytes) => {
                let w_conn = &conn_map[&msg.connection.id];
                ros1.write(w_conn, msg.timestamp, &ros1_bytes)
                    .map_err(|e| format!("Write failed: {e}"))?;
                converted += 1;
            }
            None => skipped += 1,
        }
    }

    ros1.close()
        .map_err(|e| format!("Failed to close output: {e}"))?;
    ros2.close()
        .map_err(|e| format!("Failed to close input: {e}"))?;

    if skipped > 0 {
        eprintln!("WARN: {skipped} messages failed conversion (malformed data?)");
    }
    if args.verbose {
        println!(
            "Converted {converted} messages in {:?} (total {:?})",
            t1.elapsed(),
            t0.elapsed()
        );
    } else {
        println!("Done: {converted} messages → {}", args.output.display());
    }
    Ok(())
}

// ── Per-type converters: ROS1 → CDR ──────────────────────────────────────────

fn conv_string_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let msg = ros1_msg::StringMsg::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    s.write_string(&msg.data);
    Some(s.into_bytes())
}

fn conv_vector3_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let v = ros1_msg::Vector3::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    s.write_f64(v.x);
    s.write_f64(v.y);
    s.write_f64(v.z);
    Some(s.into_bytes())
}

fn conv_point_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let p = ros1_msg::Point::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    s.write_f64(p.x);
    s.write_f64(p.y);
    s.write_f64(p.z);
    Some(s.into_bytes())
}

fn conv_quaternion_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let q = ros1_msg::Quaternion::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    s.write_f64(q.x);
    s.write_f64(q.y);
    s.write_f64(q.z);
    s.write_f64(q.w);
    Some(s.into_bytes())
}

fn conv_pose_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let p = ros1_msg::Pose::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    // Point
    s.write_f64(p.position.x);
    s.write_f64(p.position.y);
    s.write_f64(p.position.z);
    // Quaternion
    s.write_f64(p.orientation.x);
    s.write_f64(p.orientation.y);
    s.write_f64(p.orientation.z);
    s.write_f64(p.orientation.w);
    Some(s.into_bytes())
}

fn conv_pose_stamped_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let ps = ros1_msg::PoseStamped::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    write_header_cdr(&mut s, &ps.header);
    // Pose: Point + Quaternion
    s.write_f64(ps.pose.position.x);
    s.write_f64(ps.pose.position.y);
    s.write_f64(ps.pose.position.z);
    s.write_f64(ps.pose.orientation.x);
    s.write_f64(ps.pose.orientation.y);
    s.write_f64(ps.pose.orientation.z);
    s.write_f64(ps.pose.orientation.w);
    Some(s.into_bytes())
}

fn conv_imu_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let imu = ros1_msg::Imu::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    write_header_cdr(&mut s, &imu.header);
    s.write_f64(imu.orientation.x);
    s.write_f64(imu.orientation.y);
    s.write_f64(imu.orientation.z);
    s.write_f64(imu.orientation.w);
    for v in &imu.orientation_covariance {
        s.write_f64(*v);
    }
    s.write_f64(imu.angular_velocity.x);
    s.write_f64(imu.angular_velocity.y);
    s.write_f64(imu.angular_velocity.z);
    for v in &imu.angular_velocity_covariance {
        s.write_f64(*v);
    }
    s.write_f64(imu.linear_acceleration.x);
    s.write_f64(imu.linear_acceleration.y);
    s.write_f64(imu.linear_acceleration.z);
    for v in &imu.linear_acceleration_covariance {
        s.write_f64(*v);
    }
    Some(s.into_bytes())
}

/// Write a ROS1 Header as ROS2 CDR `std_msgs/msg/Header` (Time + frame_id, no seq).
fn write_header_cdr(s: &mut CdrSerializer, h: &ros1_msg::Header) {
    let sec = (h.stamp_ns / 1_000_000_000) as i32;
    let nanosec = (h.stamp_ns % 1_000_000_000) as u32;
    s.write_i32(sec);
    s.write_u32(nanosec);
    s.write_string(&h.frame_id);
}

// ── Per-type converters: CDR → ROS1 ──────────────────────────────────────────

fn conv_string_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let text = d.read_string().ok()?;
    let msg = ros1_msg::StringMsg { data: text };
    let mut s = Ros1Serializer::new();
    msg.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_vector3_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let v = ros1_msg::Vector3 {
        x: d.read_f64().ok()?,
        y: d.read_f64().ok()?,
        z: d.read_f64().ok()?,
    };
    let mut s = Ros1Serializer::new();
    v.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_point_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let p = ros1_msg::Point {
        x: d.read_f64().ok()?,
        y: d.read_f64().ok()?,
        z: d.read_f64().ok()?,
    };
    let mut s = Ros1Serializer::new();
    p.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_quaternion_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let q = ros1_msg::Quaternion {
        x: d.read_f64().ok()?,
        y: d.read_f64().ok()?,
        z: d.read_f64().ok()?,
        w: d.read_f64().ok()?,
    };
    let mut s = Ros1Serializer::new();
    q.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_pose_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let p = ros1_msg::Pose {
        position: ros1_msg::Point {
            x: d.read_f64().ok()?,
            y: d.read_f64().ok()?,
            z: d.read_f64().ok()?,
        },
        orientation: ros1_msg::Quaternion {
            x: d.read_f64().ok()?,
            y: d.read_f64().ok()?,
            z: d.read_f64().ok()?,
            w: d.read_f64().ok()?,
        },
    };
    let mut s = Ros1Serializer::new();
    p.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_pose_stamped_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let header = read_header_ros1(&mut d)?;
    let ps = ros1_msg::PoseStamped {
        header,
        pose: ros1_msg::Pose {
            position: ros1_msg::Point {
                x: d.read_f64().ok()?,
                y: d.read_f64().ok()?,
                z: d.read_f64().ok()?,
            },
            orientation: ros1_msg::Quaternion {
                x: d.read_f64().ok()?,
                y: d.read_f64().ok()?,
                z: d.read_f64().ok()?,
                w: d.read_f64().ok()?,
            },
        },
    };
    let mut s = Ros1Serializer::new();
    ps.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_imu_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let header = read_header_ros1(&mut d)?;
    let orientation = ros1_msg::Quaternion {
        x: d.read_f64().ok()?,
        y: d.read_f64().ok()?,
        z: d.read_f64().ok()?,
        w: d.read_f64().ok()?,
    };
    let mut oc = [0.0f64; 9];
    for v in &mut oc {
        *v = d.read_f64().ok()?;
    }
    let angular_velocity = ros1_msg::Vector3 {
        x: d.read_f64().ok()?,
        y: d.read_f64().ok()?,
        z: d.read_f64().ok()?,
    };
    let mut avc = [0.0f64; 9];
    for v in &mut avc {
        *v = d.read_f64().ok()?;
    }
    let linear_acceleration = ros1_msg::Vector3 {
        x: d.read_f64().ok()?,
        y: d.read_f64().ok()?,
        z: d.read_f64().ok()?,
    };
    let mut lac = [0.0f64; 9];
    for v in &mut lac {
        *v = d.read_f64().ok()?;
    }
    let imu = ros1_msg::Imu {
        header,
        orientation,
        orientation_covariance: oc,
        angular_velocity,
        angular_velocity_covariance: avc,
        linear_acceleration,
        linear_acceleration_covariance: lac,
    };
    let mut s = Ros1Serializer::new();
    imu.to_ros1(&mut s);
    Some(s.into_bytes())
}

/// Read a ROS2 CDR `std_msgs/msg/Header` (Time + frame_id) as a ROS1 Header (seq=0).
fn read_header_ros1(d: &mut CdrDeserializer<'_>) -> Option<ros1_msg::Header> {
    let sec = d.read_i32().ok()?;
    let nanosec = d.read_u32().ok()?;
    let frame_id = d.read_string().ok()?;
    Some(ros1_msg::Header {
        seq: 0,
        stamp_ns: sec as u64 * 1_000_000_000 + nanosec as u64,
        frame_id,
    })
}

/// `tf2_msgs/TFMessage`: a sequence of `geometry_msgs/TransformStamped`.
fn conv_tf_message_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let transforms = d
        .read_seq(|d| ros1_msg::TransformStamped::from_ros1(d))
        .ok()?;
    let mut s = CdrSerializer::new();
    s.write_u32(transforms.len() as u32);
    for t in &transforms {
        write_header_cdr(&mut s, &t.header);
        s.write_string(&t.child_frame_id);
        let (v, q) = (&t.transform.translation, &t.transform.rotation);
        for value in [v.x, v.y, v.z, q.x, q.y, q.z, q.w] {
            s.write_f64(value);
        }
    }
    Some(s.into_bytes())
}

fn conv_tf_message_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    use apex_io::rosbag::messages::FromCdr;
    let mut d = CdrDeserializer::new(data).ok()?;
    let transforms = d
        .read_sequence(|d| apex_io::rosbag::messages::TransformStamped::from_cdr(d))
        .ok()?;
    let mut s = Ros1Serializer::new();
    s.write_u32(transforms.len() as u32);
    for t in transforms {
        let (v, q) = (t.transform.translation, t.transform.rotation);
        ros1_msg::TransformStamped {
            header: ros1_msg::Header {
                seq: 0,
                stamp_ns: t.header.stamp.sec as u64 * 1_000_000_000 + t.header.stamp.nanosec as u64,
                frame_id: t.header.frame_id,
            },
            child_frame_id: t.child_frame_id,
            transform: ros1_msg::Transform {
                translation: ros1_msg::Vector3 {
                    x: v.x,
                    y: v.y,
                    z: v.z,
                },
                rotation: ros1_msg::Quaternion {
                    x: q.x,
                    y: q.y,
                    z: q.z,
                    w: q.w,
                },
            },
        }
        .to_ros1(&mut s);
    }
    Some(s.into_bytes())
}

fn conv_point_cloud2_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let cloud = ros1_msg::PointCloud2::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    write_header_cdr(&mut s, &cloud.header);
    s.write_u32(cloud.height);
    s.write_u32(cloud.width);
    s.write_u32(cloud.fields.len() as u32);
    for field in &cloud.fields {
        s.write_string(&field.name);
        s.write_u32(field.offset);
        s.write_u8(field.datatype);
        s.write_u32(field.count);
    }
    s.write_u8(u8::from(cloud.is_bigendian));
    s.write_u32(cloud.point_step);
    s.write_u32(cloud.row_step);
    s.write_byte_sequence(&cloud.data);
    s.write_u8(u8::from(cloud.is_dense));
    Some(s.into_bytes())
}

fn conv_point_cloud2_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    use apex_io::rosbag::messages::FromCdr;
    let mut d = CdrDeserializer::new(data).ok()?;
    let cloud = apex_io::rosbag::messages::PointCloud2::from_cdr(&mut d).ok()?;
    let header = ros1_msg::Header {
        seq: 0,
        stamp_ns: cloud.header.stamp.sec as u64 * 1_000_000_000 + cloud.header.stamp.nanosec as u64,
        frame_id: cloud.header.frame_id,
    };
    let msg = ros1_msg::PointCloud2 {
        header,
        height: cloud.height,
        width: cloud.width,
        fields: cloud.fields,
        is_bigendian: cloud.is_bigendian,
        point_step: cloud.point_step,
        row_step: cloud.row_step,
        data: cloud.data,
        is_dense: cloud.is_dense,
    };
    let mut s = Ros1Serializer::new();
    msg.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_image_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let img = ros1_msg::Image::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    write_header_cdr(&mut s, &img.header);
    s.write_u32(img.height);
    s.write_u32(img.width);
    s.write_string(&img.encoding);
    s.write_u8(img.is_bigendian);
    s.write_u32(img.step);
    s.write_byte_sequence(&img.data);
    Some(s.into_bytes())
}

fn conv_image_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let header = read_header_ros1(&mut d)?;
    let height = d.read_u32().ok()?;
    let width = d.read_u32().ok()?;
    let encoding = d.read_string().ok()?;
    let is_bigendian = d.read_u8().ok()?;
    let step = d.read_u32().ok()?;
    let pixel_data = d.read_byte_sequence().ok()?;
    let img = ros1_msg::Image {
        header,
        height,
        width,
        encoding,
        is_bigendian,
        step,
        data: pixel_data,
    };
    let mut s = Ros1Serializer::new();
    img.to_ros1(&mut s);
    Some(s.into_bytes())
}

fn conv_compressed_image_ros1_cdr(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = Ros1Deserializer::new(data);
    let img = ros1_msg::CompressedImage::from_ros1(&mut d).ok()?;
    let mut s = CdrSerializer::new();
    write_header_cdr(&mut s, &img.header);
    s.write_string(&img.format);
    s.write_byte_sequence(&img.data);
    Some(s.into_bytes())
}

fn conv_compressed_image_cdr_ros1(data: &[u8]) -> Option<Vec<u8>> {
    let mut d = CdrDeserializer::new(data).ok()?;
    let header = read_header_ros1(&mut d)?;
    let format = d.read_string().ok()?;
    let pixel_data = d.read_byte_sequence().ok()?;
    let img = ros1_msg::CompressedImage {
        header,
        format,
        data: pixel_data,
    };
    let mut s = Ros1Serializer::new();
    img.to_ros1(&mut s);
    Some(s.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_io::rosbag::messages::{FromCdr, PointField};

    fn cloud() -> ros1_msg::PointCloud2 {
        let field = |name: &str, offset: u32| PointField {
            name: name.to_string(),
            offset,
            datatype: 7,
            count: 1,
        };
        ros1_msg::PointCloud2 {
            header: ros1_msg::Header {
                seq: 0,
                stamp_ns: 1_649_764_528_163_469_000,
                frame_id: "PandarXT-32".to_string(),
            },
            height: 1,
            width: 3,
            fields: vec![field("x", 0), field("y", 4), field("z", 8)],
            is_bigendian: false,
            point_step: 12,
            row_step: 36,
            data: (0..36u8).collect(),
            is_dense: true,
        }
    }

    /// A ROS1 `TFMessage` becomes a CDR one apex-io parses to the same
    /// transforms, and comes back byte for byte.
    #[test]
    fn tf_messages_convert_both_ways() -> Result<()> {
        let transform = |child: &str, x: f64| ros1_msg::TransformStamped {
            header: ros1_msg::Header {
                seq: 0,
                stamp_ns: 1_674_207_747_863_851_000,
                frame_id: "imu_sensor_frame".to_string(),
            },
            child_frame_id: child.to_string(),
            transform: ros1_msg::Transform {
                translation: ros1_msg::Vector3 {
                    x,
                    y: -0.5,
                    z: 0.25,
                },
                rotation: ros1_msg::Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: std::f64::consts::FRAC_1_SQRT_2,
                    w: std::f64::consts::FRAC_1_SQRT_2,
                },
            },
        };
        let transforms = [transform("cam0", 0.1), transform("pandar", -0.001)];
        let mut s = Ros1Serializer::new();
        s.write_u32(transforms.len() as u32);
        for t in &transforms {
            t.to_ros1(&mut s);
        }
        let ros1 = s.into_bytes();
        let cdr = conv_tf_message_ros1_cdr(&ros1).ok_or("ROS1 to CDR failed")?;
        let mut d = CdrDeserializer::new(&cdr)?;
        let parsed =
            d.read_sequence(|d| apex_io::rosbag::messages::TransformStamped::from_cdr(d))?;
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].child_frame_id, "pandar");
        assert_eq!(parsed[1].transform.translation.x, -0.001);
        assert_eq!(
            parsed[0].transform.rotation.w,
            std::f64::consts::FRAC_1_SQRT_2
        );
        let back = conv_tf_message_cdr_ros1(&cdr).ok_or("CDR to ROS1 failed")?;
        assert_eq!(back, ros1);
        Ok(())
    }

    /// A ROS1 cloud becomes a CDR cloud apex-io's ROS2 reader parses to the
    /// same fields, and comes back byte for byte.
    #[test]
    fn point_clouds_convert_both_ways() -> Result<()> {
        let mut s = Ros1Serializer::new();
        cloud().to_ros1(&mut s);
        let ros1 = s.into_bytes();
        let cdr = conv_point_cloud2_ros1_cdr(&ros1).ok_or("ROS1 to CDR failed")?;
        let mut d = CdrDeserializer::new(&cdr)?;
        let parsed = apex_io::rosbag::messages::PointCloud2::from_cdr(&mut d)?;
        let original = cloud();
        assert_eq!(parsed.width, original.width);
        assert_eq!(parsed.fields, original.fields);
        assert_eq!(parsed.data, original.data);
        assert_eq!(parsed.header.frame_id, original.header.frame_id);
        assert_eq!(
            parsed.header.stamp.sec as u64 * 1_000_000_000 + parsed.header.stamp.nanosec as u64,
            original.header.stamp_ns
        );
        let back = conv_point_cloud2_cdr_ros1(&cdr).ok_or("CDR to ROS1 failed")?;
        assert_eq!(back, ros1);
        assert!(ros1_to_cdr_fn("sensor_msgs/PointCloud2").is_some());
        assert!(cdr_to_ros1_fn("sensor_msgs/PointCloud2").is_some());
        Ok(())
    }
}
