//! Write a dummy ROS2 bag file with all supported message types
//!
//! Demonstrates how to create a ROS2 bag file from scratch using the apex-io writer.
//! Creates dummy data for a wide variety of ROS2 message types commonly used in
//! robotics applications.
//!
//! Usage:
//!   cargo run -p apex-io --features rosbag-sqlite --bin write_dummy_bag -- \
//!       [output_path] [--compression]
//!
//! Example:
//!   cargo run -p apex-io --features rosbag-sqlite --bin write_dummy_bag -- \
//!       ./my_test_bag --compression

use apex_io::rosbag::types::{
    MessageDefinition, MessageDefinitionFormat, QosDurability, QosHistory, QosLiveliness,
    QosProfile, QosReliability, QosTime,
};
use apex_io::rosbag::{CompressionFormat, CompressionMode, StoragePlugin, Writer};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct MessageTypeInfo {
    message_type: String,
    topic: String,
    description: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let output_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("./dummy_bag")
        .to_string();
    let enable_compression = args.contains(&"--compression".to_string());

    println!("Creating dummy ROS2 bag file at: {output_path}");
    if enable_compression {
        println!("Compression: ENABLED (zstd)");
    }

    let mut writer = Writer::new(&output_path, Some(9), Some(StoragePlugin::Sqlite3))?;

    if enable_compression {
        writer.set_compression(CompressionMode::Message, CompressionFormat::Zstd)?;
    }

    writer.set_custom_data("generator".to_string(), "apex-io-example".to_string())?;
    writer.set_custom_data(
        "description".to_string(),
        "Dummy bag with all supported message types".to_string(),
    )?;

    writer.open()?;

    let message_types = get_all_message_types();
    println!("Adding {} message types...", message_types.len());

    let mut connections = Vec::new();
    for msg_info in &message_types {
        println!("  Adding: {} -> {}", msg_info.topic, msg_info.message_type);
        match writer.add_connection(
            msg_info.topic.clone(),
            msg_info.message_type.clone(),
            Some(create_message_definition(&msg_info.message_type)),
            Some(format!("hash_{}", msg_info.message_type.replace('/', "_"))),
            Some("cdr".to_string()),
            Some(create_default_qos()),
        ) {
            Ok(connection) => connections.push((connection, msg_info.description.clone())),
            Err(e) => {
                eprintln!("Warning: Failed to add {}: {}", msg_info.topic, e);
            }
        }
    }

    println!("Successfully added {} connections", connections.len());
    println!("Writing sample messages...");

    let start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;

    for (i, (connection, description)) in connections.iter().enumerate() {
        let timestamp = start_time + (i as u64 * 100_000_000);
        let message_data = create_sample_message_data(&connection.message_type);
        match writer.write(connection, timestamp, &message_data) {
            Ok(()) => println!("  OK {}: {}", connection.topic, description),
            Err(e) => eprintln!("  FAIL {}: {}", connection.topic, e),
        }
    }

    writer.close()?;

    println!(
        "Successfully created dummy bag with {} topics!",
        connections.len()
    );
    println!("Bag location: {output_path}");

    if std::fs::read_to_string(format!("{output_path}/metadata.yaml")).is_ok() {
        println!("\nBag Information:");
        println!("   Storage: SQLite3");
        if enable_compression {
            println!("   Compression: zstd (message-level)");
        }
        println!("   Message count: {}", connections.len());
        println!("   Topics: {}", connections.len());
    }

    Ok(())
}

fn get_all_message_types() -> Vec<MessageTypeInfo> {
    vec![
        MessageTypeInfo {
            message_type: "std_msgs/msg/String".to_string(),
            topic: "/demo/string".to_string(),
            description: "String message demo".to_string(),
        },
        MessageTypeInfo {
            message_type: "std_msgs/msg/Int32".to_string(),
            topic: "/demo/int32".to_string(),
            description: "32-bit integer demo".to_string(),
        },
        MessageTypeInfo {
            message_type: "std_msgs/msg/Float64".to_string(),
            topic: "/demo/float64".to_string(),
            description: "64-bit float demo".to_string(),
        },
        MessageTypeInfo {
            message_type: "std_msgs/msg/Bool".to_string(),
            topic: "/demo/bool".to_string(),
            description: "Boolean demo".to_string(),
        },
        MessageTypeInfo {
            message_type: "std_msgs/msg/Header".to_string(),
            topic: "/demo/header".to_string(),
            description: "Message header with timestamp and frame".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/Point".to_string(),
            topic: "/geometry/point".to_string(),
            description: "3D point (x, y, z)".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/Vector3".to_string(),
            topic: "/geometry/vector3".to_string(),
            description: "3D vector".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/Quaternion".to_string(),
            topic: "/geometry/quaternion".to_string(),
            description: "Quaternion orientation".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/Pose".to_string(),
            topic: "/geometry/pose".to_string(),
            description: "Position and orientation".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/PoseStamped".to_string(),
            topic: "/geometry/pose_stamped".to_string(),
            description: "Timestamped pose".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/Transform".to_string(),
            topic: "/geometry/transform".to_string(),
            description: "Transformation between frames".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/TransformStamped".to_string(),
            topic: "/tf".to_string(),
            description: "Timestamped coordinate transformation".to_string(),
        },
        MessageTypeInfo {
            message_type: "geometry_msgs/msg/Twist".to_string(),
            topic: "/cmd_vel".to_string(),
            description: "Linear and angular velocity commands".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/Image".to_string(),
            topic: "/camera/image_raw".to_string(),
            description: "Uncompressed camera image".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/CompressedImage".to_string(),
            topic: "/camera/image_compressed".to_string(),
            description: "Compressed camera image".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/CameraInfo".to_string(),
            topic: "/camera/camera_info".to_string(),
            description: "Camera calibration parameters".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/LaserScan".to_string(),
            topic: "/scan".to_string(),
            description: "2D laser scan data".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/PointCloud2".to_string(),
            topic: "/pointcloud".to_string(),
            description: "3D point cloud data".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/Imu".to_string(),
            topic: "/imu/data".to_string(),
            description: "Inertial measurement unit data".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/NavSatFix".to_string(),
            topic: "/gps/fix".to_string(),
            description: "GPS navigation satellite fix".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/MagneticField".to_string(),
            topic: "/imu/mag".to_string(),
            description: "Magnetic field measurement".to_string(),
        },
        MessageTypeInfo {
            message_type: "sensor_msgs/msg/Temperature".to_string(),
            topic: "/sensors/temperature".to_string(),
            description: "Temperature measurement".to_string(),
        },
        MessageTypeInfo {
            message_type: "nav_msgs/msg/Odometry".to_string(),
            topic: "/odom".to_string(),
            description: "Robot odometry estimate".to_string(),
        },
        MessageTypeInfo {
            message_type: "nav_msgs/msg/OccupancyGrid".to_string(),
            topic: "/map".to_string(),
            description: "2D occupancy grid map".to_string(),
        },
        MessageTypeInfo {
            message_type: "nav_msgs/msg/Path".to_string(),
            topic: "/path".to_string(),
            description: "Navigation path as sequence of poses".to_string(),
        },
        MessageTypeInfo {
            message_type: "builtin_interfaces/msg/Time".to_string(),
            topic: "/time".to_string(),
            description: "Time representation".to_string(),
        },
        MessageTypeInfo {
            message_type: "builtin_interfaces/msg/Duration".to_string(),
            topic: "/duration".to_string(),
            description: "Duration representation".to_string(),
        },
        MessageTypeInfo {
            message_type: "tf2_msgs/msg/TFMessage".to_string(),
            topic: "/tf_static".to_string(),
            description: "Static coordinate transformations".to_string(),
        },
        MessageTypeInfo {
            message_type: "diagnostic_msgs/msg/DiagnosticArray".to_string(),
            topic: "/diagnostics".to_string(),
            description: "System diagnostic information".to_string(),
        },
    ]
}

fn create_message_definition(message_type: &str) -> MessageDefinition {
    let definition_text = match message_type {
        "std_msgs/msg/String" => "string data",
        "std_msgs/msg/Int32" => "int32 data",
        "std_msgs/msg/Float64" => "float64 data",
        "std_msgs/msg/Bool" => "bool data",
        "geometry_msgs/msg/Point" => "float64 x\nfloat64 y\nfloat64 z",
        "geometry_msgs/msg/Vector3" => "float64 x\nfloat64 y\nfloat64 z",
        "geometry_msgs/msg/Quaternion" => "float64 x\nfloat64 y\nfloat64 z\nfloat64 w",
        "sensor_msgs/msg/Image" => {
            "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data"
        }
        _ => "# Auto-generated message definition",
    };

    MessageDefinition {
        format: MessageDefinitionFormat::Msg,
        data: definition_text.to_string(),
    }
}

fn create_default_qos() -> Vec<QosProfile> {
    vec![QosProfile {
        history: QosHistory::KeepLast,
        depth: 10,
        reliability: QosReliability::Reliable,
        durability: QosDurability::Volatile,
        deadline: QosTime::default(),
        lifespan: QosTime::default(),
        liveliness: QosLiveliness::Automatic,
        liveliness_lease_duration: QosTime::default(),
        avoid_ros_namespace_conventions: false,
    }]
}

fn create_sample_message_data(message_type: &str) -> Vec<u8> {
    match message_type {
        "std_msgs/msg/String" => {
            let s = "Hello from apex-io!";
            let mut data = Vec::new();
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
            while data.len() % 4 != 0 {
                data.push(0);
            }
            data
        }
        "std_msgs/msg/Int32" => 42i32.to_le_bytes().to_vec(),
        "std_msgs/msg/Float64" => std::f64::consts::PI.to_le_bytes().to_vec(),
        "std_msgs/msg/Bool" => vec![1u8],
        "geometry_msgs/msg/Point" => {
            let mut data = Vec::new();
            data.extend_from_slice(&1.0f64.to_le_bytes());
            data.extend_from_slice(&2.0f64.to_le_bytes());
            data.extend_from_slice(&3.0f64.to_le_bytes());
            data
        }
        "geometry_msgs/msg/Vector3" => {
            let mut data = Vec::new();
            data.extend_from_slice(&0.5f64.to_le_bytes());
            data.extend_from_slice(&0.5f64.to_le_bytes());
            data.extend_from_slice(&0.5f64.to_le_bytes());
            data
        }
        "geometry_msgs/msg/Quaternion" => {
            let mut data = Vec::new();
            data.extend_from_slice(&0.0f64.to_le_bytes()); // x
            data.extend_from_slice(&0.0f64.to_le_bytes()); // y
            data.extend_from_slice(&0.0f64.to_le_bytes()); // z
            data.extend_from_slice(&1.0f64.to_le_bytes()); // w (normalized)
            data
        }
        "sensor_msgs/msg/Imu" => {
            let mut data = Vec::new();
            data.extend_from_slice(&0u32.to_le_bytes()); // stamp.sec
            data.extend_from_slice(&0u32.to_le_bytes()); // stamp.nanosec
            let frame_id = "imu_link";
            data.extend_from_slice(&(frame_id.len() as u32).to_le_bytes());
            data.extend_from_slice(frame_id.as_bytes());
            while data.len() % 4 != 0 {
                data.push(0);
            }
            data.extend_from_slice(&0.0f64.to_le_bytes()); // orientation.x
            data.extend_from_slice(&0.0f64.to_le_bytes()); // orientation.y
            data.extend_from_slice(&0.0f64.to_le_bytes()); // orientation.z
            data.extend_from_slice(&1.0f64.to_le_bytes()); // orientation.w
            data
        }
        _ => vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
    }
}
