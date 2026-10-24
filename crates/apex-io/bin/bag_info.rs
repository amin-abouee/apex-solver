#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (bag summary tables).

//! Display comprehensive bag file information
//!
//! Provides detailed information about a ROS2 bag directory (via its
//! `metadata.yaml`) or a bare `.mcap` file (via its MCAP summary section),
//! including:
//! - Storage files and sizes
//! - Duration and timing information
//! - Topic details with message counts
//! - Human-readable timestamps
//!
//! Never reads message data: either the summary section or the metadata
//! file carries everything shown.
//!
//! Usage: cargo run -p apex-io --features rosbag --bin bag_info -- <bag_path>

use apex_io::rosbag::transcode::{Dialect, dialect_of, plan_connections};
use apex_io::rosbag::{BagKind, BagReader, ReaderError};
use chrono::TimeZone;
use std::env;
use std::path::Path;

fn main() -> Result<(), ReaderError> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <bag_path>", args[0]);
        eprintln!("Accepts a ROS1 .bag, a ROS2 bag directory (sqlite3/mcap), or a bare .mcap");
        std::process::exit(1);
    }

    apex_io::init_logger();
    let bag_path = Path::new(&args[1]);
    let reader = BagReader::open(bag_path)?;

    let total_size = calculate_total_size(bag_path)?;
    let storage_label = match reader.kind() {
        BagKind::Ros1Bag => "ros1 bag",
        BagKind::Ros2Sqlite => "sqlite3 (ROS2 directory)",
        BagKind::Ros2Mcap => "mcap (ROS2 directory)",
        BagKind::McapFile => "mcap (bare file)",
        BagKind::Vrs => "vrs (Project Aria)",
    };

    let duration_ns = reader.end_time().saturating_sub(reader.start_time());
    let duration_s = duration_ns as f64 / 1_000_000_000.0;

    println!("Bag size:          {}", format_size(total_size));
    println!("Storage id:        {storage_label}");
    println!("Duration:          {duration_s:.9}s");
    println!(
        "Start:             {}",
        format_timestamp(reader.start_time())
    );
    println!("End:               {}", format_timestamp(reader.end_time()));
    println!("Messages:          {}", reader.message_count());

    let mut connections = reader.connections().to_vec();
    connections.sort_by(|a, b| a.topic.cmp(&b.topic));

    // Which topics `bag_convert` could re-encode to the other ROS version?
    // (Unsupported ones are reported with a warning by the library.)
    let ros1_payload = connections
        .iter()
        .any(|c| dialect_of(&c.serialization_format) == Some(Dialect::Ros1));
    let to = if ros1_payload {
        Dialect::Ros2
    } else {
        Dialect::Ros1
    };
    let (_, unsupported) = plan_connections(&connections, to);
    for (i, c) in connections.iter().enumerate() {
        let label = if i == 0 {
            "Topic information:"
        } else {
            "                  "
        };
        let hz = if duration_s > 0.0 {
            format!("{:.1} Hz", c.message_count as f64 / duration_s)
        } else {
            "-".to_string()
        };
        let note = if unsupported.iter().any(|u| u.topic == c.topic) {
            " [not convertible]"
        } else {
            ""
        };
        println!(
            "{label} {} ({}, {}, {} msgs, {hz}){note}",
            c.topic, c.message_type, c.serialization_format, c.message_count
        );
    }

    Ok(())
}

fn calculate_total_size(bag_path: &Path) -> Result<u64, ReaderError> {
    let mut total_size = 0u64;
    if bag_path.is_file() {
        if let Ok(meta) = std::fs::metadata(bag_path) {
            total_size += meta.len();
        }
        return Ok(total_size);
    }
    if let Ok(entries) = std::fs::read_dir(bag_path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata()
                && meta.is_file()
            {
                total_size += meta.len();
            }
        }
    }
    Ok(total_size)
}

fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

fn format_timestamp(timestamp_ns: u64) -> String {
    let timestamp_s = (timestamp_ns / 1_000_000_000) as i64;
    let timestamp_ns_frac = (timestamp_ns % 1_000_000_000) as u32;
    if let Some(datetime) = chrono::Utc
        .timestamp_opt(timestamp_s, timestamp_ns_frac)
        .single()
    {
        format!(
            "{} ({}.{:09})",
            datetime.format("%b %e %Y %H:%M:%S%.9f"),
            timestamp_s,
            timestamp_ns_frac
        )
    } else {
        format!("Invalid timestamp ({timestamp_s}.{timestamp_ns_frac:09})")
    }
}
