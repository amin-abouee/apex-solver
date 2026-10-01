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

use apex_io::rosbag::{Reader, ReaderError};
use chrono::TimeZone;
use std::env;
use std::path::Path;

fn main() -> Result<(), ReaderError> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <bag_path>", args[0]);
        eprintln!("Example: {} /path/to/rosbag2_directory", args[0]);
        eprintln!("Example: {} /path/to/recording.mcap", args[0]);
        std::process::exit(1);
    }

    let bag_path = Path::new(&args[1]);

    let mut reader = Reader::new(bag_path)?;
    reader.open()?;

    let total_size = calculate_total_size(bag_path)?;
    let storage_label = match reader.metadata() {
        Some(m) => format!("{} (per metadata.yaml)", m.info().storage_identifier),
        None => "mcap (bare file)".to_string(),
    };

    let duration_ns = reader.duration();
    let duration_s = duration_ns as f64 / 1_000_000_000.0;
    let start_time_ns = reader.start_time();
    let end_time_ns = reader.end_time();
    let message_count = reader.message_count();

    println!("Bag size:          {}", format_size(total_size));
    println!("Storage id:        {storage_label}");
    println!("Duration:          {duration_s:.9}s");
    println!("Start:             {}", format_timestamp(start_time_ns));
    println!("End:               {}", format_timestamp(end_time_ns));
    println!("Messages:          {message_count}");

    let topics = reader.topics();
    if !topics.is_empty() {
        println!("Topic information: {}", format_topic_info(&topics[0]));
        for topic in topics.iter().skip(1) {
            println!("                   {}", format_topic_info(topic));
        }
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

fn format_topic_info(topic: &apex_io::rosbag::types::TopicInfo) -> String {
    format!(
        "{} ({}, {} msgs)",
        topic.name, topic.message_type, topic.message_count
    )
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
