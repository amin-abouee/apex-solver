//! Copy and filter ROS2 bag files
//!
//! Reads an existing ROS2 bag file and writes it to a new location with optional
//! topic and time-range filtering. Supports both SQLite3 and MCAP formats.
//!
//! Usage:
//!   cargo run -p apex-io --features rosbag-sqlite --bin bag_filter -- \
//!       <input_bag> <output_bag> [--topics topic1,topic2,...] [--start NS] [--end NS]
//!
//! Examples:
//!   # Copy entire bag
//!   cargo run -p apex-io --features rosbag-sqlite --bin bag_filter -- ./input ./output
//!
//!   # Copy only specific topics
//!   cargo run -p apex-io --features rosbag-sqlite --bin bag_filter -- \
//!       ./input ./output --topics /camera/image_raw,/imu/data
//!
//!   # Copy with time filtering
//!   cargo run -p apex-io --features rosbag-sqlite --bin bag_filter -- \
//!       ./input ./output --start 1000000000 --end 2000000000

use apex_io::rosbag::types::{CompressionFormat, CompressionMode, Connection, StoragePlugin};
use apex_io::rosbag::{Reader, Writer};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Arguments for copy helper functions
struct CopyArgs<'a> {
    connections: &'a [Connection],
    conn_map: &'a HashMap<String, Connection>,
    start: Option<u64>,
    end: Option<u64>,
    batch_size: usize,
    verbose: bool,
}

/// Copy a ROS2 bag file with optional topic filtering
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input bag directory
    input: PathBuf,

    /// Output bag directory
    output: PathBuf,

    /// Topics to include (comma-separated; if empty, all topics are included)
    #[arg(short, long, value_delimiter = ',')]
    topics: Vec<String>,

    /// Topics to exclude (comma-separated)
    #[arg(short = 'x', long = "exclude", value_delimiter = ',')]
    exclude_topics: Vec<String>,

    /// Start time (nanoseconds since epoch)
    #[arg(short, long)]
    start: Option<u64>,

    /// End time (nanoseconds since epoch)
    #[arg(short, long)]
    end: Option<u64>,

    /// Storage plugin for output (`sqlite3` or `mcap`)
    #[arg(long, default_value = "sqlite3")]
    storage: String,

    /// Compression mode (`none`, `file`, or `message`)
    #[arg(long, default_value = "none")]
    compression_mode: String,

    /// Compression format (`none` or `zstd`)
    #[arg(long, default_value = "none")]
    compression_format: String,

    /// Use standard (slower) copy with deserialization/serialization instead of raw copy
    #[arg(long)]
    standard_copy: bool,

    /// Buffer size in MB for raw copy mode (default: 50 MB)
    #[arg(long, default_value = "50")]
    buffer_size_mb: usize,

    /// Batch size for bulk operations in raw copy mode
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// List all topics in the bag and exit
    #[arg(long)]
    list_topics: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut reader =
        Reader::new(&args.input).map_err(|e| format!("Failed to create reader: {e}"))?;
    reader
        .open()
        .map_err(|e| format!("Failed to open input bag: {e}"))?;

    let connections = reader.connections();

    if args.list_topics {
        println!("Available topics in bag:");
        for (i, conn) in connections.iter().enumerate() {
            println!("  {}: {} ({})", i + 1, conn.topic, conn.message_type);
        }
        println!("\nTotal topics: {}", connections.len());
        return Ok(());
    }

    if args.verbose {
        println!(
            "Copying bag from {} to {}",
            args.input.display(),
            args.output.display()
        );
        if !args.standard_copy {
            println!(
                "Using raw copy mode (buffer: {}MB, batch: {})",
                args.buffer_size_mb, args.batch_size
            );
        }
    }

    let storage_plugin = match args.storage.as_str() {
        "sqlite3" => StoragePlugin::Sqlite3,
        "mcap" => StoragePlugin::Mcap,
        other => {
            return Err(
                format!("Unsupported storage plugin: '{other}'. Use 'sqlite3' or 'mcap'").into(),
            )
        }
    };

    let compression_mode = match args.compression_mode.as_str() {
        "none" => CompressionMode::None,
        "file" => CompressionMode::File,
        "message" => CompressionMode::Message,
        "storage" => CompressionMode::Storage,
        other => {
            return Err(format!(
                "Unsupported compression mode: '{other}'. Use 'none', 'file', 'message', or 'storage'"
            )
            .into())
        }
    };

    let compression_format = match args.compression_format.as_str() {
        "none" => CompressionFormat::None,
        "zstd" => CompressionFormat::Zstd,
        other => {
            return Err(
                format!("Unsupported compression format: '{other}'. Use 'none' or 'zstd'").into(),
            )
        }
    };

    let mut writer = Writer::new(&args.output, None, Some(storage_plugin))
        .map_err(|e| format!("Failed to create writer: {e}"))?;
    writer.set_compression(compression_mode, compression_format)?;

    if !args.standard_copy {
        writer.configure_buffer(args.buffer_size_mb, args.batch_size)?;
    }

    writer
        .open()
        .map_err(|e| format!("Failed to open output bag: {e}"))?;

    let filtered_connections: Vec<_> = connections
        .iter()
        .filter(|conn| {
            let include = args.topics.is_empty() || args.topics.contains(&conn.topic);
            let exclude = args.exclude_topics.contains(&conn.topic);
            include && !exclude
        })
        .cloned()
        .collect();

    if filtered_connections.is_empty() {
        println!("No topics match the filter criteria");
        return Ok(());
    }

    if args.verbose {
        println!("Selected {} topics:", filtered_connections.len());
        for conn in &filtered_connections {
            println!("  {} ({})", conn.topic, conn.message_type);
        }
    }

    let mut conn_map = HashMap::new();
    for r_conn in &filtered_connections {
        let w_conn = writer.add_connection(
            r_conn.topic.clone(),
            r_conn.message_type.clone(),
            Some(r_conn.message_definition.clone()),
            Some(r_conn.type_description_hash.clone()),
            Some(r_conn.serialization_format.clone()),
            Some(r_conn.offered_qos_profiles.clone()),
        )?;
        conn_map.insert(r_conn.topic.clone(), w_conn);
    }

    let copy_args = CopyArgs {
        connections: &filtered_connections,
        conn_map: &conn_map,
        start: args.start,
        end: args.end,
        batch_size: args.batch_size,
        verbose: args.verbose,
    };

    if args.standard_copy {
        copy_messages(&mut reader, &mut writer, &copy_args)?;
    } else {
        copy_raw_messages(&mut reader, &mut writer, &copy_args)?;
    }

    writer
        .close()
        .map_err(|e| format!("Failed to close output bag: {e}"))?;
    reader
        .close()
        .map_err(|e| format!("Failed to close input bag: {e}"))?;

    println!("Bag copy completed successfully");
    Ok(())
}

fn copy_raw_messages(reader: &mut Reader, writer: &mut Writer, args: &CopyArgs) -> Result<()> {
    if args.verbose {
        println!("Starting high-performance raw copy...");
    }
    let t0 = std::time::Instant::now();

    let raw_messages = reader
        .read_raw_messages_batch(Some(args.connections), args.start, args.end)
        .map_err(|e| format!("Failed to read raw messages: {e}"))?;

    if args.verbose {
        println!("Read {} messages in {:?}", raw_messages.len(), t0.elapsed());
    }

    let t1 = std::time::Instant::now();
    let batch: Result<Vec<(Connection, u64, Vec<u8>)>> = raw_messages
        .into_iter()
        .map(|msg| {
            let w_conn = args
                .conn_map
                .get(&msg.connection.topic)
                .ok_or_else(|| {
                    format!(
                        "Connection for topic '{}' not found in writer",
                        msg.connection.topic
                    )
                })?
                .clone();
            Ok((w_conn, msg.timestamp, msg.raw_data))
        })
        .collect();

    let mut total_written = 0;
    for chunk in batch?.chunks(args.batch_size) {
        writer
            .write_raw_messages_batch(chunk)
            .map_err(|e| format!("Failed to write raw message batch: {e}"))?;
        total_written += chunk.len();
    }

    if args.verbose {
        println!("Wrote {total_written} messages in {:?}", t1.elapsed());
        println!("Total time: {:?}", t0.elapsed());
    }
    Ok(())
}

fn copy_messages(reader: &mut Reader, writer: &mut Writer, args: &CopyArgs) -> Result<()> {
    if args.verbose {
        println!("Starting standard copy...");
    }
    let t0 = std::time::Instant::now();

    let messages = reader
        .messages_filtered(Some(args.connections), args.start, args.end)
        .map_err(|e| format!("Failed to get message iterator: {e}"))?;

    let mut count = 0;
    for message_result in messages {
        let message = message_result.map_err(|e| format!("Failed to read message: {e}"))?;
        let w_conn = args
            .conn_map
            .get(&message.connection.topic)
            .ok_or_else(|| {
                format!(
                    "Connection for topic '{}' not found in writer",
                    message.connection.topic
                )
            })?;
        writer
            .write(w_conn, message.timestamp, &message.data)
            .map_err(|e| format!("Failed to write message: {e}"))?;
        count += 1;
    }

    if args.verbose {
        println!("Copied {count} messages in {:?}", t0.elapsed());
    }
    Ok(())
}
