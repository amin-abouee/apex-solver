#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (topic lists, progress).

//! Copy a bag, keeping only selected topics and/or a time window.
//!
//! Reads any supported container (ROS1 `.bag`, ROS2 sqlite3/MCAP directory,
//! bare `.mcap`) and writes any container whose payload encoding matches:
//! message bytes are copied verbatim -- nothing is deserialized, so the
//! speed is bounded by I/O and (optional) compression. To change the payload
//! encoding (ROS1 <-> ROS2) use `bag_convert`.
//!
//! Usage:
//!   `bag_filter <INPUT> <OUTPUT> [--topics a,b] [--exclude c] [--start NS] [--end NS]
//!               [--format bag|sqlite3|mcap|mcap-file] [--compression none|zstd|lz4|bz2]`
//!
//! `--start` is inclusive and `--end` exclusive, in nanoseconds since the
//! epoch; `--start-offset`/`--end-offset` take seconds relative to the first
//! message instead.

mod common;

use apex_io::rosbag::{BagKind, BagReader, BagWriter, NewConnection, Selection};
use clap::Parser;
use common::{CliResult, parse_compression, resolve_output_kind, throughput};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input bag (ROS1 `.bag`, ROS2 directory, or `.mcap`)
    input: PathBuf,

    /// Output bag path (must not exist)
    output: Option<PathBuf>,

    /// Topics to include (comma-separated; default: all)
    #[arg(short, long, value_delimiter = ',')]
    topics: Vec<String>,

    /// Topics to exclude (comma-separated)
    #[arg(short = 'x', long = "exclude", value_delimiter = ',')]
    exclude_topics: Vec<String>,

    /// First timestamp to keep, ns since epoch (inclusive)
    #[arg(short, long)]
    start: Option<u64>,

    /// Timestamp to stop at, ns since epoch (exclusive)
    #[arg(short, long)]
    end: Option<u64>,

    /// Keep messages from this many seconds after the first message
    #[arg(long)]
    start_offset: Option<f64>,

    /// Stop this many seconds after the first message
    #[arg(long)]
    end_offset: Option<f64>,

    /// Output container: bag, sqlite3, mcap (directory) or mcap-file.
    /// Default: from the output extension (.bag/.mcap), else mcap.
    #[arg(long, alias = "storage")]
    format: Option<String>,

    /// Output compression: none, zstd (mcap / sqlite3), lz4 or bz2 (bag)
    #[arg(long, default_value = "none")]
    compression: String,

    /// List the topics of the input and exit
    #[arg(long)]
    list_topics: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> CliResult<()> {
    let args = Args::parse();
    let mut reader =
        BagReader::open(&args.input).map_err(|e| format!("cannot open input bag: {e}"))?;

    if args.list_topics {
        println!(
            "Topics in {} ({}):",
            args.input.display(),
            reader.kind().name()
        );
        for (i, c) in reader.connections().iter().enumerate() {
            println!(
                "  {}: {} ({}, {}, {} msgs)",
                i + 1,
                c.topic,
                c.message_type,
                c.serialization_format,
                c.message_count
            );
        }
        return Ok(());
    }

    let output = args.output.as_ref().ok_or("an OUTPUT path is required")?;
    let kind = resolve_output_kind(args.format.as_deref(), output, BagKind::Ros2Mcap)?;
    let compression = parse_compression(&args.compression)?;

    let start_ns = match (args.start, args.start_offset) {
        (Some(ns), _) => Some(ns),
        (None, Some(s)) => Some(reader.start_time().saturating_add((s * 1e9) as u64)),
        _ => None,
    };
    let end_ns = match (args.end, args.end_offset) {
        (Some(ns), _) => Some(ns),
        (None, Some(s)) => Some(reader.start_time().saturating_add((s * 1e9) as u64)),
        _ => None,
    };
    let selection = Selection {
        include: args.topics.clone(),
        exclude: args.exclude_topics.clone(),
        start_ns,
        end_ns,
    };

    let selected = reader.selected_connections(&selection);
    for topic in &selection.include {
        if !reader.connections().iter().any(|c| &c.topic == topic) {
            tracing::warn!(topic = %topic, "requested topic not found in input bag");
            eprintln!("WARN: requested topic '{topic}' not found in input bag");
        }
    }
    if selected.is_empty() {
        eprintln!("No topics match the filter criteria");
        return Ok(());
    }

    // Raw copy cannot change the payload encoding.
    let wants_ros1_payload = matches!(kind, BagKind::Ros1Bag);
    for c in &selected {
        let is_ros1 = c.serialization_format == "ros1";
        let compatible = match kind {
            BagKind::McapFile | BagKind::Vrs => true,
            _ => is_ros1 == wants_ros1_payload,
        };
        if !compatible {
            return Err(format!(
                "topic {} carries '{}' payloads, which a {} bag cannot hold as-is; \
                 use bag_convert to change the encoding",
                c.topic,
                c.serialization_format,
                kind.name()
            )
            .into());
        }
    }

    let mut writer = BagWriter::create(output, kind, compression)
        .map_err(|e| format!("cannot create output bag: {e}"))?;
    let mut handles = std::collections::HashMap::new();
    for c in &selected {
        if args.verbose {
            println!("  {} ({})", c.topic, c.message_type);
        }
        handles.insert(
            c.id,
            writer.add_connection(&NewConnection::from_connection(c))?,
        );
    }

    let t0 = std::time::Instant::now();
    let (mut messages, mut bytes) = (0u64, 0u64);
    reader.for_each(&selected, &selection, |c, ts, data| {
        let handle = handles
            .get(&c.id)
            .ok_or_else(|| apex_io::rosbag::BagError::connection_not_found(&c.topic))?;
        writer.write(*handle, ts, data)?;
        messages += 1;
        bytes += data.len() as u64;
        Ok(())
    })?;
    writer.close()?;

    println!(
        "Copied {} -> {}",
        throughput(messages, bytes, t0.elapsed()),
        output.display()
    );
    Ok(())
}
