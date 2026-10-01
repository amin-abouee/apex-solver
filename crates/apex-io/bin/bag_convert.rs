#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (progress, summaries).

//! Convert bags between ROS1 and ROS2 payload encodings and containers.
//!
//! Reads any supported input (ROS1 `.bag`, ROS2 sqlite3/MCAP directory, bare
//! `.mcap`) and writes `.bag`, a ROS2 directory (sqlite3 or MCAP) or a bare
//! `.mcap`. Messages are streamed: when the payload encoding is unchanged the
//! bytes are copied verbatim; otherwise they are re-encoded ROS1 <-> CDR by a
//! converter compiled from the message definition (embedded in the bag, or
//! from the built-in catalog of standard messages) -- no typed message
//! structs, and byte arrays such as images are block-copied.
//!
//! Topics whose type has no usable definition are skipped with a warning
//! (`--strict` makes that an error).
//!
//! Usage:
//!   `bag_convert <INPUT> <OUTPUT> [--format bag|sqlite3|mcap|mcap-file]
//!                [--encoding ros1|cdr] [--compression none|zstd|lz4|bz2]
//!                [--topics a,b] [--exclude c] [--start NS] [--end NS] [--strict]`

mod common;

use apex_io::rosbag::transcode::{Dialect, Transcoder, dialect_of, plan_connections};
use apex_io::rosbag::{
    BagError, BagKind, BagReader, BagWriter, ConnectionHandle, NewConnection, Selection,
};
use clap::Parser;
use common::{CliResult, parse_compression, resolve_output_kind, throughput};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input bag (ROS1 `.bag`, ROS2 directory, or `.mcap`)
    input: PathBuf,

    /// Output path (must not exist)
    output: Option<PathBuf>,

    /// Output container: bag, sqlite3, mcap (directory) or mcap-file.
    /// Default: from the output extension (.bag/.mcap); otherwise ROS1
    /// input -> mcap directory, ROS2 input -> .bag
    #[arg(long)]
    format: Option<String>,

    /// Payload encoding to write: ros1 or cdr. Only needed for `mcap-file`
    /// (a `.bag` is always ros1 and ROS2 directories are always cdr)
    #[arg(long)]
    encoding: Option<String>,

    /// Output compression: none, zstd (mcap / sqlite3), lz4 or bz2 (bag)
    #[arg(long, default_value = "none")]
    compression: String,

    /// Topics to convert (comma-separated; default: all)
    #[arg(short, long, value_delimiter = ',')]
    topics: Vec<String>,

    /// Topics to exclude (comma-separated)
    #[arg(short = 'x', long = "exclude", value_delimiter = ',')]
    exclude_topics: Vec<String>,

    /// First timestamp to convert, ns since epoch (inclusive)
    #[arg(short, long)]
    start: Option<u64>,

    /// Timestamp to stop at, ns since epoch (exclusive)
    #[arg(short, long)]
    end: Option<u64>,

    /// Fail instead of skipping topics that cannot be converted
    #[arg(long)]
    strict: bool,

    /// Show which topics can be converted and exit
    #[arg(long)]
    list_topics: bool,

    /// Verbose output (per-topic statistics)
    #[arg(short, long)]
    verbose: bool,
}

/// Payload encoding to write, or `None` to keep every message as it is
/// (containers that can hold either encoding and no `--encoding` given).
fn target_dialect(kind: BagKind, encoding: Option<&str>) -> CliResult<Option<Dialect>> {
    let requested = match encoding {
        Some(e) => {
            Some(dialect_of(e).ok_or_else(|| format!("unknown encoding '{e}' (ros1 or cdr)"))?)
        }
        None => None,
    };
    match (kind, requested) {
        (BagKind::Ros1Bag, Some(Dialect::Ros2)) => {
            Err("a .bag file can only hold ros1 payloads".into())
        }
        (BagKind::Ros2Sqlite | BagKind::Ros2Mcap, Some(Dialect::Ros1)) => {
            Err("ROS2 bag directories can only hold cdr payloads".into())
        }
        (BagKind::Ros1Bag, _) => Ok(Some(Dialect::Ros1)),
        (BagKind::Ros2Sqlite | BagKind::Ros2Mcap, _) => Ok(Some(Dialect::Ros2)),
        (BagKind::McapFile, d) => Ok(Some(d.unwrap_or(Dialect::Ros2))),
        // VRS stores raw message bytes: keep them unless asked to re-encode.
        (BagKind::Vrs, d) => Ok(d),
    }
}

enum Action {
    Copy(ConnectionHandle),
    Convert(ConnectionHandle, Box<Transcoder>),
}

fn main() -> CliResult<()> {
    apex_io::init_logger();
    let args = Args::parse();
    let mut reader =
        BagReader::open(&args.input).map_err(|e| format!("cannot open input bag: {e}"))?;

    let selection = Selection {
        include: args.topics.clone(),
        exclude: args.exclude_topics.clone(),
        start_ns: args.start,
        end_ns: args.end,
    };
    let selected = reader.selected_connections(&selection);

    // Default target: ROS1 input -> MCAP directory, ROS2 input -> .bag.
    let input_is_ros1 = selected
        .first()
        .or_else(|| reader.connections().first())
        .is_some_and(|c| c.serialization_format == "ros1");
    let default_kind = if input_is_ros1 {
        BagKind::Ros2Mcap
    } else {
        BagKind::Ros1Bag
    };

    if args.list_topics {
        let to = if input_is_ros1 {
            Dialect::Ros2
        } else {
            Dialect::Ros1
        };
        println!(
            "Topics in {} ({}):",
            args.input.display(),
            reader.kind().name()
        );
        for c in reader.connections() {
            let status = match apex_io::rosbag::transcode::plan_topic(c, to) {
                Ok(_) => "convertible".to_string(),
                Err(reason) if dialect_of(&c.serialization_format) == Some(to) => {
                    let _ = reason;
                    "already target encoding".to_string()
                }
                Err(reason) => format!("UNSUPPORTED: {reason}"),
            };
            println!(
                "  {} ({}, {}, {} msgs): {status}",
                c.topic, c.message_type, c.serialization_format, c.message_count
            );
        }
        return Ok(());
    }

    let output = args.output.as_ref().ok_or("an OUTPUT path is required")?;
    let kind = resolve_output_kind(args.format.as_deref(), output, default_kind)?;
    let to = target_dialect(kind, args.encoding.as_deref())?;
    let compression = parse_compression(&args.compression)?;

    // Split topics into pass-through and to-be-converted.
    let (same, different): (Vec<_>, Vec<_>) = match to {
        Some(to) => selected
            .iter()
            .cloned()
            .partition(|c| dialect_of(&c.serialization_format) == Some(to)),
        None => (selected.clone(), Vec::new()),
    };
    let (planned, unsupported) = match to {
        Some(to) => plan_connections(&different, to),
        None => (Vec::new(), Vec::new()),
    };
    if args.strict && !unsupported.is_empty() {
        return Err(format!(
            "{} topic(s) cannot be converted (see warnings above); rerun without --strict to skip them",
            unsupported.len()
        )
        .into());
    }
    if same.is_empty() && planned.is_empty() {
        return Err("no convertible topics found".into());
    }

    let mut writer = BagWriter::create(output, kind, compression)
        .map_err(|e| format!("cannot create output bag: {e}"))?;
    let mut actions: HashMap<u32, Action> = HashMap::new();
    let mut readable = Vec::new();
    for c in &same {
        let h = writer.add_connection(&NewConnection::from_connection(c))?;
        actions.insert(c.id, Action::Copy(h));
        readable.push(c.clone());
    }
    for (c, plan) in planned {
        let h = writer.add_connection(&plan.target)?;
        actions.insert(c.id, Action::Convert(h, Box::new(plan.transcoder)));
        readable.push(c);
    }

    let t0 = std::time::Instant::now();
    let (mut messages, mut bytes, mut failed) = (0u64, 0u64, 0u64);
    let mut per_topic: HashMap<String, u64> = HashMap::new();
    let mut scratch = Vec::new();
    reader.for_each(&readable, &selection, |c, ts, data| {
        let action = actions
            .get(&c.id)
            .ok_or_else(|| BagError::connection_not_found(&c.topic))?;
        match action {
            Action::Copy(h) => {
                writer.write(*h, ts, data)?;
                bytes += data.len() as u64;
            }
            Action::Convert(h, t) => match t.convert(data, &mut scratch) {
                Ok(()) => {
                    writer.write(*h, ts, &scratch)?;
                    bytes += scratch.len() as u64;
                }
                Err(e) => {
                    failed += 1;
                    if failed <= 5 {
                        tracing::warn!(topic = %c.topic, timestamp = ts, "message not converted: {e}");
                    }
                    return Ok(());
                }
            },
        }
        messages += 1;
        if args.verbose {
            *per_topic.entry(c.topic.clone()).or_default() += 1;
        }
        Ok(())
    })?;
    writer.close()?;

    if failed > 0 {
        tracing::warn!("{failed} message(s) could not be converted and were dropped");
    }
    if args.verbose {
        let mut topics: Vec<_> = per_topic.into_iter().collect();
        topics.sort();
        for (topic, n) in topics {
            println!("  {topic}: {n} msgs");
        }
    }
    println!(
        "Converted {} -> {} ({}){}",
        throughput(messages, bytes, t0.elapsed()),
        output.display(),
        kind.name(),
        if unsupported.is_empty() {
            String::new()
        } else {
            format!("; {} topic(s) skipped as unsupported", unsupported.len())
        }
    );
    if failed > 0 {
        std::process::exit(2);
    }
    Ok(())
}
