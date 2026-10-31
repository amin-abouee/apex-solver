#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (progress, summaries).

//! Concatenate consecutive bags of one recording into a single bag.
//!
//! Long recordings are often cut into fixed-duration files (the VBR dataset
//! splits every 80 s). This joins them back, in the order given, without
//! re-encoding: payload bytes are copied verbatim, so every input must already
//! use the encoding the output container holds (ROS1 for `.bag`, CDR for ROS2
//! directories). Convert first with `bag_convert` if they differ.
//!
//! Each topic is declared once, from the first input that has it. A later input
//! that declares the same topic with a different type or encoding is an error,
//! as is an input whose messages start before the end of the previous one
//! (more than `--tolerance-ms`): the inputs are then not consecutive and the
//! merged file would not be time-ordered.
//!
//! Usage:
//!   `bag_merge <INPUT>... --output <OUTPUT> [--format bag|sqlite3|mcap|mcap-file]
//!              [--compression none|zstd|lz4|bz2] [--tolerance-ms N]`

mod common;

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
    /// Consecutive input bags, in time order
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Output path (must not exist)
    #[arg(short, long)]
    output: PathBuf,

    /// Output container: bag, sqlite3, mcap (directory) or mcap-file.
    /// Default: from the output extension, else the first input's container
    #[arg(long)]
    format: Option<String>,

    /// Output compression: none, zstd (mcap / sqlite3), lz4 or bz2 (bag)
    #[arg(long, default_value = "none")]
    compression: String,

    /// How far (ms) an input may start before the previous one ended
    #[arg(long, default_value_t = 50)]
    tolerance_ms: u64,
}

/// What a topic was declared as, to catch inputs that disagree.
struct Declared {
    handle: ConnectionHandle,
    message_type: String,
    serialization_format: String,
}

fn default_kind(first: &BagReader) -> BagKind {
    first.kind()
}

fn main() -> CliResult<()> {
    apex_io::init_logger();
    let args = Args::parse();

    let first = BagReader::open(&args.inputs[0])
        .map_err(|e| format!("cannot open {}: {e}", args.inputs[0].display()))?;
    let kind = resolve_output_kind(args.format.as_deref(), &args.output, default_kind(&first))?;
    drop(first);
    let compression = parse_compression(&args.compression)?;
    let mut writer = BagWriter::create(&args.output, kind, compression)
        .map_err(|e| format!("cannot create output bag: {e}"))?;

    let tolerance_ns = args.tolerance_ms.saturating_mul(1_000_000);
    let selection = Selection::default();
    let mut declared: HashMap<String, Declared> = HashMap::new();
    let (mut messages, mut bytes) = (0u64, 0u64);
    let mut last_end: Option<u64> = None;
    let t0 = std::time::Instant::now();

    for input in &args.inputs {
        let mut reader =
            BagReader::open(input).map_err(|e| format!("cannot open {}: {e}", input.display()))?;
        let connections = reader.selected_connections(&selection);
        let mut handles: HashMap<u32, ConnectionHandle> = HashMap::new();
        for c in &connections {
            match declared.get(&c.topic) {
                Some(d)
                    if d.message_type != c.message_type
                        || d.serialization_format != c.serialization_format =>
                {
                    return Err(format!(
                        "{}: topic {} is {} ({}), earlier inputs had {} ({})",
                        input.display(),
                        c.topic,
                        c.message_type,
                        c.serialization_format,
                        d.message_type,
                        d.serialization_format
                    )
                    .into());
                }
                Some(d) => {
                    handles.insert(c.id, d.handle);
                }
                None => {
                    let handle = writer.add_connection(&NewConnection::from_connection(c))?;
                    handles.insert(c.id, handle);
                    declared.insert(
                        c.topic.clone(),
                        Declared {
                            handle,
                            message_type: c.message_type.clone(),
                            serialization_format: c.serialization_format.clone(),
                        },
                    );
                }
            }
        }

        let (mut first_ts, mut last_ts) = (u64::MAX, 0u64);
        reader.for_each(&connections, &selection, |c, ts, data| {
            let handle = handles
                .get(&c.id)
                .ok_or_else(|| BagError::connection_not_found(&c.topic))?;
            writer.write(*handle, ts, data)?;
            first_ts = first_ts.min(ts);
            last_ts = last_ts.max(ts);
            messages += 1;
            bytes += data.len() as u64;
            Ok(())
        })?;

        if let Some(previous_end) = last_end
            && first_ts.saturating_add(tolerance_ns) < previous_end
        {
            return Err(format!(
                "{} starts {:.3} s before the previous input ended; inputs must be consecutive",
                input.display(),
                (previous_end - first_ts) as f64 * 1e-9
            )
            .into());
        }
        last_end = Some(last_end.map_or(last_ts, |e| e.max(last_ts)));
        println!(
            "  {}: {:.3} .. {:.3} s",
            input.display(),
            first_ts as f64 * 1e-9,
            last_ts as f64 * 1e-9
        );
    }
    writer.close()?;
    println!(
        "Merged {} input(s): {} -> {} ({})",
        args.inputs.len(),
        throughput(messages, bytes, t0.elapsed()),
        args.output.display(),
        kind.name()
    );
    Ok(())
}
