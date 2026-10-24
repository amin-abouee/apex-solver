#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (progress, summary).

//! Copy, trim or re-compress a VRS file.
//!
//! Keeps tags, record formats and payload bytes; optionally selects streams
//! and a time window (seconds relative to the first data record, or absolute
//! with `--absolute`).
//!
//! Usage: `vrs_copy <IN> <OUT> [--streams 1202-1,1201-1] [--start 10 --end 20] [--compression zstd|lz4|none]`

use apex_io::vrs::{Compression, CopyOptions, StreamId, VrsReader, copy_filtered};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .vrs
    input: PathBuf,
    /// Output .vrs (must not exist)
    output: PathBuf,

    /// Streams to keep, as `type-instance` (comma-separated; default all)
    #[arg(long, value_delimiter = ',')]
    streams: Vec<String>,

    /// First data time to keep, in seconds
    #[arg(long)]
    start: Option<f64>,

    /// Data time to stop at, in seconds (exclusive)
    #[arg(long)]
    end: Option<f64>,

    /// Treat --start/--end as absolute timestamps instead of offsets from the first data record
    #[arg(long)]
    absolute: bool,

    /// Output compression: zstd, lz4 or none
    #[arg(long, default_value = "zstd")]
    compression: String,
}

fn parse_stream(s: &str) -> Result<StreamId, String> {
    let (t, i) = s
        .split_once('-')
        .ok_or_else(|| format!("bad stream '{s}', expected type-instance"))?;
    Ok(StreamId::new(
        t.parse().map_err(|_| format!("bad stream type in '{s}'"))?,
        i.parse()
            .map_err(|_| format!("bad stream instance in '{s}'"))?,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    apex_io::init_logger();
    let args = Args::parse();
    let mut src = VrsReader::open(&args.input)?;

    let first = src
        .streams()
        .into_iter()
        .filter_map(|s| src.stream_time_range(s).map(|r| r.0))
        .fold(f64::INFINITY, f64::min);
    let base = if args.absolute || !first.is_finite() {
        0.0
    } else {
        first
    };

    let compression = match args.compression.as_str() {
        "zstd" => Compression::Zstd,
        "lz4" => Compression::Lz4,
        "none" => Compression::None,
        other => return Err(format!("unknown compression '{other}'").into()),
    };
    let options = CopyOptions {
        streams: args
            .streams
            .iter()
            .map(|s| parse_stream(s))
            .collect::<Result<_, _>>()?,
        start: args.start.map(|s| base + s),
        end: args.end.map(|e| base + e),
        compression: Some(compression),
    };

    let t0 = std::time::Instant::now();
    let summary = copy_filtered(&mut src, &args.output, &options)?;
    println!(
        "Copied {} records ({:.1} MiB) -> {} in {:.2}s",
        summary.records,
        summary.bytes as f64 / (1024.0 * 1024.0),
        summary.path.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
