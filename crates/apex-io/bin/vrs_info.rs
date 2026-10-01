#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (stream tables).

//! Inspect a VRS file (Project Aria recording).
//!
//! Reads only the header, description and index -- never the sensor data --
//! so it is instant even for 20 GB recordings.
//!
//! Usage: `vrs_info <file.vrs> [--tags] [--verify-index]`

use apex_io::vrs::{IndexSource, RecordType, VrsReader};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// VRS file
    file: PathBuf,

    /// Print every file and stream tag (calibration JSON is truncated)
    #[arg(long)]
    tags: bool,

    /// Re-scan all record headers and check they agree with the index
    #[arg(long)]
    verify_index: bool,
}

fn shorten(s: &str) -> String {
    let flat = s.replace('\n', " ");
    if flat.chars().count() > 100 {
        format!(
            "{}... ({} bytes)",
            flat.chars().take(100).collect::<String>(),
            s.len()
        )
    } else {
        flat
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut vrs = VrsReader::open(&args.file)?;

    println!("File:      {}", args.file.display());
    println!(
        "Size:      {:.2} GiB",
        vrs.file_len() as f64 / (1u64 << 30) as f64
    );
    println!(
        "Index:     {} ({} records)",
        match vrs.index_source() {
            IndexSource::File => "from file",
            IndexSource::Rebuilt => "REBUILT (file has no usable index)",
        },
        vrs.records().len()
    );
    for key in [
        "device_type",
        "device_serial",
        "session_id",
        "capture_time_epoch",
    ] {
        if let Some(v) = vrs.file_tags().get(key) {
            println!("{key:<10} {v}");
        }
    }

    let streams = vrs.streams();
    println!("\nStreams ({}):", streams.len());
    println!(
        "  {:<9} {:<26} {:>9} {:>9} {:>9}  {:>10} {:>9}  data format",
        "id", "name", "config", "state", "data", "duration_s", "rate_hz"
    );
    for id in &streams {
        let tags = vrs.stream_tags(*id);
        let name = tags
            .and_then(|t| t.vrs.get("VRS_Original_Recordable_Name"))
            .map_or("?", String::as_str);
        let format = tags
            .and_then(|t| t.vrs.iter().find(|(k, _)| k.starts_with("RF:Data:")))
            .map_or("-", |(_, v)| v.as_str());
        let count = |rt: RecordType| {
            vrs.records()
                .iter()
                .filter(|r| r.stream == *id && r.record_type == rt)
                .count()
        };
        let data = vrs.data_records(*id).len();
        let (dur, rate) = match vrs.stream_time_range(*id) {
            Some((a, b)) if b > a && data > 1 => (b - a, (data - 1) as f64 / (b - a)),
            _ => (0.0, 0.0),
        };
        println!(
            "  {:<9} {:<26} {:>9} {:>9} {:>9}  {:>10.3} {:>9.2}  {}",
            id.to_string(),
            name,
            count(RecordType::Configuration),
            count(RecordType::State),
            data,
            dur,
            rate,
            format
        );
    }

    if args.tags {
        println!("\nFile tags:");
        for (k, v) in vrs.file_tags() {
            println!("  {k} = {}", shorten(v));
        }
        for id in &streams {
            if let Some(t) = vrs.stream_tags(*id) {
                println!("\nStream {id} tags:");
                for (k, v) in &t.user {
                    println!("  user {k} = {}", shorten(v));
                }
                for (k, v) in &t.vrs {
                    println!("  vrs  {k} = {}", shorten(v));
                }
            }
        }
    }

    if args.verify_index {
        let t0 = std::time::Instant::now();
        let ok = vrs.index_matches_scan()?;
        println!(
            "\nIndex vs full scan: {} ({:.1}s)",
            if ok { "IDENTICAL" } else { "MISMATCH" },
            t0.elapsed().as_secs_f64()
        );
        if !ok {
            std::process::exit(1);
        }
    }
    Ok(())
}
