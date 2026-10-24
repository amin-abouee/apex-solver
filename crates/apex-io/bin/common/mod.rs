//! Helpers shared by the bag_* command-line tools.
#![allow(dead_code)]

use apex_io::rosbag::{BagKind, OutputCompression};
use std::path::Path;
use std::time::Duration;

pub type CliResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Resolve the output container: explicit `--format`, else the extension of
/// `output` (`.bag`, `.mcap`), else `default`.
pub fn resolve_output_kind(
    format: Option<&str>,
    output: &Path,
    default: BagKind,
) -> CliResult<BagKind> {
    match format {
        Some(name) => BagKind::parse(name).ok_or_else(|| {
            format!("unknown format '{name}' (use bag, sqlite3, mcap, or mcap-file)").into()
        }),
        None => Ok(BagKind::from_output_path(output, default)),
    }
}

pub fn parse_compression(name: &str) -> CliResult<OutputCompression> {
    OutputCompression::parse(name)
        .ok_or_else(|| format!("unknown compression '{name}' (use none, zstd, lz4, or bz2)").into())
}

/// `1234 msgs, 56.7 MiB in 1.23s (46.1 MiB/s, 1003 msg/s)`
pub fn throughput(messages: u64, bytes: u64, elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let mib = bytes as f64 / (1024.0 * 1024.0);
    format!(
        "{messages} msgs, {mib:.1} MiB in {secs:.2}s ({:.1} MiB/s, {:.0} msg/s)",
        mib / secs,
        messages as f64 / secs
    )
}
