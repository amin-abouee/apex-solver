#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (verification report).

//! Check that two bags hold the same messages.
//!
//! Compares, per topic, the message count and a digest over every
//! `(timestamp, canonical payload)` pair. The canonical payload is the CDR
//! form: ROS1 payloads are normalised through the schema-driven converter, so
//! the comparison ignores container, payload encoding and the ROS1 header
//! `seq`. Use it to validate `bag_convert` / `bag_filter` output.
//!
//! Usage: `bag_verify <A> <B> [--topics a,b]`
//! Exit status: 0 identical, 1 differences, 2 error.

mod common;

use apex_io::rosbag::transcode::{Dialect, Transcoder, dialect_of, plan_topic};
use apex_io::rosbag::{BagReader, Selection};
use clap::Parser;
use common::CliResult;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    a: PathBuf,
    b: PathBuf,
    /// Only compare these topics (comma-separated)
    #[arg(short, long, value_delimiter = ',')]
    topics: Vec<String>,
}

#[derive(Default, PartialEq, Eq, Debug)]
struct Digest {
    count: u64,
    hash: u64,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn digest(path: &PathBuf, topics: &[String]) -> CliResult<BTreeMap<String, Digest>> {
    let mut reader = BagReader::open(path)?;
    let selection = Selection {
        include: topics.to_vec(),
        ..Selection::default()
    };
    let conns = reader.selected_connections(&selection);

    // Canonicalise ROS1 payloads to CDR.
    let mut converters: BTreeMap<u32, Transcoder> = BTreeMap::new();
    let mut skipped = Vec::new();
    for c in &conns {
        if dialect_of(&c.serialization_format) == Some(Dialect::Ros1) {
            match plan_topic(c, Dialect::Ros2) {
                Ok(plan) => {
                    converters.insert(c.id, plan.transcoder);
                }
                Err(reason) => skipped.push(format!("{}: {reason}", c.topic)),
            }
        }
    }
    let readable: Vec<_> = conns
        .iter()
        .filter(|c| {
            dialect_of(&c.serialization_format) != Some(Dialect::Ros1)
                || converters.contains_key(&c.id)
        })
        .cloned()
        .collect();
    for s in skipped {
        eprintln!("WARN: not comparable ({s})");
    }

    let mut out: BTreeMap<String, Digest> = BTreeMap::new();
    let mut scratch = Vec::new();
    reader.for_each(&readable, &selection, |c, ts, data| {
        let payload: &[u8] = match converters.get(&c.id) {
            Some(t) => {
                t.convert(data, &mut scratch)?;
                &scratch
            }
            None => data,
        };
        let d = out.entry(c.topic.clone()).or_insert(Digest {
            count: 0,
            hash: FNV_OFFSET,
        });
        d.count += 1;
        d.hash = fnv(fnv(d.hash, &ts.to_le_bytes()), payload);
        Ok(())
    })?;
    Ok(out)
}

fn main() -> CliResult<()> {
    let args = Args::parse();
    let (a, b) = match (digest(&args.a, &args.topics), digest(&args.b, &args.topics)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let mut differences = 0;
    let topics: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
    for topic in topics {
        match (a.get(topic), b.get(topic)) {
            (Some(x), Some(y)) if x == y => {
                println!("OK    {topic}: {} msgs, digest {:016x}", x.count, x.hash);
            }
            (Some(x), Some(y)) => {
                differences += 1;
                println!(
                    "DIFF  {topic}: A {} msgs {:016x} / B {} msgs {:016x}",
                    x.count, x.hash, y.count, y.hash
                );
            }
            (Some(_), None) => {
                differences += 1;
                println!("ONLY-A {topic}");
            }
            (None, _) => {
                differences += 1;
                println!("ONLY-B {topic}");
            }
        }
    }
    if differences > 0 {
        std::process::exit(1);
    }
    Ok(())
}
