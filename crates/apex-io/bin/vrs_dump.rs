#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Console output is this CLI's user interface (CSV on stdout).

//! Export one stream of an Aria VRS file.
//!
//! * IMU / magnetometer / barometer / GPS / time-domain streams are written
//!   as CSV to stdout.
//! * Camera streams write one `.jpg` per frame into `--images <dir>` (the
//!   original JPEG bytes) and print a CSV index of the frames.
//!
//! Usage: `vrs_dump <file.vrs> --stream 1202-1 [--start S --end E] [--images DIR]`
//! Times are absolute seconds on the device clock (see `vrs_info`).

use apex_io::vrs::StreamId;
use apex_io::vrs::aria::{AriaRecording, Sample};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// VRS file
    file: PathBuf,

    /// Stream as `type-instance`, for example `1202-1`
    #[arg(long)]
    stream: String,

    /// First time to export (seconds, inclusive)
    #[arg(long)]
    start: Option<f64>,

    /// Time to stop at (seconds, exclusive)
    #[arg(long)]
    end: Option<f64>,

    /// Directory for camera frames (created if missing)
    #[arg(long)]
    images: Option<PathBuf>,
}

fn opt3(v: Option<[f32; 3]>) -> String {
    match v {
        Some([x, y, z]) => format!("{x:.9},{y:.9},{z:.9}"),
        None => ",,".to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let (t, i) = args
        .stream
        .split_once('-')
        .ok_or("--stream must look like 1202-1")?;
    let stream = StreamId::new(t.parse()?, i.parse()?);

    let mut rec = AriaRecording::open(&args.file)?;
    let (first, last) = rec
        .vrs()
        .stream_time_range(stream)
        .ok_or("stream has no data records")?;
    let start = args.start.unwrap_or(first);
    let end = args.end.unwrap_or(last + 1.0);
    let mut cursor = rec
        .vrs()
        .cursor(&apex_io::vrs::RecordFilter::streams([stream]).between(start, end));
    if let Some(dir) = &args.images {
        std::fs::create_dir_all(dir)?;
    }

    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut header_done = false;
    while let Some(sample) = rec.next_sample(&mut cursor)? {
        match sample {
            Sample::Imu(s) => {
                if !header_done {
                    writeln!(
                        out,
                        "time_s,capture_ns,ax,ay,az,gx,gy,gz,mx,my,mz,temperature_c"
                    )?;
                    header_done = true;
                }
                writeln!(
                    out,
                    "{:.9},{},{},{},{},{:.6}",
                    s.time_s,
                    s.capture_ns,
                    opt3(s.accel),
                    opt3(s.gyro),
                    opt3(s.mag),
                    s.temperature_c
                )?;
            }
            Sample::Baro(s) => {
                if !header_done {
                    writeln!(
                        out,
                        "time_s,capture_ns,temperature_c,pressure_pa,altitude_m"
                    )?;
                    header_done = true;
                }
                writeln!(
                    out,
                    "{:.9},{},{:.6},{:.6},{:.6}",
                    s.time_s, s.capture_ns, s.temperature_c, s.pressure_pa, s.altitude_m
                )?;
            }
            Sample::Gps(s) => {
                if !header_done {
                    writeln!(
                        out,
                        "time_s,capture_ns,utc_ms,lat,lon,alt,accuracy,provider"
                    )?;
                    header_done = true;
                }
                writeln!(
                    out,
                    "{:.9},{},{},{},{},{},{},{}",
                    s.time_s,
                    s.capture_ns,
                    s.utc_time_ms,
                    s.latitude,
                    s.longitude,
                    s.altitude,
                    s.accuracy,
                    s.provider
                )?;
            }
            Sample::TimeDomain(s) => {
                if !header_done {
                    writeln!(out, "monotonic_ns,real_ns")?;
                    header_done = true;
                }
                writeln!(out, "{},{}", s.monotonic_ns, s.real_ns)?;
            }
            Sample::Image(f) => {
                if !header_done {
                    writeln!(
                        out,
                        "time_s,capture_ns,frame_number,exposure_s,gain,jpeg_bytes,file"
                    )?;
                    header_done = true;
                }
                let file = match &args.images {
                    Some(dir) => {
                        let name = format!("{}_{}.jpg", stream, f.capture_ns);
                        std::fs::write(dir.join(&name), &f.jpeg)?;
                        name
                    }
                    None => String::new(),
                };
                writeln!(
                    out,
                    "{:.9},{},{},{:.9},{:.6},{},{}",
                    f.time_s,
                    f.capture_ns,
                    f.frame_number,
                    f.exposure_s,
                    f.gain,
                    f.jpeg.len(),
                    file
                )?;
            }
            Sample::Other {
                time_s, payload, ..
            } => {
                if !header_done {
                    writeln!(out, "time_s,payload_bytes")?;
                    header_done = true;
                }
                writeln!(out, "{time_s:.9},{}", payload.len())?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
