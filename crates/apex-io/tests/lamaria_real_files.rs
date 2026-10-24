//! Checks the LaMAria readers against the dataset on disk.
//!
//! Skipped unless `APEX_LAMARIA_ROOT` points at the directory holding
//! `training/` and `test/`, e.g.
//!
//! ```text
//! APEX_LAMARIA_ROOT=/data/lamaria \
//!   cargo test -p apex-io --features vrs --release --test lamaria_real_files
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};

use apex_io::lamaria::Calibration;
use apex_io::vrs::aria::{AriaRecording, AriaSensor};
use apex_io::{TrajectoryLoader, TumNanosLoader};

fn root() -> Option<PathBuf> {
    std::env::var("APEX_LAMARIA_ROOT").ok().map(PathBuf::from)
}

/// Every `<split>/<sequence>` directory.
fn sequences(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for split in ["training", "test"] {
        let Ok(entries) = std::fs::read_dir(root.join(split)) else {
            continue;
        };
        found.extend(entries.filter_map(|e| e.ok()).map(|e| e.path()));
    }
    found.sort();
    found
}

fn name(sequence: &Path) -> String {
    sequence.file_name().unwrap().to_string_lossy().into_owned()
}

#[test]
fn every_calibration_file_parses() {
    let Some(root) = root() else {
        eprintln!("APEX_LAMARIA_ROOT not set; skipping");
        return;
    };
    let sequences = sequences(&root);
    assert!(!sequences.is_empty(), "no sequences under {root:?}");
    for sequence in &sequences {
        let file = sequence
            .join("aria_calibrations")
            .join(format!("{}.json", name(sequence)));
        let calibration = Calibration::load(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(calibration.cam0.params.len(), 16, "{file:?}");
        assert_eq!(calibration.cam1.params.len(), 16, "{file:?}");
        assert!(calibration.imu.imu_rate > 0.0, "{file:?}");
    }
    println!("{} calibration files parsed", sequences.len());
}

/// The ground truth loads, is in time order, and lies inside the recording's
/// own time range: which also settles that its unit is nanoseconds.
#[test]
fn ground_truth_loads_and_lies_inside_the_recording() {
    let Some(root) = root() else {
        eprintln!("APEX_LAMARIA_ROOT not set; skipping");
        return;
    };
    let mut checked = 0;
    for sequence in sequences(&root) {
        let id = name(&sequence);
        let truth = sequence.join("ground_truth/pGT").join(format!("{id}.txt"));
        if !truth.exists() {
            continue;
        }
        let trajectory = TumNanosLoader::load(&truth).unwrap_or_else(|e| panic!("{e}"));
        assert!(trajectory.len() > 100, "{id}: {} poses", trajectory.len());
        let (first, last) = trajectory.time_span_seconds().unwrap();

        let vrs = sequence.join("raw_data").join(format!("{id}.vrs"));
        let recording = AriaRecording::open(&vrs).unwrap_or_else(|e| panic!("{id}: {e}"));
        let (start, end) = recording.time_range(AriaSensor::ImuRight).unwrap();
        println!("{id}: truth {first:.2}..{last:.2} s, imu {start:.2}..{end:.2} s");
        assert!(
            first >= start - 1.0 && last <= end + 1.0,
            "{id}: ground truth {first}..{last} s outside the recording {start}..{end} s"
        );
        checked += 1;
    }
    assert!(checked > 0, "no sequence with ground truth under {root:?}");
}
