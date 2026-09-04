//! Minimal reader for the NCLT dataset streams used by the integration tests.
//!
//! [NCLT](https://robots.engin.umich.edu/nclt/) (University of Michigan North
//! Campus Long-Term) publishes each sensor stream separately, so the ~40 MB of
//! CSV used here downloads without the images (~100 GB) or Velodyne scans.
//!
//! Only the three streams the fusion test needs are parsed:
//!
//! | File | Columns |
//! |---|---|
//! | `groundtruth.csv` | `utime, x, y, z, roll, pitch, heading` (local metric, NED) |
//! | `odometry_mu.csv` | `utime, dx, dy, dz, droll, dpitch, dheading` (body-frame increments, ~5 Hz) |
//! | `gps.csv` | `utime, fix_mode, num_sats, lat, lng, alt, track, speed` (lat/lng in **radians**) |
//!
//! Rows with `NaN` — frequent in the GPS stream — are dropped.

use apex_solver::apex_manifolds::se3::SE3;
use nalgebra::{UnitQuaternion, Vector3};
use std::path::Path;

/// Equatorial radius used for the flat-earth GPS projection [m].
const EARTH_RADIUS: f64 = 6_378_135.0;

/// A timestamped pose from the ground-truth stream.
pub struct GtPose {
    /// Timestamp in microseconds.
    pub utime: i64,
    /// Pose in the dataset's local metric frame.
    pub pose: SE3,
}

/// A body-frame odometry increment between consecutive epochs.
pub struct OdoDelta {
    /// Timestamp in microseconds of the *end* of the increment.
    pub utime: i64,
    /// Relative transform from the previous epoch to this one.
    pub delta: SE3,
}

/// A GNSS fix projected into the dataset's local metric frame.
pub struct GpsFix {
    /// Timestamp in microseconds.
    pub utime: i64,
    /// North and east position [m] relative to the projection origin.
    pub north_east: (f64, f64),
}

fn parse_row(line: &str, expected: usize) -> Option<Vec<f64>> {
    let values: Vec<f64> = line
        .trim()
        .split(',')
        .map(|f| f.trim().parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;
    if values.len() < expected || values.iter().any(|v| v.is_nan()) {
        return None;
    }
    Some(values)
}

fn pose_from(x: f64, y: f64, z: f64, roll: f64, pitch: f64, heading: f64) -> SE3 {
    SE3::new(
        Vector3::new(x, y, z),
        UnitQuaternion::from_euler_angles(roll, pitch, heading),
    )
}

/// Read `groundtruth.csv`.
pub fn read_ground_truth(path: &Path) -> std::io::Result<Vec<GtPose>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter_map(|l| parse_row(l, 7))
        .map(|v| GtPose {
            utime: v[0] as i64,
            pose: pose_from(v[1], v[2], v[3], v[4], v[5], v[6]),
        })
        .collect())
}

/// Read `odometry_mu.csv` — body-frame increments, not absolute poses.
pub fn read_odometry(path: &Path) -> std::io::Result<Vec<OdoDelta>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter_map(|l| parse_row(l, 7))
        .map(|v| OdoDelta {
            utime: v[0] as i64,
            delta: pose_from(v[1], v[2], v[3], v[4], v[5], v[6]),
        })
        .collect())
}

/// Read `gps.csv`, keeping 3D fixes and projecting them to a local frame.
///
/// The projection is the usual flat-earth approximation about the first fix;
/// over NCLT's few-hundred-metre campus loop its error is far below the
/// receiver's own noise. The returned coordinates are north/east relative to
/// that origin, so a single translation aligns them with the ground-truth
/// frame — [`local_offset`] computes it.
pub fn read_gps(path: &Path) -> std::io::Result<Vec<GpsFix>> {
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter_map(|l| parse_row(l, 6))
        .filter(|v| v[1] >= 3.0) // 3 = 3D fix; 2D fixes have no usable altitude
        .collect();

    let Some(first) = rows.first() else {
        return Ok(Vec::new());
    };
    let (lat0, lng0) = (first[3], first[4]);

    Ok(rows
        .iter()
        .map(|v| GpsFix {
            utime: v[0] as i64,
            north_east: (
                EARTH_RADIUS * (v[3] - lat0),
                EARTH_RADIUS * lat0.cos() * (v[4] - lng0),
            ),
        })
        .collect())
}

/// The translation taking projected GPS coordinates into the ground-truth
/// frame, from the ground-truth pose at the first fix.
///
/// Both frames are north-east-down and the projection origin is a GPS fix, so
/// the two differ by a translation only.
pub fn local_offset(gps: &[GpsFix], ground_truth: &[GtPose]) -> (f64, f64) {
    let Some(first) = gps.first() else {
        return (0.0, 0.0);
    };
    let at = nearest_pose(ground_truth, first.utime);
    (
        at.pose.translation().x - first.north_east.0,
        at.pose.translation().y - first.north_east.1,
    )
}

/// The ground-truth pose nearest `utime` (the stream is ~150 Hz, so this is
/// within a few milliseconds).
pub fn nearest_pose(ground_truth: &[GtPose], utime: i64) -> &GtPose {
    let idx = ground_truth.partition_point(|g| g.utime < utime);
    let idx = idx.min(ground_truth.len() - 1);
    if idx > 0 {
        let (before, after) = (&ground_truth[idx - 1], &ground_truth[idx]);
        if (utime - before.utime).abs() <= (after.utime - utime).abs() {
            return before;
        }
    }
    &ground_truth[idx]
}

/// Median of a set of errors.
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}
