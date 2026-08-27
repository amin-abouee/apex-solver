//! TUM trajectory format.
//!
//! ```text
//! timestamp tx ty tz qx qy qz qw          whitespace-separated, w LAST
//! ```
//!
//! The interchange format for `evo` and the TUM RGB-D / VI benchmark tooling.
//!
//! # Timestamps are formatted as integers, never through `f64`
//!
//! At epoch magnitudes an `f64` second has an ulp of about 476 ns, so
//! `ns → f64 s → ns` is not the identity. Writing `{sec}.{nanos:09}` and
//! parsing the two halves separately keeps the round trip exact. Foreign files
//! using scientific notation or more than nine fractional digits fall back to
//! `f64`, which is documented as lossy.

use std::fmt::Write as _;
use std::path::Path;

use nalgebra::Vector3;

use apex_manifolds::so3::SO3;

use super::error::{Result, TrajectoryError};
use super::types::{NANOS_PER_SEC, NANOS_PER_SECOND};
use super::{Trajectory, TrajectoryLoader, TrajectoryPose, parse_field, read_file, write_file};
use crate::csv;

/// Fields in one TUM row.
const TUM_COLUMNS: usize = 8;

/// Decimal places for the pose fields.
///
/// Nine digits is 1 nm of position and 1 nrad of rotation — below any sensor
/// this crate reads, and the same width TUM's own tools emit. Deliberately not
/// the crate's `{:.17e}` house style: TUM files are consumed by `evo` and by
/// TUM's Python scripts, where scientific notation is legal but unidiomatic.
const POSE_PRECISION: usize = 9;

/// Loader for the TUM trajectory format.
pub struct TumLoader;

impl TrajectoryLoader for TumLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Trajectory> {
        let path = path.as_ref();
        let content = read_file(path)?;
        Self::parse(&content, path)
    }

    fn write<P: AsRef<Path>>(trajectory: &Trajectory, path: P) -> Result<()> {
        write_file(path.as_ref(), &Self::render(trajectory))
    }
}

impl TumLoader {
    /// Parse TUM text. `path` labels errors only.
    ///
    /// # Errors
    ///
    /// See [`TrajectoryError`].
    pub fn parse(text: &str, path: &Path) -> Result<Trajectory> {
        let rows = csv::split_rows(text, None);
        let mut poses = Vec::with_capacity(rows.len());

        for (index, row) in rows.iter().enumerate() {
            let line = index + 1;
            if row.len() != TUM_COLUMNS {
                return Err(TrajectoryError::MissingColumns {
                    path: path.to_path_buf(),
                    line,
                    expected: TUM_COLUMNS,
                    got: row.len(),
                });
            }

            let timestamp_ns = parse_timestamp(row[0], path, line)?;

            let qx = parse_field(row, 4, "qx", path, line)?;
            let qy = parse_field(row, 5, "qy", path, line)?;
            let qz = parse_field(row, 6, "qz", path, line)?;
            let qw = parse_field(row, 7, "qw", path, line)?;
            let orientation = SO3::try_from_quaternion_wxyz(qw, qx, qy, qz).ok_or_else(|| {
                TrajectoryError::InvalidQuaternion {
                    path: path.to_path_buf(),
                    line,
                    norm: (qw * qw + qx * qx + qy * qy + qz * qz).sqrt(),
                }
            })?;

            poses.push(TrajectoryPose::new(
                timestamp_ns,
                Vector3::new(
                    parse_field(row, 1, "tx", path, line)?,
                    parse_field(row, 2, "ty", path, line)?,
                    parse_field(row, 3, "tz", path, line)?,
                ),
                orientation,
            ));
        }

        Ok(Trajectory::from_poses(poses))
    }

    /// Render TUM text.
    ///
    /// Infallible: the length agreement a separate export step used to guard is
    /// now a [`Trajectory`] invariant. Any inertial columns are dropped — TUM
    /// has no place for them, and refusing would make the common export
    /// impossible.
    pub fn render(trajectory: &Trajectory) -> String {
        let mut out = String::with_capacity(trajectory.len() * 128);
        out.push_str("# trajectory written by apex-io\n");
        let _ = writeln!(out, "# Timestamp: {}", chrono::Local::now());
        out.push_str("# timestamp tx ty tz qx qy qz qw\n");

        for sample in trajectory.poses() {
            // TUM stores w LAST.
            let [qw, qx, qy, qz] = sample.orientation.coeffs();
            let seconds = sample.timestamp_ns / NANOS_PER_SEC;
            let nanos = sample.timestamp_ns % NANOS_PER_SEC;
            let p = sample.position;
            let _ = writeln!(
                out,
                "{seconds}.{nanos:09} {:.*} {:.*} {:.*} {:.*} {:.*} {:.*} {:.*}",
                POSE_PRECISION,
                p.x,
                POSE_PRECISION,
                p.y,
                POSE_PRECISION,
                p.z,
                POSE_PRECISION,
                qx,
                POSE_PRECISION,
                qy,
                POSE_PRECISION,
                qz,
                POSE_PRECISION,
                qw
            );
        }
        out
    }
}

/// Parse a TUM timestamp to exact nanoseconds where possible.
///
/// Takes the integer path for the plain `<seconds>.<fraction>` form every tool
/// in this ecosystem emits, so the round trip is exact. Falls back to `f64` for
/// scientific notation or more than nine fractional digits, which is lossy at
/// epoch magnitudes but better than refusing a readable file.
fn parse_timestamp(raw: &str, path: &Path, line: usize) -> Result<u64> {
    let invalid = || TrajectoryError::InvalidNumber {
        path: path.to_path_buf(),
        line,
        field: "timestamp".to_owned(),
        value: raw.to_owned(),
    };

    if let Some((seconds, fraction)) = raw.split_once('.') {
        if seconds.bytes().all(|b| b.is_ascii_digit())
            && !seconds.is_empty()
            && fraction.bytes().all(|b| b.is_ascii_digit())
            && fraction.len() <= 9
        {
            let seconds: u64 = seconds.parse().map_err(|_| invalid())?;
            let scale = 10u64.pow(9 - fraction.len() as u32);
            let nanos: u64 = if fraction.is_empty() {
                0
            } else {
                fraction.parse::<u64>().map_err(|_| invalid())? * scale
            };
            return seconds
                .checked_mul(NANOS_PER_SEC)
                .and_then(|s| s.checked_add(nanos))
                .ok_or_else(invalid);
        }
    } else if raw.bytes().all(|b| b.is_ascii_digit()) && !raw.is_empty() {
        let seconds: u64 = raw.parse().map_err(|_| invalid())?;
        return seconds.checked_mul(NANOS_PER_SEC).ok_or_else(invalid);
    }

    // Foreign notation: lossy, but readable.
    let seconds = raw.parse::<f64>().map_err(|_| invalid())?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(TrajectoryError::NegativeTimestamp {
            path: path.to_path_buf(),
            line,
            value: seconds,
        });
    }
    Ok((seconds * NANOS_PER_SECOND).round() as u64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use apex_manifolds::so3::SO3;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn path() -> &'static Path {
        Path::new("t.tum")
    }

    fn sample_trajectory() -> Trajectory {
        let axis = Vector3::new(0.3, 0.7, 0.1);
        Trajectory::from_poses(vec![
            TrajectoryPose::new(
                1_403_636_579_763_555_584,
                Vector3::new(1.0, -2.0, 3.0),
                SO3::from_axis_angle(&axis, 0.7),
            ),
            TrajectoryPose::new(
                1_403_636_579_813_555_584,
                Vector3::new(1.5, -2.5, 3.5),
                SO3::from_axis_angle(&axis, 0.9),
            ),
        ])
    }

    #[test]
    fn round_trip_preserves_positions_and_rotations() -> TestResult {
        let original = sample_trajectory();
        let parsed = TumLoader::parse(&TumLoader::render(&original), path())?;

        assert_eq!(parsed.len(), original.len());
        for (a, b) in parsed.poses().iter().zip(original.poses().iter()) {
            assert!((a.position - b.position).norm() < 1e-9);
            // Compare rotations, not coefficients: q and -q are the same one.
            //
            // The bound is set by the format, not by any defect: nine decimal
            // places rounds each component by up to 5e-10, which measures as
            // ~8.7e-10 rad of rotation here. That is 5e-8 degrees — far below
            // any sensor this reads, and the price of emitting the fixed-point
            // columns evo and the TUM scripts expect.
            assert!(
                a.orientation.distance(&b.orientation) < 1e-8,
                "round-trip rotation error {:e} exceeds the 9-decimal format bound",
                a.orientation.distance(&b.orientation)
            );
        }
        Ok(())
    }

    /// The absolute pin on field order. For a rotation whose `w` dominates,
    /// the last field must be the largest and the fourth must not be.
    #[test]
    fn the_file_is_w_last() -> TestResult {
        let q = SO3::from_axis_angle(&Vector3::x(), 0.2); // w ~ 0.995
        let t = Trajectory::from_poses(vec![TrajectoryPose::new(0, Vector3::zeros(), q)]);

        let text = TumLoader::render(&t);
        let row = text
            .lines()
            .find(|l| !l.starts_with('#'))
            .ok_or("a data row")?;
        let fields: Vec<f64> = row
            .split_whitespace()
            .skip(1)
            .filter_map(|f| f.parse().ok())
            .collect();

        assert_eq!(fields.len(), 7);
        let qw = fields[6];
        assert!(qw > 0.99, "field 7 must be w, got {qw}");
        assert!(
            fields[3].abs() < 0.1,
            "field 4 must be qx, got {}",
            fields[3]
        );
        Ok(())
    }

    /// The test that forces integer formatting: through `f64` seconds this
    /// fails by ~476 ns.
    #[test]
    fn timestamps_round_trip_exactly_at_euroc_magnitudes() -> TestResult {
        let ns = 1_403_636_579_763_555_584u64;
        let t = Trajectory::from_poses(vec![TrajectoryPose::new(
            ns,
            Vector3::zeros(),
            SO3::identity(),
        )]);
        let parsed = TumLoader::parse(&TumLoader::render(&t), path())?;
        assert_eq!(parsed.poses()[0].timestamp_ns, ns);
        Ok(())
    }

    #[test]
    fn a_foreign_scientific_notation_timestamp_is_read_lossily() -> TestResult {
        let text = "1.4036365797635556e9 0 0 0 0 0 0 1\n";
        let parsed = TumLoader::parse(text, path())?;
        let ns = parsed.poses()[0].timestamp_ns;
        // Within one f64 ulp at this magnitude (~476 ns), well inside 1 us.
        assert!(ns.abs_diff(1_403_636_579_763_555_584) < 1_000, "got {ns}");
        Ok(())
    }

    #[test]
    fn integer_seconds_without_a_fraction_are_accepted() -> TestResult {
        let parsed = TumLoader::parse("12 0 0 0 0 0 0 1\n", path())?;
        assert_eq!(parsed.poses()[0].timestamp_ns, 12_000_000_000);
        Ok(())
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() -> TestResult {
        let text = "# header\n\n1.0 0 0 0 0 0 0 1\n\n2.0 1 1 1 0 0 0 1\n";
        assert_eq!(TumLoader::parse(text, path())?.len(), 2);
        Ok(())
    }

    #[test]
    fn a_seven_field_row_is_rejected() -> TestResult {
        let Err(TrajectoryError::MissingColumns { expected, got, .. }) =
            TumLoader::parse("1.0 0 0 0 0 0 0\n", path())
        else {
            return Err("a 7-field row must be refused".into());
        };
        assert_eq!((expected, got), (8, 7));
        Ok(())
    }

    #[test]
    fn a_non_numeric_field_is_rejected() -> TestResult {
        let Err(TrajectoryError::InvalidNumber { field, .. }) =
            TumLoader::parse("1.0 abc 0 0 0 0 0 1\n", path())
        else {
            return Err("a non-numeric field must be refused".into());
        };
        assert_eq!(field, "tx");
        Ok(())
    }

    #[test]
    fn a_non_unit_quaternion_is_rejected() -> TestResult {
        let Err(TrajectoryError::InvalidQuaternion { .. }) =
            TumLoader::parse("1.0 0 0 0 0 0 0 2\n", path())
        else {
            return Err("a norm-2 quaternion must be refused".into());
        };
        Ok(())
    }

    #[test]
    fn a_negative_timestamp_is_rejected() -> TestResult {
        let Err(TrajectoryError::NegativeTimestamp { .. }) =
            TumLoader::parse("-1.0 0 0 0 0 0 0 1\n", path())
        else {
            return Err("a negative timestamp must be refused".into());
        };
        Ok(())
    }

    #[test]
    fn rows_are_sorted_by_timestamp() -> TestResult {
        let parsed = TumLoader::parse("2.0 0 0 0 0 0 0 1\n1.0 0 0 0 0 0 0 1\n", path())?;
        assert_eq!(parsed.poses()[0].timestamp_ns, 1_000_000_000);
        assert_eq!(parsed.poses()[1].timestamp_ns, 2_000_000_000);
        Ok(())
    }

    /// An ASL row split on whitespace yields one field, so a misdispatch fails
    /// loudly instead of misreading the first eight commas-worth of data.
    #[test]
    fn an_asl_csv_row_errors_rather_than_misparsing() -> TestResult {
        let Err(TrajectoryError::MissingColumns { got, .. }) =
            TumLoader::parse("1000,1.0,2.0,3.0,1.0,0.0,0.0,0.0\n", path())
        else {
            return Err("an ASL row must not parse as TUM".into());
        };
        assert_eq!(got, 1);
        Ok(())
    }

    #[test]
    fn writing_drops_inertial_columns_without_error() -> TestResult {
        let poses = vec![TrajectoryPose::new(0, Vector3::zeros(), SO3::identity())];
        let t = Trajectory::from_poses_and_inertial(
            poses,
            vec![super::super::InertialState::default()],
        )?;
        let parsed = TumLoader::parse(&TumLoader::render(&t), path())?;
        assert_eq!(parsed.len(), 1);
        assert!(!parsed.has_inertial());
        Ok(())
    }
}
