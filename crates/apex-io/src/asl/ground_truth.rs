//! ASL/EuRoC trajectory CSV.
//!
//! ```text
//! mocap0 (8):   ts_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z
//! EuRoC (17):   ... then v_x, v_y, v_z, b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z
//! ```
//!
//! Both store the quaternion **w first**, and the timestamp as bare integer
//! nanoseconds. The 17-column layout is EuRoC's `state_groundtruth_estimate0`;
//! the 8-column one is TUM VI's `mocap0`.
//!
//! The layout is pinned by the first data row and every following row must
//! agree. Accepting a 9-column row by reading its first eight fields is exactly
//! how the inertial columns went unnoticed for so long, and zero-filling the
//! inertial columns of an 8-column row inside a 17-column file would claim
//! measurements nobody made.

use std::fmt::Write as _;
use std::path::Path;

use apex_manifolds::so3::SO3;
use nalgebra::Vector3;

use super::super::trajectory::{
    InertialState, Result, Trajectory, TrajectoryError, TrajectoryLoader, TrajectoryPose,
    parse_field, parse_u64_field, read_file, write_file,
};
use crate::csv;

/// The EuRoC schema line, reproduced so a written file is readable by kalibr,
/// evo and the dataset's own tooling.
const EUROC_HEADER: &str = "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],\
q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],\
v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],\
b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],\
b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]";

/// The `mocap0` schema line.
const MOCAP_HEADER: &str = "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []";

/// Which on-disk layout an ASL ground-truth track uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AslLayout {
    /// EuRoC `state_groundtruth_estimate0`: 17 columns, pose plus velocity and
    /// both biases.
    Euroc17,
    /// TUM VI `mocap0`: 8 columns, pose only.
    Mocap8,
}

impl AslLayout {
    /// Number of columns this layout occupies.
    pub fn columns(self) -> usize {
        match self {
            Self::Euroc17 => 17,
            Self::Mocap8 => 8,
        }
    }
}

/// Loader for ASL/EuRoC trajectory CSV.
pub struct AslTrajectoryLoader;

impl TrajectoryLoader for AslTrajectoryLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Trajectory> {
        let path = path.as_ref();
        let content = read_file(path)?;
        Self::parse(&content, path)
    }

    /// Writes 17 columns when the trajectory carries inertial state, 8
    /// otherwise.
    fn write<P: AsRef<Path>>(trajectory: &Trajectory, path: P) -> Result<()> {
        let layout = if trajectory.has_inertial() {
            AslLayout::Euroc17
        } else {
            AslLayout::Mocap8
        };
        Self::write_as(trajectory, path, layout)
    }
}

impl AslTrajectoryLoader {
    /// Parse ASL CSV text. `path` labels errors only.
    ///
    /// The layout is pinned by the first data row; a later row of a different
    /// width is an error, never zero-filled.
    ///
    /// # Errors
    ///
    /// See [`TrajectoryError`].
    pub fn parse(text: &str, path: &Path) -> Result<Trajectory> {
        let rows = csv::split_rows(text, Some(','));
        let mut poses = Vec::with_capacity(rows.len());
        // Layout is unknown until the first row is seen.
        let mut pinned: Option<AslLayout> = None;
        let mut inertial: Option<Vec<InertialState>> = None;

        for (index, row) in rows.iter().enumerate() {
            let line = index + 1;

            let layout = match row.len() {
                got if got == AslLayout::Euroc17.columns() => AslLayout::Euroc17,
                got if got == AslLayout::Mocap8.columns() => AslLayout::Mocap8,
                got => {
                    return Err(TrajectoryError::UnexpectedColumnCount {
                        path: path.to_path_buf(),
                        line,
                        got,
                        a: AslLayout::Euroc17.columns(),
                        b: AslLayout::Mocap8.columns(),
                    });
                }
            };

            // A later row disagreeing with the pinned layout is a mixed file:
            // reject rather than zero-fill. `MissingColumns` keeps the message
            // accurate — the row has the file's width, it does not have the
            // width every row of this file must have.
            if let Some(first) = pinned
                && first != layout
            {
                return Err(TrajectoryError::MissingColumns {
                    path: path.to_path_buf(),
                    line,
                    expected: first.columns(),
                    got: layout.columns(),
                });
            }
            pinned = Some(layout);

            let timestamp_ns = parse_u64_field(row[0], "timestamp", path, line)?;

            let qw = parse_field(row, 4, "q_w", path, line)?;
            let qx = parse_field(row, 5, "q_x", path, line)?;
            let qy = parse_field(row, 6, "q_y", path, line)?;
            let qz = parse_field(row, 7, "q_z", path, line)?;
            // ASL stores w FIRST.
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
                    parse_field(row, 1, "p_x", path, line)?,
                    parse_field(row, 2, "p_y", path, line)?,
                    parse_field(row, 3, "p_z", path, line)?,
                ),
                orientation,
            ));

            if layout == AslLayout::Euroc17 {
                inertial
                    .get_or_insert_with(|| Vec::with_capacity(rows.len()))
                    .push(InertialState {
                        velocity: Vector3::new(
                            parse_field(row, 8, "v_x", path, line)?,
                            parse_field(row, 9, "v_y", path, line)?,
                            parse_field(row, 10, "v_z", path, line)?,
                        ),
                        gyro_bias: Vector3::new(
                            parse_field(row, 11, "b_w_x", path, line)?,
                            parse_field(row, 12, "b_w_y", path, line)?,
                            parse_field(row, 13, "b_w_z", path, line)?,
                        ),
                        accel_bias: Vector3::new(
                            parse_field(row, 14, "b_a_x", path, line)?,
                            parse_field(row, 15, "b_a_y", path, line)?,
                            parse_field(row, 16, "b_a_z", path, line)?,
                        ),
                    });
            }
        }

        match inertial {
            Some(states) => Trajectory::from_poses_and_inertial(poses, states),
            None => Ok(Trajectory::from_poses(poses)),
        }
    }

    /// Write in an explicit layout.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::MissingInertialColumns`] when [`AslLayout::Euroc17`]
    /// is asked for on a pose-only trajectory: padding velocity and bias with
    /// zeros would claim the body was stationary with perfect sensors.
    pub fn write_as<P: AsRef<Path>>(
        trajectory: &Trajectory,
        path: P,
        layout: AslLayout,
    ) -> Result<()> {
        let path = path.as_ref();
        let text = Self::render(trajectory, layout)?;
        write_file(path, &text)
    }

    /// Render ASL CSV in the requested layout.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::MissingInertialColumns`] — see [`Self::write_as`].
    pub fn render(trajectory: &Trajectory, layout: AslLayout) -> Result<String> {
        if layout == AslLayout::Euroc17 && !trajectory.has_inertial() {
            return Err(TrajectoryError::MissingInertialColumns);
        }

        let mut out = String::with_capacity(trajectory.len() * 256);
        out.push_str(match layout {
            AslLayout::Euroc17 => EUROC_HEADER,
            AslLayout::Mocap8 => MOCAP_HEADER,
        });
        out.push('\n');

        let inertial = trajectory.inertial();
        for (index, sample) in trajectory.poses().iter().enumerate() {
            // ASL stores w FIRST — `coeffs` is [w, x, y, z].
            let [qw, qx, qy, qz] = sample.orientation.coeffs();
            let p = sample.position;
            let _ = write!(
                out,
                "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                sample.timestamp_ns, p.x, p.y, p.z, qw, qx, qy, qz
            );

            if layout == AslLayout::Euroc17 {
                let state = inertial
                    .and_then(|s| s.get(index))
                    .cloned()
                    .unwrap_or_default();
                for v in [state.velocity, state.gyro_bias, state.accel_bias] {
                    let _ = write!(out, ",{:.17e},{:.17e},{:.17e}", v.x, v.y, v.z);
                }
            }
            out.push('\n');
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use apex_manifolds::so3::SO3;
    use std::path::Path;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn path() -> &'static Path {
        Path::new("data.csv")
    }

    /// One EuRoC row with every inertial column distinct, so a mis-indexed read
    /// cannot coincidentally pass.
    const EUROC_ROW: &str = "1403636579763555584,\
1.0,2.0,3.0,\
1.0,0.0,0.0,0.0,\
10.0,11.0,12.0,\
20.0,21.0,22.0,\
30.0,31.0,32.0\n";

    const MOCAP_ROW: &str = "1403636579763555584,1.0,2.0,3.0,1.0,0.0,0.0,0.0\n";

    #[test]
    fn all_seventeen_euroc_columns_are_read() -> TestResult {
        let t = AslTrajectoryLoader::parse(EUROC_ROW, path())?;
        assert!(t.has_inertial());

        let states = t.inertial().ok_or("inertial must be present")?;
        let s = states.first().ok_or("one row")?;
        assert!((s.velocity - Vector3::new(10.0, 11.0, 12.0)).norm() < 1e-12);
        assert!((s.gyro_bias - Vector3::new(20.0, 21.0, 22.0)).norm() < 1e-12);
        assert!((s.accel_bias - Vector3::new(30.0, 31.0, 32.0)).norm() < 1e-12);

        let p = t.poses().first().ok_or("one pose")?;
        assert!((p.position - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
        assert_eq!(p.timestamp_ns, 1_403_636_579_763_555_584);
        Ok(())
    }

    #[test]
    fn mocap_eight_columns_parse_with_no_inertial() -> TestResult {
        let t = AslTrajectoryLoader::parse(MOCAP_ROW, path())?;
        assert_eq!(t.len(), 1);
        assert!(!t.has_inertial());
        assert!(t.inertial().is_none());
        Ok(())
    }

    /// The absolute pin on field order: for a rotation whose `w` dominates,
    /// field 4 (the first quaternion column) must be the large one.
    #[test]
    fn the_file_is_w_first() -> TestResult {
        let q = SO3::from_axis_angle(&Vector3::x(), 0.2); // w ~ 0.995
        let t = Trajectory::from_poses(vec![TrajectoryPose::new(0, Vector3::zeros(), q)]);

        let text = AslTrajectoryLoader::render(&t, AslLayout::Mocap8)?;
        let row = text
            .lines()
            .find(|l| !l.starts_with('#'))
            .ok_or("a data row")?;
        let fields: Vec<f64> = row
            .split(',')
            .skip(1)
            .filter_map(|f| f.parse().ok())
            .collect();

        assert_eq!(fields.len(), 7);
        assert!(fields[3] > 0.99, "field 4 must be q_w, got {}", fields[3]);
        assert!(
            fields[6].abs() < 0.1,
            "field 7 must be q_z, got {}",
            fields[6]
        );
        Ok(())
    }

    #[test]
    fn round_trip_euroc_preserves_all_seventeen_columns() -> TestResult {
        let original = AslTrajectoryLoader::parse(EUROC_ROW, path())?;
        let text = AslTrajectoryLoader::render(&original, AslLayout::Euroc17)?;
        let parsed = AslTrajectoryLoader::parse(&text, path())?;

        assert!(parsed.has_inertial());
        let a = original.inertial().ok_or("a")?.first().ok_or("a0")?;
        let b = parsed.inertial().ok_or("b")?.first().ok_or("b0")?;
        assert!((a.velocity - b.velocity).norm() < 1e-12);
        assert!((a.gyro_bias - b.gyro_bias).norm() < 1e-12);
        assert!((a.accel_bias - b.accel_bias).norm() < 1e-12);
        assert_eq!(
            parsed.poses()[0].timestamp_ns,
            original.poses()[0].timestamp_ns
        );
        Ok(())
    }

    #[test]
    fn round_trip_mocap_preserves_eight_columns() -> TestResult {
        let original = AslTrajectoryLoader::parse(MOCAP_ROW, path())?;
        let text = AslTrajectoryLoader::render(&original, AslLayout::Mocap8)?;
        let parsed = AslTrajectoryLoader::parse(&text, path())?;
        assert_eq!(parsed.len(), 1);
        assert!(!parsed.has_inertial());
        assert!((parsed.poses()[0].position - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
        Ok(())
    }

    #[test]
    fn writing_euroc_layout_refuses_a_pose_only_trajectory() -> TestResult {
        let t = AslTrajectoryLoader::parse(MOCAP_ROW, path())?;
        let Err(TrajectoryError::MissingInertialColumns) =
            AslTrajectoryLoader::render(&t, AslLayout::Euroc17)
        else {
            return Err("padding the inertial columns with zeros must be refused".into());
        };
        Ok(())
    }

    /// A 9-column row matches no layout. Accepting it by reading the first
    /// eight fields is exactly how EuRoC's inertial columns were missed.
    #[test]
    fn a_row_of_nine_columns_is_rejected() -> TestResult {
        let row = "1000,1,2,3,1,0,0,0,9\n";
        let Err(TrajectoryError::UnexpectedColumnCount { got, a, b, .. }) =
            AslTrajectoryLoader::parse(row, path())
        else {
            return Err("a 9-column row must be refused".into());
        };
        assert_eq!((got, a, b), (9, 17, 8));
        Ok(())
    }

    #[test]
    fn a_non_unit_quaternion_is_rejected() -> TestResult {
        let row = "1000,1,2,3,2.0,0,0,0\n";
        let Err(TrajectoryError::InvalidQuaternion { norm, .. }) =
            AslTrajectoryLoader::parse(row, path())
        else {
            return Err("a norm-2 quaternion must be refused".into());
        };
        assert!((norm - 2.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn a_fractional_timestamp_is_rejected() -> TestResult {
        let row = "1000.5,1,2,3,1,0,0,0\n";
        let Err(TrajectoryError::InvalidNumber { field, .. }) =
            AslTrajectoryLoader::parse(row, path())
        else {
            return Err("ASL timestamps are integer nanoseconds".into());
        };
        assert_eq!(field, "timestamp");
        Ok(())
    }

    /// A TUM row split on commas yields one field, so a misdispatch fails
    /// loudly rather than misreading.
    #[test]
    fn a_tum_row_errors_rather_than_misparsing() -> TestResult {
        let Err(TrajectoryError::UnexpectedColumnCount { got, .. }) =
            AslTrajectoryLoader::parse("1.0 0 0 0 0 0 0 1\n", path())
        else {
            return Err("a TUM row must not parse as ASL".into());
        };
        assert_eq!(got, 1);
        Ok(())
    }

    #[test]
    fn timestamps_are_exact_u64_nanoseconds() -> TestResult {
        let t = AslTrajectoryLoader::parse(EUROC_ROW, path())?;
        let text = AslTrajectoryLoader::render(&t, AslLayout::Euroc17)?;
        let parsed = AslTrajectoryLoader::parse(&text, path())?;
        assert_eq!(parsed.poses()[0].timestamp_ns, 1_403_636_579_763_555_584);
        Ok(())
    }

    /// A 17-column file where the second row arrives as 8 columns: the mixed
    /// file must be refused, not zero-filled — the inertial columns of the
    /// wide rows would otherwise pair with zeros nobody measured.
    #[test]
    fn a_mixed_layout_file_is_rejected() -> TestResult {
        let text = "1000,1,2,3,1,0,0,0,4,5,6,7,8,9,10,11,12\n\
                    2000,1,2,3,1,0,0,0\n";
        let Err(TrajectoryError::MissingColumns { expected, got, .. }) =
            AslTrajectoryLoader::parse(text, path())
        else {
            return Err("a mixed-layout file must be refused".into());
        };
        assert_eq!((expected, got), (17, 8));
        Ok(())
    }

    /// The mirror case: an 8-column file with a 17-column interloper.
    #[test]
    fn a_mixed_layout_file_wide_row_is_rejected() -> TestResult {
        let text = "1000,1,2,3,1,0,0,0\n\
                    1000,1,2,3,1,0,0,0,4,5,6,7,8,9,10,11,12\n";
        let Err(TrajectoryError::MissingColumns { expected, got, .. }) =
            AslTrajectoryLoader::parse(text, path())
        else {
            return Err("a mixed-layout file must be refused".into());
        };
        assert_eq!((expected, got), (8, 17));
        Ok(())
    }

    /// A 17-column file parsed with rows out of timestamp order keeps pose↔
    /// inertial pairing through the sort — the parse-level twin of
    /// `unsorted_input_keeps_poses_and_inertial_aligned`.
    #[test]
    fn unsorted_17_column_rows_keep_pose_and_inertial_paired() -> TestResult {
        let text = "2000,2,0,0,1,0,0,0,20,0,0,200,0,0,0,0,0\n\
                    1000,1,0,0,1,0,0,0,10,0,0,100,0,0,0,0,0\n";
        let t = AslTrajectoryLoader::parse(text, path())?;
        let states = t.inertial().ok_or("inertial must be present")?;

        assert!((t.poses()[0].position.x - 1.0).abs() < 1e-12);
        assert!((states[0].velocity.x - 10.0).abs() < 1e-12);
        assert!((t.poses()[1].position.x - 2.0).abs() < 1e-12);
        assert!((states[1].velocity.x - 20.0).abs() < 1e-12);
        Ok(())
    }
}
