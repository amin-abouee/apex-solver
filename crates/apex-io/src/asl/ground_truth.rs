//! ASL/EuRoC trajectory CSV.
//!
//! ```text
//! mocap0 (8):   ts_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z
//! EuRoC (17):   ... then v_x, v_y, v_z, bw_x, bw_y, bw_z, ba_x, ba_y, ba_z
//! ```
//!
//! Both store the quaternion **w first**, and the timestamp as bare integer
//! nanoseconds. The 17-column layout is EuRoC's `state_groundtruth_estimate0`;
//! the 8-column one is TUM VI's `mocap0`.
//!
//! A row is accepted only at exactly one of those two widths. Reading the first
//! eight fields of a wider row is precisely how the velocity and bias columns
//! went unread for so long.

use std::fmt::Write as _;
use std::path::Path;

use nalgebra::Vector3;

use super::error::{Result, TrajectoryError};
use super::quaternion;
use super::types::AslLayout;
use super::{InertialState, Trajectory, TrajectoryLoader, TrajectoryPose};
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

/// Loader for ASL/EuRoC trajectory CSV.
pub struct AslTrajectoryLoader;

impl TrajectoryLoader for AslTrajectoryLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Trajectory> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|source| TrajectoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
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
    /// # Errors
    ///
    /// See [`TrajectoryError`].
    pub fn parse(text: &str, path: &Path) -> Result<Trajectory> {
        let rows = csv::split_rows(text, Some(','));
        let mut poses = Vec::with_capacity(rows.len());
        let mut inertial = Vec::with_capacity(rows.len());
        let mut any_inertial = false;

        for (index, row) in rows.iter().enumerate() {
            let line = index + 1;
            let layout = match row.len() {
                17 => AslLayout::Euroc17,
                8 => AslLayout::Mocap8,
                got => {
                    return Err(TrajectoryError::UnexpectedColumnCount {
                        path: path.to_path_buf(),
                        line,
                        got,
                        a: 17,
                        b: 8,
                    });
                }
            };

            let field = |i: usize, name: &str| -> Result<f64> {
                let raw = row.get(i).map(String::as_str).unwrap_or_default();
                raw.parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| TrajectoryError::InvalidNumber {
                        path: path.to_path_buf(),
                        line,
                        field: name.to_owned(),
                        value: raw.to_owned(),
                    })
            };

            let raw_stamp = row.first().map(String::as_str).unwrap_or_default();
            let timestamp_ns =
                raw_stamp
                    .parse::<u64>()
                    .map_err(|_| TrajectoryError::InvalidNumber {
                        path: path.to_path_buf(),
                        line,
                        field: "timestamp".to_owned(),
                        value: raw_stamp.to_owned(),
                    })?;

            // w FIRST — see `quaternion`, the only module allowed to know this.
            let orientation = quaternion::from_wxyz(
                path,
                line,
                field(4, "q_w")?,
                field(5, "q_x")?,
                field(6, "q_y")?,
                field(7, "q_z")?,
            )?;

            poses.push(TrajectoryPose::new(
                timestamp_ns,
                Vector3::new(field(1, "p_x")?, field(2, "p_y")?, field(3, "p_z")?),
                orientation,
            ));

            if layout == AslLayout::Euroc17 {
                any_inertial = true;
                inertial.push(InertialState {
                    velocity: Vector3::new(field(8, "v_x")?, field(9, "v_y")?, field(10, "v_z")?),
                    gyro_bias: Vector3::new(
                        field(11, "b_w_x")?,
                        field(12, "b_w_y")?,
                        field(13, "b_w_z")?,
                    ),
                    accel_bias: Vector3::new(
                        field(14, "b_a_x")?,
                        field(15, "b_a_y")?,
                        field(16, "b_a_z")?,
                    ),
                });
            } else {
                inertial.push(InertialState::default());
            }
        }

        if any_inertial {
            Trajectory::from_poses_and_inertial(poses, inertial)
        } else {
            Ok(Trajectory::from_poses(poses))
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
        std::fs::write(path, text).map_err(|source| TrajectoryError::Io {
            path: path.to_path_buf(),
            source,
        })
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
            let [qw, qx, qy, qz] = quaternion::to_wxyz(&sample.orientation);
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
    use nalgebra::UnitQuaternion;

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
        let axis = nalgebra::Unit::new_normalize(Vector3::x());
        let q = UnitQuaternion::from_axis_angle(&axis, 0.2); // w ~ 0.995
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
}
