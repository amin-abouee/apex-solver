//! Errors of the apex-io crate.
//!
//! [`IoError`] is the root. Each variant transparently wraps the error of one
//! module, so `?` lifts a module error to the root with no wrapper text and
//! [`std::error::Error::source`] walks down to the leaf:
//!
//! - [`GraphError`]: G2O and TORO pose-graph files.
//! - [`BalError`]: Bundle Adjustment in the Large files.
//! - [`TrajectoryError`]: TUM / EuRoC / KITTI trajectory files.
//! - [`AslError`]: ASL (EuRoC) dataset folders.
//! - [`LamariaError`]: LaMAria sequence folders.
//! - [`DatasetError`]: the dataset registry and its downloads.
//! - `VrsError` (feature `vrs`): VRS containers.
//! - `BagError` (feature `rosbag`): ROS1 / ROS2 / MCAP bags, itself the root of
//!   the rosbag module errors.
//!
//! Modules return their own error. Each is logged once where it is built; the
//! per-message decoders (CDR, ROS1 payloads, point clouds) are the exception:
//! a failed decode is an expected outcome there and is logged only by the
//! caller that aborts a read because of it.

pub use crate::asl::error::AslError;
pub use crate::bal::BalError;
pub use crate::graph::GraphError;
pub use crate::lamaria::LamariaError;
pub use crate::trajectory::TrajectoryError;
pub use crate::utils::DatasetError;

/// Result type for any apex-io operation.
pub type IoResult<T> = Result<T, IoError>;

/// Root error of the apex-io crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IoError {
    #[error(transparent)]
    Graph(#[from] GraphError),

    #[error(transparent)]
    Bal(#[from] BalError),

    #[error(transparent)]
    Trajectory(#[from] TrajectoryError),

    #[error(transparent)]
    Asl(#[from] AslError),

    #[error(transparent)]
    Lamaria(#[from] LamariaError),

    #[error(transparent)]
    Dataset(#[from] DatasetError),

    #[cfg(feature = "vrs")]
    #[error(transparent)]
    Vrs(#[from] crate::vrs::VrsError),

    #[cfg(feature = "rosbag")]
    #[error(transparent)]
    Bag(#[from] crate::rosbag::BagError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_error::ErrorChainExt;
    use std::error::Error;
    use std::path::PathBuf;

    type TestResult = Result<(), Box<dyn Error>>;

    fn read_graph() -> crate::graph::GraphResult<()> {
        Err(GraphError::read(
            PathBuf::from("missing.g2o"),
            std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        ))
    }

    fn root() -> IoResult<()> {
        read_graph()?;
        Ok(())
    }

    #[test]
    fn a_module_error_reaches_the_root_with_a_bare_question_mark() -> TestResult {
        let error = root().err().ok_or("expected an error")?;
        assert!(matches!(error, IoError::Graph(GraphError::Read { .. })));
        Ok(())
    }

    #[test]
    fn the_chain_shows_each_hop_once() -> TestResult {
        let error = root().err().ok_or("expected an error")?;
        assert_eq!(
            error.chain_compact(),
            "failed to read graph file missing.g2o → no such file"
        );
        Ok(())
    }

    #[test]
    fn every_root_variant_is_transparent() {
        let errors: Vec<IoError> = vec![
            GraphError::DuplicateVertex { id: 3 }.into(),
            BalError::MissingHeader.into(),
            TrajectoryError::MissingInertialColumns.into(),
            AslError::CameraNotFound { index: 2, count: 1 }.into(),
            DatasetError::DownloadDisabled {
                name: "intel".to_string(),
            }
            .into(),
        ];
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        assert_eq!(
            messages,
            [
                "duplicate vertex id 3",
                "BAL file has no header line",
                "cannot write EuRoC 17-column layout: this trajectory has no inertial columns",
                "camera index 2 not found, the dataset has 1 cameras",
                "dataset intel is not present locally and apex-io was built without the download feature",
            ]
        );
    }
}
