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

pub use logging::{ErrorChainExt, ErrorLogging};

/// Log-at-origin and error-chain helpers. Each crate keeps its own copy, so no
/// crate depends on another for them.
///
/// - [`ErrorLogging`]: log an error exactly once, where it is first built, with
///   the caller's file and line; everything above propagates it with a bare `?`.
///   An expected outcome a caller may retry is not logged where it is built;
///   the caller that gives up on it logs it once with [`ErrorLogging::log_chain`].
/// - [`ErrorChainExt`]: render an error and its `source()` chain at the
///   boundary that reports it.
mod logging {
    use std::error::Error;
    use std::fmt::Debug;
    use std::panic::Location;

    /// Log an error once, at the site that constructs it, and hand it back.
    ///
    /// The blanket impl covers every `std::error::Error`, not every `Display`
    /// type: Lie groups and tangents are `Display` and have a `log` method of
    /// their own, and a `Display` blanket impl would capture their calls.
    pub trait ErrorLogging: Error + Sized {
        /// Emit `self` at `error` level with the caller's location and return it.
        #[track_caller]
        fn log(self) -> Self {
            let at = Location::caller();
            tracing::error!(file = at.file(), line = at.line(), "{self}");
            self
        }

        /// Like [`ErrorLogging::log`], with an underlying cause that has no
        /// `#[from]` edge (a third-party error kept only as context).
        #[track_caller]
        fn log_with_source<S: Debug>(self, source: S) -> Self {
            let at = Location::caller();
            tracing::error!(file = at.file(), line = at.line(), source = ?source, "{self}");
            self
        }

        /// Like [`ErrorLogging::log`], with the whole `source()` chain as the
        /// message (`outer → cause → root cause`). For an error logged by the
        /// caller that gives up on it rather than where it was built.
        #[track_caller]
        fn log_chain(self) -> Self {
            let at = Location::caller();
            tracing::error!(
                file = at.file(),
                line = at.line(),
                "{}",
                self.chain_compact()
            );
            self
        }
    }

    impl<T: Error> ErrorLogging for T {}

    /// Render an error together with every error in its `source()` chain.
    pub trait ErrorChainExt: Error {
        /// One line per hop, outermost first: `outer\n  → cause\n  → root cause`.
        fn chain(&self) -> String {
            let mut text = self.to_string();
            let mut source = self.source();
            while let Some(cause) = source {
                text.push_str("\n  → ");
                text.push_str(&cause.to_string());
                source = cause.source();
            }
            text
        }

        /// The same hops on one line: `outer → cause → root cause`.
        fn chain_compact(&self) -> String {
            let mut text = self.to_string();
            let mut source = self.source();
            while let Some(cause) = source {
                text.push_str(" → ");
                text.push_str(&cause.to_string());
                source = cause.source();
            }
            text
        }
    }

    impl<T: Error + ?Sized> ErrorChainExt for T {}

    #[cfg(test)]
    mod tests {
        use super::*;

        #[derive(Debug, thiserror::Error)]
        #[error("leaf failed at {at}")]
        struct Leaf {
            at: usize,
        }

        #[derive(Debug, thiserror::Error)]
        enum Middle {
            #[error("middle step failed")]
            Leaf(#[from] Leaf),
        }

        #[derive(Debug, thiserror::Error)]
        enum Root {
            #[error(transparent)]
            Middle(#[from] Middle),
        }

        fn leaf() -> Result<(), Leaf> {
            Err(Leaf { at: 7 }.log())
        }

        fn middle() -> Result<(), Middle> {
            leaf()?;
            Ok(())
        }

        fn root() -> Result<(), Root> {
            middle()?;
            Ok(())
        }

        #[test]
        fn the_chain_lists_every_hop_once() {
            let chain = root().err().map(|error| error.chain());
            assert_eq!(
                chain.as_deref(),
                Some("middle step failed\n  → leaf failed at 7")
            );
        }

        #[test]
        fn the_compact_chain_is_one_line() {
            let chain = root().err().map(|error| error.chain_compact());
            assert_eq!(
                chain.as_deref(),
                Some("middle step failed → leaf failed at 7")
            );
        }

        #[test]
        fn logging_returns_the_same_error() {
            let error = Leaf { at: 2 }.log_with_source("context");
            assert_eq!(error.at, 2);
        }

        #[test]
        fn logging_the_chain_returns_the_same_error() {
            let error = Middle::from(Leaf { at: 4 }).log_chain();
            assert_eq!(
                error.chain_compact(),
                "middle step failed → leaf failed at 4"
            );
        }

        #[test]
        fn a_boxed_error_has_a_chain_too() {
            let boxed: Box<dyn Error> = Box::new(Middle::from(Leaf { at: 1 }));
            assert_eq!(
                boxed.chain_compact(),
                "middle step failed → leaf failed at 1"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
