// Module declarations
pub mod asl;
pub mod bal;
mod csv;
pub mod graph;
pub mod logger;
pub mod trajectory;
pub mod utils;

#[cfg(feature = "rosbag")]
pub mod rosbag;

#[cfg(feature = "dds")]
pub use rosbag::ros2::dds;

pub use logger::init_logger;
pub use utils::{
    DatasetRegistry, ensure_ba_dataset, ensure_odometry_dataset, ensure_sensor_dataset,
};

/// Default base directory for odometry (pose graph) datasets relative to the workspace root.
pub const ODOMETRY_DATA_DIR: &str = "data/odometry";

/// Directory for 2D odometry datasets (`data/odometry/2d`).
pub const ODOMETRY_DATA_DIR_2D: &str = "data/odometry/2d";

/// Directory for 3D odometry datasets (`data/odometry/3d`).
pub const ODOMETRY_DATA_DIR_3D: &str = "data/odometry/3d";

/// Default directory for bundle adjustment datasets relative to the workspace root.
pub const BUNDLE_ADJUSTMENT_DATA_DIR: &str = "data/bundle_adjustment";

/// Default directory for multi-sensor (IMU / GNSS / odometry) datasets.
pub const SENSOR_DATA_DIR: &str = "data/sensor";

// Re-exports
pub use asl::error::AslError;
pub use asl::{
    AslDataset, AslLayout, AslReader, AslStream, AslTrajectoryLoader, load_mav0_trajectory,
};
pub use bal::{BalCamera, BalDataset, BalLoader, BalObservation, BalPoint};
pub use graph::g2o::G2oLoader;
pub use graph::toro::ToroLoader;
pub use graph::{EdgeSE2, EdgeSE3, Graph, GraphLoader, IoError, VertexSE2, VertexSE3, load_graph};
pub use trajectory::{
    InertialState, Trajectory, TrajectoryError, TrajectoryFormat, TrajectoryLoader, TrajectoryPose,
    TumLoader, load_trajectory, load_trajectory_as,
};
