//! LiDAR odometry factors: point correspondences, point-to-plane, and
//! generalized ICP (plane-to-plane).

pub mod gicp_factor;
pub mod point_to_plane_factor;
pub mod pose_to_point_factor;

pub use gicp_factor::GicpFactor;
pub use point_to_plane_factor::{Plane, PointToPlaneFactor};
pub use pose_to_point_factor::PoseToPointFactor;
