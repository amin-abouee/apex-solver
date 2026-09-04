//! LiDAR odometry factors.
//!
//! Two graph topologies coexist deliberately: the two-pose family
//! ([`distance_field`], [`edge`]) registers a scan against a reference frame
//! with the query point baked into the factor, while the pose-and-point family
//! ([`plane`], [`point_to_point`], [`gicp`]) treats the body-frame point as a
//! variable.

pub mod distance_field;
pub mod edge;
pub mod gicp;
pub mod plane;
pub mod point_to_point;

pub use distance_field::{DistanceField, IcpFactor};
pub use edge::LidarEdgeFactor;
pub use gicp::GicpFactor;
pub use plane::{
    LidarPlaneFactor, Plane, PointToPlaneFactor, PrecomputedPlane, lidar_plane_factor_isotropic,
};
pub use point_to_point::PoseToPointFactor;
