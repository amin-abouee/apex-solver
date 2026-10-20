//! Camera factors: monocular and stereo reprojection, inverse-depth and
//! homogeneous landmark parameterizations, epipolar geometry, and
//! structure-less smart projection.

pub mod depth;
pub mod essential_matrix;
pub mod extrinsic_projection;
pub mod homogeneous_point;
pub mod inverse_depth;
pub mod inverse_depth_bearing;
pub mod inverse_depth_plane;
pub mod photometric_inverse_range;
pub mod photometric_patch;
pub mod photometric_two_view;
pub mod projection;
pub mod smart_projection;
pub mod stereo;
pub mod stereo_inverse_range;
pub mod time_offset_projection;

pub use depth::{DepthFactor, OneSidedDepthFactor, RegularDepthFactor};
pub use essential_matrix::{EssentialMatrixConstraint, EssentialMatrixFactor};
pub use extrinsic_projection::ExtrinsicProjectionFactor;
pub use homogeneous_point::HomogeneousPointFactor;
pub use inverse_depth::InverseDepthFactor;
pub use inverse_depth_bearing::InverseDepthBearingFactor;
pub use inverse_depth_plane::InverseDepthPlaneFactor;
pub use photometric_inverse_range::{
    AffinePriorFactor, GradientImage, PhotometricInverseRangeFactor, PhotometricPoint,
    PhotometricWeights,
};
pub use photometric_patch::IntensityImage;
pub use photometric_patch::PhotometricPatchFactor;
pub use photometric_two_view::{PhotometricTwoViewFactor, PlanePoint, ReferenceView};
pub use projection::{OptimizationConfig, ProjectionFactor};
pub use smart_projection::{SmartProjectionFactor, TriangulationStatus};
pub use stereo::{StereoCalibration, StereoFactor, StereoPoint2};
pub use stereo_inverse_range::{InverseRangeProjectionFactor, StereoInverseRangeFactor};
pub use time_offset_projection::TimeOffsetProjectionFactor;

#[cfg(test)]
mod extrinsic_tests;
