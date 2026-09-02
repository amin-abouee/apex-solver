//! Visual (camera-based) factors: stereo, inverse-depth landmarks, epipolar
//! geometry, and structure-less smart projection.

pub mod essential_matrix_factor;
pub mod inverse_depth_factor;
pub mod smart_projection_factor;
pub mod stereo_factor;

pub use essential_matrix_factor::{EssentialMatrixConstraint, EssentialMatrixFactor};
pub use inverse_depth_factor::InverseDepthFactor;
pub use smart_projection_factor::{SmartProjectionFactor, TriangulationStatus};
pub use stereo_factor::{StereoCalibration, StereoFactor, StereoPoint2};
