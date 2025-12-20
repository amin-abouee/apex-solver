//! Camera projection models and factors for bundle adjustment.
//!
//! This module provides a unified interface for camera projection with support for:
//! - Multiple camera models (Pinhole, Double Sphere, EUCM, etc.)
//! - Flexible optimization configurations via const generics
//! - Bundle adjustment, self-calibration, and camera calibration
//!
//! # Architecture
//!
//! The module is organized around two key abstractions:
//!
//! 1. **[`CameraModel`] trait**: Defines projection and Jacobian computation for different camera models
//! 2. **[`ProjectionFactor`]**: Generic factor that adapts to different optimization scenarios
//!
//! # Examples
//!
//! ```rust
//! use apex_solver::factors::camera::{ProjectionFactor, PinholeCamera, BundleAdjustment};
//! use nalgebra::{Matrix2xX, Vector2};
//!
//! // Create camera
//! let camera = PinholeCamera { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
//!
//! // Create observations
//! let observations = Matrix2xX::from_columns(&[
//!     Vector2::new(100.0, 150.0),
//!     Vector2::new(200.0, 250.0),
//! ]);
//!
//! // Bundle adjustment (default)
//! let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
//!     ProjectionFactor::new(observations, camera);
//! ```

pub mod pinhole;
pub mod projection_factor;
pub mod traits;

// Re-export main types
pub use traits::{
    BundleAdjustment, CameraModel, LandmarksAndIntrinsics, OnlyIntrinsics, OnlyLandmarks, OnlyPose,
    OptimizeParams, PoseAndIntrinsics, SelfCalibration,
};

pub use pinhole::{PINHOLE_INTRINSIC_DIM, PinholeCamera};

pub use projection_factor::ProjectionFactor;
