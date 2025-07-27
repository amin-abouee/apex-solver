//! Factor implementations for the factor graph
//!
//! This module provides a comprehensive set of factor types for optimization problems,
//! particularly focused on SLAM, bundle adjustment, and robotics applications.
//! The module is organized into submodules for different categories of factors.
//!
//! # Module Structure
//!
//! - `basic`: Basic factor types (unary, binary, prior)
//! - `geometry`: Geometric factors (between, relative pose)
//! - `vision`: Computer vision factors (projection, reprojection)
//! - `motion`: Motion model factors (odometry, IMU)
//! - `robust`: Robust kernels for outlier rejection

use std::fmt;
use nalgebra::{DVector, DMatrix};
use crate::core::types::{ApexResult, ApexError};

pub mod basic;
pub mod geometry;
pub mod vision;
pub mod motion;
pub mod robust;

// Re-export commonly used types
pub use basic::{UnaryFactor, BinaryFactor, PriorFactor};
pub use geometry::{
    BetweenFactor, RelativePoseFactor, SE2BetweenFactor, SE3BetweenFactor,
    SE2RelativePoseFactor, SE3RelativePoseFactor, SO2BetweenFactor, SO3BetweenFactor,
};
pub use vision::{
    ProjectionFactor, ReprojectionFactor, StereoFactor, CameraIntrinsics,
};
pub use motion::{
    OdometryFactor, ConstantVelocityFactor, VelocityFactor,
    SE2OdometryFactor, SE3OdometryFactor, SE2ConstantVelocityFactor, SE3ConstantVelocityFactor,
};
pub use robust::{
    RobustKernel, L2Kernel, HuberKernel, CauchyKernel, TukeyKernel, GemanMcClureKernel,
};

// Tests module
#[cfg(test)]
mod tests;

/// Simplified Factor trait for factor graph optimization
///
/// This trait defines the essential interface for factors in the factor graph.
/// It focuses on the core operations needed for optimization: identifying factors
/// and variables, computing residuals, and computing linearizations.
pub trait Factor: fmt::Debug + Send + Sync {
    /// Returns a unique identifier for the factor
    fn id(&self) -> usize;

    /// Returns the factor's key/identifier used in the graph
    fn key(&self) -> usize;

    /// Returns the keys/identifiers of all variables connected to this factor
    fn variable_keys(&self) -> &[usize];

    /// Evaluates both the error/residual vector and the Jacobian matrices
    /// for each connected variable at the current linearization point
    fn linearize(&self, variables: &[&dyn std::any::Any]) -> ApexResult<(DVector<f64>, DMatrix<f64>)>;

    /// Evaluates only the error/residual vector (without Jacobian computation)
    fn evaluate(&self, variables: &[&dyn std::any::Any]) -> ApexResult<DVector<f64>>;
}