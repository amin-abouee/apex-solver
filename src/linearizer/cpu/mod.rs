//! CPU linearizer implementations.
//!
//! Provides sparse and dense Jacobian assembly for CPU computation.
//! GPU equivalent lives in the sibling [`super::gpu`] module.
//!
//! This module also owns the [`LinearizationMode`] marker trait and its two
//! implementations ([`SparseMode`], [`DenseMode`]), which define the matrix
//! types used throughout the solver pipeline.

pub mod dense;
pub mod sparse;

use faer::{Mat, sparse::SparseColMat};

// ============================================================================
// LinearizationMode — static dispatch between sparse and dense paths
// ============================================================================

/// Marker trait that defines the matrix types for a linear algebra path.
///
/// Enables zero-cost static dispatch between sparse and dense linear algebra
/// backends. Optimizers are generic over `LinearizationMode`, so the compiler
/// generates one specialization per concrete mode at no runtime cost.
///
/// Two implementations are provided:
/// - [`SparseMode`]: Jacobian and Hessian are `SparseColMat<usize, f64>`
/// - [`DenseMode`]: Jacobian and Hessian are `Mat<f64>`
pub trait LinearizationMode: 'static {
    /// The Jacobian matrix type (`SparseColMat` or `Mat`)
    type Jacobian: Send + Sync;
    /// The Hessian matrix type (`SparseColMat` or `Mat`)
    type Hessian: Send + Sync;
}

/// Sparse linear algebra mode.
///
/// Uses `SparseColMat<usize, f64>` for Jacobians and Hessians.
/// Optimal for large-scale problems with sparse structure (e.g., pose graphs).
pub struct SparseMode;

impl LinearizationMode for SparseMode {
    type Jacobian = SparseColMat<usize, f64>;
    type Hessian = SparseColMat<usize, f64>;
}

/// Dense linear algebra mode.
///
/// Uses `Mat<f64>` for Jacobians and Hessians.
/// Optimal for small-to-medium problems (< 500 DOF) or dense Jacobians.
pub struct DenseMode;

impl LinearizationMode for DenseMode {
    type Jacobian = Mat<f64>;
    type Hessian = Mat<f64>;
}
