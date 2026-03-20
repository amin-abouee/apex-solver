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

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat, sparse::SparseColMat};

    #[test]
    fn test_sparse_mode_jacobian_is_sparse_col_mat() {
        let j: <SparseMode as LinearizationMode>::Jacobian =
            SparseColMat::try_new_from_triplets(1, 1, &[faer::sparse::Triplet::new(0usize, 0usize, 1.0f64)])
                .unwrap();
        assert_eq!(j.ncols(), 1);
    }

    #[test]
    fn test_sparse_mode_hessian_is_sparse_col_mat() {
        let h: <SparseMode as LinearizationMode>::Hessian =
            SparseColMat::try_new_from_triplets(
                2,
                2,
                &[
                    faer::sparse::Triplet::new(0usize, 0usize, 1.0f64),
                    faer::sparse::Triplet::new(1usize, 1usize, 2.0f64),
                ],
            )
            .unwrap();
        assert_eq!(h.nrows(), 2);
    }

    #[test]
    fn test_dense_mode_jacobian_is_mat() {
        let j: <DenseMode as LinearizationMode>::Jacobian = Mat::zeros(3, 4);
        assert_eq!(j.nrows(), 3);
        assert_eq!(j.ncols(), 4);
    }

    #[test]
    fn test_dense_mode_hessian_is_mat() {
        let h: <DenseMode as LinearizationMode>::Hessian =
            Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        assert!((h[(0, 0)] - 1.0).abs() < 1e-12);
        assert!(h[(0, 1)].abs() < 1e-12);
    }
}
