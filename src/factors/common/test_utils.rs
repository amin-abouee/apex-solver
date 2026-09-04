//! Shared helpers for factor unit tests.
//!
//! Per-block perturbation construction (manifold `right_plus` vs. plain-vector
//! `+= eps`) differs by factor and stays inline in each test; this only factors
//! out the boilerplate repeated verbatim across factor modules.

use apex_manifolds::LieGroup;
use apex_manifolds::Tangent;
use apex_manifolds::se3::{SE3, SE3Tangent};
use nalgebra::{DMatrix, DVector, UnitQuaternion};

use crate::factors::Factor;

/// Assert a single analytical Jacobian entry matches its finite-difference
/// estimate within `tol`, panicking with both values and the error on failure.
pub(crate) fn assert_close(analytical: f64, fd: f64, tol: f64, label: &str) {
    let err = (analytical - fd).abs();
    assert!(
        err < tol,
        "{label}: analytical={analytical:.8} fd={fd:.8} err={err:.2e}"
    );
}

/// Build an SE(3) parameter vector `[tx, ty, tz, qw, qx, qy, qz]`.
pub(crate) fn make_pose(tx: f64, ty: f64, tz: f64, q: UnitQuaternion<f64>) -> DVector<f64> {
    let q = q.quaternion();
    DVector::from_vec(vec![tx, ty, tz, q.w, q.i, q.j, q.k])
}

/// The SE(3) identity as a parameter vector.
pub(crate) fn identity_pose() -> DVector<f64> {
    DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
}

/// Apply a right-tangent perturbation to an SE(3) parameter vector.
pub(crate) fn perturb_se3(pose: &[f64], tangent: &[f64; 6]) -> DVector<f64> {
    let se3 = SE3::from_param_slice(pose);
    let tan = SE3Tangent::from_slice(tangent);
    DVector::from_column_slice(se3.right_plus(&tan, None, None).as_param_slice())
}

/// Evaluate a two-pose factor's residual only.
pub(crate) fn compute_residual<F: Factor>(factor: &F, t_wa: &[f64], t_wb: &[f64]) -> Vec<f64> {
    let mut residual = vec![0.0f64; factor.residual_dim()];
    factor.linearize(&[t_wa, t_wb], &mut residual, None);
    residual
}

/// Evaluate a two-pose factor's residual and dense Jacobian.
pub(crate) fn compute_with_jacobian<F: Factor>(
    factor: &F,
    t_wa: &[f64],
    t_wb: &[f64],
) -> (Vec<f64>, DMatrix<f64>) {
    let (rows, cols) = factor.jacobian_shape();
    let mut residual = vec![0.0f64; rows];
    let mut jac_buf = vec![0.0f64; rows * cols];
    let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
    factor.linearize(&[t_wa, t_wb], &mut residual, Some(jac_mut));
    let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
    (residual, jacobian)
}
