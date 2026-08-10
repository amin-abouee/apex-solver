//! Sparse linear solvers, one module per decomposition.
//!
//! Each module holds every backend for that decomposition. `cholesky.rs` has the
//! faer-based [`SparseCholeskySolver`] and, behind the `cuda` feature, the
//! cuSOLVER-based [`CudaSparseCholeskySolver`] — a sparse Cholesky is a sparse
//! Cholesky regardless of which device runs it. The CUDA device plumbing they
//! share lives in [`crate::cuda`].

pub mod cholesky;
pub mod explicit_schur;
pub mod implicit_schur;
pub mod qr;

pub use cholesky::SparseCholeskySolver;
pub use explicit_schur::{
    SchurBlockStructure, SchurOrdering, SchurPreconditioner, SchurVariant,
    SparseSchurComplementSolver,
};
pub use implicit_schur::IterativeSchurSolver;
pub use qr::SparseQRSolver;

#[cfg(feature = "cuda")]
pub use cholesky::{CholeskyAlgorithm, CudaSparseCholeskySolver};
#[cfg(feature = "cuda")]
pub use qr::CudaSparseQRSolver;

#[cfg(feature = "cuda")]
use faer::Mat;
#[cfg(feature = "cuda")]
use faer::sparse::{SparseColMat, Triplet};
#[cfg(feature = "cuda")]
use std::ops::Mul;

#[cfg(feature = "cuda")]
use crate::error::ErrorLogging;
#[cfg(feature = "cuda")]
use crate::linalg::{LinAlgError, LinAlgResult};

/// `H = JᵀJ`, matching the CPU solvers' construction exactly.
///
/// The CPU solvers inline this; the CUDA ones share it so the two paths cannot
/// drift in how they form the normal equations.
#[cfg(feature = "cuda")]
pub(crate) fn normal_matrix(
    jacobians: &SparseColMat<usize, f64>,
) -> LinAlgResult<SparseColMat<usize, f64>> {
    let jt = jacobians.as_ref().transpose();
    Ok(jt
        .to_col_major()
        .map_err(|e| {
            LinAlgError::MatrixConversion(
                "Failed to convert transposed Jacobian to column-major format".to_string(),
            )
            .log_with_source(e)
        })?
        .mul(jacobians.as_ref()))
}

/// Build `H + λI` without disturbing the caller's undamped `H`.
#[cfg(feature = "cuda")]
pub(crate) fn add_damping(
    hessian: &SparseColMat<usize, f64>,
    lambda: f64,
) -> LinAlgResult<SparseColMat<usize, f64>> {
    let n = hessian.ncols();
    let lambda_triplets: Vec<Triplet<usize, usize, f64>> =
        (0..n).map(|i| Triplet::new(i, i, lambda)).collect();
    let lambda_i = SparseColMat::try_new_from_triplets(n, n, &lambda_triplets).map_err(|e| {
        LinAlgError::SparseMatrixCreation("Failed to create lambda*I matrix".to_string())
            .log_with_source(e)
    })?;
    Ok(hessian + lambda_i)
}

/// Invert `H` on the CPU via a fresh Cholesky of the undamped matrix.
///
/// Deliberately does not reuse any factorization produced during a solve: the
/// augmented solve factorizes `H + λI`, and a reported covariance must reflect
/// `H` alone — Levenberg-Marquardt's `λ` is an internal device and must never
/// appear in an uncertainty.
#[cfg(feature = "cuda")]
pub(crate) fn invert_undamped_hessian(hessian: &SparseColMat<usize, f64>) -> Option<Mat<f64>> {
    use faer::Side;
    use faer::linalg::solvers::Solve;
    use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};

    let n = hessian.ncols();
    let symbolic = SymbolicLlt::try_new(hessian.symbolic(), Side::Lower).ok()?;
    let llt = Llt::try_new_with_symbolic(symbolic, hessian.as_ref(), Side::Lower).ok()?;
    Some(llt.solve(Mat::<f64>::identity(n, n)))
}
