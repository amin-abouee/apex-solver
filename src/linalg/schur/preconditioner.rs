//! Preconditioner selection for PCG-based Schur complement solves.
//!
//! Shared by [`ExplicitSparseSchur`](crate::linalg::sparse::schur::ExplicitSparseSchur)'s
//! `Iterative` variant (PCG on the explicit `S`) and
//! [`ImplicitSparseSchur`](crate::linalg::sparse::schur::ImplicitSparseSchur) (matrix-free
//! PCG). The three options match Ceres's `IDENTITY`/`JACOBI`/`SCHUR_JACOBI` for
//! `ITERATIVE_SCHUR`.

/// Preconditioner type for iterative solvers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchurPreconditioner {
    /// No preconditioning
    None,
    /// Block diagonal of H_cc only (fast but less effective)
    BlockDiagonal,
    /// True Schur-Jacobi: Block diagonal of S = H_cc - H_cp * H_pp^{-1} * H_cp^T
    /// This is what Ceres uses and provides much better PCG convergence
    #[default]
    SchurJacobi,
}
