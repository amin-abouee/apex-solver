//! GPU linear solvers backed by NVIDIA cuSOLVER (via [`cudarc`]).
//!
//! Enabled by the `cuda` feature. Requires an NVIDIA GPU and driver **at
//! runtime**; the code builds anywhere, because cudarc resolves the CUDA
//! libraries dynamically rather than linking them at build time.
//!
//! # What runs where
//!
//! Only the *factorization* moves to the GPU. Each optimizer iteration still
//! assembles `J` on the CPU and forms `H = JᵀJ` and `g = Jᵀr` with faer, exactly
//! as the CPU solvers do; the normal equations are then handed to cuSOLVER.
//! Jacobian assembly remains CPU-side — see [`crate::linearizer::gpu`].
//!
//! ```text
//! CPU: assemble J  ->  H = JᵀJ, g = Jᵀr
//!                          │  CSR values + cached i32 structure
//!                          ▼
//! GPU: cusolverSpDcsrlsvchol / csrlsvqr  ->  dx
//! ```
//!
//! # Host↔device traffic
//!
//! `cusolverSp*csrlsv*` takes host pointers and manages the transfer itself, so
//! the per-iteration cost is `nnz` values up, `n` right-hand-side values up, and
//! `n` solution values down. The `i32` CSR index arrays are converted once and
//! cached in [`device::CsrStructure`], since the sparsity pattern is constant
//! across iterations.
//!
//! # Known limitation: symbolic analysis is repeated
//!
//! cudarc binds `cusolverDn.h`, `cusolverSp.h`, `cusolverMg.h` and
//! `cusolverRf.h`, but **not** `cusolverSp_LOWLEVEL_PREVIEW.h`. The reusable
//! `cusolverSpDcsrcholAnalysis`/`Factor`/`Solve` triple therefore is not
//! available, and this implementation must use the one-shot
//! `cusolverSpDcsrlsvchol`, which redoes reordering and symbolic analysis on
//! every call. Since the sparsity pattern never changes, that work is repeated
//! needlessly.
//!
//! Two escape hatches, in order of effort:
//!
//! 1. [`device::Reordering`] is configurable; `Reordering::None` skips the
//!    fill-reducing permutation when analysis dominates.
//! 2. The missing symbols exist in the shipped `libcusolver` — cudarc simply
//!    does not declare them. Adding ~6 `extern "C"` declarations would enable
//!    factor-once/solve-many. That is a contained follow-up.
//!
//! Benchmark before assuming the current numbers are the ceiling.
//!
//! # Robust loss and covariance
//!
//! These solvers see the same Triggs-corrected `J` as the CPU path and cache the
//! same quantities: `get_hessian` returns the **undamped** `JᵀJ` and
//! `get_gradient` the **positive** `Jᵀr`.
//!
//! Both implement `compute_covariance_matrix` explicitly rather than inheriting
//! the trait's `None` default, so `with_compute_covariances(true)` does not
//! silently yield no result on the GPU path.
//!
//! Note the covariance is computed by inverting the **undamped** `H` on the CPU,
//! not by reusing a damped factorization — Levenberg-Marquardt's `λ` is an
//! internal device and must not appear in a reported uncertainty. See
//! `cov_issues/01-covariance-absorbs-damping.md`.

pub mod cholesky;
pub mod device;
pub mod qr;

pub use cholesky::GpuSparseCholeskySolver;
pub use device::{GpuContext, Reordering, is_available};
pub use qr::GpuSparseQRSolver;
