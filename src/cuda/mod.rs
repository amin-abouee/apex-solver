//! CUDA device plumbing for the GPU linear solvers.
//!
//! Enabled by the `cuda` feature. Requires an NVIDIA GPU and driver **at
//! runtime**; the code builds anywhere, because cudarc resolves the CUDA
//! libraries dynamically rather than linking them at build time.
//!
//! # What lives here, and what does not
//!
//! This module is infrastructure only — context, handles, device buffers, raw
//! bindings, profiling. **No decomposition logic.** The solvers themselves sit
//! with their CPU counterparts in [`crate::linalg::sparse::cholesky`] and
//! [`crate::linalg::sparse::qr`], because a sparse Cholesky is a sparse Cholesky
//! regardless of which device runs it.
//!
//! ```text
//! linalg/sparse/cholesky.rs   SparseCholeskySolver      (faer, CPU)
//!                             CudaSparseCholeskySolver  (cuSOLVER)   — no unsafe
//!         │ safe API
//!         ▼
//! cuda/context.rs             CudaContext, MatDescr, CholeskyInfo    — all unsafe
//! cuda/buffers.rs             CsrStructure, DeviceSystem
//! cuda/ffi.rs                 the 7 symbols cudarc does not declare
//! cuda/profile.rs             per-phase timings, device memory
//! ```
//!
//! [`context`] is the **only** module in the crate containing `unsafe`. Calling a
//! C library cannot be made safe — cudarc wraps handle creation and nothing else
//! — but it can be confined, and every FFI call here sits behind a checked
//! wrapper with RAII-owned handles.
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
//! GPU: analyze once -> factor + solve per iteration -> dx
//! ```
//!
//! # Host↔device traffic
//!
//! The unsuffixed cuSOLVER entry points require `csrVal`, `csrRowPtr`,
//! `csrColInd`, `b` and `x` in **device** memory; only the `singularity` and
//! `position` out-parameters stay host `int*`. (The `...Host` variants take host
//! pointers and factorize on the CPU, which would defeat the purpose.)
//!
//! Per iteration that costs `nnz` values up, `n` right-hand-side values up and
//! `n` solution values down. The `i32` CSR index arrays are converted and
//! uploaded once, since the sparsity pattern is constant across iterations —
//! including across Levenberg-Marquardt's `λ` retries, which only touch the
//! diagonal.
//!
//! # Analysis is done once, not per iteration
//!
//! cudarc binds `cusolverDn.h`, `cusolverSp.h`, `cusolverMg.h` and
//! `cusolverRf.h`, but **not** `cusolverSp_LOWLEVEL_PREVIEW.h`, where the
//! reusable `Xcsrcholanalysis`/`Dcsrcholfactor`/`Dcsrcholsolve` triple lives.
//! Those symbols are present in the shipped `libcusolver`, so [`ffi`] resolves
//! them by `dlopen`.
//!
//! This matters more than anything else here. The one-shot
//! `cusolverSpDcsrlsvchol` redoes reordering and symbolic analysis on *every*
//! call, while the CPU solver caches its `SymbolicLlt`; measured on the M3500
//! pose graph, dropping the repeated analysis cut GPU solve time from 1108 ms to
//! 436 ms, and switching the permutation from AMD to nested dissection took it
//! to 316 ms against 205 ms on the CPU.
//!
//! # When the GPU actually wins
//!
//! Measured on an RTX 3080 Laptop (16 GB) against 16 CPU cores running faer,
//! Levenberg-Marquardt, identical iteration counts, median of three runs:
//!
//! - **Wins** where the factorization has enough dense work to fill the device:
//!   `rim` 2.08x, `city10000` 1.96x, `cubicle` 1.73x, `torus3D` 1.53x,
//!   `sphere2500` 1.45x on solve time.
//! - **Loses** on small or chain-like problems, where per-call latency dominates
//!   — `ring` (434 poses) is 4.5x slower, `intel` 2.6x.
//! - **Parity at best on bundle adjustment**: BAL normal equations are dominated
//!   by a block-diagonal landmark section with almost no dense arithmetic.
//! - **Runs out of headroom** on very large problems: ladybug (485k DOF,
//!   38M nonzeros) needs a 4.3 GB cuSOLVER workspace and does not finish within
//!   25 minutes, against 11 minutes on the CPU.
//!
//! Benchmark your own problem — the crossover is driven by fill-in, not by DOF.
//! [`profile::CudaProfile`] breaks a run down by phase to show where it goes.

pub mod buffers;
pub mod context;
pub mod ffi;
pub mod profile;

pub use context::{CudaContext, Reordering, is_available};
pub use profile::{CudaProfile, DeviceMemory, Phase};
