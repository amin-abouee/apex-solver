//! GPU linearizer implementations (reserved — nothing runs here yet).
//!
//! # Nothing on the GPU is linearization today
//!
//! It is easy to assume that "the GPU solvers" put work here. They do not.
//! Building `J` from the factor graph and forming `H = JᵀJ` / `g = Jᵀr` run on
//! the CPU for **both** backends; the only thing a CUDA solver changes is the
//! factorization of the normal equations, which is linear algebra and therefore
//! lives in [`crate::linalg::sparse`] beside its CPU counterpart. The device
//! plumbing those solvers share is in [`crate::cuda`].
//!
//! ```text
//! linearizer/cpu   assemble J, form H = JᵀJ      <- CPU, both backends
//! linalg/sparse    factorize and solve           <- CPU (faer) or GPU (cuSOLVER)
//! ```
//!
//! # What would go here
//!
//! On-device Jacobian assembly, and the sparse GEMM that forms `H = JᵀJ`. The
//! second is the more valuable of the two and the smaller job: because assembly
//! is shared, it caps the end-to-end gain from any solver speedup. On city10000
//! it is 714 ms of a 1496 ms CPU run, which is why a measured 1.96x speedup in
//! the linear solve shows up as only 1.62x overall. Moving that product to
//! cuSPARSE SpGEMM is the next lever.
//!
//! Full on-device linearization — evaluating residual blocks and scattering
//! Jacobian blocks from CUDA kernels — would additionally require porting every
//! factor and manifold operation to the device.
