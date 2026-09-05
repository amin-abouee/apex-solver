//! Storage-agnostic building blocks shared by every Schur complement solver:
//! [`sparse::schur::ExplicitSparseSchur`](crate::linalg::sparse::schur::ExplicitSparseSchur),
//! [`sparse::schur::ImplicitSparseSchur`](crate::linalg::sparse::schur::ImplicitSparseSchur), and
//! [`dense::schur::ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur).
//!
//! Nothing here depends on whether the Hessian is a `SparseColMat` or a dense
//! `Mat`: the variable partition ([`SchurPartition`]), the automatic
//! eliminated/retained classification ([`SchurOrdering`]), the preconditioner
//! selection ([`SchurPreconditioner`]), and the PCG solve loop ([`pcg`]) are
//! the same regardless of storage or of what the eliminated variable
//! represents (a 3-D point, an inverse-depth landmark, or anything else).

pub mod jacobian_ops;
pub mod ordering;
pub mod partition;
pub mod pcg;
pub mod preconditioner;

pub use jacobian_ops::{column_dot, diag_jt_j, jt_j_vec_product, jt_vec};
pub use ordering::{SchurOrdering, effective_landmark_keys};
pub use partition::{BlockSpan, ColSlot, EliminatedBlocks, SchurPartition};
pub use pcg::{DEFAULT_ETA, PcgParams, PcgResult, PcgTermination, pcg as solve_pcg};
pub use preconditioner::SchurPreconditioner;
