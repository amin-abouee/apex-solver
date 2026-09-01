pub mod cholesky;
pub mod explicit_schur;
pub mod implicit_schur;
pub mod normal_eq;
pub mod schur_partition;
pub mod qr;

pub use cholesky::SparseCholeskySolver;
#[allow(deprecated)] // re-export kept so existing imports of the old name resolve
pub use explicit_schur::{
    SchurBlockStructure, SchurOrdering, SchurPreconditioner, SchurVariant,
    SparseSchurComplementSolver,
};
pub use implicit_schur::IterativeSchurSolver;
pub use schur_partition::{BlockSpan, ColSlot, EliminatedBlocks, SchurPartition};
pub use qr::SparseQRSolver;
