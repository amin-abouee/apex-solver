# 2. Proposed Multi-Backend Architecture

## 2.1 Design Philosophy

**Principle**: Implement all new backends behind the existing `SparseLinearSolver` trait to achieve **zero changes to optimizer code**.

Dense and GPU solvers accept `SparseColMat` Jacobians (as the trait requires) and convert internally. This is justified because:

- Dense solvers target small-medium problems (<1000 variables) where conversion cost is negligible
- GPU solvers need host→device transfer anyway; format conversion is part of that pipeline
- Avoids a massive refactor of Problem + all 3 optimizers + observers

## 2.2 Trait Architecture

### Option A: Thin Wrapper (Recommended for Phase 1)

Keep `SparseLinearSolver` unchanged. Dense/GPU solvers implement it directly:

```rust
// Existing trait — NO CHANGES
pub trait SparseLinearSolver {
    fn solve_normal_equation(&mut self, residuals: &Mat<f64>, jacobians: &SparseColMat<usize, f64>) -> LinAlgResult<Mat<f64>>;
    fn solve_augmented_equation(&mut self, residuals: &Mat<f64>, jacobians: &SparseColMat<usize, f64>, lambda: f64) -> LinAlgResult<Mat<f64>>;
    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>>;
    fn get_gradient(&self) -> Option<&Mat<f64>>;
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>>;
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>>;
}

// New implementations implement the SAME trait
impl SparseLinearSolver for DenseCholeskySolver { /* sparse→dense internally */ }
impl SparseLinearSolver for DenseQRSolver { /* sparse→dense internally */ }
impl SparseLinearSolver for GpuDenseCholeskySolver { /* sparse→dense→GPU */ }
impl SparseLinearSolver for GpuSparseCholeskySolver { /* sparse→GPU CSR */ }
```

**Pros**: Zero optimizer changes, backward compatible, simple
**Cons**: Trait name is misleading for dense solvers; `get_hessian()` returns sparse type even for dense solvers

### Option B: Unified Trait (Recommended for Phase 2 refactor)

Rename and generalize the trait:

```rust
/// Unified linear solver trait for all backends.
/// Replaces the old SparseLinearSolver trait.
pub trait LinearSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>>;

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>>;

    /// Returns the Hessian as sparse matrix (None if backend stores dense only).
    fn get_hessian_sparse(&self) -> Option<&SparseColMat<usize, f64>>;

    /// Returns the Hessian as dense matrix (None if backend stores sparse only).
    fn get_hessian_dense(&self) -> Option<&Mat<f64>>;

    fn get_gradient(&self) -> Option<&Mat<f64>>;
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>>;
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>>;
}

// Type alias for backward compatibility
pub type SparseLinearSolver = dyn LinearSolver;
```

**Pros**: Clean API, supports both dense and sparse Hessian access
**Cons**: Requires updating all optimizer code + tests (medium refactor)

### Recommendation

Start with **Option A** for Phase 1 (dense CPU). Migrate to **Option B** when adding GPU backends in Phase 2, as the GPU work will touch optimizer code anyway.

## 2.3 Module Structure

```
src/linalg/
├── mod.rs                     # Traits, LinearSolverType, errors
├── cholesky.rs                # SparseCholeskySolver (existing)
├── qr.rs                      # SparseQRSolver (existing)
├── explicit_schur.rs          # Schur complement (existing)
├── implicit_schur.rs          # Iterative Schur (existing)
├── dense_cholesky.rs          # DenseCholeskySolver (NEW)
├── dense_qr.rs                # DenseQRSolver (NEW)
├── utils.rs                   # sparse_to_dense(), dense_to_sparse() helpers (NEW)
└── gpu/                       # GPU backends (NEW, feature-gated)
    ├── mod.rs                 # GPU context management, shared utilities
    ├── context.rs             # CudaContext wrapper, device selection
    ├── dense_cholesky.rs      # GpuDenseCholeskySolver
    ├── dense_qr.rs            # GpuDenseQRSolver
    ├── sparse_cholesky.rs     # GpuSparseCholeskySolver
    └── sparse_qr.rs           # GpuSparseQRSolver
```

## 2.4 LinearSolverType Expansion

```rust
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverType {
    // CPU Sparse (existing)
    #[default]
    SparseCholesky,
    SparseQR,
    SparseSchurComplement,

    // CPU Dense (new)
    DenseCholesky,
    DenseQR,

    // GPU Dense (new, feature-gated)
    #[cfg(feature = "cuda")]
    GpuDenseCholesky,
    #[cfg(feature = "cuda")]
    GpuDenseQR,

    // GPU Sparse (new, feature-gated)
    #[cfg(feature = "cuda")]
    GpuSparseCholesky,
    #[cfg(feature = "cuda")]
    GpuSparseQR,
}
```

## 2.5 Factory Function Update

```rust
pub fn create_linear_solver(solver_type: &LinearSolverType) -> Box<dyn SparseLinearSolver> {
    match solver_type {
        // Existing
        LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
        LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        LinearSolverType::SparseSchurComplement => Box::new(SparseCholeskySolver::new()),

        // New CPU Dense
        LinearSolverType::DenseCholesky => Box::new(DenseCholeskySolver::new()),
        LinearSolverType::DenseQR => Box::new(DenseQRSolver::new()),

        // New GPU (feature-gated)
        #[cfg(feature = "cuda")]
        LinearSolverType::GpuDenseCholesky => {
            Box::new(GpuDenseCholeskySolver::new().expect("Failed to initialize CUDA"))
        }
        #[cfg(feature = "cuda")]
        LinearSolverType::GpuDenseQR => {
            Box::new(GpuDenseQRSolver::new().expect("Failed to initialize CUDA"))
        }
        #[cfg(feature = "cuda")]
        LinearSolverType::GpuSparseCholesky => {
            Box::new(GpuSparseCholeskySolver::new().expect("Failed to initialize CUDA"))
        }
        #[cfg(feature = "cuda")]
        LinearSolverType::GpuSparseQR => {
            Box::new(GpuSparseQRSolver::new().expect("Failed to initialize CUDA"))
        }
    }
}
```

## 2.6 The DogLeg Hessian Problem

DogLeg's `compute_predicted_reduction()` takes `&SparseColMat<usize, f64>` and performs `hessian * step`. This is a hard dependency on sparse format.

**Solution for Dense Solvers**: Store a sparse copy of the Hessian alongside the dense one.

```rust
pub struct DenseCholeskySolver {
    dense_hessian: Option<Mat<f64>>,          // Used for factorization
    sparse_hessian: Option<SparseColMat<usize, f64>>,  // Cached for get_hessian()
    gradient: Option<Mat<f64>>,
    covariance_matrix: Option<Mat<f64>>,
}
```

After computing `H_dense = J_dense^T * J_dense`, convert to sparse: `dense_to_sparse(&H_dense)`. This adds overhead but is necessary for DogLeg compatibility. For LM/GN, this overhead is minimal since `get_hessian()` is only used by observers.

**Alternative** (Phase 2): Refactor DogLeg to accept either format via the unified `LinearSolver` trait with `get_hessian_dense()`.

## 2.7 Feature Flag Strategy

### Cargo.toml Changes

```toml
# workspace Cargo.toml
[workspace.dependencies]
cudarc = { version = "0.16", features = ["driver", "cublas", "cusolver", "cusparse"] }

# root Cargo.toml
[features]
default = []
visualization = ["dep:rerun", "apex-io/visualization"]
cuda = ["dep:cudarc"]

[dependencies]
cudarc = { workspace = true, optional = true }
```

### Code Gating Pattern

```rust
// src/linalg/mod.rs
pub mod dense_cholesky;  // Always available
pub mod dense_qr;        // Always available

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports
pub use dense_cholesky::DenseCholeskySolver;
pub use dense_qr::DenseQRSolver;

#[cfg(feature = "cuda")]
pub use gpu::{GpuDenseCholeskySolver, GpuDenseQRSolver, GpuSparseCholeskySolver, GpuSparseQRSolver};
```

### Build Commands

```bash
cargo build                  # CPU sparse + CPU dense (no GPU)
cargo build --features cuda  # All backends including GPU
cargo test                   # Tests CPU backends only
cargo test --features cuda   # Tests all backends (requires NVIDIA GPU)
```
