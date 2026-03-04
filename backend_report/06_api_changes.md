# 6. API Changes & Migration Guide

## 6.1 Public API Changes

### New Enum Variants

```rust
// Before (v1.2.0)
pub enum LinearSolverType {
    SparseCholesky,       // default
    SparseQR,
    SparseSchurComplement,
}

// After (v1.3.0)
pub enum LinearSolverType {
    SparseCholesky,       // default (unchanged)
    SparseQR,             // unchanged
    SparseSchurComplement,// unchanged
    DenseCholesky,        // NEW
    DenseQR,              // NEW
    #[cfg(feature = "cuda")]
    GpuDenseCholesky,     // NEW (feature-gated)
    #[cfg(feature = "cuda")]
    GpuDenseQR,           // NEW (feature-gated)
    #[cfg(feature = "cuda")]
    GpuSparseCholesky,    // NEW (feature-gated)
    #[cfg(feature = "cuda")]
    GpuSparseQR,          // NEW (feature-gated)
}
```

### New Public Types

```rust
// CPU Dense
pub use linalg::DenseCholeskySolver;
pub use linalg::DenseQRSolver;

// GPU (feature-gated)
#[cfg(feature = "cuda")]
pub use linalg::gpu::{
    GpuContext,
    GpuDenseCholeskySolver,
    GpuDenseQRSolver,
    GpuSparseCholeskySolver,
    GpuSparseQRSolver,
};

// Utility
pub use linalg::utils::{sparse_to_dense, dense_to_sparse};
```

### New Error Variants

```rust
pub enum LinAlgError {
    // ... existing ...
    #[cfg(feature = "cuda")]
    GpuError(String),
    #[cfg(feature = "cuda")]
    GpuMemoryError(String),
    #[cfg(feature = "cuda")]
    CudaDeviceUnavailable(String),
    #[cfg(feature = "cuda")]
    GpuSingularMatrix(i32),
}
```

## 6.2 User-Facing API Examples

### Using Dense Cholesky (simplest change)

```rust
// Before
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::SparseCholesky);

// After (just change the enum variant)
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::DenseCholesky);

// Everything else is identical
let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;
```

### Using GPU Dense Cholesky

```rust
// Requires: cargo build --features cuda

#[cfg(feature = "cuda")]
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::GpuDenseCholesky);

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;
```

### Auto-selection Based on Problem Size (future)

```rust
// Future API possibility
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::Auto);
// Internally selects: Dense for <500 DOF, Sparse for >500 DOF
```

## 6.3 Backward Compatibility

### Fully Backward Compatible

- Default `LinearSolverType::SparseCholesky` is unchanged
- All existing code compiles and runs identically
- No changes to `Problem`, `Solver`, `SolverResult`, or manifold APIs
- No changes to I/O formats

### Semver Impact

- **New enum variants**: This is a **minor** version bump (1.2.0 → 1.3.0)
- **Note**: If `LinearSolverType` is `#[non_exhaustive]`, adding variants is non-breaking. Currently it's not marked `#[non_exhaustive]`, so users with exhaustive `match` statements will get a compile error. Consider adding `#[non_exhaustive]` first.

### Recommended: Add `#[non_exhaustive]`

```rust
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]  // ADD THIS — allows future variant additions without breaking
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
    SparseSchurComplement,
    DenseCholesky,
    DenseQR,
    // ... GPU variants ...
}
```

## 6.4 Complete Solver Matrix

After all phases, the full solver matrix:

| Decomposition | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense |
|--------------|-----------|----------|-----------|----------|
| **Cholesky** | `SparseCholesky` | `DenseCholesky` | `GpuSparseCholesky` | `GpuDenseCholesky` |
| **QR** | `SparseQR` | `DenseQR` | `GpuSparseQR` | `GpuDenseQR` |
| **Schur** | `SparseSchurComplement` | N/A | Future | N/A |
| **Iterative (PCG)** | `IterativeSchurSolver`* | N/A | Future | N/A |

*IterativeSchurSolver uses `StructuredSparseLinearSolver` trait, not the main `SparseLinearSolver` trait.

## 6.5 Configuration Combinatorics

All solvers work with all optimizers:

| Optimizer | Normal Eq | Augmented Eq | get_hessian() | Notes |
|-----------|-----------|-------------|---------------|-------|
| Gauss-Newton | Yes | No | Optional (observers) | All solvers work |
| Levenberg-Marquardt | No | Yes | Optional (unused currently) | All solvers work |
| Dog Leg | Yes (+ augmented with μ) | Yes | **Required** | Needs sparse Hessian |

**Dog Leg constraint**: All dense/GPU solvers must store a sparse Hessian copy for DogLeg compatibility.
