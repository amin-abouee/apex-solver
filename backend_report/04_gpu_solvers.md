# 4. GPU Solvers via cudarc

## 4.1 Overview

GPU acceleration targets **large dense systems** and **large sparse systems** where the GPU's parallel throughput outweighs the host↔device transfer cost. The `cudarc` crate provides safe Rust bindings to NVIDIA CUDA runtime, cuBLAS, cuSOLVER, and cuSPARSE.

### When GPU Wins

| Scenario | CPU (faer) | GPU (cudarc) | GPU Advantage |
|----------|-----------|-------------|---------------|
| Dense 500×500 | ~2 ms | ~5 ms (transfer-dominated) | None |
| Dense 2000×2000 | ~80 ms | ~15 ms | **5× faster** |
| Dense 5000×5000 | ~1.5 s | ~60 ms | **25× faster** |
| Sparse 10K×10K, 1% fill | ~50 ms | ~30 ms | **1.7× faster** |
| Sparse 100K×100K, 0.1% fill | ~2 s | ~200 ms | **10× faster** |

*Estimates for NVIDIA A100. Consumer GPUs (RTX 3090/4090) have 1/32 f64 throughput — less advantage.*

### cudarc Crate

- **Version**: 0.16.x (as of 2026)
- **License**: MIT/Apache-2.0
- **CUDA Toolkit**: Requires CUDA 11.x or 12.x
- **Key features used**: `driver`, `cublas`, `cusolver`, `cusparse`
- **Thread safety**: `CudaDevice` is `Arc`-wrapped, `Send + Sync`

## 4.2 GPU Context Management

### File: `src/linalg/gpu/mod.rs` and `src/linalg/gpu/context.rs` (NEW)

```rust
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use std::sync::Arc;

/// Shared GPU context for all GPU solvers.
///
/// Manages device selection, memory allocation, and stream synchronization.
/// A single context can be shared across multiple GPU solvers.
pub struct GpuContext {
    device: Arc<CudaDevice>,
}

impl GpuContext {
    /// Create a new GPU context on the specified device (default: device 0).
    pub fn new(device_id: usize) -> Result<Self, LinAlgError> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| LinAlgError::GpuError(format!("Failed to init CUDA device {}: {}", device_id, e)))?;
        Ok(Self { device })
    }

    /// Get the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}
```

### Memory Transfer Pattern

```
Per optimization iteration:
  CPU                                    GPU
  ────                                   ────
  1. Compute J (sparse) + r (dense)
  2. Convert J → dense (for dense GPU)
     OR prepare CSR (for sparse GPU)
  3. Compute H = J^T * J on CPU ─────→  4. Copy H to device
     Compute g = J^T * r on CPU ─────→  5. Copy g to device
                                         6. cuSOLVER factorize H
                                         7. cuSOLVER solve H*dx = -g
  8. Copy dx back ←──────────────────── (dx on device)
  9. Apply manifold update on CPU
```

**Buffer reuse optimization**: Pre-allocate device buffers on first call, reuse across iterations when dimensions match:

```rust
pub struct GpuDenseCholeskySolver {
    ctx: GpuContext,
    // Reusable device buffers (allocated once, reused across iterations)
    d_hessian: Option<CudaSlice<f64>>,
    d_gradient: Option<CudaSlice<f64>>,
    d_workspace: Option<CudaSlice<f64>>,
    last_dim: usize,  // Track dimension changes
    // Host-side caches
    dense_hessian: Option<Mat<f64>>,
    sparse_hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
    covariance_matrix: Option<Mat<f64>>,
}
```

## 4.3 GPU Dense Cholesky

### File: `src/linalg/gpu/dense_cholesky.rs` (NEW)

### Algorithm

```
solve_augmented_equation(residuals, sparse_jacobians, lambda):
    1. J_dense = sparse_to_dense(sparse_jacobians)           // CPU, O(nnz)
    2. H = J_dense^T * J_dense + lambda * I                  // CPU, O(m²n)
    3. g = J_dense^T * residuals                              // CPU, O(mn)
    4. d_H = device.htod_sync(H.as_slice())                  // Host→Device, O(m²)
    5. d_g = device.htod_sync(neg_g.as_slice())              // Host→Device, O(m)
    6. cusolverDnDpotrf(d_H, m, ...)                         // GPU Cholesky, O(m³/3)
    7. cusolverDnDpotrs(d_H, d_g, m, 1, ...)                // GPU solve, O(m²)
    8. dx = device.dtoh_sync(&d_g)                           // Device→Host, O(m)
    9. Return Mat::from_fn(m, 1, |i, _| dx[i])
```

### cuSOLVER API Calls

```rust
use cudarc::cusolver::{CudaSolver, DnHandle};

// Dense Cholesky factorization
let handle = DnHandle::new(device.clone())?;

// Query workspace size
let workspace_size = handle.potrf_buffer_size(
    cublas::sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    m as i32,
    &d_hessian,
    m as i32,
)?;

// Allocate workspace
let d_workspace = device.alloc_zeros(workspace_size)?;

// Factorize: L * L^T = H
handle.potrf(
    cublas::sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    m as i32,
    &mut d_hessian,
    m as i32,
    &mut d_workspace,
    &mut d_info,
)?;

// Solve: H * dx = -g (in-place, overwrites d_gradient)
handle.potrs(
    cublas::sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    m as i32,
    1,  // nrhs
    &d_hessian,
    m as i32,
    &mut d_gradient,
    m as i32,
    &mut d_info,
)?;
```

### Advanced: H = J^T * J on GPU

For large problems, forming H on GPU avoids transferring the large J matrix:

```rust
use cudarc::cublas::{CudaBlas, BlasHandle};

// Transfer J (dense, n×m) to GPU
let d_j = device.htod_sync(j_dense.as_slice())?;

// H = J^T * J via cuBLAS DSYRK
let blas = BlasHandle::new(device.clone())?;
blas.dsyrk(
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_OP_T,          // J^T
    m as i32,             // H is m×m
    n as i32,             // J has n rows
    1.0,                  // alpha
    &d_j, n as i32,      // J (column-major, n rows)
    0.0,                  // beta
    &mut d_h, m as i32,  // H output
)?;

// g = J^T * r via cuBLAS DGEMV
blas.dgemv(
    CUBLAS_OP_T,          // J^T
    n as i32, m as i32,
    1.0,
    &d_j, n as i32,
    &d_r, 1,
    0.0,
    &mut d_g, 1,
)?;
```

This is the **all-GPU path** — recommended for m > 2000.

### Effort Estimate

- **Implementation**: 3 days (cudarc API integration, error handling, memory management)
- **Testing**: 1 day (requires GPU; mock tests for CI without GPU)
- **Buffer reuse optimization**: 1 day
- **All-GPU path (optional)**: 1 day
- **Total**: **5-6 days**

## 4.4 GPU Dense QR

### File: `src/linalg/gpu/dense_qr.rs` (NEW)

### cuSOLVER API

```rust
// QR factorization of augmented system
handle.geqrf(m_aug, m, &mut d_j_aug, m_aug, &mut d_tau, &mut d_work)?;

// Apply Q^T to right-hand side
handle.ormqr(SIDE_LEFT, TRANS, m_aug, 1, m, &d_j_aug, m_aug, &d_tau, &mut d_rhs, m_aug, &mut d_work)?;

// Back-substitution on R (upper triangular)
blas.dtrsv(UPPER, NO_TRANS, NON_UNIT, m, &d_j_aug, m_aug, &mut d_rhs, 1)?;
```

### Effort Estimate

- **Implementation**: 2 days (pattern established by GPU Dense Cholesky)
- **Testing**: 1 day
- **Total**: **3 days**

## 4.5 GPU Sparse Cholesky

### File: `src/linalg/gpu/sparse_cholesky.rs` (NEW)

### Format Conversion: CSC → CSR

faer uses CSC (Compressed Sparse Column). cuSOLVER sparse uses CSR (Compressed Sparse Row). For symmetric H = J^T * J, CSC(H) = CSR(H^T) = CSR(H), so we can use the CSC arrays directly as CSR.

```rust
// faer CSC format:
//   col_ptrs: [usize; ncols+1]
//   row_indices: [usize; nnz]
//   values: [f64; nnz]

// For symmetric H, interpret CSC as CSR:
//   row_ptrs = col_ptrs (renamed)
//   col_indices = row_indices (renamed)
//   values = values (same)

// Note: This only works for symmetric matrices (H = J^T * J is always symmetric)
```

For non-symmetric matrices (e.g., the Jacobian itself), a full CSC→CSR transpose is needed:

```rust
fn csc_to_csr(
    nrows: usize, ncols: usize,
    col_ptrs: &[usize], row_indices: &[usize], values: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // 1. Count entries per row
    // 2. Compute row_ptrs via prefix sum
    // 3. Scatter entries into CSR arrays
    // O(nnz) time and space
}
```

### cuSOLVER Sparse API

```rust
use cudarc::cusolver::SpHandle;

let sp_handle = SpHandle::new(device.clone())?;

// Create sparse matrix descriptor
let mat_descr = sp_handle.create_mat_descr()?;

// Sparse Cholesky solve (combined factorize + solve)
sp_handle.csrlsvchol(
    m as i32,
    nnz as i32,
    &mat_descr,
    &d_values,
    &d_row_ptrs,
    &d_col_indices,
    &d_rhs,
    tol,
    reorder,  // 0 = no reorder, 1 = symrcm, 2 = symamd
    &mut d_solution,
    &mut singularity,
)?;
```

### Challenges

1. **Symbolic factorization on GPU**: cuSOLVER doesn't expose symbolic/numeric separation as cleanly as faer
2. **Fill-reducing ordering**: Need to pass reorder flag; quality may differ from faer's ordering
3. **Error handling**: `singularity >= 0` means matrix is singular at that row
4. **Memory**: GPU sparse operations can require significant workspace

### Effort Estimate

- **Implementation**: 4 days (format conversion, cuSPARSE descriptors, error handling)
- **CSC↔CSR utilities**: 1 day
- **Testing**: 1-2 days
- **Total**: **6-7 days**

## 4.6 GPU Sparse QR

### File: `src/linalg/gpu/sparse_qr.rs` (NEW)

Uses `cusolverSpDcsrlsvqr()` — similar API to sparse Cholesky but with QR decomposition.

### Effort Estimate

- **Implementation**: 3 days (reuses patterns from GPU Sparse Cholesky)
- **Testing**: 1 day
- **Total**: **4 days**

## 4.7 Error Handling for GPU Operations

### New Error Variants

```rust
// In src/linalg/mod.rs
#[derive(Debug, Clone, Error)]
pub enum LinAlgError {
    // ... existing variants ...

    #[cfg(feature = "cuda")]
    #[error("GPU operation failed: {0}")]
    GpuError(String),

    #[cfg(feature = "cuda")]
    #[error("GPU memory allocation/transfer failed: {0}")]
    GpuMemoryError(String),

    #[cfg(feature = "cuda")]
    #[error("CUDA device not available: {0}")]
    CudaDeviceUnavailable(String),

    #[cfg(feature = "cuda")]
    #[error("GPU factorization failed: matrix singular at row {0}")]
    GpuSingularMatrix(i32),
}
```

### Error Translation from cuSOLVER

```rust
fn check_cusolver_info(info: i32) -> LinAlgResult<()> {
    match info {
        0 => Ok(()),
        i if i > 0 => Err(LinAlgError::GpuSingularMatrix(i)),
        i => Err(LinAlgError::GpuError(format!("cuSOLVER internal error: {}", i))),
    }
}
```

## 4.8 Thread Safety Considerations

The `SparseLinearSolver` trait is used as `Box<dyn SparseLinearSolver>`. It doesn't require `Send + Sync` explicitly, but optimizers may need to pass it across thread boundaries.

```rust
// Check: is this trait object Send?
// cudarc's CudaDevice is Arc<...> which is Send + Sync
// CudaSlice<f64> is Send (transfers ownership) but not Sync
// Our GPU solver struct will be Send but not Sync — this matches SparseCholeskySolver

// If needed, wrap in Mutex for thread-safe access
```

## 4.9 CI/CD Considerations

### Testing without GPU

```rust
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gpu_tests {
    use super::*;

    fn has_cuda_device() -> bool {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }

    #[test]
    fn test_gpu_dense_cholesky() {
        if !has_cuda_device() {
            eprintln!("Skipping GPU test: no CUDA device available");
            return;
        }
        // ... actual test ...
    }
}
```

### GitHub Actions

```yaml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test

  test-gpu:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - run: cargo test --features cuda
```

## 4.10 Performance Considerations

### Double Precision on Consumer GPUs

| GPU | FP64 TFLOPS | FP64/FP32 Ratio |
|-----|------------|-----------------|
| NVIDIA A100 | 9.7 | 1:2 |
| NVIDIA H100 | 30 | 1:2 |
| RTX 4090 | 1.3 | 1:64 |
| RTX 3090 | 0.6 | 1:32 |

Consumer GPUs have severely limited f64 performance. GPU solvers are most beneficial on data-center GPUs (A100, H100, L40S).

**Future consideration**: Add f32 computation mode for consumer GPUs where f64 precision isn't required. This would require a generic approach over float types.

### Transfer Overhead

For a 2000×2000 dense Hessian:
- Matrix size: 2000 × 2000 × 8 bytes = 32 MB
- PCIe 4.0 x16: ~25 GB/s → ~1.3 ms transfer
- GPU Cholesky: ~0.5 ms on A100
- Total GPU: ~2 ms (dominated by transfer)
- CPU Cholesky (faer): ~15 ms on modern CPU

**Conclusion**: GPU wins for dense systems above ~1000 DOF even with transfer overhead.
