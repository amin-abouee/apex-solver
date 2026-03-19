# 3. CPU Dense Solvers

## 3.1 Overview

Dense solvers are optimal for **small-to-medium problems** where the Jacobian/Hessian fit in memory as full matrices. Typical use cases:

- **Visual odometry**: 2-10 keyframes, 50-200 landmarks → H is ~300-1200 DOF
- **Small pose graphs**: <500 poses → H is ~3000 DOF
- **Bundle adjustment with few cameras**: <50 cameras → Schur complement is small and dense
- **Calibration problems**: Fixed structure, small parameter count

For these sizes, dense operations (BLAS/LAPACK via faer) are often **faster** than sparse due to:
- No symbolic analysis overhead
- Better cache utilization for dense GEMM
- SIMD-optimized dense kernels in faer

**Crossover point**: Sparse typically wins above ~500-1000 DOF (problem-dependent).

## 3.2 DenseCholeskySolver

### File: `src/linalg/dense_cholesky.rs` (NEW)

```rust
use faer::{Mat, linalg::cholesky, Side, sparse::SparseColMat};
use crate::linalg::{LinAlgError, LinAlgResult, SparseLinearSolver};

#[derive(Debug, Clone)]
pub struct DenseCholeskySolver {
    /// Dense Hessian H = J^T * J (used for factorization)
    dense_hessian: Option<Mat<f64>>,
    /// Sparse copy of Hessian (for get_hessian() compatibility with DogLeg)
    sparse_hessian: Option<SparseColMat<usize, f64>>,
    /// Gradient g = J^T * r
    gradient: Option<Mat<f64>>,
    /// Covariance matrix H^{-1}
    covariance_matrix: Option<Mat<f64>>,
}
```

### Algorithm

```
solve_normal_equation(residuals, sparse_jacobians):
    1. J_dense = sparse_to_dense(sparse_jacobians)         // O(nnz)
    2. H = J_dense^T * J_dense                              // O(m²n) dense GEMM
    3. g = J_dense^T * residuals                             // O(mn)
    4. L = cholesky(H)                                       // O(m³/3) dense Cholesky
    5. dx = L.solve(-g)                                      // O(m²) forward/back substitution
    6. Cache: dense_hessian = H, sparse_hessian = dense_to_sparse(H)
    7. Return dx

solve_augmented_equation(residuals, sparse_jacobians, lambda):
    1-3. Same as above
    4. H_aug = H + lambda * I                                // O(m) diagonal addition
    5. L = cholesky(H_aug)
    6. dx = L.solve(-g)
    7. Cache as above
    8. Return dx
```

### faer Dense API Usage

```rust
use faer::linalg::cholesky::llt::compute::cholesky_in_place;
use faer::linalg::cholesky::llt::solve::solve_in_place;

// Form H = J^T * J
let h = j_dense.transpose() * &j_dense;

// In-place Cholesky factorization
let mut h_copy = h.clone();
cholesky_in_place(h_copy.as_mut(), Side::Lower, Default::default(), stack)?;

// Solve H * dx = -g
let mut dx = neg_gradient.clone();
solve_in_place(h_copy.as_ref(), Side::Lower, dx.as_mut(), Default::default(), stack)?;
```

### Complexity

| Operation | Sparse Cholesky | Dense Cholesky |
|-----------|----------------|----------------|
| H = J^T * J | O(nnz × fill) | O(m²n) |
| Factorization | O(m × fill²) | O(m³/3) |
| Solve | O(m × fill) | O(m²) |
| Memory | O(nnz_H) | O(m²) |

Where m = DOF (columns), n = residuals (rows), nnz = non-zeros, fill = fill-in after elimination.

### Effort Estimate

- **Implementation**: 1 day (straightforward faer dense API)
- **Testing**: 0.5 day (mirror existing Cholesky tests)
- **Integration**: 0.5 day (add to enum, factory, Display)
- **Total**: **2 days**

## 3.3 DenseQRSolver

### File: `src/linalg/dense_qr.rs` (NEW)

```rust
#[derive(Debug, Clone)]
pub struct DenseQRSolver {
    dense_hessian: Option<Mat<f64>>,
    sparse_hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
    covariance_matrix: Option<Mat<f64>>,
}
```

### Two Approaches

**Approach A: QR on Hessian** (normal equations)
```
H = J^T * J
QR(H) → Q, R
dx = R^{-1} * Q^T * (-g)
```

**Approach B: QR on Jacobian directly** (preferred for numerical stability)
```
QR(J) → Q, R          // J is n×m, Q is n×n, R is n×m
R_top * dx = -(Q^T * r)_top    // Only top m rows
```

Approach B avoids forming H = J^T * J, which squares the condition number. This is the advantage of QR over Cholesky for ill-conditioned problems.

**For augmented equations** (LM): Cannot use Approach B directly. Must augment:
```
[    J    ]         [ r ]
[ √λ * I ] * dx = -[ 0 ]

QR of augmented matrix, then solve.
```

### faer Dense QR API

```rust
use faer::linalg::qr::col_pivoting::compute::qr_in_place;

// QR factorization
let mut j_copy = j_dense.clone();
let (perm, tau) = qr_in_place(j_copy.as_mut(), stack)?;
// Solve via Q^T * b and back-substitution on R
```

### Effort Estimate

- **Implementation**: 1.5 days (augmented system handling adds complexity)
- **Testing**: 0.5 day
- **Integration**: Already done with DenseCholesky
- **Total**: **2 days**

## 3.4 Sparse-to-Dense Conversion Utility

### File: `src/linalg/utils.rs` (NEW)

```rust
use faer::{Mat, sparse::SparseColMat};

/// Convert a sparse column-major matrix to a dense matrix.
///
/// Efficient for small-to-medium matrices. For large matrices (>5000 rows/cols),
/// consider using sparse solvers directly instead.
pub fn sparse_to_dense(sparse: &SparseColMat<usize, f64>) -> Mat<f64> {
    let nrows = sparse.nrows();
    let ncols = sparse.ncols();
    let mut dense = Mat::zeros(nrows, ncols);

    // Iterate over columns (efficient for CSC format)
    for j in 0..ncols {
        for (i, &val) in sparse.col_indices(j).zip(sparse.col_values(j)) {
            dense[(i, j)] = val;
        }
    }
    dense
}

/// Convert a dense matrix to sparse column-major format.
///
/// Only stores entries with |value| > threshold.
pub fn dense_to_sparse(dense: &Mat<f64>, threshold: f64) -> LinAlgResult<SparseColMat<usize, f64>> {
    let nrows = dense.nrows();
    let ncols = dense.ncols();
    let mut triplets = Vec::new();

    for j in 0..ncols {
        for i in 0..nrows {
            let val = dense[(i, j)];
            if val.abs() > threshold {
                triplets.push(Triplet::new(i, j, val));
            }
        }
    }

    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .map_err(|e| LinAlgError::SparseMatrixCreation("dense_to_sparse failed".into()).log_with_source(e))
}
```

**Note**: faer may provide built-in `SparseColMatRef::to_dense()` — verify before implementing. If available, use the library version.

## 3.5 Impact on Existing Code

### Files Modified

| File | Change | Lines Affected |
|------|--------|----------------|
| `src/linalg/mod.rs` | Add module declarations, enum variants, re-exports | ~15 lines |
| `src/optimizer/mod.rs` | Update `create_linear_solver()` factory | ~5 lines |
| `src/optimizer/levenberg_marquardt.rs` | Update inline solver match | ~5 lines |

### Files Created

| File | Purpose | Estimated Lines |
|------|---------|----------------|
| `src/linalg/dense_cholesky.rs` | Dense Cholesky implementation | ~200 |
| `src/linalg/dense_qr.rs` | Dense QR implementation | ~250 |
| `src/linalg/utils.rs` | Conversion utilities | ~60 |

### Files NOT Modified

- `src/core/problem.rs` — No changes needed
- `src/optimizer/gauss_newton.rs` — Uses factory, auto-works
- `src/optimizer/dog_leg.rs` — Uses factory + `get_hessian()` which returns sparse copy
- All tests — Existing tests continue to pass unchanged

## 3.6 When to Use Dense vs Sparse

| Problem Size (DOF) | Recommended Solver | Reason |
|--------------------|-------------------|--------|
| < 100 | Dense Cholesky | Fastest, minimal overhead |
| 100 - 500 | Dense Cholesky or Sparse Cholesky | Benchmark to decide |
| 500 - 1,000 | Sparse Cholesky | Dense O(n³) becomes costly |
| > 1,000 | Sparse Cholesky | Dense memory O(n²) too large |
| Rank-deficient | Dense QR or Sparse QR | QR handles rank deficiency |
| Bundle adjustment | Schur Complement | Exploits problem structure |

A runtime warning could be emitted when dense solvers are used with >1000 DOF.
