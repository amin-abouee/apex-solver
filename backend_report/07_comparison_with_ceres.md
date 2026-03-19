# 7. Comparison with Ceres Solver and Other Libraries

## 7.1 Ceres Solver Linear Algebra Backends

Ceres Solver (Google) supports the most comprehensive set of linear solver backends in the NLLS ecosystem:

| Backend | Type | Library | Apex Equivalent |
|---------|------|---------|-----------------|
| `DENSE_NORMAL_CHOLESKY` | CPU Dense | Eigen/LAPACK | `DenseCholesky` (Phase 1) |
| `DENSE_QR` | CPU Dense | Eigen/LAPACK | `DenseQR` (Phase 1) |
| `SPARSE_NORMAL_CHOLESKY` | CPU Sparse | SuiteSparse/Eigen | `SparseCholesky` (existing) |
| `SPARSE_SCHUR` | CPU Sparse | SuiteSparse | `SparseSchurComplement` (existing) |
| `DENSE_SCHUR` | CPU Dense | Eigen/LAPACK | Not planned |
| `ITERATIVE_SCHUR` | CPU Iterative | CG + preconditioner | `IterativeSchurSolver` (existing) |
| `CGNR` | CPU Iterative | CG on normal equations | Not planned |
| CUDA backends | GPU | cuSOLVER/cuSPARSE | Phase 2-3 |

### Ceres Missing from Apex (after Phase 1-3)

- `DENSE_SCHUR`: Dense Schur complement for small BA problems
- `CGNR`: Conjugate gradient on normal equations
- Mixed precision GPU

### Apex Unique Features

- Iterative Schur with block diagonal preconditioner (matches Ceres `ITERATIVE_SCHUR`)
- Explicit + implicit Schur variants
- GPU sparse solvers (Ceres has limited GPU support)

## 7.2 GTSAM Linear Algebra Backends

GTSAM (Georgia Tech) focuses on:

| Backend | Status in GTSAM | Apex Status |
|---------|----------------|-------------|
| Dense Cholesky (Eigen) | Yes | Phase 1 |
| Sparse Cholesky (CHOLMOD) | Yes | Existing (faer) |
| Sparse QR (SuiteSparseQR) | Yes | Existing (faer) |
| Multifrontal QR | Yes (unique) | Not planned |
| Bayes tree | Yes (incremental) | Not planned |
| GPU | Limited research | Phase 2-3 |

### Key Difference

GTSAM's Bayes tree enables **incremental** optimization — updating solutions when new measurements arrive without re-solving from scratch. This is orthogonal to the linear solver backend and could be added to Apex independently.

## 7.3 g2o Linear Algebra Backends

g2o uses:

| Backend | Library | Apex Equivalent |
|---------|---------|-----------------|
| Sparse Cholesky | CHOLMOD/Eigen | `SparseCholesky` |
| Sparse PCG | Eigen | `IterativeSchurSolver` |
| Dense (Eigen) | Eigen | Phase 1 |
| Schur complement | Custom | `SparseSchurComplement` |

g2o has no GPU support. Apex would surpass g2o's backend coverage after Phase 2.

## 7.4 tiny-solver / factrs

- **tiny-solver**: Only dense Cholesky (via nalgebra). No sparse support.
- **factrs**: Sparse Cholesky only (via faer). No dense, no GPU.

Apex already has the most comprehensive backend coverage among Rust NLLS libraries. After Phase 1-3, it would be comparable to Ceres.

## 7.5 Competitive Positioning After All Phases

```
Backend Coverage Comparison (max = full coverage)

                    CPU Sparse  CPU Dense  GPU Sparse  GPU Dense  Schur  Incremental
Ceres Solver        ████████    ████████   ██████      ██████     █████  ░░░░░
GTSAM               ████████    ████████   ██          ░░░░░      █████  █████████
g2o                  ████████    ████        ░░░░░       ░░░░░      ████   ░░░░░
Apex (current)       ████████    ░░░░░      ░░░░░       ░░░░░      █████  ░░░░░
Apex (after Phase3)  ████████    ████████   ████████    ████████   █████  ░░░░░
tiny-solver          ░░░░░       ████       ░░░░░       ░░░░░      ░░░░░  ░░░░░
factrs               ████████    ░░░░░      ░░░░░       ░░░░░      ░░░░░  ░░░░░

████ = Supported    ░░░░ = Not Supported    ██ = Partial
```

After completing all three phases, Apex would have **the most comprehensive linear solver backend coverage of any Rust NLLS library**, and would match Ceres Solver's coverage (except for incremental/Bayes tree which is a fundamentally different approach).
