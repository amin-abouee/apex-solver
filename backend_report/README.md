# Multi-Backend Linear Solver Architecture Report

**Project**: Apex Solver
**Date**: 2026-02-28
**Scope**: Dense CPU, Sparse/Dense GPU linear solver backends

---

## Report Structure

| Document | Contents |
|----------|----------|
| [00 — Executive Summary](00_executive_summary.md) | High-level overview, key decisions, effort totals |
| [01 — Current Architecture](01_current_architecture.md) | Existing trait hierarchy, solver implementations, optimizer integration |
| [02 — Proposed Architecture](02_proposed_architecture.md) | New trait design, module structure, feature flags, factory updates |
| [03 — CPU Dense Solvers](03_cpu_dense_solvers.md) | DenseCholeskySolver, DenseQRSolver, conversion utilities, faer API |
| [04 — GPU Solvers](04_gpu_solvers.md) | cudarc integration, GPU Dense/Sparse Cholesky & QR, memory management |
| [05 — Effort Estimation](05_effort_estimation.md) | Per-task breakdown, phase dependencies, risk mitigation |
| [06 — API Changes](06_api_changes.md) | Public API diff, backward compatibility, migration guide |
| [07 — Comparison with Ceres](07_comparison_with_ceres.md) | Feature comparison with Ceres, GTSAM, g2o, and Rust alternatives |

## Quick Reference

### Complete Solver Matrix (After All Phases)

|  | Cholesky | QR | Schur |
|--|----------|----|-------|
| **CPU Sparse** | `SparseCholesky` | `SparseQR` | `SparseSchurComplement` |
| **CPU Dense** | `DenseCholesky` | `DenseQR` | — |
| **GPU Sparse** | `GpuSparseCholesky` | `GpuSparseQR` | Future |
| **GPU Dense** | `GpuDenseCholesky` | `GpuDenseQR` | — |

### Phase Summary

| Phase | Scope | Effort | Risk |
|-------|-------|--------|------|
| 1 | CPU Dense (Cholesky + QR) | 4 days | Low |
| 2 | GPU Dense (Cholesky + QR) | 7.5 days | Medium |
| 3 | GPU Sparse (Cholesky + QR) | 10 days | High |
| **Total** | **8 new solver types** | **~21.5 days** | **Medium** |

### Key Design Decisions

1. **Unified trait**: All backends implement `SparseLinearSolver` (accepts sparse Jacobian, converts internally)
2. **Zero optimizer changes**: LM, GN, DogLeg code unchanged
3. **Feature-gated GPU**: `cargo build --features cuda` enables GPU backends
4. **DogLeg compatibility**: Dense solvers store sparse Hessian copy for `get_hessian()`
