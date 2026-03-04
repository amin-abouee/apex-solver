# Executive Summary: Multi-Backend Linear Solver Architecture

**Date**: 2026-02-28
**Scope**: Adding Dense CPU, Sparse GPU, and Dense GPU linear solver backends to Apex Solver

---

## Current State

Apex Solver currently supports **3 sparse CPU solvers**:

| Solver | Decomposition | Use Case |
|--------|--------------|----------|
| `SparseCholeskySolver` | Cholesky (LLT) | Default, fast for SPD systems |
| `SparseQRSolver` | QR | Robust for rank-deficient systems |
| `SparseSchurComplementSolver` | Schur Complement | Bundle adjustment structure |

All solvers implement the `SparseLinearSolver` trait and operate on `faer::sparse::SparseColMat<usize, f64>` Jacobians.

## Proposed Additions

| Backend | Cholesky | QR | Schur | Effort |
|---------|----------|----|-------|--------|
| **CPU Dense** | New | New | N/A | **Small** (3-4 days) |
| **GPU Dense** | New | New | N/A | **Medium** (6-8 days) |
| **GPU Sparse** | New | New | Future | **Large** (8-12 days) |

**Total estimated effort**: 17-24 engineering days (excluding GPU Schur)

## Key Design Decision

Dense and GPU solvers will **implement the existing `SparseLinearSolver` trait** by accepting sparse Jacobians and converting internally. This enables **zero changes to optimizer code** (LM, GN, DogLeg) while maintaining full backward compatibility.

A new `LinearSolverBackend` trait will be introduced for internal polymorphism, with the `SparseLinearSolver` trait serving as the unified external interface.

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| DogLeg uses sparse Hessian directly | **High** | Store sparse Hessian copy in dense solvers |
| GPU not available on CI/dev machines | **Low** | Feature-gated behind `cuda` flag |
| Dense O(n³) for large problems | **Medium** | Runtime warnings + documentation |
| cudarc API instability | **Medium** | Pin version, wrap in internal abstractions |

## Recommended Phasing

1. **Phase 1** (3-4 days): CPU Dense Cholesky + QR
2. **Phase 2** (6-8 days): GPU Dense Cholesky + QR via cudarc/cuSOLVER
3. **Phase 3** (8-12 days): GPU Sparse Cholesky + QR via cudarc/cuSPARSE
4. **Phase 4** (future): GPU Schur, mixed precision, all-GPU pipeline
