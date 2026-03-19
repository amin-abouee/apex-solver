# 5. Effort Estimation & Implementation Plan

## 5.1 Detailed Effort Breakdown

### Phase 1: CPU Dense Solvers (3-4 days)

| Task | Files | New Lines | Effort | Risk |
|------|-------|-----------|--------|------|
| `sparse_to_dense()` / `dense_to_sparse()` utilities | `utils.rs` (new) | ~80 | 0.5 day | Low |
| `DenseCholeskySolver` implementation | `dense_cholesky.rs` (new) | ~200 | 1 day | Low |
| `DenseQRSolver` implementation | `dense_qr.rs` (new) | ~250 | 1.5 days | Low |
| `LinearSolverType` enum expansion | `mod.rs`, `optimizer/mod.rs`, `levenberg_marquardt.rs` | ~25 | 0.5 day | Low |
| Unit tests for both solvers | In each file | ~300 | 0.5 day | Low |
| **Phase 1 Total** | **3 new + 3 modified** | **~855** | **4 days** | **Low** |

### Phase 2: GPU Dense Solvers (6-8 days)

| Task | Files | New Lines | Effort | Risk |
|------|-------|-----------|--------|------|
| `cudarc` dependency + feature flag setup | `Cargo.toml` (2 files) | ~15 | 0.5 day | Low |
| `GpuContext` wrapper + device management | `gpu/mod.rs`, `gpu/context.rs` (new) | ~150 | 1 day | Medium |
| `GpuDenseCholeskySolver` | `gpu/dense_cholesky.rs` (new) | ~300 | 2 days | Medium |
| `GpuDenseQRSolver` | `gpu/dense_qr.rs` (new) | ~300 | 2 days | Medium |
| GPU error handling integration | `mod.rs` | ~30 | 0.5 day | Low |
| Factory + enum updates for GPU | `mod.rs`, `optimizer/mod.rs` | ~20 | 0.5 day | Low |
| GPU tests (conditional) | In each file | ~200 | 1 day | Medium |
| **Phase 2 Total** | **4 new + 4 modified** | **~1015** | **7.5 days** | **Medium** |

### Phase 3: GPU Sparse Solvers (8-12 days)

| Task | Files | New Lines | Effort | Risk |
|------|-------|-----------|--------|------|
| CSC ↔ CSR conversion utilities | `gpu/format.rs` (new) | ~150 | 1 day | Medium |
| cuSPARSE descriptor management | `gpu/mod.rs` | ~100 | 1 day | High |
| `GpuSparseCholeskySolver` | `gpu/sparse_cholesky.rs` (new) | ~350 | 3 days | High |
| `GpuSparseQRSolver` | `gpu/sparse_qr.rs` (new) | ~350 | 3 days | High |
| Factory + enum updates | `mod.rs`, `optimizer/mod.rs` | ~15 | 0.5 day | Low |
| GPU sparse tests | In each file | ~250 | 1.5 days | High |
| **Phase 3 Total** | **3 new + 2 modified** | **~1215** | **10 days** | **High** |

### Phase 4: Future Optimizations (Optional)

| Task | Effort | Risk |
|------|--------|------|
| `compute_residual_and_jacobian_dense()` in Problem | 2-3 days | Medium |
| All-GPU pipeline (J formation on GPU via cuBLAS) | 3-4 days | High |
| Mixed precision (f32 mode for consumer GPUs) | 3-5 days | High |
| GPU Schur complement | 5-8 days | Very High |
| Batch GPU operations for multiple problems | 3-5 days | High |

## 5.2 Grand Total

| Phase | Duration | Lines of Code | Risk Level |
|-------|----------|---------------|------------|
| Phase 1: CPU Dense | 4 days | ~855 | Low |
| Phase 2: GPU Dense | 7.5 days | ~1015 | Medium |
| Phase 3: GPU Sparse | 10 days | ~1215 | High |
| **Total (Phases 1-3)** | **~21.5 days** | **~3085** | **Medium-High** |

## 5.3 Dependency Graph

```
Phase 1: CPU Dense
├── utils.rs (conversion helpers)
├── dense_cholesky.rs (depends on utils.rs)
├── dense_qr.rs (depends on utils.rs)
└── LinearSolverType updates (depends on implementations)

Phase 2: GPU Dense (depends on Phase 1 for patterns)
├── Cargo.toml feature flag
├── gpu/context.rs (CUDA context)
├── gpu/dense_cholesky.rs (depends on context + utils.rs)
├── gpu/dense_qr.rs (depends on context + utils.rs)
└── GPU error variants

Phase 3: GPU Sparse (depends on Phase 2 for GPU context)
├── gpu/format.rs (CSC↔CSR conversion)
├── gpu/sparse_cholesky.rs (depends on context + format)
└── gpu/sparse_qr.rs (depends on context + format)
```

## 5.4 Files Impact Summary

### New Files (10 total)

| File | Phase | Lines |
|------|-------|-------|
| `src/linalg/utils.rs` | 1 | ~80 |
| `src/linalg/dense_cholesky.rs` | 1 | ~200 |
| `src/linalg/dense_qr.rs` | 1 | ~250 |
| `src/linalg/gpu/mod.rs` | 2 | ~50 |
| `src/linalg/gpu/context.rs` | 2 | ~150 |
| `src/linalg/gpu/dense_cholesky.rs` | 2 | ~300 |
| `src/linalg/gpu/dense_qr.rs` | 2 | ~300 |
| `src/linalg/gpu/format.rs` | 3 | ~150 |
| `src/linalg/gpu/sparse_cholesky.rs` | 3 | ~350 |
| `src/linalg/gpu/sparse_qr.rs` | 3 | ~350 |

### Modified Files (5 total)

| File | Phase | Changes |
|------|-------|---------|
| `Cargo.toml` (workspace) | 2 | Add cudarc workspace dep |
| `Cargo.toml` (root) | 2 | Add cuda feature, optional cudarc dep |
| `src/linalg/mod.rs` | 1-3 | Module declarations, enum variants, error variants, re-exports |
| `src/optimizer/mod.rs` | 1-3 | Update `create_linear_solver()` factory |
| `src/optimizer/levenberg_marquardt.rs` | 1-3 | Update inline solver match |

### Unchanged Files

- `src/core/problem.rs` — No modifications needed
- `src/optimizer/gauss_newton.rs` — Uses factory, auto-works
- `src/optimizer/dog_leg.rs` — Uses factory + get_hessian() returns sparse copy
- `src/linalg/cholesky.rs` — Unchanged
- `src/linalg/qr.rs` — Unchanged
- `src/linalg/explicit_schur.rs` — Unchanged
- `src/linalg/implicit_schur.rs` — Unchanged
- All manifold code — Unchanged
- All factor code — Unchanged
- All I/O code — Unchanged

## 5.5 Testing Strategy

### Unit Tests (per solver)

Each new solver gets tests mirroring `SparseCholeskySolver` tests:
- Basic solver creation
- Normal equation solve (well-conditioned)
- Augmented equation solve (various λ)
- Singular matrix handling
- Numerical accuracy with known solution
- Covariance computation
- Symbolic/numeric caching (sparse only)

### Integration Tests

- Full optimization pipeline with each solver on `sphere2500.g2o`
- Cross-solver result comparison (all solvers should converge to same solution)
- Performance regression tests

### GPU-Specific Tests

- Device availability check (skip gracefully)
- Memory transfer correctness
- Buffer reuse across iterations
- Large matrix stress tests

## 5.6 Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| DogLeg breaks with dense solver | High | High | Store sparse Hessian copy; test DogLeg with each solver |
| cudarc API changes in 0.17 | Medium | Medium | Pin `cudarc = "=0.16.x"`; wrap in internal abstraction layer |
| GPU tests flaky on CI | Medium | Low | Skip GPU tests without device; separate CI job for GPU |
| Dense solver used on large problem → OOM | Medium | Medium | Runtime warning when DOF > 1000; document size limits |
| cuSOLVER sparse reorder differs from faer | Low | Low | Allow configurable reorder strategy |
| f64 slow on consumer GPU | Certain | Low | Document in README; future f32 mode |
