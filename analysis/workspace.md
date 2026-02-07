# Workspace-Level Analysis

## Overview

The Apex Solver workspace contains 4 crates organized under `crates/`:

```
apex-manifolds (v0.1.0) ─── leaf, no internal deps
    |
    ├── apex-camera-models (v0.1.0) ─── depends on apex-manifolds
    ├── apex-io (v0.1.0) ─── depends on apex-manifolds
    └── apex-solver (v1.0.0) ─── depends on all three
```

This analysis covers cross-cutting concerns: dependency management, integration, workspace configuration, and structural issues.

---

## Dependency Management

### Good Practices

- **Workspace dependency inheritance** — All shared dependencies (nalgebra, thiserror, rayon, etc.) are declared at workspace level and inherited by crates. This ensures version consistency.
- **Dual path + version for internal crates** — `apex-manifolds = { path = "crates/apex-manifolds", version = "0.1.0" }` supports both local development and crates.io publishing.
- **No circular dependencies** — The dependency graph is strictly acyclic.
- **Workspace lints** — `unwrap_used = "deny"` and `expect_used = "deny"` enforced at workspace level.

### Issues Found

#### W1. Version Discrepancy

`apex-solver` is at version `1.0.0` while all other crates are at `0.1.0`. If this is intentional (main crate is stable, sub-crates are pre-1.0), it should be documented. If not, it suggests the version wasn't updated after the workspace split.

#### W2. Unused Dependencies in apex-io

`apex-io/Cargo.toml` lists `serde` and `serde_json` as dependencies, but neither is used in the crate's source code. These add compilation time for no benefit.

#### W3. Build Profile Configuration — Good

```toml
[profile.release]
opt-level = 3
lto = "thin"

[profile.profiling]
inherits = "release"
debug = true
```

The profiling profile correctly inherits release optimizations with debug symbols.

---

## Data Path Consistency

### Inconsistent Path References

Files in `crates/apex-solver/` reference test data using two different patterns:

| Context | Pattern | Example |
|---------|---------|---------|
| Tests | `../../data/odometry/` | Relative to crate dir (correct) |
| Benchmarks | `../../data/odometry/` | Relative to crate dir (correct) |
| Examples | `data/odometry/` | Assumes CWD is workspace root |
| Binaries | `data/odometry/` | Assumes CWD is workspace root |

**Problem:** Examples and binaries work when run via `cargo run --example` from workspace root, but break if the crate is used as an external dependency or run from a different working directory.

**Affected files:**
- `crates/apex-solver/bin/pose_graph_g2o.rs`
- `crates/apex-solver/examples/compare_optimizers.rs`
- `crates/apex-solver/examples/loss_function_comparison.rs`
- `crates/apex-solver/examples/visualize_graph_file.rs`
- `crates/apex-solver/examples/visualize_optimization.rs`

**Fix:** Use `CARGO_MANIFEST_DIR` for consistent resolution:
```rust
let manifest_dir = env!("CARGO_MANIFEST_DIR");
let data_path = format!("{}/../../data/odometry/{}.g2o", manifest_dir, dataset);
```

---

## Re-Export Strategy

### Well-Designed Backward Compatibility

`crates/apex-solver/src/lib.rs` provides three access patterns:

```rust
// 1. Direct re-export (recommended)
pub use apex_manifolds::{SE2, SE3, SO2, SO3, ...};

// 2. Module alias (backward compatible)
pub mod manifold { pub use apex_manifolds::*; }
pub mod camera_models { pub use apex_camera_models::*; }

// 3. Full crate access
pub use apex_camera_models;
pub use apex_io;
pub use apex_manifolds;
```

This allows existing code using `apex_solver::manifold::SE3` to continue working while new code can use `apex_solver::SE3` directly.

### Feature Gate Consistency

Visualization-related re-exports (`RerunObserver`, `VisualizationConfig`) are properly gated behind the `visualization` feature flag.

---

## Integration Tests

### Good Coverage

9 integration test files exist in `crates/apex-solver/tests/`:
- `integration_tests.rs` — Main optimization pipeline tests
- `camera_*_integration.rs` (8 files) — Per-camera-model integration tests
- `camera_test_utils.rs` — Shared test utilities

### Missing Integration Tests

- **Cross-crate workflow:** Load G2O -> optimize -> write G2O -> compare
- **Multi-format:** Load BAL -> create problem -> optimize with Schur
- **Error propagation:** Verify that errors from sub-crates propagate correctly through apex-solver

---

## Examples & Binaries Quality

### Strengths
- 6 examples covering loading, visualization, comparison, covariance
- 2 binaries for common workflows (pose graph, bundle adjustment)
- Feature gates properly applied for visualization examples

### Issues

#### W4. No Error Handling in Examples

Most examples use patterns like `.expect("message")` or `?` without meaningful error recovery. While acceptable for examples, they could demonstrate proper error handling patterns for users.

#### W5. No Example for Camera Model Usage

No example demonstrates camera model creation, projection, or linear estimation. Given that `apex-camera-models` is a standalone crate, a dedicated example would help users.

---

## Benchmarks

### Good Structure

Two benchmark files using Criterion:
- `odometry_pose_benchmark.rs` — Solver comparison across datasets
- `bundle_adjustment_benchmark.rs` — BAL dataset benchmarking

Both use correct relative paths and are properly configured with `harness = false`.

### Missing Benchmarks

- **Per-module microbenchmarks:** No benchmarks for sparse matrix operations, Cholesky factorization, or Jacobian computation in isolation
- **Manifold operations:** No benchmarks for SE3/SO3 operations (these are in the hot path)
- **Camera model benchmarks:** No benchmarks for projection/unprojection speed comparison across models

---

## Cross-Crate Redundancy

### Shared Patterns That Could Be Unified

#### Error Type Fragmentation

Each crate defines its own error type:
- `apex-manifolds`: `ManifoldError`
- `apex-io`: `IoError`
- `apex-camera-models`: `CameraModelError`
- `apex-solver`: `ApexSolverError`

These are independent enums with no shared traits or conversion infrastructure between them. `ApexSolverError` wraps some of these, but the wrapping is manual and incomplete.

**Consideration:** A shared `apex-core` crate with a common error trait could improve interoperability, but this may be over-engineering for the current crate count.

#### Nalgebra Version Consistency — Good

All crates use `nalgebra = "0.33"` via workspace inheritance. No version conflicts.

#### `thiserror` Used Consistently — Good

All crates use `thiserror 2.0` for error derivation, following the same patterns.

---

## Workspace Configuration Quality

### Strengths
- Resolver 2 used (required for edition 2021+)
- Shared metadata (authors, license, repository, edition, rust-version)
- Consistent lint configuration
- Build profiles well-configured

### Minor Issues

#### W6. `data_bk/` Directory at Root

A `data_bk/` directory exists at the workspace root (likely a backup). Not referenced in code but clutters the workspace. Should be removed or `.gitignore`d.

#### W7. `AGENT.md` Untracked

Git status shows `?? AGENT.md` at root. This should either be committed or added to `.gitignore`.

---

## Summary: Cross-Crate Issue Counts

| Crate | Critical | High | Medium | Low | Total |
|-------|----------|------|--------|-----|-------|
| apex-manifolds | 3 | 3 | 4 | 3 | 13 |
| apex-io | 3 | 4 | 4 | 4 | 15 |
| apex-camera-models | 3 | 4 | 5 | 4 | 16 |
| apex-solver | 4 | 5 | 6 | 5 | 20 |
| workspace | 0 | 2 | 3 | 2 | 7 |
| **Total** | **13** | **18** | **22** | **18** | **71** |

---

## Prioritized Workspace-Level Recommendations

### High
1. **Standardize data paths** using `CARGO_MANIFEST_DIR` across all examples and binaries
2. **Remove unused `serde`/`serde_json`** from apex-io Cargo.toml

### Medium
3. **Add cross-crate integration tests** (load -> optimize -> write -> compare)
4. **Add camera model usage example**
5. **Document version strategy** (1.0.0 vs 0.1.0)
6. **Add manifold and camera model microbenchmarks**

### Low
7. **Clean up `data_bk/` directory** — remove or gitignore
8. **Resolve `AGENT.md` status** — commit or gitignore
9. **Consider shared error trait** across crates (evaluate cost/benefit first)
