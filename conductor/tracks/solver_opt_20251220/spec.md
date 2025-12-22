# Specification: High-Performance Linear Solver Optimization

## Overview
This track focuses on optimizing the core sparse linear solver integration within `apex-solver`. The goal is to maximize the performance of `faer` during the linearization and solution phases of nonlinear least squares optimization.

## Goals
- Reduce wall-clock time for sparse Cholesky and QR solvers by 10-20% on large datasets.
- Minimize memory allocations during the solver loop.
- Improve the efficiency of symbolic factorization caching.

## Technical Details
- **Symbolic Factorization:** Ensure the symbolic pattern is computed only once for static-structure problems (common in pose graphs).
- **faer Integration:** Audit the use of `faer` sparse API to ensure optimal use of available SIMD and threading.
- **Buffer Reuse:** Implement persistent buffers for intermediate matrix computations to reduce GC/allocation pressure.

## Verification
- Run `cargo bench --bench solver_comparison` before and after changes.
- Profile using `samply` on the `torus3D` and `sphere2500` datasets.
