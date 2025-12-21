# Plan: High-Performance Linear Solver Optimization

## Phase 1: Profiling and Baseline Establishment
- [x] Task: Establish baseline performance metrics using existing datasets (`sphere2500`, `torus3D`). 4309070
- [ ] Task: Generate flamegraphs for the `LinearSolver::solve` path to identify current bottlenecks.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Profiling and Baseline Establishment' (Protocol in workflow.md)

## Phase 2: Solver Pattern Optimization
- [ ] Task: Optimize symbolic factorization caching in `src/linalg/cholesky.rs`.
- [ ] Task: Implement persistent buffer reuse for sparse matrix assembly.
- [ ] Task: Verify changes with unit tests and micro-benchmarks.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Solver Pattern Optimization' (Protocol in workflow.md)

## Phase 3: faer Deep Integration
- [ ] Task: Audit and refine `faer` sparse solver configuration (e.g., parallelism settings, ordering algorithms).
- [ ] Task: Optimize the conversion from problem structure to sparse Hessian format.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: faer Deep Integration' (Protocol in workflow.md)

## Phase 4: Final Verification
- [ ] Task: Run full benchmark suite and compare against baselines.
- [ ] Task: Verify 80% code test coverage for new/modified solver logic.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Verification' (Protocol in workflow.md)
