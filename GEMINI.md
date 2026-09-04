# Gemini CLI: Engineering Standards & Workflows

This document defines the foundational mandates and technical standards for the Apex Solver project. These instructions take absolute precedence over general defaults.

## Core Principles (Karpathy-Inspired)

### 1. Think Before Coding (Research & Strategy)
- **Explicit Assumptions**: Before implementing a mathematical algorithm, explicitly state the assumed constraints (e.g., "assumes objective function is convex").
- **Surface Tradeoffs**: When choosing between algorithms or linear solvers (e.g., Cholesky vs QR), present the tradeoffs regarding convergence speed, memory overhead, and numerical precision.
- **Push Back**: If a requested feature adds significant complexity for marginal gain, propose a simpler alternative.

### 2. Simplicity First (Minimalist Rust)
- **Avoid Speculative Generics**: Do not use complex trait bounds or generics unless they are required for the current implementation.
- **Minimal Abstractions**: Prefer straightforward structs and functions over deep trait hierarchies. Follow the existing patterns in `core/` and `manifold/`.
- **No "Flexibility" Bloat**: Do not add configuration parameters or hooks that weren't explicitly requested.

### 3. Surgical Changes (Maintain Integrity)
- **Style Matching**: Follow existing patterns for error handling (Layer A/B/C hierarchy with `thiserror`) and documentation.
- **Minimal Diffs**: Every changed line must trace back to the user's request. Do not reformat or "improve" unrelated functions.
- **Cleanup**: Only remove dead code or imports that *your* changes created.

### 4. Goal-Driven Execution (Verification)
- **Test-First Bug Fixes**: Every bug report must start with a failing integration or unit test in `tests/` or the relevant module.
- **Numerical Validation**: For optimization routines, success criteria must include convergence within a specified epsilon (matching Ceres verification tests where applicable).
- **Benchmark-Driven**: Performance-critical changes (especially in `linalg/` or `manifold/`) must be verified using `cargo bench` or `examples/profile_*.rs` before finalization.

## Technical Standards

### Rust Tooling & Quality
- **Edition**: Project uses **Rust 2024**.
- **Lints**: All code must pass `cargo clippy --all-targets -- -D warnings`.
- **Formatting**: Use `cargo fmt` before every commit.
- **Error Handling**: 
    - Use the strict three-layer hierarchy: `ApexSolverError` (Top) → `OptimizerError` (Logic) → `CoreError/LinAlgError` (Math).
    - **Never** `unwrap()` in library code. Use `?` or `expect()` with a detailed message for truly impossible states.
- **Unsafe Code**: Minimize `unsafe`. If required for performance in inner loops, it must be documented with a `// SAFETY:` comment.

### Numerical & Optimization Specifics
- **Manifold Operations**: Follow the [manif C++ library](https://github.com/artivis/manif) conventions. Use analytic Jacobians where possible.
- **Linear Algebra**: Use `faer` (v0.22+) for sparse matrix operations and `nalgebra` (v0.33+) for dense/vector utilities.
- **Sparsity**: Respect the symbolic structure pre-computation pattern in `Problem`. Avoid recreating symbolic factorizations inside inner loops.
- **Constraints**: 
    - Use `fix_variable()` for **hard constraints** (exact zero update, gauge fixing).
    - Use `PriorFactor` for **soft constraints** (noisy measurements, regularization).

## Workflow

1. **Research**: 
   - Map mathematical requirements.
   - Identify existing factors or manifold operations that can be reused.
   - For bugs, create a reproduction script/test.
2. **Strategy**: 
   - Propose the implementation path (algorithm, data structures).
   - Outline the testing/verification plan (e.g., "compare against Ceres", "check convergence rate").
3. **Execution**: 
   - Write reproduction/validation tests first.
   - Implement the minimal logic to pass tests.
   - Run `cargo clippy` and `cargo test`.
   - If performance is impacted, run `samply` or `cargo bench` to quantify the change.
   - **Validation is mandatory.** A task is not complete until behavioral correctness is verified.

## Specialized Tools
- **Visualization**: Use `rerun` (v0.23.4) for 3D visualization in examples.
- **Profiling**: Use `samply` for recording and analyzing performance profiles.
- **Logging**: Use `tracing` for structured logging.
