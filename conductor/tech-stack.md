# Technology Stack - Apex Solver

## Core Language & Frameworks
- **Rust (v2024):** Primary programming language, chosen for performance, memory safety, and zero-cost abstractions.

## Mathematical Foundation
- **faer (v0.22):** High-performance linear algebra library used for sparse and dense matrix operations, including Cholesky and QR decompositions.
- **nalgebra (v0.33):** Used for geometry primitives, fixed-size vectors, and basic manifold representations.

## Concurrency & Parallelism
- **rayon (v1.11):** Data-parallelism library used to parallelize residual evaluation and Jacobian computation across CPU cores.
- **num_cpus:** Used for dynamic thread-pool scaling.

## Visualization & Diagnostics
- **rerun (v0.26):** Integrated for real-time, interactive 3D visualization of optimization progress and system state.
- **tracing / tracing-subscriber:** Structured logging and diagnostic framework for production-grade visibility.

## Infrastructure & Tools
- **clap (v4.5):** Command-line argument parsing for optimization binaries.
- **serde / serde_json:** Serialization framework for configuration and data I/O.
- **criterion:** Micro-benchmarking framework for fine-grained performance tracking.
- **memmap2:** Efficient memory-mapped I/O for large-scale graph files.
