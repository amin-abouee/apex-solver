# Initial Concept
A high-performance Rust-based nonlinear least squares optimization library designed for computer vision applications including bundle adjustment, SLAM, and pose graph optimization. Built with focus on zero-cost abstractions, memory safety, and mathematical correctness.

# Product Guide - Apex Solver

## Vision & Focus
The primary focus for the current development phase is **Performance Optimization**, aiming to make Apex Solver the most efficient Rust-based solver for professional robotics and computer vision applications.

## Target Users & Use Cases
- **Real-time Robotics/SLAM:** Developers building systems for live navigation on embedded or resource-constrained hardware.
- **Large-scale Offline Mapping:** Researchers and engineers processing massive datasets for photogrammetry or global map optimization.
- **Benchmark-Driven Development:** Ensuring performance is competitive with or exceeds industry-standard C++ solvers like Ceres and GTSAM.

## Core Pillars of Performance
- **Linear Solver Efficiency:** Deep optimization of sparse Cholesky and QR decompositions, leveraging the full capabilities of the `faer` library.
- **Parallelization:** Maximizing multi-threaded performance for residual evaluation and Jacobian computation across all available CPU cores.
- **Memory Management:** Minimizing runtime allocations, implementing persistent memory pools, and improving cache locality for large-scale factor graphs.
- **Algorithmic Refinement:** Tuning Levenberg-Marquardt and other algorithms for faster, more robust convergence with minimal iterations.

## Success Metrics & Verification
- **Automated Benchmarking:** Expansion of the `criterion` suite to detect and prevent performance regressions.
- **Comparative Analysis:** Rigorous testing against Ceres, g2o, and GTSAM on standard research datasets.
- **Deep Profiling:** Systematic use of flamegraphs and memory profiling to identify and eliminate bottlenecks.
- **End-to-end Evaluation:** Tracking real-world performance (wall-clock time/memory) on full-scale optimization workflows.
