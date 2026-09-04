# Introduction

**Apex Solver** is a high-performance Rust library for nonlinear least squares
optimization on Lie groups, aimed at SLAM, bundle adjustment and computer
vision.

It bridges the gap between theoretical robotics and practical implementation:

- **Manifold-aware optimization** — full Lie group support (SO(2), SO(3), SE(2),
  SE(3), SE₂(3), SGal(3), Sim(3), ℝⁿ) with analytic Jacobians via the
  [apex-manifolds](https://github.com/amin-abouee/apex-solver/tree/main/crates/apex-manifolds)
  crate.
- **Three optimization algorithms** — Levenberg–Marquardt (with Ceres-compatible
  Nielsen and Marquardt damping policies), Gauss–Newton and Dog Leg behind a
  unified [`Optimizer`](https://docs.rs/apex-solver) trait.
- **Four sparse linear solvers** — sparse Cholesky and QR, plus explicit and
  iterative (matrix-free PCG) Schur complement solvers for large-scale bundle
  adjustment.
- **~40 factors** — pose, visual, IMU preintegration, LiDAR, GNSS, range and
  bearing, motion models and marginalization, each documented with its error
  and Jacobian in the [Factor Reference](./factors/index.md).
- **15 robust loss functions** — Huber, Cauchy, Tukey, Welsch, Barron and more,
  with correct Triggs correction of residuals *and* Jacobians.
- **Zero-copy sparse linear algebra** — parameters live in contiguous `nalgebra`
  storage viewed by `faer` without copying; `unsafe` is forbidden crate-wide.
- **Ecosystem I/O** — G2O, TORO and BAL file formats via the `apex-io` crate.
- **Live visualization** — integrated [Rerun](https://rerun.io) support for
  watching poses, landmarks, cost curves and sparsity patterns during a solve.

## Where to go next

- [Installation & Quick Start](./quick_start.md) — solve your first pose graph.
- [Problem Construction](./problem.md) — variables, factors, keys and gauge freedom.
- [Factor Reference](./factors/index.md) — every factor's error and Jacobian.
- [Optimizers](./optimizers.md) — choosing and tuning LM, GN and Dog Leg.
- [Linear Solvers](./solvers.md) — Cholesky, QR and the Schur family.
