# Examples — a learning path

Twelve runnable programs that build on each other, from a single custom factor to a
sliding-window bundle adjustment on a real dataset. Read them in order: each one
introduces a few ideas and says which example picks up from there. Every file starts
with a doc comment explaining what it shows and what to look for in its output.

```bash
cargo run --release --example <name>
```

Examples 1–4, 6, 7, 8 and 11 simulate their data and need nothing. Examples 5, 9, 10
and 12 use real datasets (`M3500`, and BAL's `ladybug` 49-camera problem), which they
download into `data/` on first use, or find there if
`cargo run -p apex-io --bin download_datasets` already fetched them.

## Part 1 — Fundamentals

| # | Example | You learn |
|---|---|---|
| 1 | [`curve_fitting`](fundamentals/curve_fitting.rs) | The whole life cycle: write a `Factor` (residual + analytic Jacobian), add an `ℝⁿ` variable and residual blocks, run Levenberg–Marquardt, read the `SolverResult`. |
| 2 | [`pose_graph_2d`](fundamentals/pose_graph_2d.rs) | `SE2` variables on a manifold, `BetweenFactor` and `PriorFactor`, `NoiseModel::from_sigmas`, and **gauge freedom**: anchoring with a prior vs `fix_variable`. |
| 3 | [`robust_losses`](fundamentals/robust_losses.rs) | False loop closures against L2, Huber, Cauchy, DCS, Geman–McClure and Tukey. Per-block losses, why redescending losses need a good start, and a graduated (Cauchy → DCS) solve. |
| 4 | [`uncertainty`](fundamentals/uncertainty.rs) | Covariances with `with_compute_covariances`: uncertainty along an odometry chain, and what a loop closure does to it. |

## Part 2 — Choosing the solver

| # | Example | You learn |
|---|---|---|
| 5 | [`optimizers`](solver_choices/optimizers.rs) | Gauss–Newton, Levenberg–Marquardt (default, presets, damping rules) and Dog Leg behind the common `Optimizer` trait, on real g2o graphs; a custom `OptObserver` that records the cost per iteration. |
| 6 | [`linear_solvers`](solver_choices/linear_solvers.rs) | `JacobianMode` (dense/sparse assembly) × `LinearSolverType` (Cholesky/QR) on growing graphs: sparsity, not size, decides. |

## Part 3 — Bundle adjustment

| # | Example | You learn |
|---|---|---|
| 7 | [`ba_synthetic`](bundle_adjustment/ba_synthetic.rs) | `SE3` cameras and `ℝ³` landmarks, `ProjectionFactor<PinholeCamera, BundleAdjustment>`, the 7-DoF monocular gauge (and why the scale coordinate needs a long baseline), and sparse Cholesky vs explicit vs implicit **Schur complement** with `mark_for_elimination`. |
| 8 | [`ba_parameterizations`](bundle_adjustment/ba_parameterizations.rs) | One factor, compile-time choice of unknowns: localization (`OnlyPose`, a single multi-observation block), triangulation (`OnlyLandmarks`), self-calibration (`SelfCalibration`, one intrinsics variable shared by every block), and a critical motion that makes calibration fail. |
| 9 | [`bal_dataset`](bundle_adjustment/bal_dataset.rs) | A real BAL problem (Ladybug, 49 cameras): axis-angle poses, per-camera `[f, k1, k2]`, `BALPinholeCameraStrict`, the BA preset with the implicit Schur solver. |

## Part 4 — Incremental and sliding-window estimation

| # | Example | You learn |
|---|---|---|
| 10 | [`incremental_pose_graph`](incremental/incremental_pose_graph.rs) | Stream a dataset pose by pose into one living `Problem`: predict, add the edges that became available, warm-start the solve and **write it back** with `update_values_from`. |
| 11 | [`fixed_lag_smoother`](incremental/fixed_lag_smoother.rs) | A constant-size window: dropping the oldest pose vs `Marginalizer::apply` (an exact Gaussian prior) vs marginalization with **first-estimate Jacobians**, measured against the batch solution. |
| 12 | [`sliding_window_ba`](incremental/sliding_window_ba.rs) | Local bundle adjustment on the BAL dataset with cameras arriving one at a time: free window, fixed anchor cameras, finalized and re-observed points, bookkeeping with `FactorKey`s, and a score against the batch solve. |

## Tools — visualization (feature `visualization`)

| Example | Shows |
|---|---|
| [`visualize_optimization`](visualization/visualize_optimization.rs) | Live cost, damping, Hessian and pose updates in [Rerun](https://rerun.io) through `RerunObserver`. |
| [`visualize_graph_file`](visualization/visualize_graph_file.rs) | A g2o/TORO graph drawn in Rerun, without optimizing. |

```bash
cargo run --release --features visualization --example visualize_optimization
```

## Beyond the examples

- The binaries `pose_graph_g2o` and `bundle_adjustment` (`bin/`) are full command-line
  front ends with every solver option exposed.
- The [cookbook](../doc/cookbook/) explains the mathematics and the design behind each
  example: noise models, the first-estimate policy, marginalization, choosing a solver.
- The factor catalogue (`doc/factor-catalog.md`) lists every built-in factor: IMU,
  LiDAR, GNSS, ranging and more, all registered exactly like the ones used here.

## Conventions used throughout

- Every example returns `Result` and propagates errors with `?`; none uses `unwrap`.
- Output goes through `writeln!(std::io::stdout().lock(), ...)`, as the crate's lint
  policy requires.
- Synthetic data uses a fixed seed: a run prints the same numbers every time.
- An example that can check its own result does, and exits with an error if the
  result is wrong — so `cargo run --example` doubles as a smoke test.
