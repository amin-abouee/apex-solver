# The Examples

The repository's `examples/` directory is a **learning path**: twelve runnable programs in
four parts, each introducing a few ideas and pointing to the next. This chapter maps every
example to the chapter that explains the idea behind it. Run one with

```bash
cargo run --release --example <name>
```

The examples use only apex-solver: no SLAM system, no external front end. Synthetic data
uses a fixed seed, so a run prints the same numbers every time, and each example checks its
own result and exits non-zero if it is wrong. The index with the full descriptions is
`examples/README.md`.

## Part 1 — Fundamentals

| Example | You learn | Read |
|---|---|---|
| `curve_fitting` | A whole problem: a hand-written `Factor` (residual and analytic Jacobian), an `ℝⁿ` variable, residual blocks, Levenberg–Marquardt, the result | [Writing a Factor](../factors/writing.md), [The Problem](../core/problem.md) |
| `pose_graph_2d` | `SE2` variables, `BetweenFactor` and `PriorFactor`, noise models, and gauge freedom: anchoring with a prior against `fix_variable` | [Noise Models](../core/noise.md), [Pose and Prior Factors](../factors/pose.md) |
| `robust_losses` | False loop closures against L2, Huber, Cauchy, DCS, Geman–McClure and Tukey; why redescending losses need a good start and a graduated solve | [Robust Loss Functions](../core/losses.md) |
| `uncertainty` | Covariances with `with_compute_covariances`, along an odometry chain and after a loop closure | [Covariance and Uncertainty](../linalg/covariance.md) |

## Part 2 — Choosing the solver

| Example | You learn | Read |
|---|---|---|
| `optimizers` | Gauss–Newton, Levenberg–Marquardt (default, presets, damping rules) and Dog Leg behind one trait, on real g2o graphs; a custom `OptObserver` | [The Optimizer Contract](../optimizer/contract.md), [Choosing a Solver](../optimizer/choosing.md), [Observers](./observers.md) |
| `linear_solvers` | `JacobianMode` and `LinearSolverType` on growing graphs: sparsity, not size, decides | [Dense](../linalg/dense.md) and [Sparse Linear Solvers](../linalg/sparse.md) |

## Part 3 — Bundle adjustment

| Example | You learn | Read |
|---|---|---|
| `ba_synthetic` | `SE3` cameras and `ℝ³` landmarks, `ProjectionFactor`, the 7-DoF monocular gauge, and sparse Cholesky against explicit and implicit Schur | [The Schur Complement](../linalg/schur.md), [Visual Factors: Geometry](../factors/visual-geometry.md) |
| `ba_parameterizations` | One factor, a compile-time choice of unknowns: localization, triangulation and self-calibration; a critical motion that makes calibration fail | [Visual Factors: Geometry](../factors/visual-geometry.md) |
| `bal_dataset` | A real BAL problem: axis-angle poses, per-camera `[f, k1, k2]`, the bundle-adjustment preset | [Choosing a Solver](../optimizer/choosing.md) |

## Part 4 — Incremental and sliding-window estimation

| Example | You learn | Read |
|---|---|---|
| `incremental_pose_graph` | Stream a dataset into one living `Problem`: predict, add the new edges, warm-start the solve and write it back with `update_values_from` | [The Problem](../core/problem.md) |
| `fixed_lag_smoother` | A constant-size window: dropping the oldest pose against `Marginalizer::apply` against marginalization with first-estimate Jacobians, measured against the batch solution | [Marginalization](../marginal/marginalization.md), [The First-Estimate Policy](../core/first-estimate.md) |
| `sliding_window_ba` | Local bundle adjustment on a real dataset with cameras arriving one at a time: free window, fixed anchors, finalized and re-observed points | [Marginalization](../marginal/marginalization.md), [The Problem](../core/problem.md) |

There is no single "smoother" type in the library: the window examples show how the
pieces (`Marginalizer`, first-estimate Jacobians, variable and block removal on `Problem`,
`AggregateFactor`) fit together, and each keeps its own bookkeeping of which blocks touch
which pose.

## Visualization (feature `visualization`)

`visualize_optimization` streams cost, damping, the Hessian and pose updates to Rerun
through a `RerunObserver`; `visualize_graph_file` draws a g2o or TORO graph without
optimizing. See [Observers: Rerun](./observers.md).

```bash
cargo run --release --features visualization --example visualize_optimization
```
