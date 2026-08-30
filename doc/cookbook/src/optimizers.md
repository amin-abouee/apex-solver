# Optimizers

All three optimizers share the same iteration skeleton — assemble → conditioning
check → Jacobi scaling → step → acceptance — and differ only in step computation
and acceptance policy. Each reports a common
[`SolverResult`](https://docs.rs/apex-solver) with `status`, `iterations`,
`initial_cost`, `final_cost` and the optimized `parameters`.

## Levenberg–Marquardt

The default choice. Two λ-update policies are available via
`DampingUpdate`:

- **`Nielsen`** (default): λ follows Nielsen's rule —
  `λ ← λ · max(⅓, 1 − (2ρ − 1)³)` on acceptance with ν reset, `λ ← λ·ν`
  geometric escalation on rejection. Reads `damping_nu`.
- **`Marquardt`**: the classic Ceres rule — multiply by
  `damping_increase_factor` when `ρ ≤ min_step_quality`, by
  `damping_decrease_factor` when `ρ ≥ good_step_quality`.

Both policies damp with Ceres' diagonal `λ·D` rather than uniform `λI`:

$$
(\mathbf{J}^\top\mathbf{J} + \lambda\, \mathbf{D})\,\Delta\mathbf{x} = -\mathbf{J}^\top\mathbf{r},
\qquad
D_{jj} = \mathrm{clamp}\!\left((\mathbf{J}^\top\mathbf{J})_{jj},\ \texttt{min\_diagonal},\ \texttt{max\_diagonal}\right)
$$

`D` makes the damped step invariant to parameter rescaling — the decisive
feature when focal lengths (${\sim}10^2$), metric landmarks (${\sim}10^0$) and
radians share one parameter vector. Setting
`with_diagonal_bounds(1.0, 1.0)` yields `D = I` and reproduces the uniform
`λI` behaviour.

Other knobs (all in `LevenbergMarquardtConfig`, defaults match Ceres):

| knob | meaning |
|---|---|
| `with_damping(λ₀)` | initial damping, `1e-4` (Ceres' `1/radius`) |
| `with_min_relative_decrease(r)` | reject steps with `ρ < r` (default `1e-3`) |
| `with_max_condition_number(c)` | terminate with `IllConditionedJacobian` when the cond lower bound exceeds `c` |
| `with_jacobi_scaling(bool)` | Jacobi column preconditioning, off by default |
| `with_compute_covariances(bool)` | attach marginal covariances to the result |

## Gauss–Newton

No damping loop — solve `JᵀJ Δx = −Jᵀr` directly. Fast near the solution,
diverges on hard residuals. Supports `SparseCholesky`/`SparseQR` in sparse mode
and `DenseCholesky`/`DenseQR` in dense mode; unsupported solver/mode
combinations are rejected with an error rather than silently substituted.

## Dog Leg

Trust-region method blending the steepest-descent (Cauchy) and Gauss–Newton
steps. The trust radius grows/shrinks from step quality and is bounded by the
configured radius.

## Convergence and status

Every optimizer returns an `OptimizationStatus`. Successful terminations
include `CostToleranceReached`, `GradientToleranceReached`,
`ParameterToleranceReached`, `Converged` and `StalledNoProgress`; failures
include `MaxIterationsReached`, `DampingFailure`, `TrustRegionFailure`,
`NumericalInstability` and `IllConditionedJacobian`.
