# Notation &amp; Conventions

This crate is the substrate under three other crates, and a convention error
here propagates silently into every trajectory built on it. This page is the
contract; every chapter uses it.

## 1. Residuals and Jacobians

<div class="apx-symbols apx-table">
<span class="apx-title">The three quantities every factor produces</span>

| Symbol | Meaning | Where |
|---|---|---|
| $\mathbf{r}$ | the residual of one factor, length `residual_dim()` | `Factor::linearize` |
| $J$ | $\partial\mathbf{r}/\partial\mathbf{x}$, shape `jacobian_shape()` | `Factor::linearize` |
| $\rho$ | a robust loss applied to $s=\|\mathbf{r}\|^2$ | `core::loss_functions` |

</div>

- **Residual convention**: $r = h(x) - z$ — predicted minus measured. A
  factor returns the residual, never the cost; $\tfrac12\|\mathbf{r}\|^2$ is the
  solver's business.
- **Whitening happens upstream of the corrector**: $\tilde r = S r$,
  $\tilde J = S J$, so the objective the optimizer minimizes is
  $\sum_i \tfrac12\rho(\|S_i r_i\|^2)$ — the $\Omega$-weighted objective
  g2o-style benchmarks report.
- **The Jacobian buffer is column-major** and optional: a factor may write
  only the residual, and the solver asks for the Jacobian only when it needs
  it (the `accumulate_normal` path avoids forming $J$ at all for some
  factor shapes).

## 2. Variables and manifolds

| Symbol | Meaning | Code |
|---|---|---|
| $X$ | a variable's value on its manifold | `ManifoldVariable` |
| $\xi$ | a tangent-space perturbation | `ManifoldType` |
| $\oplus$, $\ominus$ | manifold plus/minus | `LieGroup::plus` / `minus` |
| $d$ | degrees of freedom of a variable | `tangent_dim` |

- **Right perturbation**: $X' = X \circ \mathrm{Exp}(\xi)$. Every Jacobian in
  this crate is with respect to that $\xi$.
- **Two different sizes per manifold**: the *parameter* vector (what a factor
  receives in `params`) and the *tangent* vector (what the step and the
  Jacobian columns are indexed by). They differ wherever a quaternion is
  stored.

  | Manifold | Parameters | DoF | Parameter layout |
  |---|---:|---:|---|
  | `SE3` | 7 | 6 | `[tx, ty, tz, qw, qx, qy, qz]` |
  | `SE2` | 3 | 3 | `[tx, ty, θ]` |
  | `SE23` | 10 | 9 | `[tx, ty, tz, qw, qx, qy, qz, vx, vy, vz]` |
  | `SGal3` | 11 | 10 | the `SE23` layout plus `time` |
  | `SO3` | 4 | 3 | `[qw, qx, qy, qz]` |
  | `Sim3` | 8 | 7 | `[tx, ty, tz, qw, qx, qy, qz, scale]` |
  | `Rn(n)` | $n$ | $n$ | the plain vector |

- **A factor's Jacobian columns count DoF, not parameters.** An `SE3` block
  has **six** columns, not seven, even though the variable stores seven
  parameters. The registration path enforces this: `jacobian_shape().1` must
  equal the sum of the connected variables' `dof()`.
- **A variable is a key, not an index.** `VarKey` and `FactorKey` are
  generational handles (`slotmap`): a key to a removed variable returns
  `None` rather than someone else's data.

## 3. Noise and robustness

<div class="apx-table">
<span class="apx-title">The whitening contract</span>

| Convention | Who applies it | Registration |
|---|---|---|
| The factor returns the raw residual | most factors | give a `NoiseModel` |
| The factor whitens internally | GICP, smart factors, preintegration | **must** be `NoiseModel::null()` |

</div>

`Factor::whitens_internally()` exists so the mistake is caught at
registration: pairing an internally-whitened factor with a non-null noise
model whitens twice and silently over-weights the block.

- `NoiseModel::Null` is the zero-cost identity: no allocation, no arithmetic.
- `NoiseModel::from_sigmas(σ)` builds a diagonal model;
  `from_information(Ω)` factors a general information matrix.
- **Huber's δ is 1.345 in whitened units** — the 95 %-efficient value.

## 4. Solvers

| Symbol | Meaning | Code |
|---|---|---|
| $H = J^\top J$ | the normal matrix | assembled by the linearizer |
| $\mathbf{g} = J^\top \mathbf{r}$ | the gradient | assembled by the linearizer |
| $\lambda$ | LM damping | `Damping` |
| $D$ | Marquardt's column scaling | `diag(J^\top J)` |
| $S$ | the Schur complement's reduced Hessian | `linalg::schur` |
| $\Lambda$ | a prior's information matrix | `MarginalPriorFactor` |

- **Damping is a value, not a scalar**: `solve_augmented_equation` takes
  `&Damping`; `Damping::identity(λ)` reproduces the old uniform $\lambda I$.
- **The solve is over free columns only**: fixed coordinates are excluded from
  the step vector, so a step's norm is the norm of the step actually applied.

## 5. Units and geometry

- Angles are radians; distances metres; timestamps `f64` seconds.
- Bearings are unit vectors in the camera frame; the crate does not
  re-normalize them (the caller does).
- Camera poses are world-to-camera: $\mathbf{p}_C = R\mathbf{p}_W + \mathbf{t}$.
- Plane parameters are $(\mathbf{n}, d)$ with $\mathbf{n}$ a unit normal and
  $\mathbf{n}^\top\mathbf{x} + d = 0$; a fit may also carry the centroid
  $\mathbf{q}$ with $d = -\mathbf{n}^\top\mathbf{q}$.

## 6. Error and outcome policy

- `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!` and
  `#[allow(...)]` are denied crate-wide, tests included. Findings are fixed,
  never silenced.
- `unsafe` is forbidden.
- Structural impossibilities are typed errors (`CoreError`, `LinAlgError`,
  `MarginalizationError`, `OptimizerError`).
- Non-fatal conditions are **named outcomes**: `ConvergenceInfo`,
  `SolverResult`, `IterationStats`, `OptimizerSummary`. A caller can tell
  "converged" from "stopped at the iteration cap" from "failed to build a
  system".

## 7. Determinism

- The solver is deterministic for a fixed problem: the variable order, the
  block layout and the reduction order are fixed, not hash-dependent.
- `VarKey`/`FactorKey` are generational, so a key's identity is stable across
  a problem's lifetime and never aliases.
- RANSAC-style sampling lives in the *consumers* (apex-cv, apex-vio), which
  seed it explicitly; the solver itself draws no random numbers.

## 8. What "the crate guarantees" means

| Guarantee | Enforced by |
|---|---|
| a malformed factor is rejected at registration | `Factor::validate_variables` |
| a double-whitened factor is rejected at registration | `Factor::whitens_internally` + `try_add_residual_block_with_noise` |
| a step applies through the manifold's own plus | `apply_parameter_step` |
| a fixed variable never moves | the free-column step vector |
| a failed solve is typed | `OptimizeResult` / `SolverResult` |

<div class="apx apx-note">
<span class="apx-title">Note — the one thing the crate cannot check</span>

Correctness of a *residual* is the author's responsibility: the crate checks
shapes, block counts, whitening conventions and manifold plumbing, and it
cannot know whether your factor's sign is the sign your data has. That is
where the ecosystem's books come in: the VIO and CV books document what the
factors in *their* crates mean, and this book documents what the solver does
with them once they are written.

</div>
