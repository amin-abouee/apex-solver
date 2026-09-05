# Optimizers

This chapter documents what the three optimizers in `src/optimizer/` actually
do — the equations as implemented, the defaults as coded, and the deviations
from textbook forms with the reason the code gives for each. Every formula
cites the file and line it comes from.

## The problem all three solve

Given residual blocks $r_i$ over manifold variables, `Problem` defines

$$
\min_{x}\; f(x) \;=\; \tfrac12 \lVert r(x)\rVert_2^2 \;=\; \tfrac12\sum_i \lVert r_i(x)\rVert^2
$$

Variables live on manifolds, so the step is applied with $\boxplus$ rather
than addition: $x \leftarrow x \boxplus h$, with the per-group right-plus
from `apex-manifolds`. The Jacobian $J$ is therefore taken with respect to
the **tangent space**, and every matrix below has tangent dimension, not
parameter dimension. An `SE3` variable contributes 6 columns, not 7.

Two quantities recur:

$$
g \;=\; J^{\mathsf T} r \qquad\text{(gradient)},\qquad
H \;\approx\; J^{\mathsf T} J \qquad\text{(Gauss-Newton Hessian)}
$$

`H` drops the $\sum_i r_i \nabla^2 r_i$ term of the true Hessian, which is
the Gauss-Newton approximation and is accurate when residuals are small or
nearly linear (`gauss_newton.rs:25`).

> **Sign convention.** `LinearSolver::get_gradient` returns $+J^{\mathsf T}r$,
> **not** the right-hand side $-J^{\mathsf T}r$ (`linalg/mod.rs:311`). Each
> optimizer negates it before forming the linear system. `tests/linear_solver_contract.rs`
> pins this for every backend, because getting it backwards produces a step
> that ascends.

---

## Gauss-Newton

`optimizer/gauss_newton.rs`. Solves the normal equations exactly and takes the
full step, with no step-size control at all (`gauss_newton.rs:31`):

$$
(J^{\mathsf T} J)\, h \;=\; -J^{\mathsf T} r,
\qquad x \leftarrow x \boxplus h
$$

There is no damping, no trust region, and no line search. Convergence is
quadratic near a solution where the Gauss-Newton approximation holds, and the
method **diverges** when started far away or when $J^{\mathsf T}J$ is
ill-conditioned. Use it when you already know the problem is well-conditioned
and the initial guess is close; otherwise use LM.

### Options

| Field | Default | Meaning |
|---|---|---|
| `linear_solver_type` | `SparseCholesky` | Which backend solves the normal equations |
| `min_diagonal` | 1e-10 | Conditioning floor |
| `max_iterations` | 50 | Iteration cap (Ceres's default) |
| `cost_tolerance` | 1e-6 | Relative cost-change stop |
| `parameter_tolerance` | 1e-8 | Relative step-size stop |
| `gradient_tolerance` | 1e-10 | First-order optimality stop |
| `use_jacobi_scaling` | `false` | Column equilibration |
| `max_condition_number` | `None` | Optional conditioning guard |
| `min_cost_threshold` | `None` | Early exit once $f < \tau$ |
| `compute_covariances` | `false` | Populate `SolverResult::covariances` |
| `timeout` | `None` | Wall-clock cap |

The `schur_*` fields (`schur_variant`, `schur_preconditioner`,
`schur_cg_max_iterations`, `schur_cg_tolerance`, `schur_cg_q_tolerance`) exist
on all three configs and are forwarded to the linear solver — see
[Linear Solvers](./solvers.md).

---

## Levenberg–Marquardt

`optimizer/levenberg_marquardt.rs`. The default optimizer, and the one
`for_bundle_adjustment()` presets.

### The damped system

$$
\bigl(J^{\mathsf T} J + \lambda D\bigr)\, h \;=\; -J^{\mathsf T} r
$$

**`D` is not the identity.** It is the clamped diagonal of the Hessian
(`linalg/mod.rs:188`):

$$
D_{jj} \;=\; \operatorname{clamp}\!\bigl((J^{\mathsf T}J)_{jj},\; d_{\min},\; d_{\max}\bigr),
\qquad d_{\min} = 10^{-6},\; d_{\max} = 10^{32}
$$

This is Ceres's `LevenbergMarquardtStrategy`: damping each column in proportion
to its own curvature makes the step invariant to a rescaling of the parameters,
which uniform $\lambda I$ is not. `Damping::identity` sets
$d_{\min} = d_{\max} = 1$ and recovers plain $\lambda I$ exactly.

The lower clamp matters: without it a column with no curvature would receive no
damping at all, leaving that direction unconstrained.

### Step quality

After solving, the step is scored by the gain ratio. Predicted reduction comes
from the quadratic model (`optimizer/mod.rs:839`):

$$
\Delta_{\text{pred}} \;=\; -h^{\mathsf T} g \;-\; \tfrac12 h^{\mathsf T} H h
$$

evaluated with the **un-damped** $H$ — `hessian_vec_product` is documented
to never carry $\lambda D$, because damping in the model would corrupt
$\rho$ (`linalg/mod.rs:293`). Then (`optimizer/mod.rs:872`):

$$
\rho \;=\; \frac{f(x) - f(x \boxplus h)}{\Delta_{\text{pred}}}
$$

with two guards the textbook ratio does not have:

- $\Delta_{\text{pred}} < -10^{-15}$ → $\rho = -1$. A negative predicted
  reduction means the model says the step *increases* cost, so it is rejected
  outright rather than producing a misleading positive ratio.
- $\lvert\Delta_{\text{pred}}\rvert \le 10^{-15}$ → $\rho = 1$ if the
  actual reduction was positive, else $0$.

The step is **accepted when $\rho > $`min_relative_decrease`**, default
**1e-3** (`levenberg_marquardt.rs:447`), matching Ceres. This is not the
textbook $\rho > 0$.

### Damping update — two rules

Selected by `damping_update` (`levenberg_marquardt.rs:170`).

**`Nielsen` (default, what Ceres uses).** On an accepted step:

$$
\lambda \leftarrow \lambda \cdot \max\!\left(\tfrac13,\; 1 - (2\rho - 1)^3\right),
\qquad \nu \leftarrow \nu_0
$$

On a rejected step $\lambda \leftarrow \lambda\nu$ and $\nu \leftarrow 2\nu$,
so consecutive failures escalate geometrically. Reads `damping_nu`; ignores
`damping_increase_factor`, `damping_decrease_factor`, `min_step_quality`,
`good_step_quality`.

**`Marquardt`.** Three bands:

$$
\lambda \leftarrow
\begin{cases}
\lambda \cdot \text{decrease} & \rho \ge \text{good\_step\_quality} \\\\
\lambda \cdot \text{increase} & \rho \le \text{min\_step\_quality}\ \text{or rejected} \\\\
\lambda & \text{otherwise}
\end{cases}
$$

A rejected step **always** increases $\lambda$ whatever $\rho$ was — the
code notes that leaving it unchanged would recompute the identical step next
iteration and stall (`levenberg_marquardt.rs:989`). Afterwards $\lambda$ is
clamped to `[damping_min, damping_max]`.

### Options

| Field | Default | Notes |
|---|---|---|
| `damping` (λ₀) | **1e-4** | Ceres's `initial_trust_region_radius = 1e4` corresponds to λ = 1/radius |
| `damping_min` / `damping_max` | 1e-12 / 1e12 | Clamp after every update |
| `damping_increase_factor` | 10.0 | `Marquardt` only |
| `damping_decrease_factor` | 0.3 | `Marquardt` only |
| `damping_nu` (ν₀) | 2.0 | `Nielsen` only |
| `min_diagonal` / `max_diagonal` | 1e-6 / 1e32 | The clamp on `D` |
| `min_relative_decrease` | 1e-3 | Acceptance threshold on ρ |
| `good_step_quality` | 0.75 | `Marquardt` only |
| `min_step_quality` | 0.0 | `Marquardt` only |
| `max_consecutive_rejected_steps` | 5 | Then `StalledNoProgress` |
| `max_iterations` | 50 | Ceres's default |
| `cost_tolerance` | 1e-6 | |
| `parameter_tolerance` | 1e-8 | |
| `gradient_tolerance` | 1e-10 | |
| `use_jacobi_scaling` | **`false`** | Off because it is incompatible with Schur block structure (`levenberg_marquardt.rs:450`) |

Note the scaling default differs between optimizers: **LM defaults to `false`,
Dog Leg to `true`**. Enable it for LM on mixed-scale problems solved with
Cholesky or QR.

### `for_bundle_adjustment()` preset

`levenberg_marquardt.rs:706` overrides: `ImplicitSparseSchur` linear solver,
`SchurJacobi` preconditioner, λ₀ = 1e-3 (the plain default is 1e-4),
`max_iterations` = 20,
`cost_tolerance` = 1e-6, `parameter_tolerance` = 1e-8,
`gradient_tolerance` = 1e-10. The solver choice is justified by the measured
table in that comment; see [Benchmarks](./benchmarks.md).

---

## Dog Leg

`optimizer/dog_leg.rs`. Powell's dog leg over a spherical trust region of
radius $\Delta$, interpolating between the steepest-descent and
Gauss-Newton directions.

### Cauchy point

The unconstrained minimiser along $-g$ (`dog_leg.rs:926`):

$$
\alpha \;=\; \frac{g^{\mathsf T} g}{g^{\mathsf T} H g},
\qquad p_c \;=\; -\alpha\, g
$$

with a guard: if $\lvert g^{\mathsf T} H g\rvert \le 10^{-15}$ then
$\alpha = 1$.

### Step selection — three cases

From `dog_leg.rs:968`, given $\Delta$:

**Case 1 — GN step fits.** If $\lVert h_{gn}\rVert \le \Delta$, take
$h = h_{gn}$ (`StepType::GaussNewton`).

**Case 2 — Cauchy point already outside.** If $\lVert p_c\rVert \ge \Delta$,
take the steepest-descent direction scaled to the boundary
(`StepType::SteepestDescent`):

$$
h \;=\; \frac{\Delta}{\lVert d_{sd}\rVert}\, d_{sd}
$$

**Case 3 — interpolate.** Otherwise walk from $p_c$ toward $h_{gn}$ to
the boundary. With $v = h_{gn} - p_c$, solve $\lVert p_c + \beta v\rVert^2 = \Delta^2$,
i.e. $a\beta^2 + 2b\beta + c = 0$ with

$$
a = v^{\mathsf T} v, \qquad b = p_c^{\mathsf T} v, \qquad c = \lVert p_c\rVert^2 - \Delta^2
$$

**The implementation does not use the textbook quadratic root.** It uses
Ceres's cancellation-avoiding branch (`dog_leg.rs:1030`), with
$d^2 = b^2 - ac$:

$$
\beta \;=\;
\begin{cases}
\dfrac{-b + d}{a} & b \le 0 \\\\[2ex]
\dfrac{-c}{\,b + d\,} & b > 0
\end{cases}
$$

Choosing by the sign of $b$ avoids catastrophic cancellation in the
subtraction. Fallbacks: $\beta = 1$ when $d^2 < 0$ (geometrically
impossible, handled anyway) or $\lvert a\rvert < 10^{-15}$ (degenerate
$v$). Finally $\beta$ is clamped to $[0,1]$ and
$h = p_c + \beta v$ (`StepType::DogLeg`).

### Trust-region update

`dog_leg.rs:1059`. **Growth is Ceres's rule, not a fixed doubling:**

$$
\Delta \leftarrow \min\bigl(\Delta_{\max},\; \max(\Delta,\; 3\lVert h\rVert)\bigr)
\qquad \text{when } \rho > \text{good\_step\_quality}
$$

$$
\Delta \leftarrow \max\bigl(\Delta_{\min},\; \tfrac12 \Delta\bigr)
\qquad \text{when } \rho < \text{poor\_step\_quality}
$$

and $\Delta$ is left unchanged in between.

> **$\lVert h\rVert$ is measured in the *scaled* space** when Jacobi scaling
> is on — which is Dog Leg's default. The code documents why: the radius bounds
> the scaled step, so feeding the un-scaled norm would compare two different
> units and grow $\Delta$ by an arbitrary factor (`dog_leg.rs:1053`).

### Step reuse

On a rejected step Dog Leg caches the GN step, Cauchy point, gradient and
$\alpha$, and reuses them next iteration (`dog_leg.rs:1087`). Only
$\Delta$ changed, so the expensive linear solve does not have to be redone.
The cache is invalidated on any accepted step, because the parameters moved.
Controlled by `enable_step_reuse` (default `true`).

### Options

| Field | Default | Notes |
|---|---|---|
| `trust_region_radius` (Δ₀) | 1e4 | Ceres-style large initial radius |
| `trust_region_min` / `max` | 1e-12 / 1e12 | Δ floor triggers `TrustRegionRadiusTooSmall` |
| `trust_region_increase_factor` | 3.0 | The 3 in `max(Δ, 3‖h‖)` |
| `trust_region_decrease_factor` | 0.5 | |
| `good_step_quality` | 0.75 | Grow above this |
| `poor_step_quality` | 0.25 | Shrink below this |
| `min_step_quality` | 0.0 | Acceptance threshold |
| `use_jacobi_scaling` | **`true`** | Elliptical trust region |
| `initial_mu` / `min_mu` / `max_mu` | 1e-4 / 1e-8 / 1.0 | Adaptive regularisation for a singular $H$ |
| `mu_increase_factor` | 10.0 | μ shrinks as `μ / (0.5·factor)` on success |
| `enable_step_reuse` | `true` | See above |

---

## Jacobi column scaling

When `use_jacobi_scaling` is on, columns are equilibrated before the solve
(`optimizer/mod.rs:484`):

$$
S_{jj} \;=\; \frac{1}{1 + \lVert J_{:,j}\rVert_2},
\qquad \tilde J = J S
$$

The solve happens in the scaled space and the step is mapped back with
$h = S\,\tilde h$. The $1+$ in the denominator keeps the scaling finite
for a structurally empty column.

Ordering matters and the code is explicit about it: the **predicted reduction
is computed before un-scaling** (`levenberg_marquardt.rs:1040`), because the
solver's cached gradient and Hessian are the scaled ones and all three vectors
must live in the same space. The predicted reduction is a value of the
quadratic model and is invariant under the change of variables, so it is
equally valid for the un-scaled step.

---

## Convergence criteria

All three optimizers share `check_convergence` (`optimizer/mod.rs:627`),
evaluated in this order.

**Always checked, before anything else:**

1. **`InvalidNumericalValues`** — any of new cost, step norm or gradient norm
   is NaN or infinite.
2. **`Timeout`** — elapsed ≥ `timeout`.
3. **`MaxIterationsReached`** — iteration ≥ `max_iterations`.

**Only after an accepted step** (a rejected step cannot terminate the solve):

4. **`GradientToleranceReached`** — $\lVert g\rVert < \varepsilon_g$.
5. From iteration 1 onward, **`ParameterToleranceReached`**:
   $\lVert h\rVert \le \varepsilon_x(\lVert x\rVert + \varepsilon_x)$ — a
   *relative* test, so it scales with the parameter magnitude.
6. From iteration 1 onward, **`CostToleranceReached`**:
   $\lvert f_k - f_{k+1}\rvert / \max(f_k, 10^{-10}) < \varepsilon_f$.
7. **`MinCostThresholdReached`** — optional $f < \tau$ cutoff.
8. **`TrustRegionRadiusTooSmall`** — LM and Dog Leg only.

LM additionally reports **`StalledNoProgress`** after
`max_consecutive_rejected_steps` consecutive rejections
(`levenberg_marquardt.rs:1304`): damping has grown until every trial step is
negligible, so the remaining budget cannot change the cost.

## Output

All three return `SolverResult<T>` (`optimizer/mod.rs:244`):

| Field | Meaning |
|---|---|
| `parameters` | Final values |
| `status` | The `OptimizationStatus` above |
| `initial_cost` / `final_cost` | $f$ at start and end |
| `iterations` | Outer iterations performed |
| `elapsed_time` | Wall clock for the solve |
| `convergence_info` | Per-iteration history, when recorded |
| `covariances` | Per-variable tangent-space covariance; `None` unless `compute_covariances` |

Note that several statuses are *successful* terminations, not failures:
`Converged`, `CostToleranceReached`, `GradientToleranceReached`,
`ParameterToleranceReached` and `StalledNoProgress` are all treated as
converged by the benchmark harness. `MaxIterationsReached` means the budget ran
out with the solve still improving.
