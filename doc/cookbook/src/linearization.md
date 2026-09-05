# From Factor Graph to Linear System

This chapter follows one optimizer iteration end to end: how the factor graph is
turned into `J` and `r`, how each factor's contribution is whitened, robustified
and scattered into the global system, and how the solved step gets back onto the
manifold. It describes what the code in `core/` and `linearizer/` actually does.

## The graph

A `Problem` holds two arenas (`core::problem`):

- **Variables** — the unknowns, each a manifold element (`SE3`, `SO3`, `Rn`, …),
  stored in a `SlotMap` and addressed by a stable `VarKey`.
- **Residual blocks** — the factors, each a `ResidualBlock` holding one boxed
  `Factor`, the `VarKey`s it connects, an optional `NoiseModel`, and an optional
  robust `LossFunction`.

This *is* the factor graph: variables are nodes, residual blocks are the
(hyper-)edges connecting them. There is no separate graph structure.

```rust
let pose_i = problem.add_variable(ManifoldType::SE3, pose_i_params);
let pose_j = problem.add_variable(ManifoldType::SE3, pose_j_params);

problem.add_residual_block(
    &[pose_i, pose_j],           // the edge's endpoints
    Box::new(BetweenFactor::new(measurement)),
    Some(Box::new(HuberLoss::new(1.0)?)),
);
```

Each block is assigned a `residual_row_start_idx` — its offset in the stacked
residual vector — and each variable a column offset in the tangent space. Those
two offsets are what turn a graph into a matrix.

## What the objective actually is

Stacking every block's residual gives $r(x)$, and the problem is

$$
\min_x\; f(x) \;=\; \tfrac12 \sum_i \rho_i\!\left(\lVert \tilde r_i(x) \rVert^2\right)
$$

where $\tilde r_i$ is the *whitened* residual of block $i$ and
$\rho_i$ its robust loss (identity when none is set). The three sections
below are exactly the three transformations between "what the factor computes"
and "what the solver sees".

## Step 1 — the factor linearizes itself

Every factor implements one method (`factors::Factor`):

```rust
fn linearize(
    &self,
    params: &[&[f64]],                         // one slice per connected variable
    residual: &mut [f64],                      // output, length residual_dim()
    jacobian: Option<faer::mat::MatMut<'_, f64>>, // output, jacobian_shape()
);
```

Given the *current* parameter values, it writes

$$
r_i \;=\; h_i(x) \ominus z_i \qquad\text{and}\qquad
J_i \;=\; \left.\frac{\partial r_i}{\partial \delta}\right|_{\delta = 0}
$$

where $h_i$ is the predicted measurement, $z_i$ the observation, and
$\ominus$ whatever difference the measurement space requires — subtraction
for a Euclidean measurement, `between` + `log` for a pose measurement.

**The Jacobian is with respect to the tangent space at the current estimate,
not the parameter vector.** For an `SE3` variable stored as 7 parameters the
block has 6 columns, and it is the derivative of the residual with respect to a
local perturbation $\delta$ applied as $x \boxplus \delta$. This is what
makes the linear system well-posed on a manifold: the 7-parameter quaternion
representation is over-parameterised and would give a singular
$J^{\mathsf T}J$.

> Factors must obtain these derivatives **from the manifold**, not re-derive
> them. Every group in `apex-manifolds` reports the Jacobians of its own
> operations (`act`, `compose`, `log`, `right_plus`, `between`) through optional
> output arguments. A hand-written `[R | −R[p]ₓ]` is a second copy of a
> convention that can silently drift from the group's — `factors::common`
> deliberately contains no manifold derivatives for this reason.

The buffers are pre-allocated and reused across iterations; `linearize` never
allocates.

## Step 2 — whitening by the noise model

A measurement with covariance $\Sigma$ must be weighted by its information.
The `NoiseModel` stores the square-root information
$\Sigma^{-1/2}$ and applies it (`core::noise`):

$$
\tilde r_i \;=\; \Sigma_i^{-1/2} r_i, \qquad
\tilde J_i \;=\; \Sigma_i^{-1/2} J_i
$$

so that $\lVert \tilde r_i \rVert^2 = r_i^{\mathsf T}\Sigma_i^{-1} r_i$ —
the Mahalanobis distance. A `Diagonal` model is a per-row scale; a `Dense` model
is a triangular multiply.

**Whitening happens before the robust loss**, in both the block-evaluation path
and the assembly path, and the two are documented to keep the same ordering. The
loss must see the whitened norm, because "is this residual an outlier?" is a
question about standard deviations, not raw units.

> **A factor either whitens internally or takes a `NoiseModel` — never both.**
> Some factors' weighting is inseparable from the measurement (GICP's combined
> point covariance, a smart factor's internal elimination, an IMU
> preintegration's information matrix), so they whiten as they go and report
> `whitens_internally() == true`. Registering one of those with a non-null noise
> model would whiten twice and silently over-weight the block; the two
> conventions are indistinguishable by inspecting a residual, so registration
> rejects the combination outright.

## Step 3 — the robust loss becomes a reweighting

A robust loss $\rho$ turns the problem into a *different* least-squares
problem rather than changing the solver. `core::corrector` implements Ceres's
corrector: given $s = \lVert \tilde r\rVert^2$ and the loss derivatives
$\rho'(s), \rho''(s)$,

$$
\sqrt{\rho_1} = \sqrt{\rho'(s)}, \qquad \alpha^2 = \frac{\rho''(s)}{\rho'(s)}
$$

$$
\hat r = \sqrt{\rho_1}\,\tilde r, \qquad
\hat J = \sqrt{\rho_1}\,\tilde J + \frac{\alpha}{\lVert \tilde r\rVert}\,(\tilde J^{\mathsf T}\tilde r)\,\tilde r^{\mathsf T}
$$

so that $\lVert \hat r\rVert^2 \approx \rho(\lVert \tilde r\rVert^2)$ and the
gradient is correct. The rank-one term is what makes this more than naive
iteratively-reweighted least squares: it carries the curvature of $\rho$, so
the Gauss-Newton model of the robustified problem stays accurate.

**Ordering matters and is load-bearing**: the Jacobian correction reads the
*un-corrected* residual, so `correct_jacobian_in_place` runs before
`correct_residual_in_place`. Reversing them silently changes the rank-one term.

After this step the solver sees an ordinary least-squares problem in
$\hat r, \hat J$ — every optimizer and linear solver downstream is unaware
that a robust loss was involved at all.

## Step 4 — scattering into the global system

Each block has produced a small dense $\text{rows} \times \text{cols}$
Jacobian, where `cols` is the sum of its variables' DOFs. Assembly places those
numbers into the global $J$ (`linearizer::cpu`):

- block $i$'s rows go at its `residual_row_start_idx`,
- the sub-block for variable $v$ goes at $v$'s column offset.

The result is block-sparse: a row touches only the variables its factor
connects, which is precisely the graph's adjacency.

**The sparsity pattern is symbolic and built once.** `SymbolicStructure`
computes the pattern and a value permutation before the iteration loop, and
every subsequent iteration only refills values in that fixed layout — the
pattern is a property of the graph, not of the estimate. Recomputing it per
iteration is the single most common way to make a solver slow.

Assembly is parallelised over blocks with rayon into disjoint row ranges, so no
two threads write the same row. This is why the visiting order is load-bearing:
values are pushed positionally against the symbolic pattern's pairs, so any
disagreement between the two orders would permute $J$'s rows against
$r$ — a wrong system that still factorizes cleanly.

### Fixed variables (gauge freedom)

A pose-graph or BA problem is invariant under a global transformation, so
$J^{\mathsf T}J$ is singular unless something is pinned. `fix_variable`
holds individual tangent components fixed:

```rust
for dof in 0..6 {
    problem.fix_variable(first_pose, dof);   // anchor the gauge
}
```

Fixed components are excluded from the column space, so they never enter the
linear system and cannot move.

## Step 5 — solve, and return to the manifold

The optimizer now has $\hat J$ and $\hat r$, and forms

$$
g = \hat J^{\mathsf T}\hat r, \qquad H \approx \hat J^{\mathsf T}\hat J
$$

Which linear system gets solved is the optimizer's business
([Optimizers](./optimizers.md)) and how it is solved is the backend's
([Linear Solvers](./solvers.md)). Both produce a tangent-space step $h$.

The step is **not added** to the parameters. It is applied through the
manifold's right-plus:

$$
x \;\leftarrow\; x \boxplus h \;=\; x \cdot \exp(h)
$$

for each variable's own group, using the same right convention the Jacobians
were taken with. Mixing conventions here — taking a left-Jacobian and applying a
right update — produces a step that is wrong by a factor of the adjoint, and it
still converges, just slowly and to a slightly different place.

## The loop

```text
build symbolic structure          once, from the graph's adjacency
─── per iteration ────────────────────────────────────────────────
  for each residual block (parallel):
      factor.linearize(params) ──────────► r_i, J_i     tangent-space
      noise.whiten(...)        ──────────► r̃_i, J̃_i     Mahalanobis
      corrector.correct(...)   ──────────► r̂_i, Ĵ_i     robustified
      scatter into J, r at (row_start, col_offsets)
  ─────────────────────────────────────────────────────────────────
  solve for h   (Gauss-Newton / LM / Dog Leg × linear backend)
  x ← x ⊞ h     per-variable right-plus
  check convergence
```

Every iteration repeats steps 1–5 at the *new* estimate: the residuals, the
Jacobians and the robust weights are all recomputed, while the sparsity pattern
and every cached symbolic structure are reused.

## Why this is efficient

- **Symbolic once, numeric many.** The pattern of $J$, of
  $J^{\mathsf T}J$, and the permutation linking them are computed once and
  reused, so an iteration is a parallel value gather plus a parallel product.
- **Blocks are independent.** Linearization is embarrassingly parallel; only the
  scatter needs disjoint ranges, which the row layout guarantees.
- **No allocation in the hot path.** Factor buffers, the assembly arena and the
  solver workspaces are allocated once and reused.
- **Manifold parameters stay contiguous.** They live in `nalgebra` storage that
  `faer` views without copying, so no conversion happens between assembly and
  the linear solve.
- **Structure is exploited, not discovered.** The Schur partition, the
  elimination groups and the chunk layout all come from the graph's adjacency,
  which is known before the first iteration.
