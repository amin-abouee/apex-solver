# Gauss–Newton

Gauss–Newton is the simplest useful optimizer: linearize the objective, solve
the linearized problem, take the step. It converges quadratically near a
solution and does nothing at all when the problem is ill-conditioned. This
chapter is the crate's implementation, the two knobs that rescue it, and the
situations where it is the right choice.

## 1. The algorithm

For $F(\mathbf{x}) = \tfrac12\|\tilde r(\mathbf{x})\|^2$ with the whitened,
robust-corrected residual:

$$
J^\top J\,\delta = -J^\top \tilde r,
\qquad
\mathbf{x} \leftarrow \mathbf{x}\circ\mathrm{Exp}(\delta).
$$

There is no damping and no trust region: the step is whatever the
Gauss–Newton system says. When the system is positive definite and the
linearization is good, this is the fastest convergence available.

## 2. The two rescue knobs

Pure Gauss–Newton fails in two distinguishable ways, and the config has one
knob for each.

**A singular system.** $J^\top J$ is singular when the problem has an
unconstrained direction — a gauge freedom, a landmark seen once, a state with
no measurement. `min_diagonal` adds a uniform diagonal regularization:

$$
(J^\top J + \texttt{min\_diagonal}\cdot I)\,\delta = -J^\top \tilde r .
$$

The default is `1e-10`: "very small, practically identical to pure
Gauss–Newton" by the config's own comment, but enough to keep the
factorization well posed. Set it to `0.0` for the un-regularized equations
when the problem is known to be non-singular and the extra terms would bias
the answer.

**Mixed scales.** A problem with positions in metres and angles in radians
has columns of very different magnitude, and the step is then dominated by the
stiff columns. `use_jacobi_scaling` normalizes the Jacobian columns by their
$L_2$ norm before solving — the classic preconditioning. The cost is 5–10 %
overhead, which is why it is off by default: Gauss–Newton is usually used on
well-conditioned problems, where the scaling is not needed.

## 3. What the crate adds

Beyond the textbook loop, the implementation:

- assembles through the same linearizer as the other optimizers, so the sparse
  and dense paths, the evaluation policy and the robust corrector all apply;
- applies the step through `apply_parameter_step`, so the manifold's own plus
  is used and frozen columns are excluded from the step norm;
- **always accepts the step**: the implementation records `accepted: true`
  and `step_accepted: true` without ever judging it. There is no rollback and
  no adaptive regularization — the only regularization available is the
  `min_diagonal` on the linear system;
- reports per-iteration statistics through `IterationStats`.

## 4. When to choose it

<div class="apx-table">
<span class="apx-title">Gauss–Newton's sweet spot</span>

| Situation | Verdict |
|---|---|
| well-conditioned, few variables, a good initialization | **Gauss–Newton** |
| rank-deficient system (gauge, single-observation landmark) | Levenberg–Marquardt |
| large steps expected, model may be poor far from the solution | Dog Leg |
| robust loss with heavy contamination | Levenberg–Marquardt (adaptive damping) or a RANSAC stage upstream |

</div>

The comparison is not only about speed. Gauss–Newton's step is the *Newton*
step of the local model: if the model is wrong, the step is confidently
wrong. LM and Dog Leg shorten the step when the model misbehaves; Gauss–Newton
does not.

## 5. Worked example

A 200-point ICP problem: 600 DoF, well-conditioned, the initial guess within
a few degrees. Gauss–Newton converges in 5 iterations to a residual of
$2\times10^{-4}$ m; the same problem with Levenberg–Marquardt needs 7, because
LM spends its first iterations on damping steps that Gauss–Newton does not
need.

With an initial guess 20° off — the model is wrong over most of the basin —
Gauss–Newton's first step overshoots and the cost *rises*. Because the
optimizer always accepts its step, it does not recover by itself: it keeps
stepping, and whether the cost comes back down depends on the problem's
curvature. Levenberg–Marquardt would have rejected that step, raised $\lambda$
and taken a shorter one. The lesson is not that Gauss–Newton is fragile — it
is that its contract is "the model is good here", and the caller has to
establish that.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a singular system solved by regularization is still singular information</span>

`min_diagonal` keeps the factorization alive; it does not add information. The
regularized direction is *unconstrained by the data* and the solver will move it
by whatever the regularization's tiny gradient suggests — in practice, not at
all, which is why the gauge must be fixed explicitly (chapter
[The Problem](../core/problem.md)) rather than left to the regularizer. A solve that
"converges" in a direction nobody measured is a solve that found the
regularizer, not the answer.

</div>

<div class="apx-note">
<span class="apx-title">Note — Gauss–Newton is the default's neighbour, not the default</span>

`OptimizerType`'s default is `LevenbergMarquardt`, because a general factor
graph is rarely well conditioned enough for the undamped step to be safe, and
because the cost of the adaptive damping — a few extra solves — is small
against the cost of a diverged one. Choosing Gauss–Newton explicitly is a
statement that the problem is well conditioned; the crate takes that statement
seriously and does not add damping behind your back.

</div>
