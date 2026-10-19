# Marginalization: From Factors to a Prior

A sliding window cannot simply forget its oldest keyframe. The measurements
that constrained it also constrained its neighbours, and dropping them throws
that information away — and the gauge, the correlations and the uncertainty
with it. Marginalization eliminates the old state and leaves behind the
Gaussian it induced on the states it was connected to. This chapter is that
elimination, done densely and on purpose.

## 1. The problem

Consider a window with states $x_0, x_1, \dots, x_k$ and a set of factors.
Dropping $x_0$ and its factors loses the constraints those factors imposed on
$x_1$: the visual observations of the landmarks $x_0$ saw, the IMU interval
between $x_0$ and $x_1$, the bias random walk. The remaining graph is
*smaller* and *less informed*, and the trajectory it produces will differ from
the one the full window would have produced — a difference that accumulates
every time a window slides.

The fix is to summarize rather than discard: eliminate $x_0$ exactly, and\nkeep the Gaussian it left on the states it touched.

## 2. The mathematics

The normal equations of the affected subgraph are

$$
\begin{bmatrix}\Lambda_{kk} & \Lambda_{km}\\ \Lambda_{mk} & \Lambda_{mm}\end{bmatrix}
\begin{bmatrix}\delta x_k\\ \delta x_m\end{bmatrix}
= \begin{bmatrix}g_k\\ g_m\end{bmatrix} .
$$

Eliminating the marginalized variables $x_m$ gives the *conditional* Gaussian
on the kept variables $x_k$:

$$
\Lambda_p = \Lambda_{kk} - \Lambda_{km}\Lambda_{mm}^{-1}\Lambda_{mk},
\qquad
g_p = g_k - \Lambda_{km}\Lambda_{mm}^{-1}g_m .
$$

$\Lambda_p$ is the information the marginalized variables carried, and $g_p$
is their contribution to the gradient. Together they are the exact
information the dropped state leaves behind — no approximation, and no
assumption.

## 3. The sign

The crate is explicit about this, and the reason is worth repeating because
getting it wrong is invisible:

> `NormalEquations::gradient` is $J^\top r$, **positive**, and the system
> solved is $H\delta = -g$. The quadratic model is
> $C(\delta) = c + g^\top\delta + \tfrac12\delta^\top\Lambda\delta$,
> minimized at $\delta^* = -\Lambda^{-1}g$, and the prior's own cost
> $\tfrac12(\theta-b)^\top\Lambda_p(\theta-b)$ is minimized at $\theta = b$.
> Therefore $b = -\Lambda_p^{+}g_p$.

With the sign flipped, the prior pushes the state the wrong way by exactly
$2\Lambda^{-1}g$ — and still looks plausible: the solve converges, the cost
goes down, and the trajectory is wrong in the direction the prior should have
pulled it. The crate has a test named `offset_sign_reproduces_the_conditional_minimizer`
for exactly this.

## 4. Why it is dense, and why it does not use the Schur module

This is the chapter's most interesting design decision, and the crate's own
documentation argues it at length.

The Schur machinery (`linalg::schur`) is structurally a **landmark
eliminator**: its `ChunkLayout` rejects two eliminated variables that share a
residual row, and `SchurPartition::verify_block_diagonal` demands a
block-diagonal $\Lambda_{mm}$. A visual-inertial window eliminates
$\{\text{state}_0, \text{bias}_0, \text{hosted landmarks}\}$ — and
`CombinedImuFactor` puts `state_0` and `bias_0` in the **same fifteen rows**.
So $\Lambda_{mm}$ is not block-diagonal, and none of that machinery applies.

The alternative is to build $\Lambda$ explicitly. The Markov blanket of one
keyframe is $O(10^2 - 10^3)$ DoF, where a dense factorization is both
*correct* and *fast*. The module therefore works in `nalgebra` rather than
`faer`: it runs once per window step, not in the inner loop, and
`MarginalPriorFactor` takes its square-root information as a `DMatrix` — a
`faer` round trip would buy nothing and cost a conversion at each end.

## 5. The absorbed set

The marginal absorbs the factors **incident to a dropped variable**, not the
whole graph. Marginalizing over every block would bake information from
factors that stay in the graph into the prior — counting it twice — and would
make $\Lambda_p$ dense over the entire state.

The previous prior is itself incident to the dropped variables, so it is
re-absorbed: the marginal is recursive, exactly as the mathematics says it
should be.

The dropped variables are also required to be *observed*: a variable that
appears in no residual block is an error
(`MarginalizationError::Unobserved` — "drop it instead of marginalizing it"),
because eliminating an unobserved variable divides by a singular
$\Lambda_{mm}$.

## 6. The API

```rust
let marginalizer = Marginalizer::new();
let marginal = marginalizer.compute(&problem, &[key_0])?;   // no mutation
let applied   = marginalizer.apply(&mut problem, &[key_0])?;  // mutate
```

`compute` returns a `Marginal` without touching the problem; `apply` does the
whole transaction — remove the absorbed blocks, remove the dropped variables,
register the prior — and reports what it did in an `AppliedMarginal` (the
prior's key, the kept variables, the rank, the dimension, the removed
blocks).

<div class="apx-table">
<span class="apx-title">`Marginal`'s fields</span>

| Field | Meaning |
|---|---|
| `kept` | the retained variables, in the order the prior's blocks must be registered |
| `dims` | each kept block's tangent dimension, parallel to `kept` |
| `information` | $\Lambda_p$ |
| `gradient` | $g_p$, with $g = J^\top r$ |
| `sqrt_information` | $S$ with $S^\top S = \Lambda_p$; unobservable directions are **zero rows** |
| `offset` | $b = -\Lambda_p^+ g_p$ |
| `linearization_point` | each kept block's ambient parameters at $x_0$ |
| `manifolds` | each kept block's manifold |
| `rank` | the numeric rank of $\Lambda_p$ |
| `absorbed` | the residual blocks this marginal absorbed |

</div>

`nullity()` is `dim() - rank`: the dimension of the unobservable subspace. A
monocular visual-inertial window is *expected* to have one — global position
and global heading are not observable — and the marginal's job is to preserve
that fact rather than to hide it.

## 7. The square root

`SqrtInformation` chooses how $\Lambda_p$ is factored:

| Variant | Behaviour |
|---|---|
| `Eigen { rank_tolerance: 1e-9 }` (default) | self-adjoint eigendecomposition; eigenvalues below `rank_tolerance · λ_max` (and any negative ones) become zero, producing **zero rows** in a square $S$ |
| `Cholesky` | $S = L^\top$; faster, and **errors** on a rank-deficient $\Lambda_p$ (`MarginalizationError::RankDeficient`) |

The default is the eigen variant for exactly the reason above: a marginal is
*expected* to be rank deficient, and the zero rows are the honest encoding of
"this direction is unobserved". The Cholesky variant is for a caller who
knows the marginal is full rank and wants the speed.

The marginal is also symmetrized by default
(`with_symmetrize(true)`): the Schur complement of a symmetric matrix is
symmetric only in exact arithmetic.

## 8. Worked example

A window with 7 keyframes and 200 landmarks, marginalizing the oldest
keyframe and its bias. The Markov blanket — the states the dropped keyframe's
factors touch — is 6 states (90 DoF) plus the hosted landmarks. The
eliminated block $\Lambda_{mm}$ is (state 0, bias 0, landmarks); it is *not*
block-diagonal, because the combined IMU factor shares rows between the state
and the bias.

The marginal therefore forms $\Lambda$ densely: a $300\times300$ matrix
built from the absorbed blocks' normal equations, factorized once, and
reduced to $\Lambda_p$ — a $90\times90$ (plus landmarks) matrix with a
**zero row** for the global heading, because nothing in the window observes
it.

The prior that results carries 90 rows of real information and one row of
zeros, and the solve that follows is conditioned exactly as the window's
information really is.

## 9. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the prior is a linear model, and it ages</span>

A marginal is a *linear* statement about the world, made at $x_0$. It is
exact at $x_0$ and increasingly wrong as the estimate moves away. The
conventional answer is to rebuild the marginal — re-marginalize — when the
drift is large, and to use the first-estimate convention in the meantime
(chapter [The First-Estimate Policy](../core/first-estimate.md)). A consumer
that never re-marginalizes accumulates a stale prior, and a consumer that
re-linearizes the prior at the current estimate accumulates rank in the null
directions. Both failure modes are silent.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — the sign, again</span>

`b = -\Lambda_p^+ g_p`. Flipping the sign makes the prior push the state the
wrong way by $2\Lambda^{-1}g$, and the solve still converges. If a window's
trajectory is systematically *worse* after a marginalization than before it,
check the sign before anything else.

</div>

<div class="apx-note">
<span class="apx-title">Note — the prior addresses its blocks positionally</span>

`Marginal::to_factor` returns the kept keys in order, and the factor
addresses its blocks by *position*: the $i$-th key maps to `dims[i]`, to that
slice of $S$, and to `params[i]` inside the local-log closure. A consumer
whose problem has been rebuilt since the marginal was computed may substitute
its own keys, provided it keeps the order — and the returned `VarKey`s are
then stale and must not be used against the new problem.

</div>
