# Covariance and Uncertainty

An estimate without an uncertainty is a guess with a number attached. The
crate can compute the covariance of a solved problem — and the design of that
computation is the subject of this chapter, because the details decide whether
the number means anything.

## 1. The problem

A least-squares solve produces a point estimate. What the user usually wants
alongside it is how *uncertain* that estimate is: the standard deviation of a
pose, the correlation between two states, the size of the region the map
could occupy.

The Gauss–Newton machinery already has the ingredients. At the solution, the
normal matrix $H = J^\top J$ is the Gauss–Newton approximation of the
Hessian, and its inverse is the covariance of the estimate under the
approximation:

$$
\Sigma = H^{-1}.
$$

The catch is that the solver's $H$ is not that matrix. The optimizer solves
$H + \lambda D$ — a damped, scaled system — and the covariance must not
inherit either.

## 2. What is deliberately *not* in it

<div class="apx apx-table">
<span class="apx-title">The covariance's invariance</span>

| Quantity | In the covariance? | Why |
|---|---|---|
| the Levenberg–Marquardt damping $\lambda$ | **no** | a device that makes the *step* well-behaved; it carries no information about the estimate's uncertainty |
| the Jacobi column scaling $D$ | **no** | a preconditioner: it changes the metric of the step, not the geometry of the problem |
| the robust correction | **yes**, in the Jacobian | the corrected Jacobian $\tilde J$ is what the assembly produced, so $H = \tilde J^\top\tilde J$ |

</div>

The consequence is a property worth stating as a test: **flipping
`with_jacobi_scaling` or changing the damping must produce identical
covariances.** The covariance module therefore re-linearizes the problem
*itself*, at the given variable values, and inverts a clean $J^\top J$ — it
never reuses the optimizer's factorization, which is of $H + \lambda I$.

## 3. Robust losses

When a residual block carries a loss function, the $J$ used here is the
**Triggs-corrected** Jacobian the assembly produced, so

$$
H = \tilde J^\top \tilde J
$$

is the Gauss–Newton approximation of the Hessian of the *robust* objective.
This matches Ceres Solver, and it is the Laplace approximation of the posterior
under the robust cost.

It is deliberately **not** the Huber–White sandwich estimator
$A^{-1}BA^{-1}$ used for robust standard errors in the statistics literature.
The two answer different questions, and the crate's choice is the one a
solver can compute from the quantities it already has.

## 4. Units: when is $H^{-1}$ a covariance?

Only when the residuals are already whitened by their measurement information
matrix — the usual convention in SLAM and bundle adjustment, and the one the
crate's noise layer implements. Then $H^{-1}$ *is* the covariance.

For unweighted least squares, the noise scale has to be estimated from the
fit:

$$
\hat\sigma^2 = \frac{2\,\text{cost}}{m - n},
$$

where $m$ is the total residual dimension and $n$ the total tangent
dimension, and the covariance is $\hat\sigma^2 H^{-1}$. The option
`apply_variance_scaling` does this, and it is **off by default** — because
the crate's own convention is that residuals arrive whitened.

The estimator refuses to guess: when $m \le n$ there are no residual degrees
of freedom, so it warns and returns the *unscaled* $H^{-1}$ rather than a
number it cannot justify.

## 5. The two algorithms

<div class="apx apx-table">
<span class="apx-title">`CovarianceAlgorithm`</span>

| | `SparseCholesky` (default) | `DenseSvd` |
|---|---|---|
| what it computes | $H^{-1}$ by a fresh sparse Cholesky | the Moore–Penrose pseudo-inverse by a dense SVD |
| requires | $H$ positive definite | nothing |
| on a rank-deficient $H$ | `CovarianceError::RankDeficient` | zero variance along the null space — the correct answer for a gauge-free problem |
| cost | sparse, $O(\text{nnz}^{1.5})$ | $O(n^3)$ time, $O(n^2)$ memory |

</div>

`DenseSvd` treats singular values below
$\sigma_{\max}\cdot$ `min_reciprocal_condition_number` (default
$10^{-14}$) as zero.

The default is `SparseCholesky` because most problems *are* fully
constrained, and because the dense path's $O(n^2)$ memory is prohibitive for
a window with a thousand landmarks. The `RankDeficient` error is not a
failure to report: its message names the two causes — an unfixed gauge
freedom, or a variable no residual observes — and the two ways out (anchor
the gauge with a prior, or ask for the pseudo-inverse).

<div class="apx apx-note">
<span class="apx-title">Note — a rank-deficient Hessian is a statement about the problem</span>

The pseudo-inverse's zero variance along the null space is the *correct*
answer for a problem that is intentionally gauge-free: the estimate really is
arbitrary in that direction. A covariance that reported a finite variance
there would be claiming knowledge the problem does not contain. The error
message says as much.

</div>

## 6. Fixed coordinates

A fixed tangent coordinate is held at a known constant, so it has zero
variance and zero covariance with everything else — by construction, not by
approximation.

The implementation therefore builds $H$ over the **free columns only** and
reinserts the fixed coordinates as exact `0.0` rows and columns when handing
blocks back to the caller:

- `Covariance::block(key)` — the marginal $d \times d$ block of one variable,
  in **tangent space** (a $6 \times 6$ matrix for an `SE3`, not $7 \times 7$),
  with exact zeros at any fixed local index;
- `Covariance::block_pair(a, b)` — the cross-covariance between two
  variables, with exact zeros at a fixed index of either;
- `Covariance::per_variable()` — every variable's marginal block;
- `Covariance::full()` — the reduced $H^{-1}$ over the free columns, with
  `index_map()` giving each variable's offset in it.

The subtlety the source spells out: because the fixed columns are excluded
before the inversion, the result is a **conditional** covariance, not a
marginal that would overstate the free parameters' uncertainty.

## 7. The API

```rust
let covariance = Covariance::compute(
    CovarianceOptions::default(),
    &problem,
    &result.parameters,
)?;
let pose_block = covariance.block(pose_key);
```

`CovarianceOptions` is `Copy` and carries three plain values: the
`algorithm`, the `apply_variance_scaling` flag, and the
`min_reciprocal_condition_number`.

Two convenience paths exist:

- `Problem::compute_and_set_covariances(&mut variables, options)` — computes
  the covariance and writes each variable's marginal block into the variable
  itself (`ManifoldVariable::set_covariance`). It returns `None` on failure,
  after logging the reason; a caller that must handle the failure should
  call `Covariance::compute` directly and get the typed error.
- The optimizers' `compute_covariances` flag and `covariance_options` field —
  the optimizer computes the covariance at the end of a successful solve and
  puts it in `SolverResult::covariances` (a map from variable key to a
  covariance in tangent space).

## 8. Worked example

A pose graph over 300 keyframes, with the first keyframe's position fixed
(five of its six coordinates) and its yaw fixed by a prior.

The free space is 300 × 6 − 6 = 1 794 columns. The Cholesky path factorizes
that sparse $H$ and inverts it; the covariance of a single pose is the
$6 \times 6$ block at its offset, with the fixed yaw's row and column
exactly zero. The yaw's prior is a *soft* constraint and therefore shows up
as a small but non-zero variance — which is the difference between a fixed
coordinate and a tight prior, visible in one matrix.

Had the first pose been left entirely free, $H$ would have had a null
direction (the global gauge), the Cholesky would have failed with
`RankDeficient`, and the message would have named the cause.

## 9. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a covariance is a property of the model, not of the truth</span>

The covariance says: *if the model is right, and the residuals are white with
the stated information, the estimate's error is this covariance*. Every
assumption in that sentence can fail silently — a mis-scaled information
matrix, an unmodelled systematic error, a residual that is not white — and
the covariance will faithfully report the uncertainty of the wrong model. A
covariance that is too small is worse than none, because it is trusted.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — the first-estimate policy is deliberately ignored</span>

The covariance re-linearizes at the **current** estimate, never at a frozen
linearization point. A covariance is a property of the problem *here*, and a
frozen Jacobian would report the uncertainty of a linearization the estimate
has already left. This is the opposite trade from the marginal prior, which
must stay at its linearization point to avoid double-counting information —
and the two decisions are made independently, on purpose.

</div>

<div class="apx-note">
<span class="apx-title">Note — the covariance is not the optimizer's factorization</span>

The optimizer's Cholesky is of $H + \lambda D$ and may be cached across
iterations. The covariance's is of a clean $H$, built fresh, at the final
point, with the reduced free-column space. Reusing the first for the second
would put the damping and the scaling into the answer — which is exactly what
the module's design exists to prevent.

</div>
