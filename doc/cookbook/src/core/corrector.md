# The Corrector

A robust loss changes the objective; the optimizer still wants a least-squares
problem. The **corrector** is the bridge: it rewrites a residual and its
Jacobian so that an ordinary quadratic solve minimizes the robust objective
instead. This chapter is that rewrite — the two identities that make it
possible, and the arithmetic the crate performs.

## 1. The problem

We want to minimize

$$
\sum_i \rho\big(\|\tilde r_i\|^2\big)
$$

but the solver's machinery — normal equations, step norms, cost accounting —
is quadratic. The trick is to construct, for each block, a *corrected*
residual $\tilde r$ and a *corrected* Jacobian $\tilde J$ such that

$$
\sum_i \|\tilde r_i\|^2 = \sum_i \rho\big(\|r_i\|^2\big)
$$

**exactly** (up to the neglected higher-order terms of the kernel), and such
that the gradient of the quadratic problem is the gradient of the robust one.

That identity is what makes the robust cost a cost: the optimizer's reported
cost, its step norms, its convergence checks, and its covariance estimates all
remain meaningful.

## 2. The construction

Let $s = \|\mathbf{r}\|^2$, and let the kernel provide $\rho(s)$,
$\rho'(s)$, $\rho''(s)$. The corrector computes three scalars:

$$
\sqrt{\rho_1} = \sqrt{\rho'(s)},
\qquad
\alpha^2 = \frac{\rho''(s)}{\rho'(s)},
$$

and then

$$
\tilde r = \sqrt{\rho_1}\,\mathbf{r},
\qquad
\tilde J = \sqrt{\rho_1}\,J
+ \alpha\,\frac{(J^\top \mathbf{r})\,\mathbf{r}^\top}{\|\mathbf{r}\|}.
$$

The first term is the obvious re-weighting; the second is the correction that
makes the *gradient* right, not just the cost.

The intuition for the second term: the kernel's Hessian in the direction of
$\mathbf{r}$ is $\rho''(s)$, and the least-squares Hessian in that direction
is $1$. The corrector interpolates between them with $\alpha^2 = \rho''/\rho'$:
for a quadratic kernel ($\rho''/\rho'=1$) the correction vanishes and the
plain weighted residual is exact; for a redescending kernel the correction
dominates and the block's influence along $\mathbf{r}$ shrinks toward zero —
the kernel's influence function, expressed as a matrix.

## 3. The crate's implementation

`Corrector::new(&loss, sq_norm)` evaluates the kernel once and stores the
scalars; it is constructed **once per residual block per iteration**, not per
residual evaluation, which is why the kernel's cost is paid once per block.

The application is in-place, on flat column-major buffers:

```rust
corrector.correct_jacobian_in_place(residual.as_slice(), &mut jac_flat, rows, cols);
corrector.correct_residual_in_place(corrected_residual.as_mut_slice());
```

The residual correction and the Jacobian correction are separate calls, and
the Jacobian call takes the *original* residual (it needs $J^\top r$ before
the residual is scaled).

The crate also tracks the **robust cost** itself (`robust_cost` on the
corrector), so a solve can report $\sum\rho$ rather than $\sum\|\tilde r\|^2$
— the same number up to the kernel's higher-order terms, and the one a user
compares against a reference implementation.

<div class="apx-note">
<span class="apx-title">Note — order of operations in the pipeline</span>

Whitening comes *before* the corrector: the block's residual and Jacobian are
whitened by $S$, the squared norm of the whitened residual is what the kernel
sees, and the corrector's output is what the optimizer assembles. A corrector
applied before whitening would threshold the raw residual, and the threshold
would mean different things for different blocks.

</div>

## 4. Worked example

A one-dimensional block with residual $r = 2$ and a Huber kernel with
$\delta = 1.345$:

- $s = 4$, $\rho'(s) = \delta/\sqrt{s} = 0.6725$, $\rho''(s) = 0$;
- $\sqrt{\rho_1} = 0.820$,
  $\alpha^2 = 0$;
- $\tilde r = 0.820 \times 2 = 1.640$, and $\|\tilde r\|^2 = 2.69$, while
  $\rho(4) = 2\delta\sqrt{4} - \delta^2 = 3.578$ — the quadratic
  approximation of the robust cost at this residual, the error being the
  kernel's neglected higher-order term;
- the Jacobian is scaled by 0.820 with no correction term (Huber's
  $\rho''$ vanishes beyond the threshold), which is the correct linearization
  of the Huber cost in the residual direction.

## 5. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the correction divides by ‖r‖</span>

The Jacobian correction contains $\mathbf{r}^\top/\|\mathbf{r}\|$, which is
undefined at $\mathbf{r} = 0$. The crate handles the degenerate case (the
correction term is skipped when the residual vanishes, where $\rho'$ is maximal
and the plain scaling is right), but a factor author porting the formula must
carry the same guard. A factor that divides by a near-zero residual norm
produces a NaN in the Jacobian that appears only in the first iteration, when
the estimate is still far from the truth.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — the corrector is not a rejection mechanism</span>

The corrector makes an outlier's influence *finite*; it does not remove the
measurement from the graph. A block whose whitened residual is 100 σ is
down-weighted by a factor of about $\delta/100$ — small, but not zero, and
squared into the information. A graph whose outliers are coherent (a whole
cluster of mismatches) still moves, because every one of them pulls in the
same direction. Selection — RANSAC, in the consumer — is the mechanism that
removes.

</div>

<div class="apx-note">
<span class="apx-title">Note — the corrector is per block, not per row</span>

The kernel sees the whole residual's squared norm, so a 3-D residual is
re-weighted as one object, not row by row. A factor that wants a per-row
kernel must split itself into blocks — which is a modelling decision, not a
solver detail, and worth making deliberately.

</div>

<div class="apx-paper">
<span class="apx-title">Papers — the construction</span>

Triggs, McLauchlan, Hartley & Fitzgibbon, "Bundle Adjustment — A Modern
Synthesis", *Vision Algorithms*, 1999 — the re-weighted least-squares identity
this implements. Agarwal, Mierle et al., "Ceres Solver" — the reference
implementation the crate's doc comment cites, and the source of the exact
algorithm and the exact ordering of the operations.

</div>
