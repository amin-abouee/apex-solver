# Robust Loss Functions

Outliers dominate the ordinary least-squares objective. Apex Solver ships 15
robust kernels — `Huber`, `Cauchy`, `Tukey`, `Welsch`, `Fair`, `GemanMcClure`,
`Andrews`, `Trimmed`, `L1`, `Lp`, Barron variants and more — all in
`apex_solver::core::loss_functions`.

## Usage

```rust
use apex_solver::core::loss_functions::HuberLoss;

problem.add_residual_block(
    &[k_pose, k_landmark],
    Box::new(projection_factor),
    Some(Box::new(HuberLoss::new(1.345)?)),
);
```

## What the solver actually computes

For a squared residual norm $s = \lVert \mathbf{r} \rVert^2$, the robust
objective is

$$
F(\mathbf{x}) = \tfrac{1}{2}\sum_i \rho_i\!\left(s_i(\mathbf{x})\right)
$$

Minimizing $F$ is equivalent to an iteratively reweighted least-squares problem
whose residual and Jacobian are

$$
\tilde{\mathbf{r}} = \sqrt{\rho'(s)}\,\mathbf{r},
\qquad
\tilde{\mathbf{J}} = \sqrt{\rho'(s)}
\left(\mathbf{J} - \tfrac{\rho''(s)}{2\,\rho'(s)}\,\alpha\, \mathbf{r}\mathbf{r}^\top\mathbf{J}\right),
$$

with $\alpha$ from the Triggs correction. Apex Solver applies exactly this
correction in-place during block evaluation, and reports the cost as
$\tfrac12 \rho(s)$ — *not* as the norm of the corrected residual. The two are
different functions; conflating them silently corrupts every reported cost and
trust-region ratio on robust problems.
