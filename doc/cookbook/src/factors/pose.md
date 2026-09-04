# Pose & Priors

`factors::pose` — relative constraints and anchors. No sensor model: these
express graph topology.

Throughout, $\boxplus$ is the right perturbation $X \boxplus \delta = X\,\mathrm{Exp}(\delta)$,
$\boxminus$ its inverse $Y \boxminus X = \mathrm{Log}(X^{-1}Y)$, and $J_r$, $J_l$
the right and left Jacobians of the group. The identity used everywhere below is

$$
\mathrm{Log}\!\left(X\,\mathrm{Exp}(\boldsymbol{\epsilon})\right)
\;\approx\; \mathrm{Log}(X) + J_r^{-1}\!\big(\mathrm{Log}(X)\big)\,\boldsymbol{\epsilon}
$$

---

## `BetweenFactor<T>`

A relative measurement between two variables of the same Lie group. Odometry
edges and loop closures are both this factor, the latter usually with a robust
loss.

**Blocks** $[T_i,\ T_j]$ — any `T: LieGroup`. Residual $\mathrm{dof}(T)$,
Jacobian $\mathrm{dof}(T) \times 2\,\mathrm{dof}(T)$.

### Error

With measurement $\tilde{T}_{ij}$,

$$
\mathbf{r} \;=\; \mathrm{Log}\!\left( \tilde{T}_{ij}^{-1}\, T_i^{-1}\, T_j \right)
$$

zero exactly when the estimated relative transform equals the measured one.

### Jacobian

Write $A = T_i^{-1}T_j$. Perturbing the second variable,
$\tilde{T}^{-1}A\,\mathrm{Exp}(\delta_j)$, gives directly

$$
\frac{\partial \mathbf{r}}{\partial \delta_j} \;=\; J_r^{-1}(\mathbf{r})
$$

Perturbing the first, $T_i \leftarrow T_i\mathrm{Exp}(\delta_i)$, moves the
inverse to the *left* of $A$, and it has to be commuted through:

$$
\tilde{T}^{-1}\mathrm{Exp}(-\delta_i)A
= \tilde{T}^{-1}A\;\mathrm{Exp}\!\big(-\mathrm{Ad}_{A^{-1}}\delta_i\big)
$$

so

$$
\frac{\partial \mathbf{r}}{\partial \delta_i}
\;=\; -\,J_r^{-1}(\mathbf{r})\;\mathrm{Ad}_{T_j^{-1}T_i}
$$

For SE(3), $\mathrm{Ad}_T = \begin{bmatrix} R & [\mathbf{t}]_\times R \\ 0 & R \end{bmatrix}$
and

$$
J_r^{-1}(\boldsymbol{\rho},\boldsymbol{\theta}) =
\begin{bmatrix} J_r^{-1}(\boldsymbol{\theta}) & Q_r \\ 0 & J_r^{-1}(\boldsymbol{\theta}) \end{bmatrix},
\qquad
J_r^{-1}(\boldsymbol{\theta}) = I + \tfrac12[\boldsymbol{\theta}]_\times
+ \left(\tfrac{1}{\theta^2} - \tfrac{1+\cos\theta}{2\theta\sin\theta}\right)[\boldsymbol{\theta}]_\times^2
$$

The implementation never writes these forms out: it chains the Jacobians that
`between`, `compose` and `log` report, so it stays correct for every group.

```rust
use apex_solver::factors::pose::BetweenFactor;

problem.add_residual_block_with_noise(
    &[key_i, key_j],
    Box::new(BetweenFactor::new(measured_relative_pose)),
    Some(Box::new(HuberLoss::new(1.345)?)),
    NoiseModel::from_sigmas(&[0.05, 0.05, 0.05, 0.01, 0.01, 0.02])?,
);
```

---

## `PriorFactor<T>`

Anchors one variable to a known value in the tangent space. Prefer this to
`Problem::fix_variable`: fixed indices are honoured when the step is applied,
but the linear system still treats those coordinates as free, so variables
sharing factors with a "fixed" one are systematically under-corrected.

**Blocks** $[X]$. Residual $\mathrm{dof}(T)$, Jacobian square.

### Error and Jacobian

$$
\mathbf{r} = \mathrm{Log}\!\left(T_\text{prior}^{-1} X\right),
\qquad
\frac{\partial \mathbf{r}}{\partial \delta} = J_r^{-1}(\mathbf{r})
$$

This is a `BetweenFactor` with an identity measurement and a constant first
argument, and is implemented as exactly that — the two share one chain and
cannot drift apart. Because the residual never touches the parameters, there is
no quaternion double-cover ambiguity.

---

## `EuclideanPriorFactor`

$$
\mathbf{r} = \mathbf{x} - \mathbf{x}_\text{prior},
\qquad
\frac{\partial\mathbf{r}}{\partial\mathbf{x}} = I_n
$$

Registration **rejects any manifold but `Rn`** — on a Lie group this would
difference raw parameters, which is not a tangent-space error. Use it to anchor
velocities, biases and intrinsics.

---

## `PoseRotationPrior` / `PoseTranslationPrior`

Constrain one half of an SE(3) pose and leave the other free.

$$
\textbf{Rotation:}\quad
\mathbf{r} = \mathrm{Log}\!\left(R_\text{meas}^\top R\right),
\qquad
\frac{\partial\mathbf{r}}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \big[\;0_{3\times3} \;\big|\; J_r^{-1}(\mathbf{r})\;\big]
$$

$$
\textbf{Translation:}\quad
\mathbf{r} = \mathbf{t} - \tilde{\mathbf{t}},
\qquad
\frac{\partial\mathbf{r}}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \big[\;R \;\big|\; 0_{3\times3}\;\big]
$$

The translation block is $R$, not $I$, because the right perturbation moves the
origin by $R\,\delta\boldsymbol{\rho}$ in the world frame.

These are **not** a decomposition of `PriorFactor<SE3>`: the translation rows
here are the world-frame difference, whereas the prior's are the coupled SE(3)
logarithm $\boldsymbol{\rho}$, which mixes rotation into translation through
$J_l^{-1}(\boldsymbol{\theta})$. Stacking both does not reproduce the full
prior.

`PoseRotationPrior` is the usual rotation-only loop-closure initializer;
`PoseTranslationPrior` is how a GNSS position fix attaches to an SE(3) pose.
