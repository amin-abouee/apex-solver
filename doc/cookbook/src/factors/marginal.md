# Marginalization

`factors::marginal` — summarizing variables a sliding window has dropped.

---

## `MarginalPriorFactor`

When a window marginalizes old states, the eliminated joint distribution is
summarized as a Gaussian over the *remaining* variables. This factor carries
that Gaussian in **linear container** form: the Jacobian is a constant matrix,
fixed at the linearization point where the marginal was computed.

**Blocks** any number, any manifolds — the factor is manifold-agnostic.
Residual $\mathrm{rows}(S)$, Jacobian $\mathrm{rows}(S) \times \sum_i \mathrm{dof}_i$.

### Error

$$
\mathbf{r}(\mathbf{x}) = S\left(\boldsymbol{\theta}(\mathbf{x} \boxminus \mathbf{x}_0) - \mathbf{b}\right),
\qquad
J(\mathbf{x}) = S
$$

where

- $\boldsymbol{\theta}(\mathbf{x} \boxminus \mathbf{x}_0)$ is the concatenated
  local tangent of the connected variables relative to the marginal's
  linearization point $\mathbf{x}_0$, computed by a caller-supplied
  `local_log` closure — this is what keeps the factor independent of which
  manifolds it connects;
- $S$ is the square-root information of the marginal, $S^\top S = \Lambda$;
- $\mathbf{b} = \Lambda^{-1}\mathbf{g}$ encodes the information vector, so that
  the quadratic reproduced is
  $\tfrac12(\boldsymbol{\theta} - \mathbf{b})^\top\Lambda(\boldsymbol{\theta} - \mathbf{b})$.

### Where it comes from

Marginalizing a set $m$ out of a joint information matrix leaves the Schur
complement over the remaining set $r$:

$$
\Lambda_{rr}^{\text{marg}} = \Lambda_{rr} - \Lambda_{rm}\Lambda_{mm}^{-1}\Lambda_{mr},
\qquad
\mathbf{g}_r^{\text{marg}} = \mathbf{g}_r - \Lambda_{rm}\Lambda_{mm}^{-1}\mathbf{g}_m
$$

Factor $\Lambda^{\text{marg}} = S^\top S$ and set $\mathbf{b} = \Lambda^{-1}\mathbf{g}$.

### The catch

The Jacobian is **constant**, which is the whole point of a container — but it
is only exact at $\mathbf{x}_0$. As the estimate drifts away from the
linearization point the prior becomes progressively wrong, and worse, it is
wrong in a way that looks confident. Re-marginalize and rebuild the factor when
the estimate moves far from $\mathbf{x}_0$.

The partial-pose priors that often accompany marginalization live in
[`factors::pose`](./pose.md): they anchor a pose for loop-closure
initialization rather than summarize eliminated variables.
