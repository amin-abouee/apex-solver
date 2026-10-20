# The Marginal Prior as a Factor

The marginalization chapter produced a Gaussian over the kept variables. This
chapter is how that Gaussian enters the factor graph: as a factor, with the
linear-container semantics that keep it honest, and with the first-estimate
convention that makes it stay honest.

## 1. The linear container

`MarginalPriorFactor` is the crate's analogue of GTSAM iSAM2's
`LinearContainerFactor`. It is a factor whose residual is *linear* in the
variables' local coordinates:

$$
r(x) = S\big(\theta(x \ominus x_0) - b\big),
\qquad
J(x) = S \quad\text{(constant)},
$$

where:

- $\theta(x \ominus x_0)$ is the concatenated **local tangent** of the
  connected variables relative to the marginal's linearization point $x_0$;
- $S$ is the square-root information of the marginal, with $S^\top S = \Lambda_p$;
- $b$ encodes the information vector, $b = -\Lambda_p^+ g_p$ (chapter
  [Marginalization](./marginalization.md)).

The Jacobian is **constant**. That is the defining property of a container:
the factor is a linear model, and a linear model does not need
re-linearizing.

## 2. The manifold-agnostic local log

The factor does not know what a "pose" is, so it does not compute the local
tangent itself. Instead, the caller supplies a closure:

```rust
pub type LocalLogFn = Box<dyn Fn(&[&[f64]], &mut [f64]) + Send + Sync>;
```

The closure receives one parameter slice per block, in block order, and
writes the concatenated local tangent into the output buffer (length = the
sum of the blocks' tangent dimensions, same order). The factor's own
implementation is then a few lines: concatenate, subtract $b$, multiply by
$S$.

This design is what lets the same factor type carry a marginal over poses, a
marginal over biases, or a marginal over a mixture: the geometry is in the
closure, the linear algebra is in the factor.

## 3. Why the container semantics matter

A *re-linearized* prior — a factor that recomputes its Jacobian at the
current estimate — behaves differently in three ways:

1. **It is no longer the same measurement.** The prior was built from a
   specific linearization; re-expanding it about a different point makes the
   difference between the two expansions look like new information.
2. **It accumulates rank in the null space.** The marginal's zero rows
   (the unobservable directions) become small non-zero rows, and the
   estimator's covariance in those directions shrinks as if it had been
   measured.
3. **It is not reproducible.** The factor's contribution to the normal
   equations depends on the path the estimate took, not only on its current
   value.

The container semantics avoid all three by construction: the Jacobian is
fixed at build time.

## 4. The price: staleness

The container is exact only at the linearization point. The factor's own
documentation says it plainly:

> Like GTSAM's linear container, the Jacobian is exact only at the
> linearization point; rebuild the factor (re-marginalize) when the estimate
> drifts far from $x_0$.

So the two mechanisms are complementary:

| | Rebuild (re-marginalize) | First estimate (freeze the point) |
|---|---|---|
| when | the estimate has drifted far from $x_0$ | every solve, between rebuilds |
| cost | a dense factorization of the blanket | one extra `linearize` call per touched block |
| what it fixes | the linear model's accuracy | the double-counting of information |

A sliding-window system that uses both — re-marginalize when the drift
exceeds a threshold, and freeze the linearization points in between — is
consistent and accurate. A system that does neither accumulates the rank
drift; a system that does only the first accumulates the model error.

## 5. Worked example

A window of 7 keyframes, marginalizing the oldest. The prior's blocks are
the 6 surviving states (90 DoF) plus the hosted landmarks. The factor's
`S` is $91\times91$: 90 rows of real information and one zero row for the
global heading.

At the next solve, the estimate moves by $\delta$. The prior's residual
changes — the states have moved, and the residual must reflect that — but its
Jacobian does not: the factor reports the same $S$ it was built with. The
information the prior contributes is therefore the information the marginalized
state actually carried, counted once.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the prior does not whiten internally</span>

`MarginalPriorFactor` is registered like any other factor — and it is *not*
one of the internally-whitening factors. Its `S` is applied to the residual
and the Jacobian by the factor itself, which is why the marginalization
module produces `S` in the first place; but the factor's *whitening
convention* is the ordinary one, so the noise model that accompanies it is
`NoiseModel::null()`. Registering it with a non-null noise model whitens the
prior twice, and the prior's information is squared — a silent, confident,
wrong answer. The crate's registration check catches this, and the check is
worth reading before the registration is written.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — a stale prior keeps a dead linearization point</span>

The factor holds $x_0$ — the ambient parameters of each kept block at the
time of the marginalization. If the window's states are re-initialized (a
re-detection, a relocalization, a reset), the prior's $x_0$ refers to states
that no longer exist, and the residual is measured against a fiction. The
prior must be rebuilt, not merely re-registered.

</div>

<div class="apx-note">
<span class="apx-title">Note — the crate's own API docs</span>

`MarginalPriorFactor` carries the GTSAM `LinearContainerFactor` analogy in
its documentation, and the comparison is exact: a container factor is a
factor whose Jacobian is a constant square-root information matrix, and a
marginal prior is the container form of an eliminated Gaussian. GTSAM's iSAM2
paper (S. Agarwal et al.) is where the term comes from.

</div>
