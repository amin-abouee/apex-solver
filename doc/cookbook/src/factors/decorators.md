# Decorators: Aggregate, Held, Offset, Projected

Four factors in the crate do not describe a measurement at all: they wrap
another factor and change what the graph sees. They are the crate's escape
hatch for the situations where a factor's *shape* — how many blocks it
touches, how many rows it produces, which directions it constrains — needs to
change without changing the geometry. This chapter is all four.

## 1. `AggregateFactor` — many factors, one block

A sliding-window LiDAR adjustment has thousands of plane factors over ten
states. The stacked Jacobian has tens of thousands of rows, and forming
$J^\top J$ from it dominates the solve. The aggregate factor compresses them.

At a linearization point, the sum of many factors is fully described by three
quantities:

$$
H = \sum_i J_i^\top J_i,\qquad
g = \sum_i J_i^\top r_i,\qquad
c = \sum_i \lVert r_i \rVert^2 .
$$

With $H = U\Lambda U^\top$, the compressed factor reports

$$
J' = \begin{bmatrix}\Lambda^{1/2}U^\top \\ 0\end{bmatrix},
\qquad
r' = \begin{bmatrix}\Lambda^{-1/2}U^\top g \\ \sqrt{c - g^\top H^+ g}\end{bmatrix},
$$

so that $J'^\top J' = H$, $J'^\top r' = g$ and $\lVert r'\rVert^2 = c$. The
Gauss–Newton step and the cost — the things Levenberg–Marquardt accepts or
rejects a step on — are exactly the parts' sum, from $n+1$ rows.

Two conditions come with it. Each part's residual is taken as **already
whitened** (as with `NoiseModel::null()`), and **without a robust loss** — a
kernel cannot be applied to a sum, and the compression would otherwise change
the objective. And the compression is exact only at the linearization point:
the parts are re-linearized every iteration, and the compressed factor
re-derives $H$, $g$, $c$ from them.

`AggregatePart` is the piece type: one factor plus its block list.

## 2. `HeldBlocks` — some blocks pinned

A calibrated extrinsic, a frozen map anchor, a reference pose outside the
optimization window: a block the problem should not estimate is often simplest
to bake in than to add as a variable and pin.

`HeldBlocks<F>` wraps a factor and stores the held blocks' *parameter values*.
The wrapped factor sees them at their original positions — the factor's own
code does not change at all — and only the free blocks' Jacobian columns are
reported. The result is a factor with fewer columns, and the held values
never move.

The subtlety: the held block's *parameters* are frozen, but the factor still
needs the block to compute its residual and the held block's contribution to
the Jacobian is simply not reported. A factor that had a Jacobian coupling a
held block to a free one therefore has a *different* Jacobian when held: the
coupling term is dropped, and the residual is evaluated with the held value
substituted. The wrapper does this substitution; the factor does not know.

## 3. `RightOffset` — one block evaluated at a fixed offset

A measurement taken a few milliseconds after its state — a camera image just
after a LiDAR sweep — is often tied to that state through the motion the IMU
measured in between, held constant. `RightOffset<F, G>` evaluates the wrapped
factor's block $b$ at $X\cdot\Delta$ instead of $X$.

The Jacobian follows from the group law, and the crate states it exactly:

$$
X\,\mathrm{Exp}(\xi)\,\Delta = (X\Delta)\,\mathrm{Exp}\big(\mathrm{Ad}(\Delta^{-1})\xi\big)
\quad\Longrightarrow\quad
\frac{\partial r}{\partial \xi} = \frac{\partial r}{\partial \xi'}\,\mathrm{Ad}(\Delta^{-1}) .
$$

So the block's Jacobian columns are the inner ones multiplied by
$\mathrm{Ad}(\Delta^{-1})$ — a constant matrix, computed once at
construction.

## 4. `ProjectedFactor` — the Jacobian projected onto a subspace

In a long corridor a LiDAR scan cannot tell where along the corridor it is:
its information has a near-zero eigenvalue in that direction, and whatever it
says there is noise and mismatches. `ProjectedFactor` projects the factor's
Jacobian onto the well-constrained directions,

$$
J' = J P,\qquad P = I - \sum_k v_k v_k^\top ,
$$

where $v_k$ are the degenerate directions. The **residual is unchanged**, so
the cost is unchanged; but the factor's pull and its information along $v_k$
are removed. A Gauss–Newton step there is then decided by the other factors
— the IMU, the priors, the camera.

This is a modelling decision with a strong flavour: the factor is being told
to stop claiming to know something. It is right when the direction is truly
unobservable, and wrong when the degeneracy is a symptom of a bad match.

## 5. How the wrappers compose

A wrapping factor does not rebuild its inner factor's arithmetic; it hands
the inner factor a *view* of the normal-equation blocks and adjusts the
contribution on the way out. `NormalBlocks` provides the two views a wrapper
needs:

- `remapped` — when the inner factor's blocks are a subset or a permutation
  of the wrapper's own (the held blocks are simply dropped);
- `transformed` — when the wrapper's Jacobian of a block is the inner one
  right-multiplied, $J T$; then $H_{ab} \mathrel{+}= T_a^\top M T_b$ and
  $\mathbf{g}_a \mathrel{+}= T_a^\top \mathbf{v}$.

The `RightOffset` wrapper is exactly the second case, with $T = \mathrm{Ad}(\Delta^{-1})$;
`HeldBlocks` is the first. This is why the wrappers are cheap: they change
where a number is written, not what it is.

## 6. Worked example

A LiDAR window with 10 states and 4 000 plane matches. Registering the
matches individually gives 4 000 residual blocks over 10 states; the
aggregate factor gives **one** block, whose compressed Jacobian has $n+1$
rows for $n$ summed DoF. The solve's linearization cost is now proportional
to the state count, not the match count — the difference between a window
that refines in 40 ms and one that refines in 4 s.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the aggregate factor drops the robust loss</span>

A kernel is applied per block, and the aggregate factor's block is the
*sum* of many parts. The compression preserves the Gauss–Newton step and the
cost of the plain sum; it does not preserve a robust objective over the
parts. A window with 30 % mismatched plane matches therefore needs the
mismatches removed *before* aggregation (by a gate or an inlier mask in the
consumer), not after — the kernel cannot be applied to the compressed factor
and recover the parts' individual weights.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — a projected Jacobian is a claim about the world</span>

`ProjectedFactor` removes a direction from the graph's information. If the
direction was merely *weakly* observed rather than unobserved, the projection
throws away real measurements, and the answer in that direction comes from
whichever factor is left. The projection is a statement that the direction is
unobservable, and it should be made where the degeneracy is diagnosed — with
the evidence — not as a default.

</div>

<div class="apx-note">
<span class="apx-title">Note — decorators compose</span>

`HeldBlocks<RightOffset<ProjectedFactor<...>>>` is legal: each wrapper changes
one aspect of the factor's presentation and passes the rest through. The
order matters, though: holding a block *after* an offset is holding the
offset's value, and holding it *before* means the offset is applied to a
frozen value. The crate's wrappers are designed so that the composition reads
in the order the semantics apply.

</div>

<div class="apx-paper">
<span class="apx-title">Papers — the degeneracy projection</span>

Zhang, Kaess & Singh, "On degeneracy of optimization-based state estimation
problems", *ICRA* 2016 — the observation-remapping idea: a degenerate
measurement's solution contribution should be removed rather than trusted.
The crate's `ProjectedFactor` implements it as a projection of the Jacobian.
</div>
