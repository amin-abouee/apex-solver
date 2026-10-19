# Pose and Prior Factors

Some factors carry no sensor model at all. They express the *topology* of a
problem: this pose is relative to that one, this variable is anchored here,
this heading agrees with that one. They are the pose-graph's vocabulary, and
they are shared by every front end in the ecosystem. This chapter is the five
families the crate ships.

## 1. The family

<div class="apx-table">
<span class="apx-title">`factors::pose`</span>

| Type | Blocks | Residual |
|---|---|---|
| `BetweenFactor` | two poses | the relative pose between them, against a measurement |
| `PriorFactor` | one pose | a tangent-space anchor at a given value |
| `EuclideanPriorFactor` | one `RN` variable | an ambient-space prior on the parameters |
| `PoseRotationPrior` | one pose | the rotation part only |
| `PoseTranslationPrior` | one pose | the translation part only |
| `Relative4DofFactor` | two poses | position and heading, no roll or pitch |

</div>

## 2. Priors: two different things called "prior"

`PriorFactor` and `EuclideanPriorFactor` look interchangeable and are not.

**`PriorFactor`** is a **tangent-space anchor**: it penalizes the *difference*
between the variable and a reference value, measured on the manifold:

$$
r = \log\!\left(X_{\text{ref}}^{-1} \circ X\right) \in \mathbb{R}^{d},
$$

with $d$ the variable's **DoF**, and the Jacobian
$J = J_{\log}(\cdot)\cdot J_{\text{between}}$. The residual vanishes if
and only if $X = X_{\text{ref}}$, and because the error lives in the tangent
space there is no quaternion double-cover ambiguity and the rotation and
translation are coupled the way the manifold couples them — a rotation of
$0.1$ rad is $0.1$ rad whatever the translation, and a translation of
$0.1$ m is $0.1$ m.

**`EuclideanPriorFactor`** anchors an `Rn` variable's **parameters**: $r = x -
x_{\text{ref}}$ in ambient coordinates. It is the right tool for a vector
variable (a scale, a time offset, a 4-DoF pose node) and the wrong tool for a
pose, where the parameter difference is not the geometric difference. The
registration path *rejects* any other manifold: the factor is `Rn`-only by
construction, not by convention.

The crate's migration chapter records the 1.5-era change: `PriorFactor` became
the tangent-space anchor, generic over the manifold, and the ambient version
was split out as `EuclideanPriorFactor` (`Rn` only). Code written against the
old ambient `PriorFactor` must change — a pose prior and a parameter prior are
not the same measurement.

## 3. `BetweenFactor`

The pose-graph edge: a measurement of the relative pose between two states,
applied as a residual on the *difference*:

$$
r = \log\!\left(T_{\text{meas}}^{-1} \circ T_i^{-1} \circ T_j\right),
$$

in the tangent space of the group. The factor is generic over the group
(`SE3`, `SE2`, `SO3`, `SO2`, even `Rn`) with static dispatch, and the residual
dimensionality follows the group's DoF: 6 for `SE3` (`[v_x, v_y, v_z,
ω_x, ω_y, ω_z]`), 3 for `SE2` (`[dx, dy, dθ]`), 3 for `SO3`.

The measurement carries its own information through the noise model, so the
same factor type serves a tight odometry edge and a loose loop closure — the
weights differ, not the type.

The convention: poses are world-to-camera in the ecosystem's VIO and CV code
(chapter 0), and the factor is indifferent as long as both poses and the
measurement use the same one. What is *not* indifferent is the order of the
two poses in the measurement, and the direction of the residual — both are
properties of the factor's documentation, and both are easy to get backwards.

## 4. `Relative4DofFactor`

A four-degree-of-freedom relative pose: position and heading, with roll and
pitch excluded. This is the factor the SLAM pose graph uses in the VIO stack,
because a visual-inertial odometry observes gravity — its roll and pitch do
not drift — so only position and the heading about gravity are free. A pose
graph that let a loop closure bend roll and pitch too would trade away what
the odometry knows for noise in the loop.

The factor works on **`Rn(4)` nodes** `[x, y, z, ψ]` rather than on poses:
the node's tilt is a *constant of the factor*, so the node's rotation is

$$
R_i = R_z(\psi_i)\,\mathrm{Tilt}_i ,
$$

and the residual is

```
r_t = R_iᵀ (p_j − p_i) − Δp      3 rows, expressed in node i's frame
r_ψ = wrap(ψ_j − ψ_i − Δψ)       1 row
```

Four rows in total. The factor's residual is **raw** — it is registered with
the noise model that whitens `[Δp (m), Δψ (rad)]`, so the two components carry
their own units and their own sigmas.

The VIO stack builds it exactly this way: the odometry edges carry a
`Relative4DofFactor` with the nodes' tilts and a `NoiseModel::from_sigmas`
over the relative position and yaw, the loop closures carry the same factor
type with a looser model and a `HuberLoss`, and the graph's gauge is an
`EuclideanPriorFactor` on the first node.

## 5. Partial priors

`PoseRotationPrior` and `PoseTranslationPrior` anchor *part* of a pose. They
exist for loop-closure initialization, where a candidate match should fix the
heading but leave the position free for the graph to decide (or the reverse).
A prior on the whole pose would over-constrain the very quantities the
closure is supposed to determine.

## 6. Worked example

A pose graph over 300 keyframes: 299 odometry edges (`BetweenFactor` with a
tight noise model) and 20 loop closures (the same type with a looser model),
plus one `PriorFactor` anchoring the first keyframe's position and heading —
the gauge. The first keyframe's fixed coordinates could do the same job more
cheaply, but a prior expresses the same intent in a way that survives a
re-initialization.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a prior on a gauge direction is a weak prior</span>

Anchoring the first pose with a `PriorFactor` at its initial value is a
*soft* gauge: the graph can move the whole trajectory if the data pull it, and
the prior resists only weakly. Fixing the coordinates (chapter
[The Problem](../core/problem.md)) is a hard gauge: the trajectory cannot move
at all. The two are not interchangeable, and a soft gauge on a graph whose
first pose is not well determined is a common source of slow drift.

</div>

<div class="apx-note">
<span class="apx-title">Note — the same factor type serves very different edges</span>

`BetweenFactor` is the odometry edge and the loop edge; only the noise model
differs. That is a feature — one implementation, one set of tests — and a
hazard when a graph's edges are built from a table: the weight is the only
thing distinguishing a tight edge from a loose one, and a table that forgets
the column produces a graph that looks fine and is wrong everywhere.

</div>
