# IMU Factors

An IMU measures angular rate and specific force; the estimator needs the
*relative motion* between two states. The bridge is preintegration: the
measurements between two keyframes are integrated once, into a rotation, a
velocity increment and a position increment, plus the Jacobians that say how
those change when the bias changes. This chapter is that bridge, the four
factor types built on it, and the two modelling choices that decide which one
a system should use.

## 1. Preintegration

`ImuPreintegration` integrates a sample sequence over $[t_0, t_1]$ with the
midpoint rule, accumulating

$$
\Delta R,\quad \Delta\mathbf{v},\quad \Delta\mathbf{p},
\qquad
\mathbf{C}_1 = \int R\,\mathrm{d}\tau,
\qquad
\mathbf{C}_2 = \int\!\!\int R\,\mathrm{d}\tau^2,
$$

together with the first-order bias Jacobians
$\partial\Delta R/\partial b_g$,
$\partial\Delta\mathbf{v}/\partial b_g$,
$\partial\Delta\mathbf{p}/\partial b_g$ and the accelerometer terms
$-\mathbf{C}_1\,\partial b_a$, $-\mathbf{C}_2\,\partial b_a$.

The two $\mathbf{C}$ integrals are what make the accelerometer bias's effect
on the *position* correct: it enters through $\int\!\!\int R$, not through a
linear $t$.

The factor that uses it is the crate's most reused: `apex-vio` and `apex-cv`
both build their IMU factors from it.

## 2. The residual

The interval is compared as a right-minus on the `SE23` group, with gravity
folded into the state so the comparison is a plain group difference:

$$
gc_i = \left(\mathbf{t}_i + \mathbf{v}_i\Delta t - \tfrac12\mathbf{g}\Delta t^2,\;
\mathbf{v}_i - \mathbf{g}\Delta t,\; R_i\right),
\qquad
\mathbf{r} = (gc_i^{-1}\circ X_j)\boxminus\left(\Delta\boxplus\xi(\delta b)\right),
$$

with the first-order bias correction

$$
\xi(\delta b) = \begin{bmatrix}
J^p_{b_g}\delta b_g - \mathbf{C}_2\delta b_a\
-J^\alpha_{b_g}\delta b_g\
J^v_{b_g}\delta b_g - \mathbf{C}_1\delta b_a
\end{bmatrix}.
$$

## 3. Two groups

<div class="apx-table">
<span class="apx-title">`se23` and `sgal3`</span>

| | `se23` | `sgal3` |
|---|---|---|
| state | $(R, t, v)$ | $(R, t, v, s)$ — a time coordinate |
| interval | given by the keyframes | an estimated quantity |
| use when | the camera and IMU are synchronized | sensor time-offset or rolling-shutter must be calibrated |

</div>

The crate's IMU module docs make the choice a two-question decision: *which
group* (`se23` unless the time offset is a thing you must estimate) and
*combined or not* (below). Both groups ship `ImuFactor` and
`CombinedImuFactor` with the same residual and the same bias correction, so
switching groups does not change anything else.

## 4. Combined or not

**`ImuFactor`** shares **one** bias variable across the interval and leaves
the bias's evolution to a separate `bias_random_walk` edge. The factor's
residual is 9 rows: $[\rho, \theta, \nu]$.

**`CombinedImuFactor`** takes a bias **per keyframe** and embeds the random
walk in its trailing six rows. The residual is 15 rows:
$[\rho, \theta, \nu, \Delta b_g, \Delta b_a]$.

The module's documentation is explicit about the consequence: "Using both at
once would count that uncertainty twice." The trade is graph size — the
combined factor saves a factor per interval — against the coupling: a
combined factor's two bias blocks and two state blocks share rows, which is
precisely what makes the *marginalization* of a window's first keyframe
ineligible for the Schur path (chapter
[The Schur Complement](../linalg/schur.md)).

## 5. Gravity and scale

Two wrapper factors extend the IMU factor with additional unknowns:

**`GravityFrameImuFactor`** adds a three-degree-of-freedom gravity direction
$R_g \in SO(3)$ to the factor — gravity is $R_g(0,0,-g)$ — and evaluates the
inner factor on the states expressed in the gravity frame,
$X' = T_g^{-1}X$ with $T_g = SE_2(3)(R_g, 0, 0)$. Its columns are

$$
\frac{\partial \mathbf{r}}{\partial \boldsymbol{\psi}}
= \sum_{X'}\frac{\partial \mathbf{r}}{\partial X'}
\left(-\mathrm{Ad}(X'^{-1})\right)_{[:,\,\theta]} ,
\qquad
\boldsymbol{\psi} = (0,\psi,0),
$$

the gravity rotation's effect on the interval residual, propagated through the
states' adjoints. The LIO stack uses it: a map built from a still start
inherits that start's tilt — at rest, an accelerometer bias across gravity is
indistinguishable from tilt — so the world frame stays the map's and gravity's
direction *in it* is a variable. Rotating the rig makes the tilt observable
and the bias separable.

**One warning from the source**: the yaw of $R_g$ is **unobservable**, because
gravity is invariant to it. It must be held or priored.

**`ScaleGravityImuFactor`** adds a **scale** variable on top: the factor sees
$X' = T_g^{-1}S(s)X$ with $S(s)(R, p, v) = (R, s\,p, s\,v)$, where $s$ is a
one-dimensional `Rn` variable. A monocular visual system has no scale *and*
starts in a frame whose vertical is wherever the first image put it; making
the scale and the gravity direction two variables means a global scale change
costs the solver **one coordinate** instead of every pose and every depth.
This is the DM-VIO construction (von Stumberg & Cremers, 2022), and it is
how a simulation-to-reality scale is estimated.

## 6. Whitening

The IMU factors whiten **internally**: the preintegration's information matrix
is part of the measurement, and the factor applies it itself. They must
therefore be registered with `NoiseModel::null()` — the crate rejects the
other combination at registration (chapter [Noise Models](../core/noise.md)).

## 7. Worked example

A keyframe pair 100 ms apart, 10 IMU samples. `ImuPreintegration::try_new`
produces $\Delta R$, $\Delta\mathbf{v}$, $\Delta\mathbf{p}$ at the reference
bias, and the factor's residual compares the state's implied motion against
it, with the bias correction for the difference between the current and
reference biases. The 15 rows of a `CombinedImuFactor` land in the graph as
one block over four variables: $[X_i, b_i, X_j, b_j]$.

## 8. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the interval's IMU samples must actually cover the interval</span>

Preintegration zero-order-holds the nearest sample at each boundary, and a
buffer lying wholly outside $[t_0, t_1]$ would be extrapolated across the
whole span and returned as if it had been measured. The crate rejects that
case (`EmptyIntegration`) rather than integrating nothing. A *partially*
covered interval is not rejected — the missing part is held — which is why a
window with gaps produces a factor that is quietly wrong over the gap.

</div>

<div class="apx-note">
<span class="apx-title">Note — the factor's blocks decide the marginalization's fate</span>

The combined factor's four blocks over two states is what makes a window's
first keyframe ineligible for Schur elimination: the eliminated block is not
block-diagonal. That is a *design* consequence, not a bug, and it is why the
crate has a separate dense marginalization path. Choosing between `ImuFactor`
and `CombinedImuFactor` is therefore also a choice about how the window will be
marginalized.

</div>
