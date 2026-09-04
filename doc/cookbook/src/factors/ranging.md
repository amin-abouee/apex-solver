# Range & Bearing

`factors::ranging` — distance and direction measurements: UWB, radar, sonar,
acoustic angle-of-arrival, and bearing-only landmark observation.

Bearing lives here rather than under `visual` because a direction measurement
is not camera-specific — and keeping the two bearing residuals adjacent is what
keeps them on one SE(3) convention.

**Convention.** Poses are body-in-world, so the body-frame direction to a world
landmark is

$$
\mathbf{p}_\text{body} = R^\top\!\left(\mathbf{p}_j - \mathbf{t}_i\right)
$$

Getting this backwards ($R$ instead of $R^\top$) produces a residual that is
self-consistent with its own Jacobian, so finite differences pass and only a
hand-computed value catches it.

---

## `PosePoseRangeFactor`

Distance between two poses' origins — a UWB or acoustic baseline.

**Blocks** $[T_i,\ T_j]$ — residual 1, Jacobian $1\times12$.

$$
\mathbf{q} = \mathbf{t}_i - \mathbf{t}_j,
\qquad
d = \lVert\mathbf{q}\rVert,
\qquad
r = d - \tilde{d}
$$

$$
\frac{\partial r}{\partial \delta\boldsymbol{\rho}_i} = \hat{\mathbf{q}}^\top R_i,
\qquad
\frac{\partial r}{\partial \delta\boldsymbol{\rho}_j} = -\hat{\mathbf{q}}^\top R_j,
\qquad
\frac{\partial r}{\partial \delta\boldsymbol{\theta}} = 0,
\qquad
\hat{\mathbf{q}} = \frac{\mathbf{q}}{d}
$$

The rotation columns are exactly zero: a range between origins carries no
attitude information. Below $d < 10^{-12}$ the direction is undefined and the
Jacobian is zeroed rather than divided by zero.

---

## `PosePointRangeFactor`

Distance from a pose to a landmark.

**Blocks** $[T,\ \mathbf{p}]$ — residual 1, Jacobian $1\times9$.

$$
r = \lVert\mathbf{t} - \mathbf{p}\rVert - \tilde{d},
\qquad
\frac{\partial r}{\partial\delta\boldsymbol{\rho}} = \hat{\mathbf{q}}^\top R,
\qquad
\frac{\partial r}{\partial\mathbf{p}} = -\hat{\mathbf{q}}^\top
$$

One constraint per measurement, along the line of sight only — trilateration
needs three non-collinear anchors.

---

## `BearingFactor`

Direction-only observation, on the unit sphere $S^2$.

**Blocks** $[T,\ \mathbf{p}]$ — residual 2, Jacobian $2\times9$. Whitens
internally.

### Error

$$
\mathbf{p}_\text{body} = R^\top(\mathbf{p} - \mathbf{t}),
\qquad
\mathbf{n} = \frac{\mathbf{p}_\text{body}}{\lVert\mathbf{p}_\text{body}\rVert},
\qquad
\mathbf{r} = W\,E^\top\left(\mathbf{n} - \tilde{\mathbf{n}}\right) \in \mathbb{R}^2
$$

where $E \in \mathbb{R}^{3\times2}$ is an orthonormal basis for the tangent
plane at $\tilde{\mathbf{n}}$. The projection matters: the raw difference
$\mathbf{n} - \tilde{\mathbf{n}}$ is a 3-vector confined to a 2-manifold, so
using it directly would give a structurally rank-2 residual with three rows —
a singular information block.

### Jacobian

With $d = \lVert\mathbf{p}_\text{body}\rVert$ and the normalization derivative
$\dfrac{\partial\mathbf{n}}{\partial\mathbf{p}_\text{body}} = \dfrac{1}{d}\left(I - \mathbf{n}\mathbf{n}^\top\right)$,

$$
\frac{\partial\mathbf{p}_\text{body}}{\partial\delta\boldsymbol{\rho}} = -I,
\qquad
\frac{\partial\mathbf{p}_\text{body}}{\partial\delta\boldsymbol{\theta}} = [\mathbf{p}_\text{body}]_\times,
\qquad
\frac{\partial\mathbf{p}_\text{body}}{\partial\mathbf{p}} = R^\top
$$

all pre-multiplied by $W E^\top \frac{1}{d}(I - \mathbf{n}\mathbf{n}^\top)$.

---

## `BearingRangeFactor`

Bearing and range together — a radar or sonar return, or a camera plus depth.

**Blocks** $[T,\ \mathbf{p}]$ — residual 4, Jacobian $4\times9$. Takes an
external `NoiseModel`.

$$
\mathbf{r} =
\begin{bmatrix}
\mathbf{b} - \tilde{\mathbf{b}} \\[2pt]
d - \tilde{d}
\end{bmatrix},
\qquad
\mathbf{b} = \frac{R^\top(\mathbf{p} - \mathbf{t})}{d},
\qquad
d = \lVert\mathbf{p} - \mathbf{t}\rVert
$$

The three bearing rows are the raw direction difference rather than a
tangent-plane projection, so they are structurally rank 2 — give the bearing
rows a common $\sigma$ and let the noise model handle it, or use
`BearingFactor` plus `PosePointRangeFactor` if you need a full-rank
information block.

Bearing rows share `BearingFactor`'s derivatives; the range row is
`PosePointRangeFactor`'s.
