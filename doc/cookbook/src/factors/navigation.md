# GNSS & Navigation

`factors::navigation` — satellite positioning and the auxiliary sensors that
bound inertial drift.

---

## `GpsFactor` — position fix with a lever arm and frame transform

The receiver reports position in *its* frame, and the antenna is not at the
body origin. Both matter: the frame transform is often unsurveyed, and the
lever arm couples attitude into the position measurement.

**Blocks** $[T_{WS},\ T_{GW}]$ — residual 3, Jacobian $3\times12$. Whitens
internally.

### Error

$$
\mathbf{t}_{A}^{W} = \mathbf{t}_{WS} + C_{WS}\,\mathbf{r}_{SA},
\qquad
\hat{\mathbf{z}} = C_{GW}\,\mathbf{t}_{A}^{W} + \mathbf{t}_{GW},
\qquad
\mathbf{r} = W\left(\tilde{\mathbf{z}} - \hat{\mathbf{z}}\right)
$$

with $\mathbf{r}_{SA}$ the sensor-to-antenna lever arm in the sensor frame.

### Jacobian

$$
\frac{\partial\mathbf{e}}{\partial\delta\boldsymbol{\rho}_{WS}} = -C_{GW}C_{WS},
\qquad
\frac{\partial\mathbf{e}}{\partial\delta\boldsymbol{\theta}_{WS}} = C_{GW}C_{WS}\,[\mathbf{r}_{SA}]_\times
$$

$$
\frac{\partial\mathbf{e}}{\partial\delta\boldsymbol{\rho}_{GW}} = -C_{GW},
\qquad
\frac{\partial\mathbf{e}}{\partial\delta\boldsymbol{\theta}_{GW}} = C_{GW}\,[\mathbf{t}_{A}^{W}]_\times
$$

then scaled by $W$. Note the attitude column is proportional to the lever arm:
with $\mathbf{r}_{SA} = 0$ a position fix says nothing about orientation.

> **Observability.** Each pose owns three translation degrees of freedom and
> receives one 3D fix, so the poses can absorb *any* frame transform — with
> $T_{GW}$ fully free the graph has a six-dimensional null space and a
> zero-cost family of solutions. Survey the frame rotation (or anchor enough
> poses) before estimating it.

---

## `GpsAsyncFactor` — a fix that lands between keyframes

GNSS epochs rarely coincide with keyframes. This propagates the keyframe state
forward with an IMU preintegration to the fix timestamp, then applies the same
position error.

**Blocks** $[T_{WS},\ \text{speed-and-bias}(9),\ T_{GW}]$ — residual 3,
Jacobian $3\times21$. Whitens internally.

### Error

$$
\mathbf{p}_g = \mathbf{p}_k + \mathbf{v}_k\Delta t
\;-\; \tfrac12\,\mathbf{g}\,\Delta t^2
\;+\; C_{WS,k}\,\Delta\mathbf{p}_\text{body},
\qquad
C_{WS,g} = C_{WS,k}\,\Delta R
$$

$$
\Delta\mathbf{p}_\text{body} = \Delta\mathbf{p}_\text{nom}
+ \frac{\partial\Delta\mathbf{p}}{\partial\mathbf{b}_g}\delta\mathbf{b}_g
+ C_{\!\iint}\,\delta\mathbf{b}_a
$$

then $\mathbf{t}_A^W = \mathbf{p}_g + C_{WS,g}\mathbf{r}_{SA}$ and the same
projection into the GNSS frame as `GpsFactor`.

The $-\tfrac12\mathbf{g}\Delta t^2$ term is easy to lose: every preintegrated
quantity is **gravity-free**, so gravity has to be reinstated here exactly as
the IMU factors do when they build their gravity-corrected state. Omitting it
biases the predicted antenna position by 0.44 m after only 0.3 s, growing
quadratically.

---

## `GpsVelocityFactor`

**Blocks** $[\mathbf{v}]$ — residual 3, Jacobian $3\times3 = I$.

$$
\mathbf{r} = \mathbf{v} - \tilde{\mathbf{v}}
$$

A Doppler-derived velocity in the local tangent frame, on a plain
$\mathbb{R}^3$ variable. For a *raw* range-rate along a satellite line of sight
use `DopplerFactor` instead.

---

## `PseudorangeFactor`

Raw ranging to one satellite, before any position solution has been formed.

**Blocks** $[\mathbf{x},\ b]$ — receiver position and clock bias, the latter in
**metres** (already range-equivalent). Residual 1, Jacobian $1\times4$.

$$
r = \lVert\mathbf{s} - \mathbf{x}\rVert + b - \tilde{\rho},
\qquad
\frac{\partial r}{\partial\mathbf{x}} = -\hat{\mathbf{f}}^\top,
\qquad
\frac{\partial r}{\partial b} = 1,
\qquad
\hat{\mathbf{f}} = \frac{\mathbf{s} - \mathbf{x}}{\lVert\mathbf{s} - \mathbf{x}\rVert}
$$

The position and clock columns are separable only when the line-of-sight
directions span three dimensions — the reason a fix needs four satellites, and
why poor geometry is penalized rather than silently accepted.

---

## `DopplerFactor`

Range rate along the same line of sight.

**Blocks** $[\mathbf{x},\ \mathbf{v}]$ — residual 1, Jacobian $1\times6$.

$$
r = \hat{\mathbf{f}}^\top(\mathbf{v}_s - \mathbf{v}_r) - \tilde{\dot{\rho}},
\qquad
\frac{\partial r}{\partial\mathbf{v}_r} = -\hat{\mathbf{f}}^\top
$$

The position column carries the derivative of $\hat{\mathbf{f}}$ itself,

$$
\frac{\partial \hat{\mathbf{f}}}{\partial\mathbf{x}}
= -\frac{1}{\lVert\mathbf{s}-\mathbf{x}\rVert}\left(I - \hat{\mathbf{f}}\hat{\mathbf{f}}^\top\right)
$$

which is small at typical satellite ranges — velocity is what this measurement
really informs.

---

## `BarometricFactor`

**Blocks** $[T,\ b]$ — pose and a slowly-varying altimeter bias. Residual 1,
Jacobian $1\times7$.

$$
r = (z_\text{pose} + b) - \tilde{z},
\qquad
\frac{\partial r}{\partial\delta\boldsymbol{\rho}} = \mathbf{e}_3^\top R,
\qquad
\frac{\partial r}{\partial b} = 1
$$

The pose block is the third row of $R$, not $\mathbf{e}_3^\top$, because the
right perturbation moves the origin by $R\,\delta\boldsymbol{\rho}$.

Height and bias are only separable if the bias is otherwise constrained — give
it a prior and a random walk, or the pair drifts together.

---

## `AttitudeFactor`

Constrains rotation so a known world direction maps onto the direction the
sensor measures in the body frame — gravity from an accelerometer at rest, or
the field from a magnetometer.

**Blocks** $[T]$ — residual 3, Jacobian $3\times6$.

$$
\mathbf{r} = R\,\mathbf{d}_\text{world} - \tilde{\mathbf{d}}_\text{body},
\qquad
\frac{\partial\mathbf{r}}{\partial\delta\boldsymbol{\rho}} = 0,
\qquad
\frac{\partial\mathbf{r}}{\partial\delta\boldsymbol{\theta}} = -R\,[\mathbf{d}_\text{world}]_\times
$$

The Jacobian is rank 2: rotation about the reference direction leaves it
unchanged, so **one such factor pins two of the three rotational degrees of
freedom**. Add a gravity factor and a magnetometer factor for a full AHRS.
