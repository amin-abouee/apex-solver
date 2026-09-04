# Motion Models

`factors::motion` — what the platform's own dynamics assert, with no sensor
involved.

That makes them unusually cheap and unusually easy to misapply. Each is valid
only while its condition holds, so it is added by whatever detects that
condition: a stationarity detector for ZUPT and ZARU, vehicle knowledge for the
other two. Applied where the assumption is violated they are confidently wrong,
in the way only a near-zero-noise constraint can be.

---

## `ZeroVelocityFactor` — ZUPT

The platform is at rest. Applying this bounds the drift inertial integration
would otherwise accumulate through a stop, at no sensor cost.

**Blocks** $[\,\text{SE}_2(3)\ \text{state}\,]$ — residual 3, Jacobian $3\times9$.

$$
\mathbf{r} = \mathbf{v}_\text{world},
\qquad
\frac{\partial\mathbf{r}}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta},\delta\boldsymbol{\nu})}
= \big[\;0 \;\big|\; 0 \;\big|\; R\;\big]
$$

Velocity is stored in the world frame, so the residual is it verbatim; the
Jacobian block is $R$ because the group's right perturbation moves it by
$R\,\delta\boldsymbol{\nu}$.

---

## `ZeroAngularRateFactor` — ZARU

While at rest, the gyroscope reads its own bias. This is what makes a stop
*observe* the bias rather than merely stop accumulating it, so it complements
ZUPT rather than duplicating it.

**Blocks** $[\,\text{bias}(6)\,]$ — residual 3, Jacobian $3\times6$.

$$
\mathbf{r} = \tilde{\boldsymbol{\omega}} - \mathbf{b}_g,
\qquad
\frac{\partial\mathbf{r}}{\partial\mathbf{b}}
= \big[\;-I_3 \;\big|\; 0_{3\times3}\;\big]
$$

with $\tilde{\boldsymbol{\omega}}$ the gyro reading averaged over the stationary
interval. The accelerometer half is untouched — at rest the accelerometer reads
gravity, not its bias, so a stop says nothing about $\mathbf{b}_a$ without an
attitude reference.

---

## `NonholonomicFactor`

A wheeled vehicle does not slide sideways or leave the ground: in the body
frame its velocity has no lateral or vertical component.

**Blocks** $[\,\text{SE}_2(3)\ \text{state}\,]$ — residual 2, Jacobian $2\times9$.

$$
\mathbf{v}_\text{body} = R^\top\mathbf{v}_\text{world},
\qquad
\mathbf{r} = \begin{bmatrix} v_{\text{body},y} \\ v_{\text{body},z} \end{bmatrix}
$$

$$
\frac{\partial\mathbf{v}_\text{body}}{\partial\delta\boldsymbol{\rho}} = 0,
\qquad
\frac{\partial\mathbf{v}_\text{body}}{\partial\delta\boldsymbol{\theta}} = [\mathbf{v}_\text{body}]_\times,
\qquad
\frac{\partial\mathbf{v}_\text{body}}{\partial\delta\boldsymbol{\nu}} = I
$$

taking rows $y$ and $z$. The $\boldsymbol{\theta}$ block is what makes this
informative about *attitude*: a heading error turns forward motion into apparent
lateral slip, so the constraint corrects the heading, not just the velocity.

Two nearly-free constraints per state, and on a ground vehicle they bound the
lateral drift inertial integration is worst at. They are violated by skidding,
by lifting a wheel, and by any vehicle that can translate sideways.

---

## `PlanarMotionFactor`

A vehicle confined to a horizontal plane: fixed height, no roll, no pitch.

**Blocks** $[\,\text{SE}(3)\ \text{pose}\,]$ — residual 3, Jacobian $3\times6$.

$$
\mathbf{r} =
\begin{bmatrix}
t_z - h \\[2pt]
\mathbf{e}_z^\top R\,\mathbf{e}_x \\[2pt]
\mathbf{e}_z^\top R\,\mathbf{e}_y
\end{bmatrix}
=
\begin{bmatrix}
t_z - h \\ R_{31} \\ R_{32}
\end{bmatrix}
$$

The two tilt rows are the world-$z$ components of the body's $x$ and $y$ axes,
**not** roll and pitch angles. They vanish on exactly the same set, but Euler
extraction introduces a gimbal singularity and a discontinuity that a
least-squares residual should not carry; these rows are smooth everywhere.

$$
\frac{\partial r_0}{\partial\delta\boldsymbol{\rho}} = \mathbf{e}_z^\top R,
\qquad
\frac{\partial r_1}{\partial\delta\boldsymbol{\theta}} = -\mathbf{e}_z^\top R\,[\mathbf{e}_x]_\times,
\qquad
\frac{\partial r_2}{\partial\delta\boldsymbol{\theta}} = -\mathbf{e}_z^\top R\,[\mathbf{e}_y]_\times
$$

with $\partial r_0/\partial\delta\boldsymbol{\theta} = 0$ and
$\partial r_{1,2}/\partial\delta\boldsymbol{\rho} = 0$. The rotation blocks
follow from $R \leftarrow R\,\mathrm{Exp}(\delta\boldsymbol{\theta}) \approx R(I + [\delta\boldsymbol{\theta}]_\times)$
and $R[\delta\boldsymbol{\theta}]_\times\mathbf{e}_a = -R[\mathbf{e}_a]_\times\delta\boldsymbol{\theta}$.

Three constraints per pose, removing the three least-observable degrees of
freedom in a 6-DOF estimate of an indoor or road vehicle.
