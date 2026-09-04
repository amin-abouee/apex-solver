# IMU

`factors::imu` — inertial preintegration. Two factors per group and nothing
else, so a choice here is two questions rather than a catalogue.

## Preintegration

Integrating raw IMU samples between keyframes would re-integrate the whole
interval every time the state changes. Preintegration instead accumulates the
measurements into a *relative* quantity that does not depend on the
states — only on the bias linearization point — so a keyframe pair can be
re-evaluated in constant time.

`ImuPreintegration::new(measurements, params, t0, t1, &bias_ref)` accumulates
the delta $\Delta$, its $15\times15$ covariance, and the bias-correction
Jacobians $\partial\Delta/\partial\mathbf{b}$. Everything it produces is
**gravity-free**; the factors reinstate gravity when they build the
gravity-corrected state.

## Two axes

**Which group?** `se23` models a keyframe as $(R, \mathbf{t}, \mathbf{v})$;
`sgal3` adds a time coordinate, $(R, \mathbf{t}, \mathbf{v}, s)$, making the
inter-keyframe interval an estimated quantity — pick it for time-offset or
rolling-shutter calibration, `se23` otherwise.

**Combined or not?** This is the split GTSAM draws between `ImuFactor` and
`CombinedImuFactor`, and it decides the residual dimension.

| | `ImuFactor` | `CombinedImuFactor` |
|---|---|---|
| `se23` | 9D, `(SE23, SE23, bias)` | 15D, `(SE23, bias, SE23, bias)` |
| `sgal3` | 10D, `(SGal3, SGal3, bias)` | 16D, `(SGal3, bias, SGal3, bias)` |

`ImuFactor` shares one bias variable across the interval and leaves its
evolution to a separate edge; `CombinedImuFactor` takes a bias per keyframe and
embeds the random walk in six extra residual rows. **Doing both counts that
uncertainty twice.**

Weighting follows the same split: the shared-bias form uses a $9\times9$
information built from measurement noise alone, the combined form the full
$15\times15$ including the random walk.

A keyframe is a **single** variable on the group, not separate pose and
velocity blocks — the optimizer's update is then a group right-plus, and the
pose/velocity coupling inertial integration produces is the group's job. The
practical consequence is that aiding measurements attach to that state: an
SE(3)-only factor cannot connect to an `SE23` variable, while
`PriorFactor<SE23>` and `MarginalPriorFactor` are generic and do.

## Error

$$
\begin{aligned}
\text{gc}_i &= \mathrm{SE}_2(3)\!\left(\mathbf{p}_i + \mathbf{v}_i\Delta t - \tfrac12 \mathbf{g}\,\Delta t^2,\;\; \mathbf{v}_i - \mathbf{g}\,\Delta t,\;\; R_i\right) \\
\text{predicted} &= \text{gc}_i^{-1} \circ \text{state}_j \\
\mathbf{r} &= \text{predicted} \boxminus \big(\Delta \boxplus \mathbf{c}(\mathbf{b})\big) \in \mathbb{R}^9
\end{aligned}
$$

Folding gravity into the state rather than the residual is what makes the
comparison a plain group right-minus. The bias correction $\mathbf{c}(\mathbf{b})$
is the first-order update of the preintegrated delta for a bias away from its
linearization point,

$$
\mathbf{c} = \left(\tfrac{\partial\Delta\mathbf{p}}{\partial\mathbf{b}_g}\delta\mathbf{b}_g - C_{\!\iint}\delta\mathbf{b}_a,\;\;
-\tfrac{\partial\Delta\boldsymbol{\alpha}}{\partial\mathbf{b}_g}\delta\mathbf{b}_g,\;\;
\tfrac{\partial\Delta\mathbf{v}}{\partial\mathbf{b}_g}\delta\mathbf{b}_g - C_{\!\int}\delta\mathbf{b}_a\right)
$$

The combined form appends $\mathbf{b}_i - \mathbf{b}_j$ (6 rows).

## Jacobian

All four factors are thin wrappers over one `evaluate`, which returns the
residual and three tangent-space blocks. Write
$J_{\boxminus}^{\text{pred}}, J_{\boxminus}^{\Delta}$ for the two Jacobians
`right_minus` reports and $J_{\boxplus}$ for the tangent Jacobian of
`right_plus`. Then

$$
\frac{\partial\mathbf{r}}{\partial\,\text{gc}_i}
= J_{\boxminus}^{\text{pred}}\cdot\big({-}\mathrm{Ad}_{\text{predicted}^{-1}}\big),
\qquad
\frac{\partial\mathbf{r}}{\partial\,\text{state}_j} = J_{\boxminus}^{\text{pred}},
\qquad
\frac{\partial\mathbf{r}}{\partial\mathbf{b}}
= J_{\boxminus}^{\Delta}\,J_{\boxplus}\,\frac{\partial\mathbf{c}}{\partial\mathbf{b}}
$$

The $-\mathrm{Ad}$ comes from differentiating $X^{-1}Y$ with respect to the
left argument: $(X\,\mathrm{Exp}(\delta))^{-1}Y
= \mathrm{Exp}(-\mathrm{Ad}_{(X^{-1}Y)^{-1}}\delta)\,X^{-1}Y$.

The bias block expands to

$$
\frac{\partial\mathbf{c}}{\partial\mathbf{b}} =
\begin{bmatrix}
\partial\Delta\mathbf{p}/\partial\mathbf{b}_g & -C_{\!\iint} \\[2pt]
-\partial\Delta\boldsymbol{\alpha}/\partial\mathbf{b}_g & 0 \\[2pt]
\partial\Delta\mathbf{v}/\partial\mathbf{b}_g & -C_{\!\int}
\end{bmatrix} \in \mathbb{R}^{9\times6}
$$

in the $[\boldsymbol{\rho},\boldsymbol{\theta},\boldsymbol{\nu}]$ tangent order.

The lift from $\text{state}_i$ to the gravity-corrected state is a **constant**
matrix, independent of the linearization point, because $\text{gc}_i$ shares
$\text{state}_i$'s rotation and shifts the rest by gravity terms only:

$$
\frac{\partial\,\text{gc}_i}{\partial\,\text{state}_i} =
\begin{bmatrix} I & 0 & \Delta t\,I \\ 0 & I & 0 \\ 0 & 0 & I \end{bmatrix},
\qquad
(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta},\delta\boldsymbol{\nu})
\mapsto (\delta\boldsymbol{\rho} + \Delta t\,\delta\boldsymbol{\nu},\ \delta\boldsymbol{\theta},\ \delta\boldsymbol{\nu})
$$

The combined form appends the bias-walk rows, whose Jacobian is constant:

$$
\frac{\partial(\mathbf{b}_i - \mathbf{b}_j)}{\partial\mathbf{b}_i} = I_6,
\qquad
\frac{\partial(\mathbf{b}_i - \mathbf{b}_j)}{\partial\mathbf{b}_j} = -I_6
$$

Every $J$ above is reported by the group's own `right_plus`, `right_minus` and
`adjoint`; none of it is hand-derived.

## Bias evolution

The non-combined factors need a companion edge, which
`factors::imu::bias` builds as a `BetweenFactor<Rn>` with zero measurement:

$$
\mathbf{b}_j = \mathbf{b}_i + \mathbf{w}, \qquad
\mathbf{w} \sim \mathcal{N}\!\left(0,\ \mathrm{diag}(\sigma_{gw}^2, \sigma_{aw}^2)\,\Delta t\right)
$$

```rust
use apex_solver::factors::imu::{se23, bias_random_walk, bias_random_walk_noise};

problem.add_residual_block(
    &[state_i, state_j, bias],
    Box::new(se23::ImuFactor::new(preintegration)),
    None,
);
problem.add_residual_block_with_noise(
    &[bias, bias_next],
    Box::new(bias_random_walk()),
    None,
    bias_random_walk_noise(&params, dt)?,
);
```

An initial bias prior (an `EuclideanPriorFactor` on $\mathbb{R}^6$) is required
for observability in every configuration.

## SGal(3) and the time row

SGal(3)'s tangent carries a time coordinate, so the residual gains a row
$(t_j - t_i) - \Delta t$, weighted by $1/\sigma_t$ (`with_time_sigma`, default
100 µs). That row is the reason to choose SGal(3): it makes the interval a
quantity the optimizer can move.

> **Known limitation.** SGal(3)'s group law is
> $\mathbf{t} = R_1(\mathbf{t}_2 + s_1\mathbf{v}_2) + \mathbf{t}_1$ — the left
> operand's time couples the right operand's velocity into translation. The
> residual composes $\text{gc}_i^{-1}\circ\text{state}_j$, so it depends on the
> absolute $s_i$, while the preintegrated delta corresponds to $s_i = 0$. A
> **single interval** with $s_i = 0,\ s_j = \Delta t$ is correct; chaining these
> factors across keyframes on a common absolute clock is not. Prefer the
> `se23` factors for multi-keyframe chains until the residual is re-derived in
> terms of $\Delta s = s_j - s_i$.

