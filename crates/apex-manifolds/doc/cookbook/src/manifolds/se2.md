# SE(2) — 2D Rigid Transforms

The special Euclidean group $SE(2)$ is the group of planar rigid-body motions
(rotation + translation). It has $\text{DOF} = 3$ and is stored as
$[x, y, \theta]$ ($\text{REP\_SIZE} = 3$).

$$
SE(2) = \left\{\, \mathbf{T} = \begin{bmatrix} \mathbf{R}(\theta) & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \;\middle|\; \mathbf{R}(\theta) \in SO(2),\ \mathbf{t} \in \mathbb{R}^2 \,\right\}.
$$

## Representation

| Storage | Layout |
|---|---|
| Group `SE2` | `SVector<f64, 3>` = $[x, y, \theta]$ |
| Tangent `SE2Tangent` | `Vector3<f64>` = $[\rho_x, \rho_y, \theta]$ |
| Lie algebra | $3\times 3$ matrix $\begin{bmatrix} [\theta]_\times & \boldsymbol{\rho} \\ \mathbf{0}^\top & 0 \end{bmatrix}$ |

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $[0, 0, 0]$ | `SE2::identity()` |
| Composition | $\mathbf{T}_1 \mathbf{T}_2 = (\mathbf{R}_1\mathbf{R}_2,\ \mathbf{R}_1\mathbf{t}_2 + \mathbf{t}_1)$ | `compose` |
| Inverse | $(\mathbf{R}^\top,\ -\mathbf{R}^\top \mathbf{t})$ | `inverse` |
| Action | $\mathbf{R}\,\mathbf{v} + \mathbf{t}$ | `act` |

## Lie Algebra (Hat / Vee)

$$
\boldsymbol{\tau}^\wedge =
\begin{bmatrix} 0 & -\theta & \rho_x \\ \theta & 0 & \rho_y \\ 0 & 0 & 0 \end{bmatrix},
\qquad
\boldsymbol{\tau} = (\rho_x, \rho_y, \theta).
$$

<a id="exp"></a>
## Exponential Map

The rotation exponentiates directly; the translation is mapped through the
$SE(2)$ **$\mathbf{V}$ matrix**. With $a = \dfrac{\sin\theta}{\theta}$ and
$b = \dfrac{1 - \cos\theta}{\theta}$,

$$
\mathbf{V}(\theta) = \begin{bmatrix} a & -b \\ b & a \end{bmatrix},
\qquad
\mathrm{Exp}(\boldsymbol{\tau}) = \big(\mathbf{R}(\theta),\ \mathbf{V}(\theta)\,\boldsymbol{\rho}\big).
$$

For $\theta^2 < \epsilon_\theta$: $a \approx 1 - \tfrac{\theta^2}{6}$,
$b \approx \tfrac{\theta}{2} - \tfrac{\theta^3}{24}$.

<a id="log"></a>
## Logarithmic Map

$$
\mathrm{Log}(\mathbf{T}) = \big(\mathbf{V}^{-1}(\theta)\,\mathbf{t},\ \theta\big),
\qquad
\theta = \operatorname{atan2}(\sin\theta, \cos\theta).
$$

## Adjoint

$$
\mathrm{Ad}_{\mathbf{T}} =
\begin{bmatrix} \cos\theta & -\sin\theta & \phantom{-}t_y \\ \sin\theta & \cos\theta & -t_x \\ 0 & 0 & 1 \end{bmatrix}.
$$

<a id="jacobians"></a>
## Jacobians

The right Jacobian is $3\times 3$. Its rotational block reuses $a, b$ above; the
coupling column mixes the translation with $\theta$ (implemented in
`SE2Tangent::right_jacobian`):

$$
\mathbf{J}_r(\boldsymbol{\tau}) =
\begin{bmatrix}
a & b & J_{02} \\
-b & a & J_{12} \\
0 & 0 & 1
\end{bmatrix},
\quad
\begin{aligned}
J_{02} &= \frac{-\rho_y + \theta\rho_x + \rho_y\cos\theta - \rho_x\sin\theta}{\theta^2}, \\
J_{12} &= \frac{\rho_x + \theta\rho_y - \rho_x\cos\theta - \rho_y\sin\theta}{\theta^2},
\end{aligned}
$$

with the small-angle limits $J_{02} \approx -\tfrac{\rho_y}{2} + \tfrac{\theta\rho_x}{6}$,
$J_{12} \approx \tfrac{\rho_x}{2} + \tfrac{\theta\rho_y}{6}$. The left Jacobian and
both inverses are available as `left_jacobian()`, `right_jacobian_inv()`,
`left_jacobian_inv()`.

## Plus and Minus

$$
\mathbf{T} \oplus \boldsymbol{\tau} = \mathbf{T}\,\mathrm{Exp}(\boldsymbol{\tau}),
\qquad
\mathbf{T}_1 \ominus \mathbf{T}_2 = \mathrm{Log}\!\left(\mathbf{T}_2^{-1}\mathbf{T}_1\right).
$$

## Example

```rust
use apex_solver::manifold::se2::{SE2, SE2Tangent};
use apex_solver::manifold::{LieGroup, Tangent};

let pose = SE2::from_xy_angle(1.0, 2.0, 0.5);
let delta = SE2Tangent::new(0.1, 0.0, 0.05);   // (ρx, ρy, θ)
let moved = pose.plus(&delta, None, None);

let tau = pose.log(None);                       // SE2Tangent
assert!(pose.is_approx(&tau.exp(None), 1e-9));
```

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
