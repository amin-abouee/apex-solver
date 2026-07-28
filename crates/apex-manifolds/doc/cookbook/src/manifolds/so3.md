# SO(3) — 3D Rotations

The special orthogonal group $SO(3)$ is the group of 3D rotations. It has
$\text{DOF} = 3$ and is stored as a **unit quaternion** ($\text{REP\_SIZE} = 4$).
It is the rotational backbone of $SE(3)$, $SE_2(3)$, $SGal(3)$ and $Sim(3)$, so
the [Jacobians defined here](./conventions.md#so3-jacobians-shared-building-block)
are reused throughout the book.

$$
SO(3) = \{\, \mathbf{R} \in \mathbb{R}^{3\times 3} \mid \mathbf{R}^\top \mathbf{R} = \mathbf{I},\ \det \mathbf{R} = 1 \,\}.
$$

## Representation

| Storage | Layout |
|---|---|
| Group `SO3` | `SVector<f64, 4>` = $[q_w, q_x, q_y, q_z]$ (unit quaternion, $w$-first) |
| Tangent `SO3Tangent` | `Vector3<f64>` = $[\theta_x, \theta_y, \theta_z]$ (axis–angle) |
| Lie algebra | $3\times 3$ skew-symmetric matrix $[\boldsymbol{\theta}]_\times$ |

The matrix form is the rotation matrix $\mathbf{R} = \mathbf{R}(\mathbf{q})$
(`rotation_matrix()`).

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $\mathbf{q} = [1, 0, 0, 0]$ | `SO3::identity()` |
| Composition | $\mathbf{R}_1 \mathbf{R}_2$ (quaternion product $\mathbf{q}_1 \otimes \mathbf{q}_2$) | `compose` |
| Inverse | $\mathbf{R}^\top$ (quaternion conjugate) | `inverse` |
| Action | $\mathbf{R}\,\mathbf{v}$ | `act` |

The point action has $\dfrac{\partial (\mathbf{R}\mathbf{v})}{\partial \mathbf{v}} = \mathbf{R}$ and
$\dfrac{\partial (\mathbf{R}\mathbf{v})}{\partial \boldsymbol{\theta}} = -\mathbf{R}\,[\mathbf{v}]_\times$
(right perturbation).

## Lie Algebra (Hat / Vee)

$$
\boldsymbol{\theta}^\wedge = [\boldsymbol{\theta}]_\times =
\begin{bmatrix} 0 & -\theta_z & \theta_y \\ \theta_z & 0 & -\theta_x \\ -\theta_y & \theta_x & 0 \end{bmatrix},
\qquad
\left([\boldsymbol{\theta}]_\times\right)^\vee = \boldsymbol{\theta}.
$$

<a id="exp"></a>
## Exponential Map

With $\theta = \lVert\boldsymbol{\theta}\rVert$, the exponential is **Rodrigues'
formula** (implemented via a unit quaternion from the scaled axis):

$$
\mathrm{Exp}(\boldsymbol{\theta}) = \exp\!\big([\boldsymbol{\theta}]_\times\big)
= \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\theta}]_\times
+ \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\theta}]_\times^2 .
$$

For $\theta^2 < \epsilon_\theta$ the quaternion is built from
$[1, \tfrac12\theta_x, \tfrac12\theta_y, \tfrac12\theta_z]$ (the small-angle limit).

<a id="log"></a>
## Logarithmic Map

Given a unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ with vector part
$\mathbf{q}_v = [q_x, q_y, q_z]$ and $s = \lVert\mathbf{q}_v\rVert$, the log is

$$
\mathrm{Log}(\mathbf{R}) = \frac{2\,\phi}{s}\,\mathbf{q}_v,
\qquad
\phi = \operatorname{atan2}(s, q_w),
$$

with the sign of both `atan2` arguments flipped when $q_w < 0$ so the recovered
angle stays in $[-\pi, \pi]$ (matching manif). As $s \to 0$ the coefficient tends
to $2$.

## Adjoint

$$
\mathrm{Ad}_{\mathbf{R}} = \mathbf{R} \in \mathbb{R}^{3\times 3},
\qquad
\mathrm{Ad}_{\mathbf{R}}\,\boldsymbol{\theta} = \big(\mathbf{R}\,[\boldsymbol{\theta}]_\times \mathbf{R}^\top\big)^\vee = \mathbf{R}\,\boldsymbol{\theta}.
$$

<a id="jacobians"></a>
## Jacobians

The right/left Jacobians and their inverses are the SO(3) forms collected in the
[conventions page](./conventions.md#so3-jacobians-shared-building-block):

$$
\mathbf{J}_l(\boldsymbol{\theta}) = \mathbf{I} + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\theta}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\theta}]_\times^2,
\qquad
\mathbf{J}_r(\boldsymbol{\theta}) = \mathbf{J}_l(\boldsymbol{\theta})^\top .
$$

$$
\mathbf{J}_l^{-1}(\boldsymbol{\theta}) = \mathbf{I} - \tfrac12[\boldsymbol{\theta}]_\times + \left(\frac{1}{\theta^2} - \frac{1 + \cos\theta}{2\theta\sin\theta}\right)[\boldsymbol{\theta}]_\times^2,
\qquad
\mathbf{J}_r^{-1}(\boldsymbol{\theta}) = \mathbf{J}_l^{-1}(\boldsymbol{\theta})^\top .
$$

These are exposed on `SO3Tangent` as `right_jacobian()`, `left_jacobian()`,
`right_jacobian_inv()`, `left_jacobian_inv()`.

## Plus and Minus

Right retraction (`plus` delegates to `right_plus`):

$$
\mathbf{R} \oplus \boldsymbol{\theta} = \mathbf{R}\,\mathrm{Exp}(\boldsymbol{\theta}),
\qquad
\mathbf{R}_1 \ominus \mathbf{R}_2 = \mathrm{Log}\!\left(\mathbf{R}_2^\top \mathbf{R}_1\right).
$$

The geodesic distance is `distance()` $= \lVert \mathbf{R}_1 \ominus \mathbf{R}_2 \rVert$.

## Example

```rust
use apex_solver::manifold::so3::{SO3, SO3Tangent};
use apex_solver::manifold::{LieGroup, Tangent};
use nalgebra::Vector3;

// Build from Euler angles and retract by an axis–angle tangent.
let r = SO3::from_euler_angles(0.1, 0.2, 0.3);
let delta = SO3Tangent::new(Vector3::new(0.01, 0.0, -0.02));
let r_updated = r.plus(&delta, None, None);

// Round-trip through log / exp.
let tau = r.log(None);        // SO3Tangent
let r_back = tau.exp(None);   // SO3
assert!(r.is_approx(&r_back, 1e-9));
```

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537 (Eqs. 143–144 for the SO(3) Jacobians).
- Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press.
