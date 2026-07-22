# SGal(3) — Galilean Group (Pose + Velocity + Time)

$SGal(3)$ is the **special Galilean group**: a rotation, a position, a velocity
(boost), and a time offset. It has $\text{DOF} = 10$ ($\text{REP\_SIZE} = 11$) and
models a full inertial state including the clock, which makes it a convenient
group for continuous-time and IMU-driven estimation.

## Representation

| Storage | Layout |
|---|---|
| Group `SGal3` | `SVector<f64, 11>` = $[\mathbf{t}, q_w, q_x, q_y, q_z, \mathbf{v}, t]$ (position, quaternion, velocity, time) |
| Tangent `SGal3Tangent` | `Vector10<f64>` = $[\boldsymbol{\rho},\ \boldsymbol{\nu},\ \boldsymbol{\theta},\ s]$ |
| Lie algebra | $6\times 6$ matrix (`hat()`), larger than the $5\times 5$ group embedding |

The tangent order is $[\boldsymbol{\rho}, \boldsymbol{\nu}, \boldsymbol{\theta}, s]$
(position, velocity, rotation, time-scalar) — note it differs both from
[SE₂(3)](./se23.md) and from the parameter storage order. The group matrix
(`matrix()`) is the $5\times 5$ homogeneous form

$$
\mathbf{X} = \begin{bmatrix} \mathbf{R} & \mathbf{v} & \mathbf{p} \\ \mathbf{0} & 1 & t \\ \mathbf{0} & 0 & 1 \end{bmatrix},
$$

with the time offset $t$ in the boost row.

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $\mathbf{R} = \mathbf{I}$, $\mathbf{p} = \mathbf{v} = \mathbf{0}$, $t = 0$ | `SGal3::identity()` |
| Composition | $\mathbf{X}_1 \mathbf{X}_2$ (matrix product) | `compose` |
| Inverse | matrix inverse of $\mathbf{X}$ | `inverse` |
| Action | $\mathbf{R}\,\mathbf{x} + \mathbf{p}$ | `act` |

## Lie Algebra (Hat / Vee)

`hat()` embeds $\boldsymbol{\tau} = (\boldsymbol{\rho}, \boldsymbol{\nu}, \boldsymbol{\theta}, s)$
into a $6\times 6$ matrix carrying the rotation generator $[\boldsymbol{\theta}]_\times$,
the position $\boldsymbol{\rho}$, the boost $\boldsymbol{\nu}$, and the time
scalar $s$; `vee` is its inverse.

<a id="exp"></a>
## Exponential Map

Rotation exponentiates as in SO(3); position and boost are mapped through the
left Jacobian $\mathbf{J}_l(\boldsymbol{\theta})$; the time scalar passes through
unchanged:

$$
\mathrm{Exp}(\boldsymbol{\tau}) = \Big(\mathrm{Exp}(\boldsymbol{\theta}),\ \ \mathbf{p} = \mathbf{J}_l(\boldsymbol{\theta})\,\boldsymbol{\rho},\ \ \mathbf{v} = \mathbf{J}_l(\boldsymbol{\theta})\,\boldsymbol{\nu},\ \ t = s\Big).
$$

<a id="log"></a>
## Logarithmic Map

$$
\mathrm{Log}(\mathbf{X}) = \Big(\mathbf{J}_l^{-1}(\boldsymbol{\theta})\,\mathbf{p},\ \ \mathbf{J}_l^{-1}(\boldsymbol{\theta})\,\mathbf{v},\ \ \boldsymbol{\theta},\ \ t\Big),
\qquad \boldsymbol{\theta} = \mathrm{Log}(\mathbf{R}).
$$

## Adjoint

$\mathrm{Ad}_{\mathbf{X}}$ is the $10\times 10$ matrix returned by `adjoint()`. It
generalises the [SE₂(3) adjoint](./se23.md#adjoint): the rotation block acts on
each of $\boldsymbol{\rho}, \boldsymbol{\nu}, \boldsymbol{\theta}$, position and
boost couple with rotation through $[\mathbf{p}]_\times\mathbf{R}$ and
$[\mathbf{v}]_\times\mathbf{R}$, and the time scalar couples the boost.

<a id="jacobians"></a>
## Jacobians

The $10\times 10$ right Jacobian is block-structured, reusing the SE(3)-style
coupling block $\mathbf{Q}(\cdot, \boldsymbol{\theta})$
(`SGal3Tangent::q_matrix`) for the position–rotation and boost–rotation blocks,
with $\mathbf{J}_r(\boldsymbol{\theta})$ on the rotational diagonal and $1$ on the
time diagonal. It is exposed as `right_jacobian()`, with `left_jacobian()` and the
two inverses available as usual.

## Plus and Minus

$$
\mathbf{X} \oplus \boldsymbol{\tau} = \mathbf{X}\,\mathrm{Exp}(\boldsymbol{\tau}),
\qquad
\mathbf{X}_1 \ominus \mathbf{X}_2 = \mathrm{Log}\!\left(\mathbf{X}_2^{-1}\mathbf{X}_1\right).
$$

## Example

```rust
use apex_solver::manifold::sgal3::{SGal3, SGal3Tangent};
use apex_solver::manifold::{LieGroup, Tangent};
use nalgebra::{Vector3, UnitQuaternion};

let state = SGal3::new(
    Vector3::new(1.0, 0.0, 0.0),   // position
    Vector3::new(0.0, 0.0, 9.8),   // velocity
    0.1,                            // time
    UnitQuaternion::identity(),
);
let delta = SGal3Tangent::new(
    Vector3::new(0.01, 0.0, 0.0),  // ρ
    Vector3::new(0.0, 0.0, 0.05),  // ν
    Vector3::new(0.0, 0.02, 0.0),  // θ
    0.01,                           // s
);
let updated = state.plus(&delta, None, None);
assert!(state.is_approx(&state.log(None).exp(None), 1e-9));
```

## References

- Kelly, J. (2024). *All About the Galilean Group SGal(3)* / *On the Galilean invariance of inertial navigation*. (Galilean group for state estimation.)
- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
