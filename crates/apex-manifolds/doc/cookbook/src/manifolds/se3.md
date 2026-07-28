# SE(3) — 3D Rigid Transforms

The special Euclidean group $SE(3)$ is the group of 3D rigid-body motions. It has
$\text{DOF} = 6$ and is stored as a translation plus a unit quaternion
($\text{REP\_SIZE} = 7$). It is the workhorse pose type for SLAM and bundle
adjustment.

$$
SE(3) = \left\{\, \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \;\middle|\; \mathbf{R} \in SO(3),\ \mathbf{t} \in \mathbb{R}^3 \,\right\}.
$$

## Representation

| Storage | Layout |
|---|---|
| Group `SE3` | `SVector<f64, 7>` = $[t_x, t_y, t_z, q_w, q_x, q_y, q_z]$ |
| Tangent `SE3Tangent` | `Vector6<f64>` = $[\boldsymbol{\rho},\ \boldsymbol{\theta}]$ (translation first) |
| Lie algebra | $4\times 4$ matrix $\begin{bmatrix} [\boldsymbol{\theta}]_\times & \boldsymbol{\rho} \\ \mathbf{0}^\top & 0 \end{bmatrix}$ |

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $\mathbf{t} = \mathbf{0}$, $\mathbf{q} = [1,0,0,0]$ | `SE3::identity()` |
| Composition | $(\mathbf{R}_1\mathbf{R}_2,\ \mathbf{R}_1\mathbf{t}_2 + \mathbf{t}_1)$ | `compose` |
| Inverse | $(\mathbf{R}^\top,\ -\mathbf{R}^\top \mathbf{t})$ | `inverse` |
| Action | $\mathbf{R}\,\mathbf{v} + \mathbf{t}$ | `act` |

## Lie Algebra (Hat / Vee)

$$
\boldsymbol{\tau}^\wedge =
\begin{bmatrix} [\boldsymbol{\theta}]_\times & \boldsymbol{\rho} \\ \mathbf{0}^\top & 0 \end{bmatrix} \in \mathbb{R}^{4\times 4},
\qquad
\boldsymbol{\tau} = (\boldsymbol{\rho}, \boldsymbol{\theta}) \in \mathbb{R}^6 .
$$

<a id="exp"></a>
## Exponential Map

The rotation uses the SO(3) exponential; the translation is mapped through the
SO(3) **left Jacobian** $\mathbf{J}_l(\boldsymbol{\theta})$ (the $SE(3)$
$\mathbf{V}$ matrix):

$$
\mathrm{Exp}(\boldsymbol{\tau}) = \big(\mathrm{Exp}(\boldsymbol{\theta}),\ \mathbf{J}_l(\boldsymbol{\theta})\,\boldsymbol{\rho}\big),
$$

with $\mathbf{J}_l$ from the [conventions page](./conventions.md#so3-jacobians-shared-building-block).

<a id="log"></a>
## Logarithmic Map

$$
\mathrm{Log}(\mathbf{T}) = \big(\mathbf{J}_l^{-1}(\boldsymbol{\theta})\,\mathbf{t},\ \boldsymbol{\theta}\big),
\qquad
\boldsymbol{\theta} = \mathrm{Log}(\mathbf{R}).
$$

## Adjoint

$$
\mathrm{Ad}_{\mathbf{T}} =
\begin{bmatrix} \mathbf{R} & [\mathbf{t}]_\times \mathbf{R} \\ \mathbf{0} & \mathbf{R} \end{bmatrix} \in \mathbb{R}^{6\times 6}.
$$

<a id="jacobians"></a>
## Jacobians

The right Jacobian is block-upper-triangular, coupling translation and rotation
through the **$\mathbf{Q}$ matrix**:

$$
\mathbf{J}_r(\boldsymbol{\tau}) =
\begin{bmatrix} \mathbf{J}_r(\boldsymbol{\theta}) & \mathbf{Q}(\boldsymbol{\rho}, \boldsymbol{\theta}) \\ \mathbf{0} & \mathbf{J}_r(\boldsymbol{\theta}) \end{bmatrix},
$$

where $\mathbf{J}_r(\boldsymbol{\theta})$ is the SO(3) right Jacobian and, writing
$P = [\boldsymbol{\rho}]_\times$, $\Theta = [\boldsymbol{\theta}]_\times$,
$\theta = \lVert\boldsymbol{\theta}\rVert$ (Barfoot's form, implemented as a
numerically-stable series in `SE3Tangent::q_block_jacobian_matrix`):

$$
\begin{aligned}
\mathbf{Q} = \tfrac12 P
&+ \frac{\theta - \sin\theta}{\theta^3}\big(\Theta P + P\Theta + \Theta P\Theta\big) \\
&- \frac{1 - \tfrac{\theta^2}{2} - \cos\theta}{\theta^4}\big(\Theta^2 P + P\Theta^2 - 3\,\Theta P\Theta\big) \\
&- \tfrac12\!\left(\frac{1 - \tfrac{\theta^2}{2} - \cos\theta}{\theta^4} - 3\,\frac{\theta - \sin\theta - \tfrac{\theta^3}{6}}{\theta^5}\right)\!\big(\Theta P\Theta^2 + \Theta^2 P\Theta\big).
\end{aligned}
$$

The left Jacobian has the same block structure with the left SO(3) Jacobian and
the corresponding $\mathbf{Q}$; the inverses are `right_jacobian_inv()` /
`left_jacobian_inv()`.

## Plus and Minus

$$
\mathbf{T} \oplus \boldsymbol{\tau} = \mathbf{T}\,\mathrm{Exp}(\boldsymbol{\tau}),
\qquad
\mathbf{T}_1 \ominus \mathbf{T}_2 = \mathrm{Log}\!\left(\mathbf{T}_2^{-1}\mathbf{T}_1\right).
$$

In the optimiser, fixing DOF indices $\{0,1,2\}$ freezes translation and
$\{3,4,5\}$ freezes rotation (the tangent order is $[\boldsymbol{\rho}, \boldsymbol{\theta}]$).

## Example

```rust
use apex_solver::manifold::se3::{SE3, SE3Tangent};
use apex_solver::manifold::{LieGroup, Tangent};
use nalgebra::{Vector3, Quaternion, UnitQuaternion};

let pose = SE3::from_translation_quaternion(
    Vector3::new(1.0, 2.0, 3.0),
    UnitQuaternion::identity().into_inner(),
);
let delta = SE3Tangent::new(Vector3::new(0.1, 0.0, 0.0), Vector3::new(0.0, 0.05, 0.0));
let moved = pose.plus(&delta, None, None);

let tau = pose.log(None);                        // SE3Tangent = [ρ, θ]
assert!(pose.is_approx(&tau.exp(None), 1e-9));
```

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
- Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press (Eq. 7.86 for the $\mathbf{Q}$ matrix).
