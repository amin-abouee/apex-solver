# SO(2) — 2D Rotations

The special orthogonal group $SO(2)$ is the group of planar rotations. It is the
simplest non-trivial Lie group: **abelian**, with $\text{DOF} = 1$, stored as a
single angle ($\text{REP\_SIZE} = 1$). Because it is abelian, its adjoint and all
its Jacobians are the identity.

$$
SO(2) = \{\, \mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \mid \theta \in \mathbb{R} \,\}.
$$

## Representation

| Storage | Layout |
|---|---|
| Group `SO2` | `f64` = $\theta$ (radians) |
| Tangent `SO2Tangent` | `f64` = $\dot\theta$ |
| Lie algebra | $2\times 2$ skew matrix $\theta \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ |

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $\theta = 0$ | `SO2::identity()` |
| Composition | $\theta_1 + \theta_2$ (angles add — abelian) | `compose` |
| Inverse | $-\theta$ | `inverse` |
| Action | $\mathbf{R}(\theta)\,\mathbf{v}$ | `act` |

## Lie Algebra (Hat / Vee)

$$
\theta^\wedge = \begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix},
\qquad
\left(\theta^\wedge\right)^\vee = \theta .
$$

<a id="exp"></a>
## Exponential Map

$$
\mathrm{Exp}(\theta) = \mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}.
$$

<a id="log"></a>
## Logarithmic Map

$$
\mathrm{Log}\big(\mathbf{R}(\theta)\big) = \operatorname{atan2}(\sin\theta, \cos\theta) \in (-\pi, \pi].
$$

## Adjoint

$SO(2)$ is abelian, so the adjoint is the scalar identity:
$\mathrm{Ad}_{\mathbf{R}} = 1$.

<a id="jacobians"></a>
## Jacobians

All Jacobians are the scalar $1$:

$$
\mathbf{J}_l(\theta) = \mathbf{J}_r(\theta) = \mathbf{J}_l^{-1}(\theta) = \mathbf{J}_r^{-1}(\theta) = 1 .
$$

The Lie bracket vanishes ($[\cdot,\cdot] = 0$) — a direct consequence of being abelian.

## Plus and Minus

$$
\mathbf{R}(\theta) \oplus \delta = \mathbf{R}(\theta + \delta),
\qquad
\mathbf{R}(\theta_1) \ominus \mathbf{R}(\theta_2) = \operatorname{atan2}\!\big(\sin(\theta_1 - \theta_2), \cos(\theta_1 - \theta_2)\big).
$$

## Example

```rust
use apex_solver::manifold::so2::{SO2, SO2Tangent};
use apex_solver::manifold::{LieGroup, Tangent};

let r = SO2::from_angle(0.5);
let delta = SO2Tangent::new(0.1);
let r_updated = r.plus(&delta, None, None);   // angle 0.6

let tau = r.log(None);                         // 0.5
assert!((tau.angle() - 0.5).abs() < 1e-12);
```

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
