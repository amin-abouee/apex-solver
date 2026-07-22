# Sim(3) — 3D Similarity Transforms

$Sim(3)$ is the group of 3D **similarity** transforms: rotation, translation, and
a positive uniform scale. It has $\text{DOF} = 7$ ($\text{REP\_SIZE} = 8$) and is
the standard state for monocular SLAM loop closure and scale-drift correction,
where the reconstruction is known only up to a scale.

$$
Sim(3) = \left\{\, \mathbf{S} = \begin{bmatrix} s\,\mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \;\middle|\; \mathbf{R} \in SO(3),\ \mathbf{t} \in \mathbb{R}^3,\ s > 0 \,\right\}.
$$

## Representation

| Storage | Layout |
|---|---|
| Group `Sim3` | `SVector<f64, 8>` = $[\mathbf{t}, q_w, q_x, q_y, q_z, s]$ (translation, quaternion, scale) |
| Tangent `Sim3Tangent` | `Vector7<f64>` = $[\boldsymbol{\rho},\ \boldsymbol{\theta},\ \sigma]$ |
| Lie algebra | $4\times 4$ matrix $\begin{bmatrix} \sigma\mathbf{I} + [\boldsymbol{\theta}]_\times & \boldsymbol{\rho} \\ \mathbf{0}^\top & 0 \end{bmatrix}$ |

The scale is stored multiplicatively as $s$; the tangent carries the **log-scale**
$\sigma = \ln s$.

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $\mathbf{R} = \mathbf{I}$, $\mathbf{t} = \mathbf{0}$, $s = 1$ | `Sim3::identity()` |
| Composition | $\big(\mathbf{R}_1\mathbf{R}_2,\ s_1\mathbf{R}_1\mathbf{t}_2 + \mathbf{t}_1,\ s_1 s_2\big)$ | `compose` |
| Inverse | $\big(\mathbf{R}^\top,\ -s^{-1}\mathbf{R}^\top\mathbf{t},\ s^{-1}\big)$ | `inverse` |
| Action | $s\,\mathbf{R}\,\mathbf{x} + \mathbf{t}$ | `act` |

## Lie Algebra (Hat / Vee)

$$
\boldsymbol{\tau}^\wedge =
\begin{bmatrix} \sigma\mathbf{I} + [\boldsymbol{\theta}]_\times & \boldsymbol{\rho} \\ \mathbf{0}^\top & 0 \end{bmatrix} \in \mathbb{R}^{4\times 4},
\qquad
\boldsymbol{\tau} = (\boldsymbol{\rho}, \boldsymbol{\theta}, \sigma) \in \mathbb{R}^7 .
$$

<a id="exp"></a>
## Exponential Map

The rotation exponentiates as in SO(3), the scale as $s = e^\sigma$, and the
translation through the **scale-coupled $\mathbf{V}$ matrix**
$\mathbf{V}(\boldsymbol{\theta}, \sigma)$:

$$
\mathrm{Exp}(\boldsymbol{\tau}) = \big(\mathrm{Exp}(\boldsymbol{\theta}),\ \mathbf{V}(\boldsymbol{\theta}, \sigma)\,\boldsymbol{\rho},\ e^\sigma\big).
$$

With $\theta = \lVert\boldsymbol{\theta}\rVert$ and $\alpha^2 = \sigma^2 + \theta^2$,
the general form is $\mathbf{V} = a\,\mathbf{I} + b\,[\boldsymbol{\theta}]_\times + c\,[\boldsymbol{\theta}]_\times^2$ with

$$
a = \frac{e^\sigma - 1}{\sigma}, \quad
b = \frac{e^\sigma(\sigma\sin\theta - \theta\cos\theta) + \theta}{\theta\,\alpha^2}, \quad
c = \frac{a - \dfrac{e^\sigma(\sigma\cos\theta + \theta\sin\theta) - \sigma}{\alpha^2}}{\theta^2}.
$$

It reduces to the two limiting cases

$$
\mathbf{V}(\mathbf{0}, \sigma) = \frac{e^\sigma - 1}{\sigma}\,\mathbf{I}
\quad\text{(pure scale)}, \qquad
\mathbf{V}(\boldsymbol{\theta}, 0) = \mathbf{J}_l(\boldsymbol{\theta})
\quad\text{(pure rotation, the SE(3) case)},
$$

and $\mathbf{V}(\mathbf{0}, 0) = \mathbf{I}$ (implemented in `Sim3Tangent::v_matrix`).

<a id="log"></a>
## Logarithmic Map

$$
\mathrm{Log}(\mathbf{S}) = \big(\mathbf{V}^{-1}(\boldsymbol{\theta}, \sigma)\,\mathbf{t},\ \boldsymbol{\theta},\ \sigma\big),
\qquad
\boldsymbol{\theta} = \mathrm{Log}(\mathbf{R}),\quad \sigma = \ln s,
$$

with $\mathbf{V}^{-1}$ from `Sim3::compute_v_inv`.

## Adjoint

$\mathrm{Ad}_{\mathbf{S}}$ is the $7\times 7$ matrix from `adjoint()`. The rotation
block acts on $\boldsymbol{\rho}$ and $\boldsymbol{\theta}$, the scale rescales the
translation part, and translation couples with rotation and scale (the scale row
of the twist is invariant, $\mathrm{Ad}$ acting as $1$ on $\sigma$).

<a id="jacobians"></a>
## Jacobians

The $7\times 7$ right Jacobian couples translation with both rotation and scale
through a $\mathbf{Q}$-type block `Sim3Tangent::q_matrix`, with
$\mathbf{J}_r(\boldsymbol{\theta})$ on the rotational diagonal and $1$ on the
scale diagonal. Available as `right_jacobian()`, `left_jacobian()`,
`right_jacobian_inv()`, `left_jacobian_inv()`.

## Plus and Minus

$$
\mathbf{S} \oplus \boldsymbol{\tau} = \mathbf{S}\,\mathrm{Exp}(\boldsymbol{\tau}),
\qquad
\mathbf{S}_1 \ominus \mathbf{S}_2 = \mathrm{Log}\!\left(\mathbf{S}_2^{-1}\mathbf{S}_1\right).
$$

## Example

```rust
use apex_solver::manifold::sim3::{Sim3, Sim3Tangent};
use apex_solver::manifold::{LieGroup, Tangent};
use nalgebra::{Vector3, UnitQuaternion};

let sim = Sim3::new(
    Vector3::new(1.0, 2.0, 3.0),    // translation
    UnitQuaternion::identity(),
    1.5,                             // scale s
);
let delta = Sim3Tangent::new(
    Vector3::new(0.1, 0.0, 0.0),    // ρ
    Vector3::new(0.0, 0.05, 0.0),   // θ
    0.02,                            // σ = log-scale
);
let updated = sim.plus(&delta, None, None);
assert!(sim.is_approx(&sim.log(None).exp(None), 1e-9));
```

## References

- Strasdat, H., Montiel, J. M. M. & Davison, A. J. (2010). *Scale Drift-Aware Large Scale Monocular SLAM*. RSS 2010.
- Lovegrove, S. (2012). *Sim(3) exponential and logarithm maps* (appendix), and Strasdat, H. (2012). *Local Accuracy and Global Consistency for Efficient Visual SLAM*, PhD thesis.
- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
