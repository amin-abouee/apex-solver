# Rⁿ — Euclidean Vectors

$\mathbb{R}^n$ under addition is the trivial (flat, abelian) manifold: the tangent
space *is* the group, and exp/log are the identity. It models points, landmarks,
velocities, biases, and camera intrinsics — anything that lives in ordinary
Euclidean space. Its dimension is **dynamic** (`REP_SIZE` / `DOF` are stored as
the sentinel $0$; the real size is `dim()` / `tangent_dim()`).

## Representation

| Storage | Layout |
|---|---|
| Group `Rn` | `DVector<f64>` = $[v_1, \dots, v_n]$ |
| Tangent `RnTangent` | `DVector<f64>` = $[v_1, \dots, v_n]$ |
| Lie algebra | the vector itself (no matrix embedding needed) |

## Group Operations

| Operation | Formula | Method |
|---|---|---|
| Identity | $\mathbf{0}$ | `Rn::identity()` / `Rn::zeros(n)` |
| Composition | $\mathbf{a} + \mathbf{b}$ | `compose` |
| Inverse | $-\mathbf{a}$ | `inverse` |
| Action | $\mathbf{a} + \mathbf{v}$ (translation) | `act` |

## Lie Algebra (Hat / Vee)

Both maps are the identity — there is no non-trivial matrix algebra:
$\mathbf{v}^\wedge = \mathbf{v}$, $(\mathbf{v})^\vee = \mathbf{v}$. The Lie bracket
is zero (abelian).

<a id="exp"></a>
## Exponential Map

$$
\mathrm{Exp}(\mathbf{v}) = \mathbf{v}.
$$

<a id="log"></a>
## Logarithmic Map

$$
\mathrm{Log}(\mathbf{a}) = \mathbf{a}.
$$

## Adjoint

$\mathrm{Ad}_{\mathbf{a}} = \mathbf{I}_n$ (identity) — translations commute.

<a id="jacobians"></a>
## Jacobians

Every Jacobian is the $n\times n$ identity:

$$
\mathbf{J}_l = \mathbf{J}_r = \mathbf{J}_l^{-1} = \mathbf{J}_r^{-1} = \mathbf{I}_n .
$$

Consequently the $\oplus$ / $\ominus$ Jacobians are also identity, which is why
$\mathbb{R}^n$ variables behave like a classical (non-manifold) least-squares
unknown.

## Plus and Minus

$$
\mathbf{a} \oplus \mathbf{v} = \mathbf{a} + \mathbf{v},
\qquad
\mathbf{a}_1 \ominus \mathbf{a}_2 = \mathbf{a}_1 - \mathbf{a}_2 .
$$

## Interpolation

$\mathbb{R}^n$ is the only group in the crate that implements `Interpolatable`:

$$
\mathrm{interp}(\mathbf{a}, \mathbf{b}, t) = (1 - t)\,\mathbf{a} + t\,\mathbf{b},
$$

and `slerp` falls back to the same linear blend.

## Example

```rust
use apex_solver::manifold::rn::{Rn, RnTangent};
use apex_solver::manifold::{LieGroup, Tangent};
use nalgebra::DVector;

let p = Rn::from_vec(vec![1.0, 2.0, 3.0]);
let delta = RnTangent::from_vec(vec![0.1, 0.0, -0.1]);
let p_updated = p.plus(&delta, None, None);       // [1.1, 2.0, 2.9]

assert_eq!(p.tangent_dim(), 3);
```

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537 (§ trivial groups).
