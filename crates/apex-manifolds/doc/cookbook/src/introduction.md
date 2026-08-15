# Introduction

This book is the mathematical reference for the **`apex-manifolds`** crate — the
Lie-group / manifold library used by Apex Solver. It documents every manifold and
every operation the crate defines: composition, inverse, exponential and
logarithmic maps, adjoints, left/right Jacobians, and the `⊞`/`⊟` retraction.

The crate source stays focused on the implementation; the equations, derivations,
and conventions live here.

## Why Manifold Operations?

Consider a 3D rotation. Representing it as a 3×3 matrix (9 numbers) for only 3
degrees of freedom is:

- **Overconstrained** — 9 parameters for 3 DOF,
- **Numerically unstable** — you must re-project onto $SO(3)$ after each update,
- **Inefficient** — computation is spent enforcing constraints.

Working *on the manifold* instead:

- represents updates in the **tangent space** (3 parameters for $SO(3)$),
- applies them with the **exponential map** so the result stays on the manifold,
- and provides **analytic Jacobians** for the geometry.

## Key Concepts

| Term | Definition |
|------|------------|
| **Manifold** | A smooth space where each point has a local Euclidean structure. |
| **Lie group** | A manifold that is also a group (composition, inverse, identity). |
| **Tangent space** | The local linear approximation at a point (the Lie algebra at the identity). |
| **Exponential map** | Tangent → group, $\mathrm{Exp}: \mathbb{R}^{\text{DOF}} \to \mathcal{G}$. |
| **Logarithmic map** | Group → tangent, $\mathrm{Log}: \mathcal{G} \to \mathbb{R}^{\text{DOF}}$. |
| **Plus / Minus** | Retraction $g \oplus \boldsymbol{\tau} = g \circ \mathrm{Exp}(\boldsymbol{\tau})$ and its inverse $g_1 \ominus g_2 = \mathrm{Log}(g_2^{-1} \circ g_1)$. |

## The Groups

`apex-manifolds` implements eight manifolds, all sharing one `LieGroup` /
`Tangent` trait interface:

| Group | DOF | Description |
|---|---|---|
| [SO(2)](./manifolds/so2.md) | 1 | 2D rotations |
| [SO(3)](./manifolds/so3.md) | 3 | 3D rotations |
| [SE(2)](./manifolds/se2.md) | 3 | 2D rigid transforms |
| [SE(3)](./manifolds/se3.md) | 6 | 3D rigid transforms |
| [SE₂(3)](./manifolds/se23.md) | 9 | Extended poses (rotation + position + velocity) |
| [SGal(3)](./manifolds/sgal3.md) | 10 | Galilean group (pose + velocity + time) |
| [Sim(3)](./manifolds/sim3.md) | 7 | Similarity transforms (rotation + translation + scale) |
| [Rⁿ](./manifolds/rn.md) | n | Euclidean vectors (trivial manifold) |

Start with the [**Conventions**](./manifolds/conventions.md) page — it defines the
notation, storage layout (quaternion order, twist order), the `⊞`/`⊟` operators,
and the shared SO(3) Jacobians that every other chapter reuses.

## Build the Book

```bash
cargo install mdbook --locked
cargo install mdbook-katex --locked

mdbook serve crates/apex-manifolds/doc/cookbook --open
mdbook build crates/apex-manifolds/doc/cookbook
```

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
- Deray, J. & Solà, J. (2020). *manif: A micro Lie theory library for state estimation in robotics applications*. JOSS 5(46), 1371.
