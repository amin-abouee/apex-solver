# Manifold Conventions

This page codifies the conventions shared by every manifold chapter in this book.
Each group page (SO(2), SO(3), SE(2), SE(3), SE₂(3), SGal(3), Sim(3), Rⁿ) follows
the **same section template** and reuses the definitions collected here. The
implementation lives in the [`apex-manifolds`](https://docs.rs/apex-manifolds)
crate and follows the [manif](https://artivis.github.io/manif/) /
[*A micro Lie theory*](https://arxiv.org/abs/1812.01537) conventions.

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{G}$ | A Lie group (e.g. $SO(3)$, $SE(3)$) |
| $\mathfrak{g}$ | Its Lie algebra (tangent space at the identity) |
| $g, \mathcal{X}$ | A group element |
| $\boldsymbol{\tau}, \boldsymbol{\xi}$ | A tangent (twist) vector in $\mathbb{R}^{\text{DOF}}$ |
| $\boldsymbol{\theta} \in \mathbb{R}^3$ | Rotation part of a twist (axis–angle) |
| $\boldsymbol{\rho} \in \mathbb{R}^3$ | Translation part of a twist |
| $[\cdot]_\times$ / $(\cdot)^\wedge$ | **Hat**: vector $\to$ Lie-algebra matrix |
| $(\cdot)^\vee$ | **Vee**: Lie-algebra matrix $\to$ vector |
| $\mathrm{Exp}, \mathrm{Log}$ | Capitalised maps $\mathbb{R}^{\text{DOF}} \leftrightarrow \mathcal{G}$ (composition of $\exp/\log$ with hat/vee) |
| $\mathbf{J}_r, \mathbf{J}_l$ | Right / left Jacobian of the group |
| $\mathrm{Ad}_g$ | Adjoint matrix of $g$ |
| $\oplus, \ominus$ | Right **plus** / **minus** (retraction and its inverse) |
| $\epsilon_\theta$ | Small-angle threshold, `SMALL_ANGLE_THRESHOLD` $= 10^{-10}$ (compared against $\theta^2$) |

## Storage Conventions

- **Quaternions are stored $w$-first**, as $[q_w, q_x, q_y, q_z]$ (nalgebra order).
  Helpers such as `from_quaternion_coeffs(x, y, z, w)` accept the G2O
  $[x, y, z, w]$ order; `coeffs()` and `params()` return $w$-first.
- **Twists use translation-before-rotation**, $\boldsymbol{\tau} = (\boldsymbol{\rho}, \boldsymbol{\theta})$, matching manif.
- The default retraction is the **right** convention (`⊞`/`⊟`); left variants are
  also provided.
- Below the small-angle threshold every closed form falls back to its Taylor
  expansion for numerical stability.

## The Hat Operator on $\mathbb{R}^3$

The rotational hat is the skew-symmetric (cross-product) matrix

$$
[\boldsymbol{\theta}]_\times =
\begin{bmatrix}
0 & -\theta_z & \theta_y \\
\theta_z & 0 & -\theta_x \\
-\theta_y & \theta_x & 0
\end{bmatrix},
\qquad
[\boldsymbol{\theta}]_\times \mathbf{v} = \boldsymbol{\theta} \times \mathbf{v}.
$$

## SO(3) Jacobians (shared building block)

Every group with a rotational part reuses the SO(3) left/right Jacobians. With
$\theta = \lVert\boldsymbol{\theta}\rVert$ and $[\boldsymbol{\theta}]_\times$ the hat above:

$$
\mathbf{J}_l(\boldsymbol{\theta}) = \mathbf{I}
+ \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\theta}]_\times
+ \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\theta}]_\times^2,
$$

$$
\mathbf{J}_r(\boldsymbol{\theta}) = \mathbf{J}_l(\boldsymbol{\theta})^\top = \mathbf{I}
- \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\theta}]_\times
+ \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\theta}]_\times^2,
$$

with inverses

$$
\mathbf{J}_l^{-1}(\boldsymbol{\theta}) = \mathbf{I}
- \tfrac12 [\boldsymbol{\theta}]_\times
+ \left(\frac{1}{\theta^2} - \frac{1 + \cos\theta}{2\,\theta \sin\theta}\right)[\boldsymbol{\theta}]_\times^2,
\qquad
\mathbf{J}_r^{-1}(\boldsymbol{\theta}) = \mathbf{J}_l^{-1}(\boldsymbol{\theta})^\top .
$$

As $\theta \to 0$ these tend to $\mathbf{J}_{l,r}(\boldsymbol{\theta}) \approx \mathbf{I} \pm \tfrac12[\boldsymbol{\theta}]_\times$.

## Right Plus and Minus

The **retraction** and its inverse (right convention) are

$$
g \oplus \boldsymbol{\tau} = g \circ \mathrm{Exp}(\boldsymbol{\tau}),
\qquad
g_1 \ominus g_2 = \mathrm{Log}\!\left(g_2^{-1} \circ g_1\right).
$$

Their analytic Jacobians (used throughout the optimiser) are, with $\boldsymbol{\tau} = g_1 \ominus g_2$,

$$
\frac{\partial (g \oplus \boldsymbol{\tau})}{\partial g} = \mathrm{Ad}_{\mathrm{Exp}(\boldsymbol{\tau})}^{-1},
\quad
\frac{\partial (g \oplus \boldsymbol{\tau})}{\partial \boldsymbol{\tau}} = \mathbf{J}_r(\boldsymbol{\tau}),
\quad
\frac{\partial (g_1 \ominus g_2)}{\partial g_1} = \mathbf{J}_r^{-1}(\boldsymbol{\tau}),
\quad
\frac{\partial (g_1 \ominus g_2)}{\partial g_2} = -\mathbf{J}_l^{-1}(\boldsymbol{\tau}).
$$

The **left** variants are $\boldsymbol{\tau} \oplus g = \mathrm{Exp}(\boldsymbol{\tau}) \circ g$ and
$g_1 \ominus g_2 = \mathrm{Log}(g_1 \circ g_2^{-1})$.

## Chapter Template

Every group page has the same sections:

1. **Representation** — parameter storage layout and tangent layout.
2. **Group Operations** — composition $\circ$, inverse, identity, action on points.
3. **Lie Algebra (Hat / Vee)** — the $\wedge$ / $\vee$ maps and the algebra matrix.
4. **Exponential Map** — $\mathrm{Exp}: \mathbb{R}^{\text{DOF}} \to \mathcal{G}$.
5. **Logarithmic Map** — $\mathrm{Log}: \mathcal{G} \to \mathbb{R}^{\text{DOF}}$.
6. **Adjoint** — $\mathrm{Ad}_g$.
7. **Jacobians** — right/left Jacobian and inverses.
8. **Plus and Minus** — the $\oplus$/$\ominus$ retraction.
9. **Example** — a minimal Rust snippet.
10. **References**.

## Dimensions at a Glance

| Group | `REP_SIZE` (params) | `DOF` (tangent) | Parameter layout | Tangent layout |
|---|---|---|---|---|
| $SO(2)$ | 1 | 1 | $[\theta]$ | $[\theta]$ |
| $SO(3)$ | 4 | 3 | $[q_w, q_x, q_y, q_z]$ | $[\theta_x, \theta_y, \theta_z]$ |
| $SE(2)$ | 3 | 3 | $[x, y, \theta]$ | $[\rho_x, \rho_y, \theta]$ |
| $SE(3)$ | 7 | 6 | $[\mathbf{t}, q_w, q_x, q_y, q_z]$ | $[\boldsymbol{\rho}, \boldsymbol{\theta}]$ |
| $SE_2(3)$ | 10 | 9 | $[\mathbf{t}, \mathbf{q}, \mathbf{v}]$ | $[\boldsymbol{\rho}, \boldsymbol{\theta}, \boldsymbol{\nu}]$ |
| $SGal(3)$ | 11 | 10 | $[\mathbf{t}, \mathbf{q}, \mathbf{v}, t]$ | $[\boldsymbol{\rho}, \boldsymbol{\nu}, \boldsymbol{\theta}, s]$ |
| $Sim(3)$ | 8 | 7 | $[\mathbf{t}, \mathbf{q}, \text{scale}]$ | $[\boldsymbol{\rho}, \boldsymbol{\theta}, \sigma]$ |
| $\mathbb{R}^n$ | $n$ (dynamic) | $n$ | $[v_1, \dots, v_n]$ | $[v_1, \dots, v_n]$ |

## References

- Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537.
- Deray, J. & Solà, J. (2020). *manif: A micro Lie theory library for state estimation in robotics applications*. JOSS 5(46), 1371.
- Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press.
