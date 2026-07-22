# Introduction

This book is the mathematical reference for every camera projection model shipped
in the `apex-camera-models` crate. The crate source code stays focused on the
implementation; the equations, derivations, and references live here.

## Coordinate Frame

All models in this book use the **RDF** computer-vision convention unless a
model states otherwise:

- $X$ axis: points to the right of the image.
- $Y$ axis: points down the image.
- $Z$ axis: points forward, into the scene. The camera looks down the **+Z** axis.

**Exception — BAL Pinhole.** The BAL family follows the Bundler convention:
the camera looks down the **-Z** axis. See the [BAL Pinhole chapter](./bal-pinhole.md).

A 3D point $p_{\mathrm{cam}} = (x, y, z)$ is in the camera frame, with the origin
at the optical centre and $z > 0$ in front of the camera.

A 2D point $p_{uv} = (u, v)$ is in pixel coordinates, with the origin at the
top-left of the image. The principal point $c = (c_x, c_y)$ and the focal
lengths $f_x, f_y$ are in pixels.

## Notation

| Symbol | Meaning |
|---|---|
| $p_{\mathrm{cam}} = (x, y, z)$ | 3D point in the camera frame |
| $p_{uv} = (u, v)$ | 2D point in pixel coordinates |
| $K = (f_x, f_y, c_x, c_y)$ | Linear intrinsics (focal lengths + principal point) |
| $d$ | Distortion parameter vector, model-specific |
| $\theta$ | Incidence angle between a ray and the optical axis, $\theta = \arctan(r / z)$ with $r = \sqrt{x^2 + y^2}$ |
| $\phi$ | Azimuth in the image plane, $\phi = \arctan(y / x)$ |
| $r$ | Lateral distance in the normalised image plane, $r = \sqrt{x^2 + y^2}$ |
| $r_d$ | Distorted radius, mapped to pixels by $f(r_d)$ |
| $\pi$ | The projection function $p_{\mathrm{cam}} \mapsto p_{uv}$ |
| $m_x, m_y$ | Normalised image coordinates, $m_x = (u - c_x)/f_x$, $m_y = (v - c_y)/f_y$ |
| $\partial_a b$ | Partial derivative of $b$ with respect to $a$ |
| $\epsilon_g$ | Geometric-precision threshold, $\epsilon_g = 10^{-6}$ (crate constant `GEOMETRIC_PRECISION`) |

## Pose Jacobians (SE(3))

The crate exposes a pose Jacobian through [`CameraModel::jacobian_pose`](https://docs.rs/apex-camera-models).
The pose is treated as a **world-to-camera** transform $T_{wc}$ so that
$p_{\mathrm{cam}} = R \cdot p_{\mathrm{world}} + t$, with $R \in \mathrm{SO}(3)$
and $t \in \mathbb{R}^3$.

The pose perturbation is a right perturbation in the Lie algebra $\mathfrak{se}(3)$:

$$
T'(\delta\xi) = T \cdot \mathrm{Exp}(\delta\xi), \quad \delta\xi = (\delta\rho, \delta\theta) \in \mathbb{R}^6
$$

The first three components $\delta\rho$ are a translation, the last three
$\delta\theta$ are a rotation in the tangent space at the identity. The
perturbed camera-frame point is

$$
p_{\mathrm{cam}}' = R \cdot (I + [\delta\theta]_\times) \cdot p_{\mathrm{world}} + (t + R \cdot \delta\rho)
$$

To first order, the partials of $p_{\mathrm{cam}}$ with respect to $\delta\xi$
are

$$
\frac{\partial p_{\mathrm{cam}}}{\partial \delta\rho} = R, \qquad
\frac{\partial p_{\mathrm{cam}}}{\partial \delta\theta} = -R \cdot [p_{\mathrm{world}}]_\times
$$

The 2×6 pixel-vs-pose Jacobian is obtained by the chain rule

$$
\frac{\partial p_{uv}}{\partial \delta\xi}
= \frac{\partial p_{uv}}{\partial p_{\mathrm{cam}}}
\cdot \frac{\partial p_{\mathrm{cam}}}{\partial \delta\xi}
$$

with $\partial p_{uv} / \partial p_{\mathrm{cam}}$ given by the point Jacobian
of the chosen model and the cross-product matrix

$$
[v]_\times = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}
$$

## A Unified Chapter Layout

Every camera-model chapter follows the **same eight-section structure** so the
models can be read and compared side by side:

1. **Parameters** — the intrinsic vector, its units and admissible ranges.
2. **Projection** — the forward map $p_{\mathrm{cam}} \mapsto p_{uv}$, with the
   **validity conditions of the forward map merged in** (depth, denominator, and
   the error each violation raises).
3. **Inverse Projection** — the map $p_{uv} \mapsto \mathbf{r}$ (a unit ray),
   with the **validity conditions of the inverse map merged in** (the domain on
   which a real solution exists).
4. **Point Jacobian** — $\partial p_{uv} / \partial p_{\mathrm{cam}}$ (2×3).
5. **Intrinsic Jacobian** — $\partial p_{uv} / \partial (\text{parameters})$.
6. **Linear Estimation** — the closed-form / least-squares initialiser, if any.
7. **Example** — a minimal constructor snippet.
8. **References**.

Validity is therefore **not** a separate section: a condition that guards the
forward map lives inside *Projection*, and a condition that guards the inverse
map lives inside *Inverse Projection*.

## Validity Conventions

For every model a 3D point must be in front of the camera. The depth test uses
the geometric-precision threshold $\epsilon_g = 10^{-6}$: a point is rejected
when $z < \epsilon_g$ (or $z > -\epsilon_g$ for the BAL $-Z$ convention). Models
built on a virtual sphere (UCM, EUCM, Double Sphere) add a geometric inequality
that excludes points inside that sphere, and the algebraic inverses of those
models are only real on a bounded disc $r \le r_{\max}(\text{parameters})$. Each
chapter states the exact inequality and the error it raises:

| Error variant | Raised when |
|---|---|
| `PointBehindCamera { z, min_z }` | depth test $z < \epsilon_g$ fails |
| `ProjectionOutOfBounds` | the sphere/FOV geometric condition fails |
| `DenominatorTooSmall { denom, threshold }` | the projection denominator $< \epsilon_g$ |
| `PointOutsideImage { x, y }` | the inverse map leaves the real domain |
| `NumericalError { .. }` | an iterative inverse fails to converge |

## Number of Parameters

| Model | Linear $K$ | Distortion | Total |
|---|---|---|---|
| Pinhole | 4 | 0 | 4 |
| RadTan | 4 | 5 | 9 |
| Kannala-Brandt | 4 | 4 | 8 |
| FOV | 4 | 1 | 5 |
| UCM | 4 | 1 | 5 |
| EUCM | 4 | 2 | 6 |
| Double Sphere | 4 | 2 | 6 |
| F-Theta | 2 (only $c_x, c_y$) | 4 | 6 |
| BAL Pinhole (strict) | 1 ($f$ only) | 2 | 3 |

## Build the Book

```bash
# install once
cargo install mdbook --locked
cargo install mdbook-katex --locked

# from the cookbook directory
mdbook serve doc/cookbook --open
mdbook build doc/cookbook
```
