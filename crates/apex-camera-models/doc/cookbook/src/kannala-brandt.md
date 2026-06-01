# Kannala-Brandt Fisheye

A widely-used fisheye model that maps the **incidence angle** $\theta$ of a
3D ray to a **distorted radius** $d(\theta)$ through an odd-order
polynomial. It is the model that OpenCV calls the "fisheye" model and that
Kalibr calls "equidistant". It handles FOVs up to about $180°$.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $k_1, k_2, k_3, k_4$ | Polynomial coefficients | — | finite |

Total: **8 parameters**.

## Projection

Let $r = \sqrt{x^2 + y^2}$ and $\theta = \arctan(r / z)$. The forward
polynomial is

$$
d(\theta) = \theta + k_1 \theta^3 + k_2 \theta^5 + k_3 \theta^7 + k_4 \theta^9
$$

The unit direction in the image plane is $(x / r, y / r)$ (with safe handling
for $r = 0$). The pixel coordinates are

$$
u = f_x \cdot d(\theta) \cdot \frac{x}{r} + c_x, \qquad
v = f_y \cdot d(\theta) \cdot \frac{y}{r} + c_y
$$

## Unprojection

**Iterative.** Given $(u, v)$ and $(m_x, m_y) = ((u - c_x) / f_x, (v - c_y) / f_y)$,
let $r_d = \sqrt{m_x^2 + m_y^2}$. We need $\theta$ such that $d(\theta) = r_d$.
This is solved by Newton-Raphson on $d(\theta) - r_d = 0$ with
derivative $d'(\theta) = 1 + 3 k_1 \theta^2 + 5 k_2 \theta^4 + 7 k_3 \theta^6 + 9 k_4 \theta^8$.

The unit ray is

$$
\mathbf{r} = \frac{1}{\sqrt{m_x^2 + m_y^2 + 1}} (m_x, m_y, 1)^\top
$$

## Validity

- $z > 0$.
- $r > 0$ is handled separately (the on-axis case projects to the principal
  point).

## Point Jacobian

With $r = \sqrt{x^2 + y^2}$, $\theta = \arctan(r / z)$,

$$
\partial r / \partial x = x / r, \quad
\partial r / \partial y = y / r, \quad
\partial r / \partial z = 0
$$

$$
\partial \theta / \partial x = \frac{z x}{r (x^2 + y^2 + z^2)},
\quad
\partial \theta / \partial y = \frac{z y}{r (x^2 + y^2 + z^2)},
\quad
\partial \theta / \partial z = -\frac{r}{x^2 + y^2 + z^2}
$$

The point Jacobian is then obtained by the chain rule through $d(\theta)$
and the unit direction $(x / r, y / r)$. The exact expression is implemented
in the crate.

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4]$.

The $f_x, f_y, c_x, c_y$ columns are obtained from the chain rule of the
distorted-radius projection; the $k_i$ columns are
$(f_x \theta^{2i+1} \cdot x / r, f_y \theta^{2i+1} \cdot y / r)$.

## Linear Estimation

For each correspondence, compute $\theta = \arctan(r / z)$ with
$r = \sqrt{x^2 + y^2}$. If $r > 0$, the unit direction is $(x / r, y / r)$.
Define $a_i = f_x \theta^{2i+1} x / r$ and $b_i = f_y \theta^{2i+1} y / r$
for $i = 1, 2, 3, 4$ (mapping to $k_1, k_2, k_3, k_4$). The linear system is

$$
\begin{bmatrix} a_1 & a_2 & a_3 & a_4 \\ b_1 & b_2 & b_3 & b_4 \\ \vdots & & & \vdots \end{bmatrix}
\begin{bmatrix} k_1 \\ k_2 \\ k_3 \\ k_4 \end{bmatrix}
=
\begin{bmatrix} (u - c_x) - f_x \theta \cdot x / r \\ (v - c_y) - f_y \theta \cdot y / r \\ \vdots \end{bmatrix}
$$

Solved by SVD. **At least 4 correspondences** are required.

## Example

```rust
use apex_camera_models::{KannalaBrandtCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::KannalaBrandt {
    k1: -0.02, k2: 0.0, k3: 0.0, k4: 0.0,
};
let camera = KannalaBrandtCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f_x, f_y > 0$ and finite.
- $c_x, c_y$ finite.
- $k_1, k_2, k_3, k_4$ finite.

## References

- Kannala, J. & Brandt, S. S. (2006). *A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses*. IEEE TPAMI 28(8), 1335–1340.
