# Double Sphere

Usenko et al. (2018). A point is first projected onto a unit sphere
displaced by $\xi$ along the optical axis, and then onto a second unit sphere.
The resulting projection is closed-form and has been shown empirically to
outperform UCM and EUCM on cameras with FOV > 180° (e.g. catadioptric
panoramic rigs and 360° lenses).

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\xi$ | First-sphere offset | — | $-1 \le \xi \le 1$ |
| $\alpha$ | Second-sphere coupling | — | $0 < \alpha \le 1$ |

Total: **6 parameters**.

## Projection

Let $d_1 = \sqrt{x^2 + y^2 + z^2}$ and $w = \xi d_1 + z$. Then

$$
d_2 = \sqrt{x^2 + y^2 + w^2}, \qquad
\mathrm{denom} = \alpha d_2 + (1 - \alpha) w
$$

The pixel coordinates are

$$
u = f_x \cdot \frac{x}{\mathrm{denom}} + c_x, \qquad
v = f_y \cdot \frac{y}{\mathrm{denom}} + c_y
$$

## Unprojection

**Algebraic.** Let $m_x, m_y$ be the normalised pixel coordinates and
$r^2 = m_x^2 + m_y^2$. The intermediate $m_z$ is

$$
m_z = \frac{1 - \alpha^2 r^2}{\alpha \sqrt{1 - (2\alpha - 1) r^2} + (1 - \alpha)}
$$

Recover the world point via

$$
k = \frac{m_z \xi + \sqrt{m_z^2 + (1 - \xi^2) r^2}}{m_z^2 + r^2}, \qquad
(x, y, z) = (k m_x, k m_y, k m_z - \xi)
$$

followed by normalisation.

## Validity

- $z > 0$ and the projection condition $z > -w_2 d_1$ where
  $w_2 = (w_1 + \xi) / \sqrt{2 w_1 \xi + \xi^2 + 1}$ and
  $w_1 = \min(\alpha, 1 - \alpha) / \max(\alpha, 1 - \alpha)$.
- $\mathrm{denom} > 0$ (otherwise `DenominatorTooSmall`).
- For $\alpha > 0.5$ and $r^2 > 1 / (2\alpha - 1)$ the unprojection has no
  real solution; the point is `PointOutsideImage`.

## Point Jacobian

The chain rule runs through $d_1$, $w = \xi d_1 + z$, $d_2$, and
$\mathrm{denom} = \alpha d_2 + (1 - \alpha) w$:

$$
\frac{\partial d_1}{\partial x_i} = \frac{x_i}{d_1}
$$

$$
\frac{\partial w}{\partial x_i} = \xi \frac{x_i}{d_1} \;\; (i \in \{x, y\}), \qquad
\frac{\partial w}{\partial z} = \xi \frac{z}{d_1} + 1
$$

$$
\frac{\partial d_2}{\partial x_i} = \frac{x_i + w \cdot \partial w / \partial x_i}{d_2}
$$

$$
\frac{\partial \mathrm{denom}}{\partial x_i} = \alpha \frac{\partial d_2}{\partial x_i} + (1 - \alpha) \frac{\partial w}{\partial x_i}
$$

The point Jacobian is then the quotient rule on
$u = f_x x / \mathrm{denom} + c_x$ and similarly for $v$.

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y, \xi, \alpha]$.

- The $f_x, f_y, c_x, c_y$ columns are the pinhole form with respect to
  $x / \mathrm{denom}$.
- The $\xi$ column uses $\partial w / \partial \xi = d_1$ and propagates
  through $d_2$ and $\mathrm{denom}$.
- The $\alpha$ column uses $\partial \mathrm{denom} / \partial \alpha = d_2 - w$,
  giving

$$
\frac{\partial (u, v)}{\partial \alpha} =
\begin{bmatrix} -f_x x (d_2 - w) / \mathrm{denom}^2 \\ -f_y y (d_2 - w) / \mathrm{denom}^2 \end{bmatrix}
$$

## Linear Estimation

A linear least-squares estimate of $\alpha$ is available, with $\xi$ fixed
to 0 (the UCM limit). For each correspondence, let
$d = \sqrt{x^2 + y^2 + z^2}$, $u_c = u - c_x$, $v_c = v - c_y$, and form the
two rows of a 1-unknown linear system

$$
(u_c) \cdot (d - z) = (f_x x) - (u_c) z
$$
$$
(v_c) \cdot (d - z) = (f_y y) - (v_c) z
$$

Stack across all correspondences and solve by SVD.

## Example

```rust
use apex_camera_models::{DoubleSphereCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::DoubleSphere { xi: -0.2, alpha: 0.6 };
let camera = DoubleSphereCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f_x, f_y > 0$ and finite.
- $c_x, c_y$ finite.
- $\xi$ finite, $-1 \le \xi \le 1$.
- $\alpha$ finite, $0 < \alpha \le 1$.

## References

- Usenko, V., Demmel, N., Schubert, D., Stückler, J. & Cremers, D. (2018). *The Double Sphere Camera Model*. 3DV 2018, 552–560. arXiv:1807.08957.
