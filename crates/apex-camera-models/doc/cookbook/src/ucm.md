# Unified Camera Model (UCM)

Geyer & Daniilidis (2000). The point is first projected onto a unit sphere
and then onto the perspective image plane, parameterised by a single
$\alpha \in [0, 1]$ that controls the trade-off between para-perspective
($\alpha = 0$) and perspective ($\alpha = 1$). Catadioptric and fisheye
lenses with FOV between 90° and 180° are well served by UCM.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\alpha$ | Sphere-plane coupling | — | $0 \le \alpha \le 1$ |

Total: **5 parameters**.

## Projection

Let $d = \sqrt{x^2 + y^2 + z^2}$. The denominator blends the sphere radius
and the $z$ coordinate:

$$
\mathrm{denom} = \alpha d + (1 - \alpha) z
$$

The pixel coordinates are

$$
u = f_x \cdot \frac{x}{\mathrm{denom}} + c_x, \qquad
v = f_y \cdot \frac{y}{\mathrm{denom}} + c_y
$$

## Unprojection

**Algebraic.** Given $(m_x, m_y) = ((u - c_x) / f_x, (v - c_y) / f_y)$ and
$r^2 = m_x^2 + m_y^2$, recover $m_z$ from

$$
m_z = \frac{1 - \alpha^2 r^2}{\alpha \sqrt{1 - (2\alpha - 1) r^2} + (1 - \alpha)}
$$

The 3D ray is $(m_x, m_y, m_z) / \|(m_x, m_y, m_z)\|$.

## Validity

- $z > 0$.
- $\mathrm{denom} > 0$ (rejected as `DenominatorTooSmall`).
- $\alpha \in [0, 1]$ (enforced by validation).
- For $\alpha > 0.5$ and $r^2 > 1 / (2\alpha - 1)$ the unprojection has no
  real solution; the point is `PointOutsideImage`.

## Point Jacobian

The partials of $d$ and $\mathrm{denom}$ are

$$
\frac{\partial d}{\partial x_i} = \frac{x_i}{d}, \qquad
\frac{\partial \mathrm{denom}}{\partial x_i} = \alpha \frac{x_i}{d} + (1 - \alpha) \delta_{i,z}
$$

The point Jacobian is the standard quotient rule applied to
$u = f_x x / \mathrm{denom} + c_x$ (and $v$).

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y, \alpha]$.

The $f_x, f_y, c_x, c_y$ columns are the pinhole form (with respect to
$x / \mathrm{denom}$). The $\alpha$ column is

$$
\frac{\partial \mathrm{denom}}{\partial \alpha} = d - z
\Rightarrow
\frac{\partial (u, v)}{\partial \alpha} =
\begin{bmatrix} -f_x x (d - z) / \mathrm{denom}^2 \\ -f_y y (d - z) / \mathrm{denom}^2 \end{bmatrix}
$$

## Linear Estimation

A linear least-squares estimate of $\alpha$ is available. For each
correspondence, set $d = \sqrt{x^2 + y^2 + z^2}$, $u_c = u - c_x$, $v_c = v - c_y$, and form the row

$$
(u_c) \cdot (d - z) = (f_x x) - (u_c) z
$$
$$
(v_c) \cdot (d - z) = (f_y y) - (v_c) z
$$

which is a scalar equation in the single unknown $\alpha$. Stack the rows for
all correspondences and solve by SVD.

## Example

```rust
use apex_camera_models::{UcmCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::UCM { alpha: 0.6 };
let camera = UcmCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f_x, f_y > 0$ and finite.
- $c_x, c_y$ finite.
- $\alpha$ finite, $0 \le \alpha \le 1$.

## References

- Geyer, C. & Daniilidis, K. (2000). *A Unifying Theory for Central Panoramic Systems and Practical Implications*. ECCV 2000, LNCS 1843, 445–461.
- Mei, C. & Rives, P. (2007). *Single View Point Omnidirectional Camera Calibration from Planar Grids*. ICRA 2007, 3945–3950.
