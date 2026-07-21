# Unified Camera Model (UCM)

Geyer & Daniilidis (2000); Mei & Rives (2007). A point is first projected onto a
unit sphere and then onto the perspective image plane, parameterised by a single
$\alpha \in [0, 1]$ that trades off para-perspective ($\alpha = 0$) against
perspective ($\alpha = 1$). Catadioptric and fisheye lenses with FOV between
$90°$ and $180°$ are well served by UCM.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\alpha$ | Sphere-plane coupling | — | $0 \le \alpha \le 1$ |

Total: **5 parameters**, in the order $[f_x, f_y, c_x, c_y, \alpha]$.

## Projection

With the ray length $d = \sqrt{x^2 + y^2 + z^2}$, the denominator blends the
sphere radius and the depth,

$$
D = \alpha \, d + (1 - \alpha)\, z,
$$

and the pixel is

$$
u = f_x \, \frac{x}{D} + c_x, \qquad
v = f_y \, \frac{y}{D} + c_y .
$$

**Validity.** Define the sphere half-angle bound
$w = \alpha/(1-\alpha)$ if $\alpha \le \tfrac12$, else $w = (1-\alpha)/\alpha$.
The point must satisfy the geometric condition $z > -w\,d$ (a point inside the
virtual sphere is behind the mirror), which raises `PointBehindCamera` on
failure. Independently, the denominator must satisfy $D \ge \epsilon_g$, else
`DenominatorTooSmall`.

<a id="unprojection"></a>
## Inverse Projection

**Algebraic** (Mei $\xi$-sphere inverse). Let $\gamma = 1 - \alpha$ and the
derived offset $\xi = \alpha/\gamma$. Scale the normalised coordinates by
$\gamma$,

$$
m_x = \gamma\,\frac{u - c_x}{f_x}, \qquad
m_y = \gamma\,\frac{v - c_y}{f_y}, \qquad
r^2 = m_x^2 + m_y^2,
$$

and form the lifting coefficient

$$
\kappa = \frac{\xi + \sqrt{1 + (1 - \xi^2)\, r^2}}{1 + r^2} .
$$

The bearing is $(\kappa m_x,\; \kappa m_y,\; \kappa - \xi)$, normalised to a unit
ray.

**Validity.** For $\alpha > \tfrac12$ a real solution exists only on the disc
$r^2 \le \dfrac{\gamma^2}{2\alpha - 1}$; outside it the point is
`PointOutsideImage`. For $\alpha \le \tfrac12$ the inverse is unconstrained.

<a id="jacobians"></a>
## Point Jacobian

The denominator partials are

$$
\frac{\partial d}{\partial x_i} = \frac{x_i}{d}, \qquad
\frac{\partial D}{\partial x_i} = \alpha \frac{x_i}{d} + (1 - \alpha)\,\delta_{i,z},
$$

and the $2 \times 3$ point Jacobian is the quotient rule applied to
$u = f_x\, x/D + c_x$ and $v = f_y\, y/D + c_y$, e.g.
$\partial u/\partial x_i = f_x\,(D\,\delta_{i,x} - x\,\partial D/\partial x_i)/D^2$.

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y, \alpha]$, the linear columns are the
pinhole form (with respect to $x/D$), and the $\alpha$ column uses
$\partial D/\partial \alpha = d - z$:

$$
\frac{\partial (u, v)}{\partial \alpha}
=
\begin{bmatrix}
-f_x\, x\,(d - z)/D^2 \\
-f_y\, y\,(d - z)/D^2
\end{bmatrix} .
$$

## Linear Estimation

A linear least-squares estimate of $\alpha$ is available (the intrinsics
$f_x, f_y, c_x, c_y$ must already be set). With $d = \sqrt{x^2 + y^2 + z^2}$,
$u_c = u - c_x$, $v_c = v - c_y$, each correspondence contributes two scalar
equations linear in $\alpha$,

$$
u_c\,(d - z)\,\alpha = f_x\, x - u_c\, z, \qquad
v_c\,(d - z)\,\alpha = f_y\, y - v_c\, z .
$$

Stack across all correspondences and solve by SVD. **At least 1 correspondence**
is required.

## Example

```rust
use apex_camera_models::{UcmCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::UCM { alpha: 0.6 };
let camera = UcmCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Geyer, C. & Daniilidis, K. (2000). *A Unifying Theory for Central Panoramic Systems and Practical Implications*. ECCV 2000, LNCS 1843, 445–461.
- Mei, C. & Rives, P. (2007). *Single View Point Omnidirectional Camera Calibration from Planar Grids*. ICRA 2007, 3945–3950.
