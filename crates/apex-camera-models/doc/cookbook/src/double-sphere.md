# Double Sphere

Usenko et al. (2018). A point is first projected onto a unit sphere displaced by
$\xi$ along the optical axis, then onto a second unit sphere. The resulting
projection is closed-form and empirically outperforms UCM and EUCM on cameras
with FOV $> 180°$ (catadioptric panoramic rigs, $360°$ lenses).

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\xi$ | First-sphere offset | — | $-1 \le \xi \le 1$ |
| $\alpha$ | Second-sphere coupling | — | $0 < \alpha \le 1$ |

Total: **6 parameters**, in the order $[f_x, f_y, c_x, c_y, \xi, \alpha]$.

## Projection

With $r^2 = x^2 + y^2$, the two sphere distances and the blended denominator are

$$
d_1 = \sqrt{x^2 + y^2 + z^2}, \qquad
w = \xi\, d_1 + z, \qquad
d_2 = \sqrt{x^2 + y^2 + w^2},
$$

$$
D = \alpha\, d_2 + (1 - \alpha)\, w,
$$

and the pixel is $u = f_x \, x/D + c_x$, $v = f_y \, y/D + c_y$.

**Validity.** Define
$w_1 = (1-\alpha)/\alpha$ if $\alpha > \tfrac12$, else $w_1 = \alpha/(1-\alpha)$,
and $w_2 = \dfrac{w_1 + \xi}{\sqrt{2 w_1 \xi + \xi^2 + 1}}$. The forward map
requires the geometric condition $z > -w_2\, d_1$; failing it raises
`ProjectionOutOfBounds`. The denominator must also satisfy $D \ge \epsilon_g$,
else `DenominatorTooSmall`.

<a id="unprojection"></a>
## Inverse Projection

**Algebraic.** From $m_x = (u - c_x)/f_x$, $m_y = (v - c_y)/f_y$ and
$r^2 = m_x^2 + m_y^2$, undo the second sphere,

$$
m_z = \frac{1 - \alpha^2\, r^2}{\alpha \sqrt{1 - (2\alpha - 1)\, r^2} + (1 - \alpha)},
$$

then undo the first with the lifting coefficient

$$
\kappa = \frac{m_z\, \xi + \sqrt{m_z^2 + (1 - \xi^2)\, r^2}}{m_z^2 + r^2},
$$

giving the point $(\kappa m_x,\; \kappa m_y,\; \kappa m_z - \xi)$, normalised to a
unit ray.

**Validity.** For $\alpha > \tfrac12$ a real solution requires
$r^2 \le \dfrac{1}{2\alpha - 1}$; otherwise (or when $m_z^2 + r^2 < \epsilon_g$)
the point is `PointOutsideImage`.

<a id="jacobians"></a>
## Point Jacobian

The chain rule runs through $d_1$, $w = \xi d_1 + z$, $d_2$, and $D = \alpha d_2 + (1-\alpha) w$:

$$
\frac{\partial d_1}{\partial x_i} = \frac{x_i}{d_1}, \qquad
\frac{\partial w}{\partial x_i} = \xi\,\frac{x_i}{d_1}\ (i \in \{x, y\}), \qquad
\frac{\partial w}{\partial z} = \xi\,\frac{z}{d_1} + 1,
$$

$$
\frac{\partial d_2}{\partial x_i} = \frac{x_i + w\,\partial w/\partial x_i}{d_2}, \qquad
\frac{\partial D}{\partial x_i} = \alpha\,\frac{\partial d_2}{\partial x_i} + (1 - \alpha)\,\frac{\partial w}{\partial x_i} .
$$

The $2 \times 3$ point Jacobian is then the quotient rule on $u = f_x x/D + c_x$
and $v = f_y y/D + c_y$.

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y, \xi, \alpha]$, the linear columns are
the pinhole form. The $\xi$ column propagates $\partial w/\partial \xi = d_1$
through $d_2$ and $D$. The $\alpha$ column uses $\partial D/\partial \alpha = d_2 - w$:

$$
\frac{\partial (u, v)}{\partial \alpha}
=
\begin{bmatrix} -f_x\, x\,(d_2 - w)/D^2 \\ -f_y\, y\,(d_2 - w)/D^2 \end{bmatrix} .
$$

## Linear Estimation

A linear least-squares estimate of $\alpha$ is available, with $\xi$ fixed to $0$
(the UCM limit). It is the UCM system with $d = \sqrt{x^2 + y^2 + z^2}$; per
correspondence, with $u_c = u - c_x$, $v_c = v - c_y$,

$$
u_c\,(d - z)\,\alpha = f_x\, x - u_c\, z, \qquad
v_c\,(d - z)\,\alpha = f_y\, y - v_c\, z .
$$

Stack across all correspondences and solve by SVD. **At least 1 correspondence**
is required.

## Example

```rust
use apex_camera_models::{DoubleSphereCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::DoubleSphere { xi: -0.2, alpha: 0.6 };
let camera = DoubleSphereCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Usenko, V., Demmel, N., Schubert, D., Stückler, J. & Cremers, D. (2018). *The Double Sphere Camera Model*. 3DV 2018, 552–560. arXiv:1807.08957.
