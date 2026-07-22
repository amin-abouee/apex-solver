# Extended UCM (EUCM)

Khomutenko et al. (2016). UCM with one extra parameter $\beta$ that reshapes the
projection surface: the cross-section of the virtual sphere can become an ellipse
when $\beta \ne 1$, which makes EUCM more accurate on fisheye lenses with strong
tangential distortion.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\alpha$ | Sphere-plane coupling | — | $0 \le \alpha \le 1$ |
| $\beta$ | Ellipticity | — | $\beta > 0$ |

Total: **6 parameters**, in the order $[f_x, f_y, c_x, c_y, \alpha, \beta]$.

## Projection

With $r^2 = x^2 + y^2$, the elliptic distance and the blended denominator are

$$
d = \sqrt{\beta\,(x^2 + y^2) + z^2}, \qquad
D = \alpha\, d + (1 - \alpha)\, z,
$$

and the pixel is

$$
u = f_x \, \frac{x}{D} + c_x, \qquad
v = f_y \, \frac{y}{D} + c_y .
$$

Setting $\beta = 1$ recovers UCM exactly.

**Validity.** The denominator must satisfy $D \ge \epsilon_g$, else
`DenominatorTooSmall`. For $\alpha > \tfrac12$ there is an additional geometric
condition $z \ge D\,\dfrac{\alpha - 1}{2\alpha - 1}$; a point failing it is behind
the projection surface and raises `PointBehindCamera`.

<a id="unprojection"></a>
## Inverse Projection

**Algebraic** (Usenko et al., 3DV 2018, Eq. 41). From
$m_x = (u - c_x)/f_x$, $m_y = (v - c_y)/f_y$ and $r^2 = m_x^2 + m_y^2$,

$$
m_z = \frac{1 - \beta\,\alpha^2\, r^2}
{\alpha \sqrt{1 - (2\alpha - 1)\,\beta\, r^2} + (1 - \alpha)},
$$

and the bearing is $(m_x, m_y, m_z)$ normalised to a unit ray.

**Validity.** For $\alpha > \tfrac12$ a real solution requires
$r^2 \le \dfrac{1}{(2\alpha - 1)\,\beta}$; violating this (or a negative
radicand) raises `PointOutsideImage`, and a vanishing denominator raises
`NumericalError`.

<a id="jacobians"></a>
## Point Jacobian

The elliptic-distance partials are
$\partial d/\partial x = \beta x/d$, $\partial d/\partial y = \beta y/d$,
$\partial d/\partial z = z/d$, so
$\partial D/\partial x_i = \alpha\,\partial d/\partial x_i + (1-\alpha)\,\delta_{i,z}$.
The $2 \times 3$ point Jacobian is then the quotient rule applied to
$u = f_x\, x/D + c_x$ and $v = f_y\, y/D + c_y$.

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y, \alpha, \beta]$, the linear columns are
the pinhole form, and the two distortion columns use
$\partial D/\partial \alpha = d - z$ and
$\partial D/\partial \beta = \alpha\,(x^2 + y^2)/(2 d)$:

$$
\frac{\partial (u, v)}{\partial \alpha}
=
\begin{bmatrix} -f_x\, x\,(d - z)/D^2 \\ -f_y\, y\,(d - z)/D^2 \end{bmatrix},
\qquad
\frac{\partial (u, v)}{\partial \beta}
=
\begin{bmatrix} -f_x\, x\,\alpha\,(x^2 + y^2)/(2 d\, D^2) \\ -f_y\, y\,\alpha\,(x^2 + y^2)/(2 d\, D^2) \end{bmatrix} .
$$

## Linear Estimation

A linear least-squares estimate of $\alpha$ is available, with $\beta$ reset to
$1$ (the UCM limit). It is the UCM system with $d = \sqrt{\beta\,(x^2 + y^2) + z^2}$;
per correspondence, with $u_c = u - c_x$, $v_c = v - c_y$,

$$
u_c\,(d - z)\,\alpha = f_x\, x - u_c\, z, \qquad
v_c\,(d - z)\,\alpha = f_y\, y - v_c\, z .
$$

Stack across all correspondences and solve by SVD. **At least 1 correspondence**
is required.

## Example

```rust
use apex_camera_models::{EucmCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::EUCM { alpha: 0.6, beta: 1.0 };
let camera = EucmCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Khomutenko, B., Garcia, G. & Martinet, P. (2016). *An Enhanced Unified Camera Model*. IEEE Robotics and Automation Letters 1(1), 137–144.
- Usenko, V., Demmel, N. & Cremers, D. (2018). *The Double Sphere Camera Model*. 3DV 2018 (Eq. 41 for the EUCM inverse).
