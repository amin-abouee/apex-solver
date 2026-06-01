# Extended UCM (EUCM)

Khomutenko et al. (2016). UCM with one extra parameter $\beta$ that controls
the shape of the projection sphere. The forward projection is

$$
d = \sqrt{\beta (x^2 + y^2) + z^2}, \qquad
\mathrm{denom} = \alpha d + (1 - \alpha) z
$$

Compared to UCM, the cross-section of the virtual sphere can be an ellipse
(instead of a circle) when $\beta \ne 1$, which makes EUCM more accurate
on fisheye lenses with strong tangential distortion.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\alpha$ | Sphere-plane coupling | — | $0 \le \alpha \le 1$ |
| $\beta$ | Ellipticity | — | $\beta > 0$ |

Total: **6 parameters**.

## Projection

$$
u = f_x \cdot \frac{x}{\mathrm{denom}} + c_x, \qquad
v = f_y \cdot \frac{y}{\mathrm{denom}} + c_y
$$

where $d$ and $\mathrm{denom}$ are defined above.

## Unprojection

**Algebraic.** With $m_x, m_y$ as before and $r^2 = m_x^2 + m_y^2$,

$$
m_z = \frac{1 - \alpha^2 r^2}{\alpha \sqrt{1 - (2\alpha - 1) \beta r^2} + (1 - \alpha)}
$$

The 3D ray is $(m_x, m_y, m_z) / \|(m_x, m_y, m_z)\|$.

## Validity

- $z > 0$.
- $\mathrm{denom} > 0$.
- $\alpha \in [0, 1]$ and $\beta > 0$ (enforced).
- For $\alpha > 0.5$ and $\beta r^2 > 1 / (2\alpha - 1)$ the unprojection has
  no real solution; the point is `PointOutsideImage`.

## Point Jacobian

The partials of $d$ are $\partial d / \partial x = \beta x / d$,
$\partial d / \partial y = \beta y / d$, $\partial d / \partial z = z / d$.
The rest of the derivation follows the UCM chain rule on
$\mathrm{denom} = \alpha d + (1 - \alpha) z$.

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y, \alpha, \beta]$.

- The $f_x, f_y, c_x, c_y$ columns are the pinhole form.
- The $\alpha$ column is $-f_x x (d - z) / \mathrm{denom}^2$ (and $-f_y y \ldots$ for $v$).
- The $\beta$ column uses $\partial d / \partial \beta = (x^2 + y^2) / (2 d)$,
  leading to

$$
\frac{\partial \mathrm{denom}}{\partial \beta} = \alpha \frac{x^2 + y^2}{2 d}
\Rightarrow
\frac{\partial (u, v)}{\partial \beta} =
\begin{bmatrix} -f_x x \cdot \alpha (x^2 + y^2) / (2 d \, \mathrm{denom}^2) \\ -f_y y \cdot \alpha (x^2 + y^2) / (2 d \, \mathrm{denom}^2) \end{bmatrix}
$$

## Linear Estimation

A linear least-squares estimate of $\alpha$ is available (with $\beta$
initialised to 1.0, the UCM limit). The system is the same as UCM, except
the $d$ used in each row is now $d = \sqrt{\beta (x^2 + y^2) + z^2}$:

$$
(u_c) \cdot (d - z) = (f_x x) - (u_c) z
$$
$$
(v_c) \cdot (d - z) = (f_y y) - (v_c) z
$$

Solve by SVD. **At least 1 correspondence** is required.

## Example

```rust
use apex_camera_models::{EucmCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::EUCM { alpha: 0.6, beta: 1.0 };
let camera = EucmCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f_x, f_y > 0$ and finite.
- $c_x, c_y$ finite.
- $\alpha$ finite, $0 \le \alpha \le 1$.
- $\beta$ finite, $\beta > 0$.

## References

- Khomutenko, B., Garcia, G. & Martinet, P. (2016). *An Enhanced Unified Camera Model*. IEEE Robotics and Automation Letters 1(1), 137–144.
