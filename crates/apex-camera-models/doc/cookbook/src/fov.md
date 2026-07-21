# Field-of-View (FOV)

Devernay & Faugeras (2001). A single scalar $\omega$ controls the amount of
distortion. The projection is an arctangent of the lateral distance scaled by
$\tan(\omega/2)$, which gives near-uniform angular resolution around the optical
axis and a graceful saturation at the edges of a wide lens. It is popular in
visual-SLAM stacks that need a wide lens without a higher-order polynomial.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\omega$ | Field-of-view parameter | rad | $0 < \omega \le \pi$ |

Total: **5 parameters**, in the order $[f_x, f_y, c_x, c_y, \omega]$.

## Projection

With the lateral radius $r = \sqrt{x^2 + y^2}$, define the auxiliary angle
argument $\psi = \dfrac{2 \tan(\omega/2)\, r}{z}$. The distortion scale is

$$
r_d =
\begin{cases}
\dfrac{\arctan \psi}{\omega \, r}, & r \ge \epsilon_g, \\[2ex]
\dfrac{2 \tan(\omega/2)}{\omega}, & r < \epsilon_g,
\end{cases}
$$

and the pixel is $u = f_x \, r_d \, x + c_x$, $v = f_y \, r_d \, y + c_y$ (note
$r_d$ already carries the $1/r$ factor, so it multiplies $x$ and $y$ directly).

**Validity.** The depth must satisfy $z \ge \epsilon_g$, else
`ProjectionOutOfBounds`. The parameter is constrained to $0 < \omega \le \pi$
(the map degenerates at $\omega = 0$ and saturates beyond $\pi$); the on-axis
case $r < \epsilon_g$ uses the finite limit above.

<a id="unprojection"></a>
## Inverse Projection

Given $m_x = (u - c_x)/f_x$, $m_y = (v - c_y)/f_y$ and $r_d = \sqrt{m_x^2 + m_y^2}$,
invert the arctangent to recover the undistorted radius
$r_u = \dfrac{\tan(r_d\,\omega)}{2 \tan(\omega/2)}$. With $s = \sqrt{1 + r_u^2}$
the unit ray is

$$
\mathbf{r} =
\begin{bmatrix}
\dfrac{m_x \, r_u}{r_d \, s} \\[1.5ex]
\dfrac{m_y \, r_u}{r_d \, s} \\[1.5ex]
\dfrac{1}{s}
\end{bmatrix} .
$$

**Validity.** If $r_d < \epsilon_g$ the ray is exactly $(0, 0, 1)$. The inverse
is otherwise closed-form on $r_d \, \omega < \pi/2$ (where $\tan$ stays finite).

<a id="jacobians"></a>
## Point Jacobian

The $2 \times 3$ point Jacobian is the chain rule on the forward map, whose main
dependencies are $\partial r_d / \partial r$ and $\partial r_d / \partial z$
through $\psi = 2 \tan(\omega/2)\, r / z$. The exact assembled expression is
implemented in the crate.

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y, \omega]$, the linear columns match the
pinhole form (with respect to $r_d x$ and $r_d y$), and the $\omega$ column
follows from

$$
\frac{\partial r_d}{\partial \omega}
= \frac{1}{\omega}\left( \frac{\psi}{1 + \psi^2}\,\frac{\partial \psi}{\partial \omega} - \frac{\arctan\psi}{\omega} \right) \frac{1}{r},
$$

evaluated per pixel (the complete expression is in the implementation).

## Linear Estimation

**Not** a closed-form linear system — the dependence on $\omega$ is non-linear.
The implementation runs a 1-D grid search over
$\omega \in \{0.10, 0.11, \ldots, 2.99\}$ rad, picks the value minimising the
mean reprojection error, and assigns it to the camera. **At least 2
correspondences** are required.

## Example

```rust
use apex_camera_models::{FovCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::FOV { w: 1.0 };
let camera = FovCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Devernay, F. & Faugeras, O. (2001). *Straight Lines Have to Be Straight: Automatic Calibration and Removal of Distortion from Scenes of Structured Environments*. Machine Vision and Applications 13(1), 14–24.
