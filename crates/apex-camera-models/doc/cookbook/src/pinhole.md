# Pinhole

The simplest perspective camera. There is no lens distortion: a 3D point
$p_{\mathrm{cam}} = (x, y, z)$ in the camera frame is mapped to a pixel
$p_{uv} = (u, v)$ by a single division by the depth $z$, followed by the focal
lengths $f_x, f_y$ and the principal point $(c_x, c_y)$.

It is the right baseline for narrow field-of-view lenses ($\text{FOV} < 90°$)
and the natural initial estimate when calibrating any of the richer models in
this book.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |

Total: **4 parameters**, in the order $[f_x, f_y, c_x, c_y]$. No distortion vector.

## Projection

Writing the normalised (retinal) coordinates as $x' = x/z$ and $y' = y/z$, the
pinhole map applies the focal length and principal point directly:

$$
u = f_x \, \frac{x}{z} + c_x, \qquad
v = f_y \, \frac{y}{z} + c_y .
$$

Equivalently, with the calibration matrix
$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$,
the projection in homogeneous form is
$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \begin{bmatrix} x \\ y \\ z \end{bmatrix}$.

**Validity.** The point must lie in front of the camera: the map requires
$z \ge \epsilon_g$ ($\epsilon_g = 10^{-6}$). A depth $z < \epsilon_g$ is rejected
with `PointBehindCamera { z, min_z: `$\epsilon_g$` }`.

<a id="unprojection"></a>
## Inverse Projection

Algebraic and defined everywhere in the image plane. From the normalised
coordinates $m_x = (u - c_x)/f_x$ and $m_y = (v - c_y)/f_y$, the back-projected
unit ray is

$$
\mathbf{r} = \frac{1}{\sqrt{m_x^2 + m_y^2 + 1}}
\begin{bmatrix} m_x \\ m_y \\ 1 \end{bmatrix} .
$$

**Validity.** None — there is no iteration and no bounded domain, so the inverse
never fails (its `Result` exists only for trait uniformity).

<a id="jacobians"></a>
## Point Jacobian

Differentiating $(u, v)$ with respect to $(x, y, z)$ gives the $2 \times 3$ matrix

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
=
\begin{bmatrix}
f_x / z & 0 & -f_x \, x / z^2 \\
0 & f_y / z & -f_y \, y / z^2
\end{bmatrix} .
$$

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y]$ and $x' = x/z$, $y' = y/z$,

$$
\frac{\partial (u, v)}{\partial (f_x, f_y, c_x, c_y)}
=
\begin{bmatrix}
x' & 0 & 1 & 0 \\
0 & y' & 0 & 1
\end{bmatrix} .
$$

## Linear Estimation

Not provided — the pinhole model has no distortion parameters to estimate. Once
a planar or checkerboard calibration tool produces $f_x, f_y, c_x, c_y$, the
camera is fully specified.

## Example

```rust
use apex_camera_models::{PinholeCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- Scaramuzza, D. & Fraundorfer, F. (2011). *Visual Odometry: Part I — The First 30 Years and Fundamentals*. IEEE Robotics & Automation Magazine.
