# Pinhole

The simplest perspective camera. No lens distortion: a 3D point $(x, y, z)$ in
the camera frame is mapped to pixel coordinates through a single division by $z$.

The model is the right baseline for narrow field-of-view lenses (FOV < ~90°)
and as an initial estimate when calibrating more complex models.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |

Total: **4 parameters**. No distortion vector.

## Projection

$$
u = f_x \cdot \frac{x}{z} + c_x, \qquad
v = f_y \cdot \frac{y}{z} + c_y
$$

The matrix form with $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$
is

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \begin{bmatrix} x \\ y \\ z \end{bmatrix}
$$

## Unprojection

$$
m_x = \frac{u - c_x}{f_x}, \qquad m_y = \frac{v - c_y}{f_y}, \qquad
\mathbf{r} = \frac{(m_x, m_y, 1)^\top}{\sqrt{m_x^2 + m_y^2 + 1}}
$$

Algebraic — no iteration. Defined everywhere in the image plane.

## Validity

- $z > 0$ (point in front of the camera).
- The crate rejects $z \le \sqrt{\varepsilon}$ (`PointAtCameraCenter`).

## Point Jacobian

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
=
\begin{bmatrix}
f_x / z & 0 & -f_x \, x / z^2 \\
0 & f_y / z & -f_y \, y / z^2
\end{bmatrix}
$$

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y]$.

$$
\frac{\partial (u, v)}{\partial (f_x, f_y, c_x, c_y)}
=
\begin{bmatrix}
x / z & 0 & 1 & 0 \\
0 & y / z & 0 & 1
\end{bmatrix}
$$

## Linear Estimation

Not provided — the pinhole model has no distortion parameters to estimate.
Once a planar or checkerboard calibration tool produces $f_x, f_y, c_x, c_y$,
the camera is fully specified.

## Example

```rust
use apex_camera_models::{PinholeCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f_x > 0$, $f_y > 0$ and finite.
- $c_x$, $c_y$ finite.
- `DistortionModel::None` only.

## References

- Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- Scaramuzza, D. & Fraundorfer, F. (2011). *Visual Odometry: Part I — The First 30 Years and Fundamentals*. IEEE Robotics & Automation Magazine.
