# Radial-Tangential (Brown-Conrady)

The OpenCV / Brown-Conrady distortion model. It augments the pinhole camera
with a third-order **radial** polynomial ($k_1, k_2, k_3$) plus two **tangential**
(decentring) coefficients ($p_1, p_2$). It is the default calibration model for
most narrow-to-moderate field-of-view cameras.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $k_1, k_2, k_3$ | Radial distortion | — | finite |
| $p_1, p_2$ | Tangential distortion | — | finite |

Total: **9 parameters**, in the order $[f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3]$.

## Projection

Start from the normalised coordinates $x' = x/z$, $y' = y/z$ and their squared
radius $r^2 = x'^2 + y'^2$. The **radial factor** and **tangential correction**
are

$$
r' = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6,
$$

$$
\delta_x = 2 p_1 x' y' + p_2 (r^2 + 2 x'^2), \qquad
\delta_y = p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' .
$$

Writing the distorted retinal point as
$(m_x, m_y) = (r' x' + \delta_x,\; r' y' + \delta_y)$, the pixel is

$$
u = f_x \, m_x + c_x, \qquad v = f_y \, m_y + c_y .
$$

**Validity.** The depth must satisfy $z \ge \epsilon_g$, else
`PointBehindCamera`. In addition, extreme radial coefficients can drive the
radial factor non-positive; if $r' \le \epsilon_g$ the model is no longer
monotonic in $r$ and the point is rejected with `PointOutsideImage`.

<a id="unprojection"></a>
## Inverse Projection

**Iterative.** The forward map is a degree-6 polynomial in $(x', y')$, so the
inverse has no closed form. Given the distorted target
$(m_x, m_y) = (\,(u - c_x)/f_x,\; (v - c_y)/f_y\,)$, the crate runs Newton-Raphson
on the 2-D residual $\mathbf{g}(x', y') = (r' x' + \delta_x - m_x,\; r' y' + \delta_y - m_y)$,
starting from $(x', y') = (m_x, m_y)$ and inverting the analytic $2 \times 2$
Jacobian $\partial \mathbf{g} / \partial (x', y')$ at each step (up to 100
iterations). Once $(x', y')$ has converged, the unit ray is

$$
\mathbf{r} = \frac{1}{\sqrt{x'^2 + y'^2 + 1}}
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} .
$$

**Validity.** A singular step Jacobian ($|\det| < \epsilon_g$) or failure to
converge within 100 iterations raises `NumericalError`. Real solutions exist
only for moderate distortion.

<a id="jacobians"></a>
## Point Jacobian

The point Jacobian is the chain rule through the normalised coordinates, the
radial factor, and the tangential correction. With
$(m_x, m_y) = (r' x' + \delta_x,\; r' y' + \delta_y)$,

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
=
\begin{bmatrix}
f_x \, \partial_x m_x & f_x \, \partial_y m_x & -f_x \, m_x / z \\
f_y \, \partial_x m_y & f_y \, \partial_y m_y & -f_y \, m_y / z
\end{bmatrix},
$$

where $\partial_x, \partial_y$ denote the partials with respect to the normalised
coordinates $(x', y')$ (the standard derivatives of the Brown-Conrady
polynomial), and the $z$ column follows from $\partial x'/\partial z = -x'/z$.

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3]$ and the
distorted retinal point $(m_x, m_y)$,

$$
\frac{\partial (u, v)}{\partial (f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3)}
=
\begin{bmatrix}
m_x & 0 & 1 & 0 & f_x x' r^2 & f_x x' r^4 & 2 f_x x' y' & f_x (r^2 + 2 x'^2) & f_x x' r^6 \\
0 & m_y & 0 & 1 & f_y y' r^2 & f_y y' r^4 & f_y (r^2 + 2 y'^2) & 2 f_y x' y' & f_y y' r^6
\end{bmatrix} .
$$

The first four columns are the pinhole form with respect to the distorted
coordinates $(m_x, m_y)$.

## Linear Estimation

A linear least-squares estimate of the radial coefficients $k_1, k_2, k_3$ is
available (the tangential $p_1, p_2$ are initialised to zero). For each
correspondence, form the undistorted pinhole prediction
$u_0 = f_x x' + c_x$, $v_0 = f_y y' + c_y$; the residual is linear in the $k_i$:

$$
\begin{bmatrix} u - u_0 \\ v - v_0 \end{bmatrix}
=
\begin{bmatrix}
f_x x' r^2 & f_x x' r^4 & f_x x' r^6 \\
f_y y' r^2 & f_y y' r^4 & f_y y' r^6
\end{bmatrix}
\begin{bmatrix} k_1 \\ k_2 \\ k_3 \end{bmatrix} .
$$

Stack the two rows across all correspondences and solve by SVD. **At least 3
correspondences** are required.

## Example

```rust
use apex_camera_models::{RadTanCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
let distortion = DistortionModel::BrownConrady {
    k1: -0.2, k2: 0.1, p1: 0.0, p2: 0.0, k3: 0.0,
};
let camera = RadTanCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Brown, D. C. (1966). *Decentering Distortion of Lenses*. Photogrammetric Engineering 32(3), 444–462.
- Brown, D. C. (1971). *Close-Range Camera Calibration*. Photogrammetric Engineering 37(8), 855–866.
- Conrady, A. E. (1919). *Decentred Lens-Systems*. Monthly Notices of the Royal Astronomical Society 79(5), 384–390.
- OpenCV Camera Calibration and 3D Reconstruction — `cv::calibrateCamera` documentation.
