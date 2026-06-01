# Radial-Tangential (Brown-Conrady)

The OpenCV / Brown-Conrady distortion model. It augments the pinhole camera
with a third-order radial polynomial plus two tangential (decentring)
coefficients. It is the default calibration model for most narrow-to-moderate
field-of-view cameras.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $k_1, k_2, k_3$ | Radial distortion | — | finite |
| $p_1, p_2$ | Tangential distortion | — | finite |

Total: **9 parameters**.

## Projection

Normalised image coordinates

$$
x' = \frac{x}{z}, \qquad y' = \frac{y}{z}, \qquad r^2 = x'^2 + y'^2
$$

Radial factor

$$
r' = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6
$$

Tangential correction

$$
\delta_x = 2 p_1 x' y' + p_2 (r^2 + 2 x'^2), \qquad
\delta_y = p_1 (r^2 + 2 y'^2) + 2 p_2 x' y'
$$

Distorted pixel coordinates

$$
u = f_x (r' x' + \delta_x) + c_x, \qquad
v = f_y (r' y' + \delta_y) + c_y
$$

## Unprojection

Iterative. The inverse is non-linear in $k_1, k_2, k_3, p_1, p_2$, so the
crate solves it with a few Newton steps on the radial-tangential residual,
starting from the un-distorted pinhole estimate.

## Validity

- $z > 0$.
- Real solutions exist only for moderate distortion; extreme $k_i$ values
  make the radial factor $r'$ negative near the principal point and the model
  is no longer monotonic in $r$.

## Point Jacobian

Computed through the chain rule on the normalised coordinates, the radial
factor, and the tangential correction. The full matrix is given by

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
=
\begin{bmatrix}
f_x \partial_x m_x & f_x \partial_y m_x & -f_x m_x / z \\
f_y \partial_x m_y & f_y \partial_y m_y & -f_y m_y / z
\end{bmatrix}
$$

where $(m_x, m_y) = (r' x' + \delta_x, r' y' + \delta_y)$ and the partials of
$m_x, m_y$ w.r.t. the normalised coordinates $(x', y')$ are the standard
derivatives of the Brown-Conrady polynomial.

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3]$.

$$
\frac{\partial (u, v)}{\partial (f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3)}
$$

The first four columns match the pinhole model (with respect to the distorted
coordinates $(m_x, m_y)$). The columns for $k_1, k_2, k_3$ are
$(f_x x' r^2, f_x x' r^4, f_x x' r^6)$ in the $u$-row (and similarly for $v$).
The $p_1, p_2$ columns are
$(2 f_x x' y', f_x (r^2 + 2 x'^2))$ and
$(f_y (r^2 + 2 y'^2), 2 f_y x' y')$ respectively.

## Linear Estimation

A linear least-squares estimate of $k_1, k_2, k_3$ is available. For each
correspondence, write $x' = x / z$, $y' = y / z$, $u_0 = f_x x' + c_x$,
$v_0 = f_y y' + c_y$, and the residual

$$
\begin{bmatrix} u - u_0 \\ v - v_0 \end{bmatrix}
=
\begin{bmatrix}
f_x x' r^2 & f_x x' r^4 & f_x x' r^6 \\
f_y y' r^2 & f_y y' r^4 & f_y y' r^6
\end{bmatrix}
\begin{bmatrix} k_1 \\ k_2 \\ k_3 \end{bmatrix}
$$

Stack the rows across all correspondences and solve the linear system by SVD.
$p_1, p_2$ are initialised to zero. **At least 3 correspondences** are
required.

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

## Validation Rules

- $f_x, f_y > 0$ and finite.
- $c_x, c_y$ finite.
- $k_1, k_2, p_1, p_2, k_3$ finite.

## References

- Brown, D. C. (1966). *Decentering Distortion of Lenses*. Photogrammetric Engineering 32(3), 444–462.
- Brown, D. C. (1971). *Close-Range Camera Calibration*. Photogrammetric Engineering 37(8), 855–866.
- Conrady, A. E. (1919). *Decentred Lens-Systems*. Monthly Notices of the Royal Astronomical Society 79(5), 384–390.
- OpenCV Camera Calibration and 3D Reconstruction — `cv::calibrateCamera` documentation.
