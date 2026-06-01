# F-Theta (NVIDIA)

The NVIDIA f-theta fisheye model. A degree-4 polynomial maps the **incidence
angle** $\theta$ of a 3D ray to an **image-plane radius**, while the
**azimuth** $\phi$ of the ray is preserved exactly. The model is isotropic
(no separate $f_x$ / $f_y$ — the polynomial absorbs the focal length) and is
the convention used by NVIDIA DriveWorks for automotive surround-view
cameras.

The model handles FOVs up to about $220°$.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $k_1$ | Linear coefficient (focal length) | pixels/rad | $k_1 > 0$ |
| $k_2$ | Quadratic coefficient | pixels/rad² | finite |
| $k_3$ | Cubic coefficient | pixels/rad³ | finite |
| $k_4$ | Quartic coefficient | pixels/rad⁴ | finite |

Total: **6 parameters**. The parameter vector is
$[c_x, c_y, k_1, k_2, k_3, k_4]$.

## Projection

Let $d = \sqrt{x^2 + y^2 + z^2}$, $r_p = \sqrt{x^2 + y^2}$, and
$\theta = \arccos(z / d) \in [0, \pi]$. The forward polynomial is

$$
f(\theta) = k_1 \theta + k_2 \theta^2 + k_3 \theta^3 + k_4 \theta^4
$$

The unit direction in the image plane is $(x / r_p, y / r_p)$ (with safe
handling for $r_p = 0$):

$$
u = c_x + f(\theta) \cdot \frac{x}{r_p}, \qquad
v = c_y + f(\theta) \cdot \frac{y}{r_p}
$$

**Special case** $r_p \approx 0$ (on the optical axis): the point projects
to the principal point $(u, v) = (c_x, c_y)$.

## Unprojection

**Iterative.** Given $(u, v)$, set $\delta_x = u - c_x$, $\delta_y = v - c_y$,
and $r_d = \sqrt{\delta_x^2 + \delta_y^2}$. If $r_d \approx 0$, the ray is
$(0, 0, 1)$. Otherwise solve $f(\theta) = r_d$ for $\theta$ by Newton-Raphson
with derivative $f'(\theta) = k_1 + 2 k_2 \theta + 3 k_3 \theta^2 + 4 k_4 \theta^3$
and initial guess $\theta_0 = r_d / k_1$. The unit ray is

$$
\mathbf{r} = (\sin\theta \cdot \delta_x / r_d, \; \sin\theta \cdot \delta_y / r_d, \; \cos\theta)^\top
$$

## Validity

- $z > 0$ (`MIN_DEPTH` threshold, default $10^{-6}$).
- $r_p \approx 0$ is handled separately (returns the principal point).

## Point Jacobian

With $c_\phi = x / r_p$ and $s_\phi = y / r_p$ the chain rule gives

$$
\frac{\partial (u, v)}{\partial (x, y, z)} =
\begin{bmatrix}
f'(\theta) \cdot c_\phi \cdot \partial\theta/\partial x + f(\theta) \cdot \partial(c_\phi)/\partial x & \ldots & \ldots \\
f'(\theta) \cdot s_\phi \cdot \partial\theta/\partial x + f(\theta) \cdot \partial(s_\phi)/\partial x & \ldots & \ldots
\end{bmatrix}
$$

where

$$
\frac{\partial \theta}{\partial x} = -\frac{x}{d^2} \cdot \frac{1}{\sin\theta}, \quad
\frac{\partial \theta}{\partial y} = -\frac{y}{d^2} \cdot \frac{1}{\sin\theta}, \quad
\frac{\partial \theta}{\partial z} = \frac{d - z^2 / d}{d^2 \sin\theta}
$$

and the partials of $c_\phi = x / r_p$, $s_\phi = y / r_p$ are the standard
derivatives of a planar unit vector.

**At the optical axis** ($r_p < \varepsilon$): the Jacobian degenerates to
$\begin{bmatrix} k_1 / z & 0 & 0 \\ 0 & k_1 / z & 0 \end{bmatrix}$.

## Intrinsic Jacobian

Parameter order: $[c_x, c_y, k_1, k_2, k_3, k_4]$.

$$
\frac{\partial (u, v)}{\partial (c_x, c_y, k_1, k_2, k_3, k_4)}
=
\begin{bmatrix}
1 & 0 & \theta c_\phi & \theta^2 c_\phi & \theta^3 c_\phi & \theta^4 c_\phi \\
0 & 1 & \theta s_\phi & \theta^2 s_\phi & \theta^3 s_\phi & \theta^4 s_\phi
\end{bmatrix}
$$

**At the optical axis** the $k_i$ columns are zero.

## Linear Estimation

For each correspondence, compute $\theta = \arccos(z / d)$ and $r = \sqrt{(u - c_x)^2 + (v - c_y)^2}$.
The linear system

$$
\begin{bmatrix}
\theta_i & \theta_i^2 & \theta_i^3 & \theta_i^4
\end{bmatrix}
\begin{bmatrix} k_1 \\ k_2 \\ k_3 \\ k_4 \end{bmatrix}
= r_i
$$

is stacked across all correspondences and solved by SVD.
**At least 4 correspondences** are required. Returns a fresh
`FThetaCamera` value rather than mutating in place (different signature
from the other models).

## Example

```rust
use apex_camera_models::FThetaCamera;

let camera = FThetaCamera::new(
    640.0, 400.0,
    apex_camera_models::DistortionModel::FTheta {
        k1: 800.0, k2: -0.5, k3: 0.1, k4: -0.01,
    },
)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $k_1 > 0$ and finite.
- $k_2, k_3, k_4$ finite.
- $c_x, c_y$ finite.

## References

- NVIDIA, *The f-theta Camera Model*, internal whitepaper.
- Scaramuzza, D., Martinelli, A. & Siegwart, R. (2006). *A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion*. ICVS 2006.
- Abraham, S. & Förstner, W. (2005). *Fish-Eye-Stereo Calibration and Epipolar Rectification*. ISPRS Journal of Photogrammetry and Remote Sensing 59(5), 278–288.
