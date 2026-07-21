# F-Theta (NVIDIA)

The NVIDIA f-theta fisheye model. A degree-4 polynomial maps the **incidence
angle** $\theta$ of a 3D ray to an **image-plane radius** $f(\theta)$, while the
**azimuth** $\phi$ of the ray is preserved exactly. The model is isotropic — the
polynomial absorbs the focal length, so there is no separate $f_x$ / $f_y$ — and
is the convention used by NVIDIA DriveWorks for automotive surround-view cameras.
It handles fields of view up to about $220°$.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $k_1$ | Linear coefficient (focal length) | px/rad | $k_1 > 0$, finite |
| $k_2$ | Quadratic coefficient | px/rad² | finite |
| $k_3$ | Cubic coefficient | px/rad³ | finite |
| $k_4$ | Quartic coefficient | px/rad⁴ | finite |

Total: **6 parameters**, in the order $[c_x, c_y, k_1, k_2, k_3, k_4]$.

## Projection

With $d = \sqrt{x^2 + y^2 + z^2}$, the incidence angle
$\theta = \arccos(z/d) \in [0, \pi]$ and the image-plane radius
$r_p = \sqrt{x^2 + y^2}$, the forward polynomial is

$$
f(\theta) = k_1 \theta + k_2 \theta^2 + k_3 \theta^3 + k_4 \theta^4 .
$$

Spreading $f(\theta)$ along the image-plane unit direction $(x/r_p,\, y/r_p)$,

$$
u = c_x + f(\theta)\, \frac{x}{r_p}, \qquad
v = c_y + f(\theta)\, \frac{y}{r_p} .
$$

**Validity.** The depth must satisfy $z \ge \epsilon_g$ (`MIN_DEPTH`), else
`PointBehindCamera`. The on-axis case $r_p < \epsilon_g$ has no defined azimuth
and projects exactly to the principal point $(u, v) = (c_x, c_y)$.

<a id="unprojection"></a>
## Inverse Projection

**Iterative.** With $\delta_x = u - c_x$, $\delta_y = v - c_y$ and
$r_d = \sqrt{\delta_x^2 + \delta_y^2}$, solve $f(\theta) = r_d$ by Newton-Raphson
from $\theta_0 = r_d / k_1$, using
$f'(\theta) = k_1 + 2 k_2 \theta + 3 k_3 \theta^2 + 4 k_4 \theta^3$ (up to 100
iterations). The unit ray is

$$
\mathbf{r} = \operatorname{normalize}
\begin{bmatrix} \sin\theta \, \delta_x / r_d \\ \sin\theta \, \delta_y / r_d \\ \cos\theta \end{bmatrix} .
$$

**Validity.** If $r_d < \epsilon_g$ the ray is $(0, 0, 1)$. If Newton-Raphson
returns a non-finite or negative $\theta$, the inverse raises `NumericalError`.

<a id="jacobians"></a>
## Point Jacobian

With $r_p = \sqrt{x^2 + y^2}$, $d^2 = x^2 + y^2 + z^2$, and the two scalars

$$
A = \frac{f'(\theta)\, z}{r_p^2\, d^2}, \qquad
B = \frac{f(\theta)}{r_p^3},
$$

the $2 \times 3$ point Jacobian is

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
=
\begin{bmatrix}
A x^2 + B y^2 & (A - B)\,x y & -f'(\theta)\, x / d^2 \\
(A - B)\,x y & A y^2 + B x^2 & -f'(\theta)\, y / d^2
\end{bmatrix} .
$$

**At the optical axis** ($r_p < \epsilon_g$) this degenerates to
$\begin{bmatrix} k_1/z & 0 & 0 \\ 0 & k_1/z & 0 \end{bmatrix}$.

## Intrinsic Jacobian

With parameter order $[c_x, c_y, k_1, k_2, k_3, k_4]$ and the azimuth cosines
$c_\phi = x/r_p$, $s_\phi = y/r_p$,

$$
\frac{\partial (u, v)}{\partial (c_x, c_y, k_1, k_2, k_3, k_4)}
=
\begin{bmatrix}
1 & 0 & \theta\, c_\phi & \theta^2 c_\phi & \theta^3 c_\phi & \theta^4 c_\phi \\
0 & 1 & \theta\, s_\phi & \theta^2 s_\phi & \theta^3 s_\phi & \theta^4 s_\phi
\end{bmatrix} .
$$

**At the optical axis** the four $k_i$ columns are zero.

## Linear Estimation

For each correspondence compute $\theta = \arccos(z/d)$ and the observed radius
$r = \sqrt{(u - c_x)^2 + (v - c_y)^2}$. The polynomial is linear in the $k_i$,
giving a Vandermonde row

$$
\begin{bmatrix} \theta & \theta^2 & \theta^3 & \theta^4 \end{bmatrix}
\begin{bmatrix} k_1 \\ k_2 \\ k_3 \\ k_4 \end{bmatrix} = r .
$$

Stack across all correspondences and solve by SVD. **At least 4 correspondences**
are required. The estimator returns a fresh `FThetaCamera` value (rather than
mutating in place, unlike the other models).

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

## References

- NVIDIA, *The f-theta Camera Model*, DriveWorks technical documentation.
- Scaramuzza, D., Martinelli, A. & Siegwart, R. (2006). *A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion*. ICVS 2006.
- Abraham, S. & Förstner, W. (2005). *Fish-Eye-Stereo Calibration and Epipolar Rectification*. ISPRS Journal of Photogrammetry and Remote Sensing 59(5), 278–288.
