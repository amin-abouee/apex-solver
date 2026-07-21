# Kannala-Brandt Fisheye

A widely-used fisheye model that maps the **incidence angle** $\theta$ of a 3D
ray to a **distorted radius** $d(\theta)$ through an odd-order polynomial. It is
the model OpenCV calls "fisheye" and Kalibr calls "equidistant", and it handles
fields of view up to about $180°$.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$, finite |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$, finite |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $k_1, k_2, k_3, k_4$ | Polynomial coefficients | — | finite |

Total: **8 parameters**, in the order $[f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4]$.

## Projection

With the lateral radius $r = \sqrt{x^2 + y^2}$ and the incidence angle
$\theta = \operatorname{atan2}(r, z)$, the forward polynomial is

$$
d(\theta) = \theta + k_1 \theta^3 + k_2 \theta^5 + k_3 \theta^7 + k_4 \theta^9 .
$$

The distorted radius $d(\theta)$ is spread along the image-plane unit direction
$(x/r,\, y/r)$, giving

$$
u = f_x \, d(\theta) \, \frac{x}{r} + c_x, \qquad
v = f_y \, d(\theta) \, \frac{y}{r} + c_y .
$$

**Validity.** The depth must satisfy $z \ge \epsilon_g$, else
`PointBehindCamera`. The on-axis case $r < \epsilon_g$ has no defined azimuth
and is handled separately: the point collapses to the pinhole limit
$u = f_x x/z + c_x$, $v = f_y y/z + c_y$ (i.e. the principal point as
$x, y \to 0$).

<a id="unprojection"></a>
## Inverse Projection

**Iterative.** From $m_x = (u - c_x)/f_x$, $m_y = (v - c_y)/f_y$, let
$r_d = \sqrt{m_x^2 + m_y^2}$ (clamped to $\pi/2$). We need the $\theta$ with
$d(\theta) = r_d$, found by Newton-Raphson on $d(\theta) - r_d = 0$ using

$$
d'(\theta) = 1 + 3 k_1 \theta^2 + 5 k_2 \theta^4 + 7 k_3 \theta^6 + 9 k_4 \theta^8,
$$

started at $\theta_0 = r_d$ (up to 10 iterations, tolerance $10^{-6}$). With the
recovered $\theta$ and the scale $s = \sin\theta / r_d$, the unit ray is

$$
\mathbf{r} = \operatorname{normalize}
\begin{bmatrix} s \, m_x \\ s \, m_y \\ \cos\theta \end{bmatrix} .
$$

**Validity.** If $r_d < \epsilon_g$ the ray is exactly $(0, 0, 1)$. A vanishing
derivative $|d'(\theta)| < \varepsilon_{\text{machine}}$ raises `NumericalError`.

<a id="jacobians"></a>
## Point Jacobian

With $r = \sqrt{x^2 + y^2}$, $\theta = \arctan(r/z)$ and $\rho^2 = x^2 + y^2 + z^2$,
the intermediate partials are

$$
\frac{\partial r}{\partial x} = \frac{x}{r}, \quad
\frac{\partial r}{\partial y} = \frac{y}{r}, \quad
\frac{\partial r}{\partial z} = 0,
$$

$$
\frac{\partial \theta}{\partial x} = \frac{z\,x}{r\,\rho^2}, \quad
\frac{\partial \theta}{\partial y} = \frac{z\,y}{r\,\rho^2}, \quad
\frac{\partial \theta}{\partial z} = -\frac{r}{\rho^2} .
$$

The $2 \times 3$ point Jacobian follows by the chain rule through $d(\theta)$ and
the unit direction $(x/r,\, y/r)$; the exact assembled expression is implemented
in the crate.

## Intrinsic Jacobian

With parameter order $[f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4]$, the linear
columns are the distorted-radius pinhole form and the polynomial columns are

$$
\frac{\partial (u, v)}{\partial k_i}
=
\begin{bmatrix}
f_x \, \theta^{2i+1} \, x / r \\
f_y \, \theta^{2i+1} \, y / r
\end{bmatrix},
\qquad i = 1, 2, 3, 4 .
$$

## Linear Estimation

For each correspondence compute $\theta = \arctan(r/z)$ with
$r = \sqrt{x^2 + y^2}$ and, for $r > 0$, the unit direction $(x/r,\, y/r)$. The
$k_i$ enter linearly through

$$
a_i = f_x \, \theta^{2i+1} \, \frac{x}{r}, \qquad
b_i = f_y \, \theta^{2i+1} \, \frac{y}{r}, \qquad i = 1, 2, 3, 4,
$$

giving, per correspondence, the two rows

$$
\begin{bmatrix} a_1 & a_2 & a_3 & a_4 \\ b_1 & b_2 & b_3 & b_4 \end{bmatrix}
\begin{bmatrix} k_1 \\ k_2 \\ k_3 \\ k_4 \end{bmatrix}
=
\begin{bmatrix} (u - c_x) - f_x \, \theta \, x / r \\ (v - c_y) - f_y \, \theta \, y / r \end{bmatrix} .
$$

Stack across all correspondences and solve by SVD. **At least 4 correspondences**
are required.

## Example

```rust
use apex_camera_models::{KannalaBrandtCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::KannalaBrandt {
    k1: -0.02, k2: 0.0, k3: 0.0, k4: 0.0,
};
let camera = KannalaBrandtCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Kannala, J. & Brandt, S. S. (2006). *A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses*. IEEE TPAMI 28(8), 1335–1340.
- Usenko, V., Demmel, N. & Cremers, D. (2018). *The Double Sphere Camera Model*. 3DV 2018 (Eqs. 27–32 for the Kannala-Brandt forward/inverse).
