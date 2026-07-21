# BAL Pinhole

The "Bundle Adjustment in the Large" pinhole variant used by Bundler, Ceres, and
GTSAM. Two differences from the standard pinhole model:

1. The camera looks down the **$-Z$ axis** — the world is in front of the camera
   when $z < 0$.
2. The intrinsic vector is reduced to **3** scalars in the strict form: a single
   focal length $f = f_x = f_y$, a radial pair $k_1, k_2$, and a fixed principal
   point at the image origin ($c_x = c_y = 0$). This is the BAL dataset
   parameterisation, enforced by `BALPinholeCameraStrict::new`.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f$ | Shared focal length | pixels | $f > 0$, finite |
| $k_1$ | Radial distortion (2nd order) | — | finite |
| $k_2$ | Radial distortion (4th order) | — | finite |

Total: **3 parameters**, in the order $[f, k_1, k_2]$. The strict form additionally
requires $|f_x - f_y| \le 10^{-10}$ and $|c_x|, |c_y| \le 10^{-10}$, and accepts
only `DistortionModel::Radial { k1, k2 }`.

## Projection

Because the camera looks down $-Z$, divide by $w = -z > 0$ to get the normalised
coordinates $x' = x/(-z)$, $y' = y/(-z)$ and their squared radius
$r^2 = x'^2 + y'^2$. The radial factor and pixel are

$$
D = 1 + k_1 r^2 + k_2 r^4, \qquad
u = f \, D \, x', \qquad v = f \, D \, y' .
$$

There is no $c_x, c_y$ offset — the principal point sits at the image origin.

**Validity.** The point must be in front of the camera under the BAL convention:
$z < -\epsilon_g$ (`MIN_DEPTH`); a depth failing this raises
`ProjectionOutOfBounds`.

<a id="unprojection"></a>
## Inverse Projection

**Iterative** (fixed point). From $x_d = u/f$, $y_d = v/f$, initialise
$(x', y') = (x_d, y_d)$ and iterate $x' = x_d / D$, $y' = y_d / D$ five times,
re-evaluating $D = 1 + k_1 r^2 + k_2 r^4$ each pass. With $s = \sqrt{1 + x'^2 + y'^2}$
the unit ray points down $-Z$:

$$
\mathbf{r} = \frac{1}{s}\begin{bmatrix} x' \\ y' \\ -1 \end{bmatrix} .
$$

**Validity.** None beyond the fixed number of iterations; the radial inverse is
well conditioned for the small BAL distortion coefficients.

<a id="jacobians"></a>
## Point Jacobian

Let $s = 1/(-z) = 1/w$, $D' = k_1 + 2 k_2 r^2$, and the $2 \times 2$ retinal block

$$
J_2 = \frac{\partial (x_d, y_d)}{\partial (x', y')}
=
\begin{bmatrix}
D + 2 D' x'^2 & 2 D' x' y' \\
2 D' x' y' & D + 2 D' y'^2
\end{bmatrix},
$$

where $(x_d, y_d) = (D x',\, D y')$ is the distorted retinal point. Since
$\partial x'/\partial x = s$ and $\partial x'/\partial z = x' s$ (and likewise for
$y'$), the $2 \times 3$ point Jacobian is

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
= f \, s \left[\; J_2 e_x \;\;\middle|\;\; J_2 e_y \;\;\middle|\;\; J_2 \begin{bmatrix} x' \\ y' \end{bmatrix} \right],
$$

i.e. the first two columns are $f s$ times the columns of $J_2$, and the $z$
column is $f s \, J_2 (x', y')^\top$ (positive sign, because the chain rule runs
through $w = -z$).

## Intrinsic Jacobian

With parameter order $[f, k_1, k_2]$,

$$
\frac{\partial (u, v)}{\partial (f, k_1, k_2)}
=
\begin{bmatrix}
D x' & f \, x' \, r^2 & f \, x' \, r^4 \\
D y' & f \, y' \, r^2 & f \, y' \, r^4
\end{bmatrix} .
$$

## Linear Estimation

Not provided. BAL-format datasets ship with pre-calibrated $f, k_1, k_2$; the
strict form is a drop-in replacement, not a calibrator.

## Example

```rust
use apex_camera_models::{BALPinholeCameraStrict, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(500.0, 500.0, 0.0, 0.0)?;
let distortion = DistortionModel::Radial { k1: -0.1, k2: 0.01 };
let camera = BALPinholeCameraStrict::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## References

- Snavely, N., Seitz, S. M. & Szeliski, R. (2006). *Photo Tourism: Exploring Photo Collections in 3D*. ACM SIGGRAPH 2006.
- Agarwal, S., Snavely, N., Simon, I., Seitz, S. M. & Szeliski, R. (2009). *Building Rome in a Day*. ICCV 2009.
