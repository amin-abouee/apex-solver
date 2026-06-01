# BAL Pinhole

The "Bundle Adjustment in the Large" pinhole variant used by Bundler, Ceres,
and GTSAM. Two important differences from the standard pinhole model:

1. The camera looks down the **-Z axis** (the world is in front of the
   camera when $z < 0$).
2. The intrinsic parameter vector is reduced to **3** scalars in the strict
   form: a single focal length $f$ shared by $f_x$ and $f_y$, a single
   radial pair $k_1, k_2$, and a fixed principal point at the image
   origin. This is the parameterisation used by the BAL dataset.

The crate enforces the strict constraints in `BALPinholeCameraStrict::new`.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f$ | Shared focal length | pixels | $f > 0$ |
| $k_1$ | Radial distortion (second order) | — | finite |
| $k_2$ | Radial distortion (fourth order) | — | finite |

Total: **3 parameters**. The parameter vector is $[f, k_1, k_2]$.
By convention $f_x = f_y = f$ and $c_x = c_y = 0$.

## Projection

For a 3D point $p_{\mathrm{cam}} = (x, y, z)$ with $z < 0$ (in front of the
camera):

$$
x' = \frac{x}{-z}, \qquad y' = \frac{y}{-z}, \qquad r^2 = x'^2 + y'^2
$$

Radial factor

$$
d = 1 + k_1 r^2 + k_2 r^4
$$

Distorted coordinates and pixel projection

$$
u = f \cdot d \cdot x', \qquad v = f \cdot d \cdot y'
$$

(No $c_x$, $c_y$ offset — the principal point is at the image origin.)

## Unprojection

**Iterative** for the radial part, with the same Newton-Raphson scheme as
the other radial models. The azimuth is recovered directly from
$(m_x, m_y) = (u / f, v / f)$.

## Validity

- $z < 0$ (point in front of the camera under the BAL convention).
- $|f_x - f_y| < 10^{-10}$ and $|c_x|, |c_y| < 10^{-10}$ for the strict form.

## Point Jacobian

With $w = -z > 0$ and the standard pinhole quotient rule, the model differs
from the regular pinhole only by a global sign in the $z$ partial:

$$
\frac{\partial (u, v)}{\partial (x, y, z)}
=
\begin{bmatrix}
f \, d \, / w + f \, x' \cdot \partial d / \partial x' & \ldots & f \, x \, d / w^2 \\
\ldots & f \, d \, / w + f \, y' \cdot \partial d / \partial y' & f \, y \, d / w^2
\end{bmatrix}
$$

(where the third column picks up a positive sign because the chain rule runs
through $w = -z$). The full expression is implemented in the crate.

## Intrinsic Jacobian

Parameter order: $[f, k_1, k_2]$.

$$
\frac{\partial (u, v)}{\partial (f, k_1, k_2)}
=
\begin{bmatrix}
d \, x' & f \, x' \, r^2 & f \, x' \, r^4 \\
d \, y' & f \, y' \, r^2 & f \, y' \, r^4
\end{bmatrix}
$$

## Linear Estimation

Not provided. BAL-format datasets ship with pre-calibrated $f, k_1, k_2$
and the strict form is meant to be a drop-in replacement, not a calibrator.

## Example

```rust
use apex_camera_models::{BALPinholeCameraStrict, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(500.0, 500.0, 0.0, 0.0)?;
let distortion = DistortionModel::Radial { k1: -0.1, k2: 0.01 };
let camera = BALPinholeCameraStrict::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f > 0$ and finite (enforced via $f_x = f_y$).
- $k_1, k_2$ finite.
- $|f_x - f_y| \le 10^{-10}$ (single focal length).
- $|c_x|, |c_y| \le 10^{-10}$ (no principal point offset).
- `DistortionModel::Radial { k1, k2 }` only.

## References

- Snavely, N., Seitz, S. M. & Szeliski, R. (2006). *Photo Tourism: Exploring Photo Collections in 3D*. ACM SIGGRAPH 2006.
- Agarwal, S., Snavely, N., Simon, I., Seitz, S. M. & Szeliski, R. (2009). *Building Rome in a Day*. ICCV 2009.
