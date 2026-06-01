# Field-of-View (FOV)

Devernay & Faugeras (2001). A single scalar $\omega$ controls the amount of
distortion. The projection is an arctangent of the lateral distance scaled
by $\tan(\omega / 2)$, which gives a uniform angular resolution near the
optical axis and a graceful saturation near the edges of a wide lens.

The model is popular in visual-SLAM stacks that need to handle a wide lens
without paying the cost of a higher-order polynomial.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x$ | Focal length, $x$ | pixels | $f_x > 0$ |
| $f_y$ | Focal length, $y$ | pixels | $f_y > 0$ |
| $c_x$ | Principal point, $x$ | pixels | finite |
| $c_y$ | Principal point, $y$ | pixels | finite |
| $\omega$ | Field-of-view parameter | rad | $0 < \omega \le \pi$ |

Total: **5 parameters**.

## Projection

Let $r = \sqrt{x^2 + y^2}$ and $r_p = \sqrt{x^2 + y^2 + z^2}$. Define the
auxiliary quantity

$$
\psi = \frac{2 \tan(\omega / 2) \cdot r}{z}
$$

The distorted radius is

$$
r_d = \frac{\arctan \psi}{\omega \cdot r} \quad \text{if } r > 0, \qquad
r_d = \frac{2 \tan(\omega / 2)}{\omega} \quad \text{if } r \approx 0
$$

The pixel coordinates are

$$
u = f_x \cdot r_d \cdot x + c_x, \qquad
v = f_y \cdot r_d \cdot y + c_y
$$

## Unprojection

Given $(u, v)$, compute $(m_x, m_y) = ((u - c_x) / f_x, (v - c_y) / f_y)$
and $r_d = \sqrt{m_x^2 + m_y^2}$. The point lies on a ray of half-angle
$\alpha = \arctan(\psi) / 2 = r_d \omega / 2$ from the optical axis. The unit
ray is reconstructed from $\alpha$ and the azimuth $\phi = \arctan(m_y / m_x)$.

## Validity

- $z > 0$.
- $\omega \in (0, \pi]$ (the model degenerates at $\omega = 0$ and saturates
  for $\omega > \pi$).

## Point Jacobian

The point Jacobian is obtained by the chain rule on the forward polynomial.
The exact expression is implemented in the crate; the main dependencies are
$\partial r_d / \partial r$ and $\partial r_d / \partial z$ through $\psi$.

## Intrinsic Jacobian

Parameter order: $[f_x, f_y, c_x, c_y, \omega]$.

The $f_x, f_y, c_x, c_y$ columns match the pinhole model. The $\omega$
column involves $\partial r_d / \partial \omega$:

$$
\frac{\partial r_d}{\partial \omega} = \frac{1}{\omega^2}
\left( \frac{\psi}{1 + \psi^2} - \arctan \psi \right) \cdot \frac{1}{r} + \ldots
$$

(complete expression in the implementation).

## Linear Estimation

**Not** a closed-form linear system — the dependence on $\omega$ is
non-linear. The implementation performs a 1-D grid search over
$\omega \in \{0.10, 0.11, \ldots, 2.99\}$ rad, picks the $\omega$ that
minimises the average reprojection error, and assigns the result to the
camera. **At least 2 correspondences** are required.

## Example

```rust
use apex_camera_models::{FovCamera, PinholeParams, DistortionModel};

let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
let distortion = DistortionModel::FOV { w: 1.0 };
let camera = FovCamera::new(pinhole, distortion)?;
# Ok::<(), apex_camera_models::CameraModelError>(())
```

## Validation Rules

- $f_x, f_y > 0$ and finite.
- $c_x, c_y$ finite.
- $\omega$ finite, $0 < \omega \le \pi$.

## References

- Devernay, F. & Faugeras, O. (2001). *Straight Lines Have to Be Straight: Automatic Calibration and Removal of Distortion from Scenes of Structured Environments*. Machine Vision and Applications 13(1), 14–24.
