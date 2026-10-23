# Fisheye624 (Project Aria)

The fisheye model of every Project Aria SLAM and RGB camera, and so of the
LaMAria dataset, where it is called `RAD_TAN_THIN_PRISM_FISHEYE`. A
[Kannala-Brandt](./kannala-brandt.md)-style polynomial in the **incidence angle**
$\theta$, with six radial terms instead of four, followed by **tangential** and
**thin-prism** corrections. The name reads off the parameters: 6 radial, 2
tangential, 4 thin-prism.

## Parameters

| Symbol | Name | Units | Range |
|---|---|---|---|
| $f_x, f_y$ | Focal lengths | pixels | $> 0$, finite |
| $c_x, c_y$ | Principal point | pixels | finite |
| $k_0 \dots k_5$ | Radial coefficients | — | finite |
| $p_0, p_1$ | Tangential coefficients | — | finite |
| $s_0 \dots s_3$ | Thin-prism coefficients | — | finite |

Total: **16 parameters**, in the order
$[f_x, f_y, c_x, c_y, k_0 \dots k_5, p_0, p_1, s_0 \dots s_3]$. This is also the
layout of LaMAria's calibration files.

**Aria's own vector** has one focal length and 15 entries,
$[f, c_x, c_y, k_0 \dots k_5, p_0, p_1, s_0 \dots s_3]$. It is the special case
$f_x = f_y$:
`Fisheye624Camera::from_aria_params` reads it and `to_aria_params` writes it
(an error if $f_x \neq f_y$).

## Projection

With $\rho = \sqrt{x^2 + y^2}$ and $\theta = \operatorname{atan2}(\rho, z)$:

$$
R(\theta) = \theta + k_0 \theta^3 + k_1 \theta^5 + k_2 \theta^7 + k_3 \theta^9 + k_4 \theta^{11} + k_5 \theta^{13},
\qquad
\begin{bmatrix} x_r \\ y_r \end{bmatrix} = \frac{R(\theta)}{\rho} \begin{bmatrix} x \\ y \end{bmatrix} .
$$

With $r_d^2 = x_r^2 + y_r^2$, the two corrections are

$$
\mathbf{t} = \begin{bmatrix} p_0 (2 x_r^2 + r_d^2) + 2 p_1 x_r y_r \\ p_1 (2 y_r^2 + r_d^2) + 2 p_0 x_r y_r \end{bmatrix},
\qquad
\mathbf{s} = \begin{bmatrix} s_0 r_d^2 + s_1 r_d^4 \\ s_2 r_d^2 + s_3 r_d^4 \end{bmatrix},
$$

and the pixel is

$$
u = f_x (x_r + t_x + s_x) + c_x, \qquad v = f_y (y_r + t_y + s_y) + c_y .
$$

**Validity.** $z \ge \epsilon_g$, else `PointBehindCamera`. On the axis
($\rho < \epsilon_g$) $R(\theta)/\rho \to 1/z$, so the point lands on the
principal point.

## Inverse Projection

Two Newton iterations, as in the reference implementation.

1. **Undo the corrections.** With $\mathbf{m} = ((u - c_x)/f_x,\ (v - c_y)/f_y)$,
   solve $\boldsymbol{\xi} + \mathbf{t}(\boldsymbol{\xi}) + \mathbf{s}(\boldsymbol{\xi}) = \mathbf{m}$ for
   $\boldsymbol{\xi} = (x_r, y_r)$, started at $\mathbf{m}$, with the $2 \times 2$
   Jacobian of the corrections.
2. **Undo the radial polynomial.** Solve $R(\theta) = \lVert\boldsymbol{\xi}\rVert$ by
   Newton on $R'(\theta) = 1 + \sum_i (2i + 3) k_i \theta^{2i + 2}$, started at
   $\lVert\boldsymbol{\xi}\rVert$. If Newton does not converge, bisection on
   $[0, \pi/2)$ is tried before giving up.

The unit ray is $(\sin\theta\, \boldsymbol{\xi}/\lVert\boldsymbol{\xi}\rVert,\ \cos\theta)$.

A pixel whose angle would reach $\pi/2$ is **rejected** with
`NumericalError`, never clamped: the model ends there, and `project` needs
$z \ge \epsilon_g$.

## Point Jacobian

With $g = R(\theta)/\rho$ and $\boldsymbol{\xi} = g\,(x, y)$,

$$
\frac{\partial \boldsymbol{\xi}}{\partial \mathbf{p}}
= g \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
+ \begin{bmatrix} x \\ y \end{bmatrix} \nabla g^{\top},
\qquad
\nabla g = \frac{R'(\theta)}{\rho} \nabla\theta - \frac{R(\theta)}{\rho^2} \nabla\rho ,
$$

with $\nabla\theta = \big(\tfrac{zx}{\rho(\rho^2+z^2)},\ \tfrac{zy}{\rho(\rho^2+z^2)},\ -\tfrac{\rho}{\rho^2+z^2}\big)$
and $\nabla\rho = (x/\rho,\ y/\rho,\ 0)$. The pixel Jacobian is
$\operatorname{diag}(f_x, f_y)\,(I + \partial(\mathbf{t}+\mathbf{s})/\partial\boldsymbol{\xi})\,\partial\boldsymbol{\xi}/\partial\mathbf{p}$,
where

$$
\frac{\partial(\mathbf{t}+\mathbf{s})}{\partial\boldsymbol{\xi}} =
\begin{bmatrix}
6 p_0 x_r + 2 p_1 y_r + 2 x_r (s_0 + 2 s_1 r_d^2) & 2 p_0 y_r + 2 p_1 x_r + 2 y_r (s_0 + 2 s_1 r_d^2) \\
2 p_1 x_r + 2 p_0 y_r + 2 x_r (s_2 + 2 s_3 r_d^2) & 6 p_1 y_r + 2 p_0 x_r + 2 y_r (s_2 + 2 s_3 r_d^2)
\end{bmatrix}.
$$

On the axis the limit $\partial\boldsymbol{\xi}/\partial\mathbf{p} = [\,1/z,\ 0,\ -x/z^2;\ 0,\ 1/z,\ -y/z^2\,]$ is used.

## Intrinsic Jacobian

Columns follow the parameter order. With $\mathbf{d} = \boldsymbol{\xi} + \mathbf{t} + \mathbf{s}$:

| Column | $\partial(u, v)$ |
|---|---|
| $f_x,\ f_y$ | $(d_x, 0),\ (0, d_y)$ |
| $c_x,\ c_y$ | $(1, 0),\ (0, 1)$ |
| $k_i$ | $\operatorname{diag}(f_x, f_y)\,(I + \partial(\mathbf{t}+\mathbf{s})/\partial\boldsymbol{\xi})\ \theta^{2i+3}\,(x, y)/\rho$ |
| $p_0$ | $\big(f_x (2 x_r^2 + r_d^2),\ f_y\, 2 x_r y_r\big)$ |
| $p_1$ | $\big(f_x\, 2 x_r y_r,\ f_y (2 y_r^2 + r_d^2)\big)$ |
| $s_0, s_1$ | $(f_x r_d^2, 0),\ (f_x r_d^4, 0)$ |
| $s_2, s_3$ | $(0, f_y r_d^2),\ (0, f_y r_d^4)$ |

## Example

```rust
use apex_camera_models::{CameraModel, Fisheye624Camera};
use nalgebra::Vector3;

// LaMAria's 16-entry layout: fx fy cx cy k0..k5 p0 p1 s0..s3.
let camera = Fisheye624Camera::try_from(&params[..])?;
let pixel = camera.project(&Vector3::new(0.3, -0.2, 1.0))?;
let ray = camera.unproject(&pixel)?;            // back to the unit ray
let j = camera.jacobian_intrinsics(&Vector3::new(0.3, -0.2, 1.0));  // 2 x 16
```

## Relation to other models

| This model with … | is |
|---|---|
| $k_4 = k_5 = 0$, $p = s = 0$ | [Kannala-Brandt](./kannala-brandt.md) with $k_{1..4} = k_{0..3}$, exactly |
| $s = 0$ | [Fisheye62](./fisheye62.md), exactly |

Both are `From` conversions that lose nothing; see [Conversion](./conversion.md)
for fitting to models that cannot hold it exactly.

## Source

The model, its parameter order and the two Newton inversions are Meta's
Project Aria models:

- [Camera intrinsic models](https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models), Project Aria documentation.
- `core/calibration/camera_projections/FisheyeRadTanThinPrism.h`
  (`Fisheye624 = FisheyeRadTanThinPrism<6, true, true, true>`), in
  [facebookresearch/projectaria_tools](https://github.com/facebookresearch/projectaria_tools).

This crate is an independent implementation with analytic Jacobians. Its tests
compare projection against values computed from the reference's formulas, the
Jacobians against central differences, and the whole model against
Kannala-Brandt where the two coincide.
