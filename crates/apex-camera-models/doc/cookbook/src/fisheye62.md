# Fisheye62 (Project Aria)

[Fisheye624](./fisheye624.md) without the thin-prism terms: a fisheye
polynomial in the incidence angle $\theta$ with **six radial** terms and
**two tangential** terms. Aria's name for it reads off the parameters.

## Parameters

| Symbol | Name |
|---|---|
| $f_x, f_y, c_x, c_y$ | Linear intrinsics |
| $k_0 \dots k_5$ | Radial coefficients |
| $p_0, p_1$ | Tangential coefficients |

Total: **12 parameters**, in the order
$[f_x, f_y, c_x, c_y, k_0 \dots k_5, p_0, p_1]$. Project Aria's own vector has
one focal length and 11 entries, $[f, c_x, c_y, k_0 \dots k_5, p_0, p_1]$:
`Fisheye62Camera::from_aria_params` and `to_aria_params`.

## Projection and inverse

Those of [Fisheye624](./fisheye624.md) with $s_0 \dots s_3 = 0$; the two types
share one implementation, so they cannot drift apart. With
$R(\theta) = \theta + k_0 \theta^3 + \dots + k_5 \theta^{13}$ and
$(x_r, y_r) = R(\theta)\,(x, y)/\rho$:

$$
u = f_x \big(x_r + p_0 (2 x_r^2 + r_d^2) + 2 p_1 x_r y_r\big) + c_x, \qquad
v = f_y \big(y_r + p_1 (2 y_r^2 + r_d^2) + 2 p_0 x_r y_r\big) + c_y .
$$

## Jacobians

The point Jacobian is Fisheye624's. The intrinsic Jacobian is its first 12
columns, $2 \times 12$.

## Conversions

- [Kannala-Brandt](./kannala-brandt.md) → Fisheye62: exact (`From`).
- Fisheye62 → [Fisheye624](./fisheye624.md): exact (`From`).
- Fisheye624 → Fisheye62: `TryFrom`, an error unless $s_0 \dots s_3 = 0$, so
  that no distortion is dropped silently. To approximate a camera with
  thin-prism terms, fit it: [Conversion](./conversion.md).

## Source

Meta's Project Aria:
[Camera intrinsic models](https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models)
and `FisheyeRadTanThinPrism.h`
(`Fisheye62 = FisheyeRadTanThinPrism<6, true, false, true>`) in
[facebookresearch/projectaria_tools](https://github.com/facebookresearch/projectaria_tools).
