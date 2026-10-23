# Conversion

Two ways to move a camera from one model to another.

## Exact

Some models are special cases of another, and the conversion is a `From`:

| From | To | How |
|---|---|---|
| [Kannala-Brandt](./kannala-brandt.md) | [Fisheye62](./fisheye62.md), [Fisheye624](./fisheye624.md) | $k_{0..3} = k_{1..4}$, the rest zero |
| [Fisheye62](./fisheye62.md) | [Fisheye624](./fisheye624.md) | $s = 0$ |
| [Fisheye624](./fisheye624.md) | [Fisheye62](./fisheye62.md) | `TryFrom`, only if $s = 0$ |

Project Aria's other models are covered by existing types: `Linear` is the
[Pinhole](./pinhole.md), `KannalaBrandtK3` is the Kannala-Brandt model with its
four coefficients, and `Spherical` ($r = f\theta$) is Kannala-Brandt with every
$k_i = 0$.

## Fitted

```rust
use apex_camera_models::{convert_camera, ConversionOptions};

let (target, report) = convert_camera(&source, &initial, 640, 480, &ConversionOptions::default())?;
println!("rms {:.3} px, max {:.3} px over {} rays", report.rms_px, report.max_px, report.samples);
```

`convert_camera` fits a camera of the family of `initial` to `source`, for any
pair of models:

1. Sample the source's image on a $41 \times 41$ grid and unproject every pixel
   to a ray; a ray that does not project back to its pixel is outside the
   source's domain and is dropped.
2. Minimise $\sum \lVert \pi_{\text{target}}(\text{ray}) - \text{pixel} \rVert^2$ over
   the target's intrinsics with Levenberg-Marquardt, using the target's analytic
   `jacobian_intrinsics`. A step that leaves the model's valid parameter range
   is rejected.
3. Report the residual in pixels.

The result is the closest camera of the target family. It is exact only when
the target can hold the source; **read `rms_px`**, because a model cannot always
represent another.

`initial` fixes the family and the starting point. Use the source's
`get_pinhole_params()` and neutral distortion, with values inside the domain
for models that have one (UCM, EUCM and double sphere need $\alpha > 0$).

`ConversionOptions::max_theta_deg` limits the fitted rays to the centre of the
lens, which is what a pinhole or radial-tangential target needs: it has no field
of view to match a fisheye's edge.

### What it costs

Fitting LaMAria's Fisheye624 `cam0` (`R_06_medium`, about 150° across), on the
default grid:

| Target | rms [px] | max [px] |
|---|---|---|
| Fisheye624 | $3 \times 10^{-14}$ | $2 \times 10^{-13}$ |
| Fisheye62 | 0.14 | 0.51 |
| Kannala-Brandt | 0.26 | 0.80 |
| EUCM | 0.45 | 0.94 |
| Double sphere, UCM | 1.87 | 10.2 |
| Radial-tangential, $\theta \le 25°$ / $35°$ / $50°$ | 0.014 / 0.030 / 0.094 | 0.033 / 0.072 / 0.25 |

Over all 172 cameras of the dataset the worst rms is 0.24 px to Fisheye62, 0.55
px to Kannala-Brandt and 0.66 px to EUCM. The double sphere's optimum on this
lens is $\xi \approx 0$, which is UCM; the two errors are the same because
the fit lands on the same camera from every starting point tried.
