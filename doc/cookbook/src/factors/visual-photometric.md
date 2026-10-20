# Visual Factors: Photometric

The photometric family measures *intensities* instead of geometry: a patch of
pixels must look the same in two frames, and the difference between how it
looks is the error. The family's five factor types share three pieces — an
image container, a point description, and a weighting — and this chapter is
those pieces and the four factors built on them.

## 1. The image container

`IntensityImage` is a single-channel image that returns the intensity and its
gradient in one bilinear lookup:

```rust
pub fn sample(&self, u: f64, v: f64) -> Option<(f64, Vector2<f64>)>;
//                                                       the value and [∂I/∂u, ∂I/∂v]
```

The gradient is the **exact derivative of the same bilinear interpolant** the
intensity comes from. That matters: a residual evaluated with one
interpolation and differentiated with another is inconsistent at the sub-pixel
level, and the inconsistency shows up as a step that drifts along the texture.

`GradientImage` is the solver-side counterpart — a shared buffer of
`[f32; 3]` triples (intensity and two gradient components) with its own
`sample(x, y) -> Option<[f64; 3]>`. It is what the *factor* consumes; the
`IntensityImage` is what the *consumer* builds it from.

## 2. Points and weights

<div class="apx-table">
<span class="apx-title">The shared types</span>

| Type | Contents |
|---|---|
| `PhotometricPoint` | a point's position, its inverse range, and the pattern offsets used to sample it |
| `PhotometricWeights` | the per-sample weights: Huber threshold, sigma, the dynamic weight, the exposure ratio |
| `ReferenceView` | the reference image and the **body pose** it was taken at |
| `PlanePoint` | a point on a plane: its world position and the plane's normal |

</div>

`ReferenceView` holding a *body* pose is the same design as the visual map's
reference patch: the camera pose is recomposed from the body pose and the
current extrinsic, so an online extrinsic update re-renders the reference
correctly instead of invalidating it.

## 3. The four factors

<div class="apx-table">
<span class="apx-title">`factors::visual`, photometric part</span>

| Factor | What it measures |
|---|---|
| `PhotometricPatchFactor` | a reference patch against the current image, with the pose as the unknown |
| `PhotometricTwoViewFactor` | the same, with **both** views rendered through the extrinsic — what makes online extrinsic calibration honest |
| `PhotometricInverseRangeFactor` | a map point's photometric appearance, with the point's inverse range in the state |
| `AffinePriorFactor` | a prior on the per-frame affine brightness parameters $(a, b)$ |

</div>

**`PhotometricPatchFactor`** is the workhorse. Its blocks are
`[X ∈ G, T_SC ∈ SE3]` — the image's state and the camera extrinsic (camera →
body) — and its residual dimension is the number of patch pixels. A reference
patch of intensities is compared with the current image's patch at the
predicted projection, and the residual is the difference, made zero-mean on
both sides so a global brightness offset cancels.

**`PhotometricTwoViewFactor`** exists because the single-view form has a blind
spot. If the reference patch was rendered with the *current* extrinsic, then a
change in the extrinsic changes the reference and the residual with it — the
measurement partly defines its own truth, and an estimate chases its own
references. The two-view form renders **both** views through the same extrinsic
at evaluation time, so the extrinsic's effect is visible in the residual
instead of absorbed by the reference.

**`PhotometricInverseRangeFactor`** is the LIVO form: the visual
measurement is about a *map* point whose depth is a state variable, and the
factor's residual is the photometric difference at the point's predicted
projection. It is the factor the LIVO stack uses, with the LiDAR supplying
the geometry and the camera the appearance.

**`AffinePriorFactor`** is the brightness model made explicit: a prior on each
frame's affine transfer $(a, b)$, keeping the brightness parameters from
drifting into the geometry.

## 4. Weights

`PhotometricWeights` carries the pieces that decide how much a sample counts:

- the Huber threshold, in grey levels;
- the residual scale $\sigma$;
- the **dynamic weight**, the mechanism that tells the solve "this frame's
  residuals are worse than I thought — trust the other measurements more";
- the **exposure ratio** between the two views, which absorbs a brightness
  change that the reference's own normalization cannot.

These are the photometric analogue of a noise model, and they are applied
*inside* the factor: the photometric family whitens internally and must be
registered with `NoiseModel::null()` (chapter [Noise
Models](../core/noise.md)).

## 5. Worked example

A patch factor on an 8-pixel pattern with a 3-level pyramid. The factor
evaluates the pattern at the predicted projection in the current image, samples
the reference pattern (already warped and gain-matched by the consumer), and
writes eight rows. The Jacobian is 8 × 6 for an `SE3` block: the derivative
of each sample's intensity with respect to the pose, through the projection's
Jacobian. Eight rows and a 7-column block is a good ratio — the factor is
strongly determined in the directions the texture constrains and weak in the
others, which the robust weighting cannot fix and the optimizer's damping has
to.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — brightness constancy is an assumption, not a measurement</span>

Every photometric factor assumes the same physical surface produces the same
intensity (up to the affine transfer) in both views. Specular highlights,
saturation, rolling shutter, and auto-exposure break it, and the residual then
measures the violation rather than the motion. The robust weighting bounds the
damage; the *correct* handling of a violated assumption is to detect it
(an energy gate) and drop the measurement, which the consumer crates do.

</div>

<div class="apx-note">
<span class="apx-title">Note — the two-view form costs a second rendering</span>

`PhotometricTwoViewFactor` renders both views through the extrinsic at every
evaluation, so its cost is roughly double the single-view form's for the same
number of samples. That is the price of an honest extrinsic estimate, and it is
why the LIVO stack uses the single-view form when the extrinsic is held and
the two-view form only when it is estimated.

</div>
