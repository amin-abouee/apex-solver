# Visual Factors: Geometry

The visual family is the largest in the crate: seventeen factor types covering
the ways a camera can constrain an unknown. They differ in three choices — the
unknown's parameterization, the measurement's representation, and which
variables are involved — and this chapter organizes them by those choices
rather than listing them.

## 1. The parameterization of the unknown

A point in the world can be parameterized in three ways, and the choice
changes the geometry of the residual:

<div class="apx-table">
<span class="apx-title">Point parameterizations</span>

| Type | Unknown | Good for |
|---|---|---|
| `HomogeneousPointFactor` | a 3-D point | unconstrained scenes; the classic formulation |
| `InverseDepthFactor` | $\rho = 1/z$ along the optical axis | points in front of the camera, where $z$ is positive and $\rho$ varies smoothly |
| `InverseDepthSe23Factor`, `InverseDepthBearingSe23Factor` | $\rho$ in an `SE23` frame | visual-inertial graphs, where the point's frame follows the body |

</div>

Inverse depth is the workhorse: it keeps the unknown positive in front of the
camera and makes the *Jacobian* well behaved when the point is far away. Its
weakness is the ray: a point off the optical axis has an inverse depth that
is not the inverse of its range, and a residual built on the wrong one
mis-weights the observation.

## 2. The measurement

Three representations coexist, and the crate is explicit about which is which:

- **pixels** — `ProjectionFactor`, `SmartProjectionFactor`,
  `DepthFactor`, `HomogeneousPointFactor`, the stereo pair; they take a
  calibration matrix $K$ and work in image space;
- **unit bearings** — `InverseDepthBearingSe23Factor`,
  `InverseRangeProjectionSe23Factor`, `StereoInverseRangeFactor`; the
  measurement is a direction, and the residual lives on the sphere;
- **relative pose** — `EssentialMatrixFactor` and
  `EssentialMatrixConstraint`, which take the epipolar constraint itself as
  the measurement.

<div class="apx-widget" data-widget="epipolar" data-title="The epipolar constraint, and its residual" data-foot="Move a correspondence off the epipolar line and watch the Sampson distance the estimators threshold."></div>

## 3. The factors, by job

<div class="apx-table">
<span class="apx-title">The visual family</span>

| Factor | What it constrains |
|---|---|
| `ProjectionFactor` | a 3-D point's projection into an image, against a pixel measurement |
| `SmartProjectionFactor` | the same, with the point's depth eliminated *inside* the factor (structure-less) |
| `HomogeneousPointFactor` | a homogeneous 3-D point, with the projection and the depth folded together |
| `DepthFactor` | a point's depth along the ray, against a measured depth |
| `InverseDepthFactor` | a point's inverse depth, against an inverse-depth measurement |
| `InverseDepthSe23Factor` | a point in an `SE23` (body-anchored) frame, with a bearing measurement |
| `InverseDepthBearingSe23Factor` | the same, with the residual on the bearing's tangent plane |
| `EssentialMatrixFactor` | the epipolar constraint between two views, as a factor over the two poses |
| `EssentialMatrixConstraint` | the same constraint, packaged for a pose-graph edge |
| `StereoFactor` (+ `StereoCalibration`) | a stereo pair's relative pose, with the calibration as part of the state |
| `InverseRangeProjectionSe23Factor` | a point seen by camera 1, with the rig's baseline in the chain |
| `StereoInverseRangeFactor` | a point seen by both cameras of a rig, with the body state cancelling |
| `ExtrinsicProjectionFactor` | the camera-to-body extrinsic, calibrated by the projections it appears in |
| `TimeOffsetProjectionFactor` | a temporal offset between camera and IMU, calibrated by the same projections |

</div>

## 4. What the ecosystem uses, and why

The VIO backend builds its graph from the bearing-based family
(`InverseDepthBearingSe23Factor`, `InverseRangeProjectionSe23Factor`,
`StereoInverseRangeFactor`) and the stereo calibration factors; the LIVO
stack adds `ExtrinsicProjectionFactor` when the extrinsic is estimated online.
The camera-agnostic choice — bearings instead of pixels — is what lets the same
factor serve a pinhole and a fisheye.

`ExtrinsicProjectionFactor` and `TimeOffsetProjectionFactor` deserve a note:
they are *calibration* factors. Their unknown is not a scene point but a
parameter of the rig (the extrinsic, the time offset), and the measurement is
the projection of the scene points that depend on it. They are the factors that
make online calibration a *measurement* rather than a guess.

## 5. Worked example

A stereo rig observing a landmark: the point is anchored in the host camera's
`SE23` frame with an inverse range along its bearing; the observation in the
other camera is a bearing; the residual is the angle between the predicted and
measured bearings, projected on the measured bearing's tangent plane. The
rig's baseline enters through the extrinsics, and the body state cancels when
both observations come from the same keyframe — which is why the same-keyframe
stereo factor has fewer blocks than the cross-keyframe one.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — pixels and bearings are not interchangeable</span>

A pixel measurement belongs to a calibrated camera; a bearing measurement
belongs to the camera's frame. Mixing them in one graph means the residuals
live in different units, the thresholds mean different things, and a single
Huber scale cannot serve both. The crate keeps the two families separate for
exactly this reason, and the ecosystem's books state which one each front end
uses.

</div>

<div class="apx-note">
<span class="apx-title">Note — the depth parameterization follows the camera's frame</span>

`InverseDepthFactor` measures depth along the **optical axis**; the bearing
family measures inverse *range* along the *ray*. On the axis they agree; at
the rim of a fisheye they differ by $1/\cos\theta$ and diverge. A factor
documented as "inverse depth" and used where a range is meant produces a
residual that is small in the middle of the image and wrong at the edges —
a bias that looks like a lens distortion.

</div>
