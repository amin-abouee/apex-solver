# The Factor Catalogue

Every public factor type in the crate, by family, with the chapter that
explains it. The crate ships **54** `*Factor` types across nine families,
plus the companion types they need and the four decorators.

## 1. Pose and priors — [chapter](./pose.md)

| Type | Blocks | Residual |
|---|---|---|
| `PriorFactor<T>` | one | a tangent-space anchor at a reference value |
| `EuclideanPriorFactor` | one `RN` | an ambient parameter-space prior |
| `BetweenFactor<T>` | two | a relative pose against a measurement |
| `Relative4DofFactor` | two | position and heading, no roll or pitch |
| `PoseRotationPrior` | one | the rotation part only |
| `PoseTranslationPrior` | one | the translation part only |

## 2. Visual geometry — [chapter](./visual-geometry.md)

| Type | Blocks | Residual |
|---|---|---|
| `ProjectionFactor<CAM, OP>` | pose, point | the point's projection against a pixel |
| `SmartProjectionFactor<CAM>` | pose | the same, with the depth eliminated inside the factor |
| `HomogeneousPointFactor` | pose, point | a homogeneous 3-D point, projection and depth folded |
| `DepthFactor<ONESIDED>` | pose, point | the point's depth along the ray |
| `InverseDepthFactor<CAM>` | pose, point | the point's inverse depth |
| `InverseDepthPlaneFactor` | state, point | a point in an `SE23` frame, with a bearing |
| `InverseDepthBearingFactor` | state, point | the same, on the bearing's tangent plane |
| `InverseRangeProjectionFactor<G, CAM>` | state | a point seen by one camera of a rig |
| `StereoInverseRangeFactor<CAM>` | state, point | a point seen by both cameras of a rig |
| `EssentialMatrixFactor` | two poses | the epipolar constraint |
| `ExtrinsicProjectionFactor<CAM>` | extrinsic, point | the extrinsic, calibrated by the projections it appears in |
| `TimeOffsetProjectionFactor` | offset, point | a camera–IMU time offset, calibrated the same way |
| `StereoFactor` | two poses, calibration | a stereo pair's relative pose, calibration in the state |

Companions: `EssentialMatrixConstraint`, `StereoCalibration`, `RegularDepthFactor`
/ `OneSidedDepthFactor` (the `DepthFactor` aliases), `StereoPoint2`.

## 3. Visual photometric — [chapter](./visual-photometric.md)

| Type | Blocks | Residual |
|---|---|---|
| `PhotometricPatchFactor<G, CAM>` | pose | a reference patch against the current image |
| `PhotometricTwoViewFactor<G, CAM>` | pose, extrinsic | both views rendered through the extrinsic |
| `PhotometricInverseRangeFactor` | state, point | a map point's photometric appearance |
| `AffinePriorFactor` | brightness params | a prior on the affine brightness model |

Companions: `IntensityImage`, `GradientImage`, `PhotometricPoint`,
`PhotometricWeights`, `ReferenceView`, `PlanePoint`.

## 4. IMU — [chapter](./imu.md)

| Type | Blocks | Residual |
|---|---|---|
| `ImuFactor` (in `se23` and `sgal3`) | two states | the preintegrated interval, 9 rows |
| `CombinedImuFactor` (in `se23` and `sgal3`) | two states, two biases | the interval plus the bias random walk, 15 rows |
| `GravityFrameImuFactor` | two states, gravity | the interval, with gravity as a variable |
| `ScaleGravityImuFactor` | two states, scale, gravity | the interval, with scale and gravity as variables |

Companions: `ImuPreintegration`, `ImuParameters`, `ImuSensorReadings`,
`ImuMeasurement`.

## 5. LiDAR — [chapter](./lidar.md)

| Type | Blocks | Residual |
|---|---|---|
| `LidarMapPlaneFactor` | pose, point | one point against one map plane |
| `LidarMapPlanesFactor` | pose | a whole sweep's matches, one block |
| `LidarPlaneBundleFactor` | poses, plane | one plane over several frames, plane eliminated |
| `LidarPlaneMomentFactor` | poses, plane | the same, summarized by each frame's moments |
| `LidarSubmapFieldFactor` | pose | a scan against a submap's field |
| `SubmapAlignmentFactor` | poses | a submap's alignment as a variable |
| `IcpFactor` | two poses | a scan registered against a reference frame's field |
| `LidarEdgeFactor` | two poses | a scan against a reference frame's edges |
| `GicpFactor` | pose, point | the GICP point-to-plane residual |
| `PointToPlaneFactor` | pose, point | a point against a plane |
| `PoseToPointFactor` | pose, point | a point against a point |

Companions: `Plane`, `PrecomputedPlane`, `LidarPointChain`, `PlaneMatch`,
`DistanceField`, `PointMoments`.

## 6. Range and bearing — [chapter](./ranging.md)

| Type | Blocks | Residual |
|---|---|---|
| `PosePoseRangeFactor` | two poses | the distance between two origins |
| `PosePointRangeFactor` | pose, point | the distance from an origin to a point |
| `BearingRangeFactor` | pose, point | a distance and a direction together |
| `BearingFactor<G>` | any state that moves points | the direction from an origin to a point |

## 7. Motion models — [chapter](./motion.md)

| Type | Blocks | Residual |
|---|---|---|
| `NonholonomicFactor` | state | the lateral velocity is zero |
| `PlanarMotionFactor` | state | the motion stays in a plane |
| `ZeroVelocityFactor` | state | the velocity is zero |
| `ZeroAngularRateFactor` | state | the angular rate is zero |

## 8. Navigation — [chapter](./navigation.md)

| Type | Blocks | Residual |
|---|---|---|
| `GpsFactor` | pose | a GNSS position with its covariance |
| `GpsAsyncFactor` | two poses | a GNSS position between two frames |
| `GpsVelocityFactor` | pose | a GNSS velocity |
| `PseudorangeFactor` | pose, satellite | a raw pseudorange |
| `DopplerFactor` | pose, satellite | a raw Doppler measurement |
| `BarometricFactor` | pose | a barometric altitude |
| `AttitudeFactor` | pose | an attitude observation |

## 9. Marginalization — [chapter](../marginal/prior-factor.md)

| Type | Blocks | Residual |
|---|---|---|
| `MarginalPriorFactor` | one or more | a linear container over a marginalized set |

## 10. Decorators — [chapter](./decorators.md)

| Type | What it does |
|---|---|
| `AggregateFactor` | compresses many factors into one exact block |
| `ProjectedFactor` | projects a Jacobian onto the well-constrained directions |
| `HeldBlocks<F>` | holds some blocks at fixed values |
| `RightOffset<F, G>` | evaluates one block at $X\cdot\Delta$ |

Companion: `AggregatePart`.

## 11. The count

<div class="apx-table">
<span class="apx-title">The families, by size</span>

| Family | `*Factor` types | Chapter |
|---|---:|---|
| visual (geometry + photometric) | 17 | [geometry](./visual-geometry.md), [photometric](./visual-photometric.md) |
| LiDAR | 11 | [LiDAR](./lidar.md) |
| navigation | 7 | [navigation](./navigation.md) |
| pose and priors | 4 (+2 partial priors) | [pose](./pose.md) |
| IMU | 4 | [IMU](./imu.md) |
| range and bearing | 4 (+1 `SE23` variant) | [ranging](./ranging.md) |
| motion models | 4 | [motion](./motion.md) |
| marginalization | 1 | [prior factor](../marginal/prior-factor.md) |
| decorators | 1 (as a factor) + 3 wrappers | [decorators](./decorators.md) |
| **total** | **54** | |

</div>

## 12. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a catalogue is an index, not a decision</span>

Every type here is a tool, and none of them is right by default. The choice
between `InverseDepthFactor` and `InverseDepthBearingFactor` is a choice
about the camera model and the state formulation; the choice between
`LidarMapPlanesFactor` and `LidarPlaneMomentFactor` is a choice about
whether the individual point weights matter. Reading the catalogue is how
you find the candidates; the family chapters are how you choose between them.

</div>

<div class="apx-note">
<span class="apx-title">Note — the crate's own API list</span>

The authoritative list is `cargo doc`; this page is a map of it. The two can
drift — a new factor type will appear in the API docs before it appears
here — and the crate's own `doc/factor-catalog.md` at the repository root
serves the same purpose for the older documentation surface.

</div>
