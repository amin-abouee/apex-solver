# Range and Bearing Factors

Not every measurement is a camera. Radar, sonar, UWB, acoustic angle-of-arrival
and laser rangefinders all produce *ranges* and *bearings* — and the bearing
family lives in its own module rather than under `visual`, because a bearing
is not a camera concept. This chapter is that family: three range factors and
two bearing factors, and the convention they share.

## 1. The family

<div class="apx-table">
<span class="apx-title">`factors::ranging`</span>

| Factor | Blocks | Measurement |
|---|---|---|
| `PosePoseRangeFactor` | two poses | the distance between two frames' origins |
| `PosePointRangeFactor` | a pose, a point | the distance from a pose's origin to a known point |
| `BearingRangeFactor` | a pose, a point | a distance **and** a direction, together |
| `BearingFactor` | a pose | the direction from a pose's origin to a known point |
| `BearingFactorSe23` | an `SE23` state | the same, for the `SE23` formulation |

</div>

The module's own comment explains the grouping: "Bearing-only measurements
are not camera-specific, so they live here rather than under `visual`; keeping
`bearing` next to `bearing_range` also keeps the two bearing residuals on a
single SE(3) convention."

## 2. The two conventions in one module

A **range** factor measures a distance. A **bearing** factor measures a
direction. Both are usually expressed *relative to a frame's origin* — a
rangefinder measures the distance from the sensor, a radar measures the
direction from the sensor — and the factor's residual is the difference between
that and what the state predicts.

The subtlety is the **frame of the origin**. A radar mounted on a vehicle
measures from the vehicle's reference point; a factor that assumes the camera
centre will be wrong by the mount's lever arm. The crate's convention is that
the origin is the frame's own origin, and the lever arm is a property of the
model that composes the frames (an `SE3` between the sensor and the body),
not of the factor.

## 3. `BearingRangeFactor`

A combined measurement: a distance *and* a direction to the same point, in
one factor. The two pieces are not redundant — a range alone constrains the
radial direction, a bearing alone constrains the angular direction, and
together they pin the point. A factor that combined them incorrectly (for
example, weighting them as one residual) would lose the distinction between
"the range is wrong" and "the direction is wrong", which matters when one of
the two is corrupted.

## 4. `BearingFactor` versus `BearingFactorSe23`

The same measurement in two state formulations: a plain `SE3` pose, and an
`SE23` state that carries the velocity along with the pose. The `SE23` form
is what a visual-inertial or LiDAR-inertial graph uses, because the interval
between two states already carries the velocity, and a bearing measured in the
middle of that interval belongs to a state that has one.

The residual in the `SE23` case is written on the sphere — the perpendicular
component of the difference between the measured and predicted directions —
which is the same projector $I - \mathbf{b}\mathbf{b}^\top$ the visual bearing
factors use. The crate keeps one convention for the two, which is why the
module's comment insists on it.

## 5. Worked example

A UWB tag on a backpack: the range factor measures 12.4 m, the bearing factor
measures the direction. The two are registered on the same graph, and their
noise models say how much each is trusted — a UWB range is often good to
10 cm and its *bearing* poor, because the antenna array's baseline is short.
A graph that weights them equally throws away the information the hardware
provides.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a range factor is not a point-to-point factor</span>

`PosePoseRangeFactor` measures the distance between two frames' origins and
nothing else: it is invariant to how the two frames are rotated about each
other, and it constrains exactly one scalar per pair. A graph built from
range factors alone has a large null space, and the states it does constrain
can be rotated freely about the baseline. Using range factors as if they were
full relative-pose constraints is a silent way to under-constrain a graph.

</div>

<div class="apx-note">
<span class="apx-title">Note — the same lever-arm trap as the visual factors</span>

Both factors measure from the *frame's origin*, and a sensor is rarely at the
frame's origin. The composition between the sensor frame and the body frame
carries the offset, and a factor that is given the body frame where the
sensor frame belongs will be wrong by exactly that offset, in a way that looks
like a scale error.

</div>
