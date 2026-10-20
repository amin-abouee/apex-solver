# Motion Model Factors

Some constraints come from no sensor at all. They assert what the platform's
*dynamics* must be: a car cannot move sideways, a platform on the ground
cannot roll, a platform that has stopped cannot be accelerating. The module's
documentation is unusually blunt about the hazard, and this chapter repeats it:
these factors are cheap, easy to misapply, and confidently wrong when their
assumption is violated.

## 1. The family

<div class="apx-table">
<span class="apx-title">`factors::motion`</span>

| Factor | Asserts | Valid when |
|---|---|---|
| `NonholonomicFactor` | the platform's lateral velocity is zero | a wheeled vehicle on a plane |
| `PlanarMotionFactor` | the motion stays in a plane, without rotating out of it | a platform constrained to a plane |
| `ZeroVelocityFactor` | the velocity is zero | the platform has stopped |
| `ZeroAngularRateFactor` | the angular rate is zero | the platform is not rotating |

</div>

The module's own words:

> These carry no measurement of their own (beyond a stationary gyro reading),
> which makes them unusually cheap — and unusually easy to misapply. Each is
> only valid while its assumption holds, so they are added by whatever
> detects that condition: a stationarity detector for `ZeroVelocityFactor` and
> `ZeroAngularRateFactor`, vehicle knowledge for `NonholonomicFactor` and
> `PlanarMotionFactor`. Applied where the assumption is violated they are
> confidently wrong, in the way only a zero-noise constraint can be.

## 2. Why a zero-noise constraint is dangerous

A measurement's uncertainty says how much to trust it. A motion-model factor
has *no* uncertainty: it says the lateral velocity is exactly zero, and the
solver is told to believe it as exactly as it is told to believe a
millimetre-accurate lidar. That is the right behaviour when the assumption
holds and catastrophic when it does not: a car that is skidding, a platform on
a slope, a drone that is *supposed* to roll, all get contradicted by a factor
with no slack.

The mitigation is structural, and it is the consumer's job: the factor is
added **only while the detector says the assumption holds**, and removed when
it stops. A motion-model factor permanently in the graph is a bug.

## 3. `NonholonomicFactor`

The classic: a wheeled vehicle's velocity lies along its heading, so the
lateral component vanishes. The residual is one scalar per frame — the
component of the velocity perpendicular to the heading. It is a strong
constraint (it says the platform cannot slip) and it is exactly wrong during
a skid, a turn at low speed, or on a slope.

## 4. `PlanarMotionFactor`

The platform's motion stays in a plane and its orientation does not roll out of
it. Two components are constrained rather than one, and the assumption is
about the *platform*, not about the trajectory: a vehicle on a banked road
still satisfies it (the plane is tilted, but fixed), while a vehicle on a
changing slope does not.

## 5. The stationarity factors

`ZeroVelocityFactor` asserts the velocity is zero; `ZeroAngularRateFactor`
that the angular rate is zero. The second is the one that can be measured
directly: a stationary platform's gyro reads its bias, so the factor is a
*measurement of the gyro bias* in disguise — which is why it is cheap and why
it is dangerous when the platform is rotating slowly.

## 6. Worked example

A handheld rig that the user sets down for a second. The stationarity
detector fires; `ZeroVelocityFactor` is added with a tight weight and
`ZeroAngularRateFactor` with a looser one (the gyro's bias is known to
$0.01$ rad/s, the motion is not known at all). When the user picks the rig
up, the detector stops firing and the factors are removed. During the stop, the
factors contribute a large fraction of the graph's information — and that is
exactly the point, and exactly the risk.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a motion model is a modelling decision, not a measurement</span>

Every other factor in the crate has a noise model and can be re-weighted when
it disagrees with the data. These cannot: they assert, and the solver believes
them absolutely. That makes them the most valuable factors in a graph where
they hold and the most destructive where they do not. A factor with no
uncertainty is a statement about the world, and the statement's validity is
the caller's responsibility to establish — continuously, not once.

</div>

<div class="apx-note">
<span class="apx-title">Note — the cheapest factors are the most-used</span>

Because they carry no measurement, motion-model factors are nearly free, and a
graph can be improved by adding several. That is also how a graph acquires
several mutually inconsistent assertions, each of which is individually
plausible and collectively impossible.

</div>
