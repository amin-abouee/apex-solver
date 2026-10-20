# Navigation Factors

GNSS and its aids are the measurements that break the gauge. A camera-and-IMU
system is defined up to a global position and a global heading; a GNSS
position fixes the first, a velocity fixes the second, and everything else in
this family is a refinement of that idea. This chapter is the six factors that
bring absolute information into a graph.

## 1. The family

<div class="apx-table">
<span class="apx-title">`factors::navigation`</span>

| Factor | Measurement | Typical use |
|---|---|---|
| `GpsFactor` | a GNSS position with its covariance | the position observation of a multi-GNSS session |
| `GpsAsyncFactor` | the same, for an asynchronous (out-of-frame) measurement | a GNSS fix that arrives between two frames |
| `GpsVelocityFactor` | a GNSS velocity | the heading observation a position fix cannot give |
| `PseudorangeFactor` | a raw pseudorange to a satellite | the tightest GNSS observable, and the hardest to model |
| `DopplerFactor` | a raw Doppler measurement | the velocity observable, in the satellite's own frame |
| `BarometricFactor` | a barometric altitude | a height that is better than the GNSS's, in a building |
| `AttitudeFactor` | an attitude from gravity or a magnetometer | a roll and pitch observation independent of the IMU's estimate |

</div>

## 2. Position fixes and the gauge

A GNSS position factor says "this frame's origin is at $\mathbf{p}$". Two
consequences make it different from every other factor in the crate:

- it is **absolute**, so it fixes the gauge the rest of the graph is free in;
- it is **correlated in time** — the receiver's error drifts slowly, so
  successive fixes are not independent, and a graph that treats them as
  independent is over-confident in the slow modes.

The first is why a GNSS factor is often the only thing that makes a
camera-only graph's trajectory absolute. The second is why the factor's
covariance is the receiver's, and why a session with twenty fixes of the same
error does not constrain twenty times as much as twenty independent fixes
would.

## 3. `GpsAsyncFactor`

A GNSS fix that arrives *between* two camera frames. The factor's blocks
then straddle the interval, and the state it refers to is not the frame's
state but the state interpolated to the fix's time — which is why this factor
exists separately rather than as a flag on `GpsFactor`. The crate's
factor-family convention (pose-and-point versus two-pose) is the same
distinction: an asynchronous measurement does not belong to a frame.

## 4. `PseudorangeFactor` and `DopplerFactor`

The *raw* GNSS observables, rather than the solution the receiver computes.
A pseudorange is the measured distance to a satellite, including the clock
bias and the atmosphere; a Doppler is the range rate. Working with the raw
observables removes the receiver's own error processing from the loop — at the
cost of modelling satellite positions, the clock, and the atmosphere
explicitly.

The payoff is accuracy: a receiver's filtered solution has already
discarded information, and the graph can use the raw measurement with a
covariance that reflects the *observation* rather than the *solution*.

## 5. `GpsVelocityFactor`

A GNSS velocity fixes the heading — the direction the gravity vector cannot
see. A system that knows its acceleration in world axes but not its heading has
a one-dimensional gauge freedom that no amount of visual or inertial
measurement resolves. A velocity measurement closes it.

## 6. `BarometricFactor` and `AttitudeFactor`

Two cheap aids with narrow jobs. A barometer gives a *height* far better than
GNSS does in a building (GNSS's vertical error is two to three times its
horizontal one), which is what makes it worth a factor. An attitude factor
gives roll and pitch from gravity or from a magnetometer — the IMU's own
estimate of gravity is integrated and drifts, and an absolute attitude
observation stops it.

## 7. Worked example

A vehicle with a camera, an IMU and a GNSS receiver. The graph carries the
camera factors, the IMU factors, one `GpsFactor` per fix (with the
receiver's covariance), and one `GpsVelocityFactor` per velocity solution. The
GNSS factors are what make the trajectory absolute; without them the graph is
defined up to a global rigid motion, and any alignment against a map has to
estimate it.

## 8. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a GNSS factor's covariance is not independent of the others</span>

A receiver's errors are temporally correlated, and a graph that treats each
fix as an independent measurement will produce a covariance that is too
optimistic along the slow error modes — a slow drift in the position, which is
precisely the mode that matters for a long trajectory. The fix is not in the
factor (which is honest about the *measurement*) but in the graph's structure:
a factor that models the correlation, or a covariance that is deliberately
inflated to account for it.

</div>

<div class="apx-note">
<span class="apx-title">Note — the frame of the GNSS measurement is the receiver's antenna</span>

Every factor here measures from a *sensor*, and the sensor is not at the
state's origin. The antenna's lever arm is part of the model, and a factor
that ignores it is wrong by that offset — which for a vehicle's roof-mounted
receiver is metres.

</div>
