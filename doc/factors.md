# Factors

Every factor shipped in apex-solver, grouped by sensor modality. A factor is one
measurement's contribution to the objective: given the variables it connects, it
produces a residual vector and its analytic Jacobian. For the mathematics of each
factor — construction, error, and Jacobian written out — see the
[Factor Reference](cookbook/src/factors/index.md) chapters of the cookbook. For
the GTSAM audit behind this library (what was implemented, what was skipped, and
why) see the [Factor Catalog](factor-catalog.md).

## How to read these tables

- **Variables** are the parameter blocks the factor expects, in registration
  order. `T_wb`-style poses are body-in-world; camera factors that use a
  world-to-camera pose say so in their docs.
- **Residual** is `Factor::residual_dim()`. The Jacobian columns are the
  *minimal* (tangent) dimensions: an `SE3` block is 7 parameters but 6 columns.
- **Self-whitening** factors weight their own residual (built-in square-root
  information or covariance) and must be registered with `NoiseModel::null()`.
  All other factors take their weighting from the [`NoiseModel`](cookbook/src/noise.md)
  registered with the residual block.

## Pose & priors — [`src/factors/pose/`](../src/factors/pose/) · [derivation](cookbook/src/factors/pose.md)

| Factor | Variables | Residual | What it does |
|---|---|---|---|
| `PriorFactor<T>` | 1 manifold `T` | `dof(T)` | Tangent-space anchor `Log(T_prior⁻¹ ∘ X)` on any Lie group |
| `EuclideanPriorFactor` | 1 `Rⁿ` | `n` | Ambient-space anchor for `Rn` states (landmarks, velocities, biases) |
| `BetweenFactor<T>` | 2 manifolds `(T_i, T_j)` | `dof(T)` | Relative constraint — odometry, loop closure (pair with a robust loss) |
| `PoseRotationPrior` | 1 `SE3` | 3 | Rotation-only anchor on a pose |
| `PoseTranslationPrior` | 1 `SE3` | 3 | World-frame translation-only anchor |

## IMU — [`src/factors/imu/`](../src/factors/imu/) · [derivation](cookbook/src/factors/imu.md)

All four share one `ImuPreintegration` (preintegrated ΔR, Δv, Δp between two
keyframes). Pick the group (`se23` = `(R,t,v)`, `sgal3` adds an estimated time
coordinate) and the bias handling (`ImuFactor` shares one bias variable across
the interval — evolve it with a `bias_random_walk()` edge; `CombinedImuFactor`
takes a bias per keyframe and embeds the random walk in its trailing rows).

| Factor | Variables | Residual |
|---|---|---|
| `imu::se23::ImuFactor` | `(SE23_i, SE23_j, bias)` | 9 |
| `imu::se23::CombinedImuFactor` | `(SE23_i, bias_i, SE23_j, bias_j)` | 15 |
| `imu::sgal3::ImuFactor` | `(SGal3_i, SGal3_j, bias)` | 10 |
| `imu::sgal3::CombinedImuFactor` | `(SGal3_i, bias_i, SGal3_j, bias_j)` | 16 |

Helpers: `bias::bias_random_walk()` builds the `BetweenFactor<Rn>` bias-evolution
edge; `bias::bias_random_walk_noise()` builds its `NoiseModel` from
`ImuParameters` and `Δt`.

## Visual — [`src/factors/visual/`](../src/factors/visual/) · [derivation](cookbook/src/factors/visual.md)

| Factor | Variables | Residual | What it does |
|---|---|---|---|
| `ProjectionFactor<CAM, OP>` | `(pose, landmark)` + intrinsics per the `OP` config | `2·N` obs | Standard reprojection; `OP` toggles which of pose/landmark/intrinsics are optimized |
| `ExtrinsicProjectionFactor<CAM>` | `(T_WB, T_BC, p_world)` | 2 | Reprojection with estimated camera extrinsics |
| `TimeOffsetProjectionFactor<CAM>` | `(SE23 state, T_BC, p_world, t_d)` | 2 | Reprojection with an estimated camera-to-IMU time offset |
| `StereoFactor` | `(pose, landmark)` | 3 | Rectified stereo `(u_L, u_R, v)` reprojection |
| `InverseDepthFactor<CAM>` | `(pose_i, anchor, pose_j)` | 2 | Inverse-depth landmark anchored at a pixel in the first view |
| `SmartProjectionFactor<CAM>` | `N` poses (no landmark) | `2·N` | Structure-less multi-view factor; triangulates internally, self-whitening |
| `EssentialMatrixFactor` | 1 relative pose `T_21` | `N` pairs | 2D–2D epipolar residual per normalized point pair |
| `EssentialMatrixConstraint` | 1 relative pose `T_21` | 6 | Pose realizes a measured essential matrix (up to scale) |
| `DepthFactor<ONESIDED>` | `(T_WS, homogeneous point, T_SC)` | 1 | RGB-D-style depth reading; `ONESIDED` penalizes only too-close points; self-whitening |
| `HomogeneousPointFactor` | 1 homogeneous 4D point | 3 | Unary prior on a dehomogenized position; self-whitening |

## LiDAR — [`src/factors/lidar/`](../src/factors/lidar/) · [derivation](cookbook/src/factors/lidar.md)

Scan-matching factors over `[T_wr, p_body]` — a body pose and the scan point as a
variable block (hold it fixed to treat it as a measurement).

| Factor | Residual | What it does |
|---|---|---|
| `PoseToPointFactor` | 3 | Point-to-point correspondence |
| `PointToPlaneFactor` | 1 | Point-to-plane against a target plane (no distance field) |
| `GicpFactor` | 3 | Plane-to-plane via combined-covariance whitening; self-whitening |
| `LidarEdgeFactor` | 3 | LOAM point-to-edge (point-to-line) against a matched edge; self-whitening |
| `IcpFactor<F: DistanceField>` | 1 | Point against a distance field in frame A; self-whitening |

## GNSS & navigation — [`src/factors/navigation/`](../src/factors/navigation/) · [derivation](cookbook/src/factors/navigation.md)

| Factor | Variables | Residual | What it does |
|---|---|---|---|
| `GpsFactor` | `(T_WS, T_GW)` | 3 | Synchronous GPS position with an estimated GPS-to-world frame; self-whitening |
| `GpsAsyncFactor` | `(T_WS, speed-and-bias, T_GW)` | 3 | Asynchronous GPS position with lever arm and propagation; self-whitening |
| `GpsVelocityFactor` | 1 `R³` velocity | 3 | GNSS velocity measurement |
| `PseudorangeFactor` | `(R³ position, R¹ clock bias)` | 1 | Raw pseudorange with fixed satellite ephemeris |
| `DopplerFactor` | `(R³ position, R³ velocity)` | 1 | Doppler range-rate |
| `BarometricFactor` | `(SE3 pose, bias)` | 1 | Altimeter height with a drifting pressure-reference bias |
| `AttitudeFactor` | 1 `SE3` pose | 3 | Gravity/magnetometer direction constraint (pair two for full yaw) |

## Range & bearing — [`src/factors/ranging/`](../src/factors/ranging/) · [derivation](cookbook/src/factors/ranging.md)

| Factor | Variables | Residual | What it does |
|---|---|---|---|
| `PosePoseRangeFactor` | `(pose_i, pose_j)` | 1 | Distance between two pose origins |
| `PosePointRangeFactor` | `(pose, point)` | 1 | Distance from a pose origin to a landmark |
| `BearingRangeFactor` | `(pose, point)` | 4 | Bearing (3 rows) + range (1 row) to a landmark |
| `BearingFactor` | `(pose, point)` | 2 | Unit direction from a pose to a landmark; self-whitening |

## Motion models — [`src/factors/motion/`](../src/factors/motion/) · [derivation](cookbook/src/factors/motion.md)

| Factor | Variables | Residual | What it does |
|---|---|---|---|
| `NonholonomicFactor` | 1 `SE23` state | 2 | No lateral/vertical velocity in the body frame |
| `PlanarMotionFactor` | 1 `SE3` pose | 3 | Planar (SE2-like) motion constraint |
| `ZeroVelocityFactor` | 1 `SE23` state | 3 | Zero velocity while at rest (ZUPT) |
| `ZeroAngularRateFactor` | 1 IMU bias `[b_g, b_a]` | 3 | Zero-angular-rate update — at rest the gyro reads its own bias |

## Marginalization — [`src/factors/marginal/`](../src/factors/marginal/) · [derivation](cookbook/src/factors/marginal.md)

| Factor | Variables | Residual | What it does |
|---|---|---|---|
| `MarginalPriorFactor` | `k` blocks (any manifolds) | prior rank | Gaussian marginal over eliminated variables (iSAM2 `LinearContainerFactor` analogue); manifold-agnostic via a caller-supplied local-log closure; self-whitening |

---

*Back to [README](../README.md)*
