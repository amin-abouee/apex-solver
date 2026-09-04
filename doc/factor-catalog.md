# Factor Catalog

A critical audit of the GTSAM factor library mapped onto apex-solver: what is
implemented, what was deliberately skipped, and why. Each factor implements
the buffer-writing [`Factor`](../src/factors/mod.rs) trait with analytical
manifold Jacobians and finite-difference unit tests.

**Layout.** A factor lives in the folder named for its sensor modality, and is
addressed by that path — `factors::visual::StereoFactor`,
`factors::imu::se23::ImuFactor`. `apex_solver::prelude` re-exports only the
common set. `factors::common/` holds what several families share; manifold
derivatives are *not* among them, since every group in `apex-manifolds` already
reports the Jacobians of its own operations.

**Two conventions worth knowing.** Poses are body-in-world (`T_wb`), so a
body-frame direction is `Rᵀ(p_world − t)`. And a factor either whitens
internally or expects a `NoiseModel` — never both; the ones that do report
`Factor::whitens_internally()`, and registering them with anything but
`NoiseModel::null()` is rejected rather than silently double-weighted.

## Implemented

### Pose / SLAM core (`src/factors/pose/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `PriorFactor<T>` | `PriorFactor<T>` | tangent-space, any Lie group |
| `EuclideanPriorFactor` | `PriorFactor<Rn>` / `PartialPriorFactor` | plain Rⁿ anchoring |
| `BetweenFactor<T>` | `BetweenFactor<T>` | relative constraints; loop closure = between + robust loss |
| `PoseRotationPrior` | `PoseRotationPrior` | rotation-only anchor (moved here from `marginal/`) |
| `PoseTranslationPrior` | `PoseTranslationPrior` | world-frame translation anchor |

### IMU (`src/factors/imu/`) — shared `ImuPreintegration`

Two factors per group and nothing else, so the choice is two questions.

**Which group?** `se23` models a keyframe as `(R, t, v)`. `sgal3` adds a time
coordinate, `(R, t, v, s)`, making the inter-keyframe interval an estimated
quantity — pick it for sensor time-offset or rolling-shutter calibration.

**Combined or not?** `ImuFactor` shares one bias variable across the interval
and leaves its evolution to a `bias::bias_random_walk` edge.
`CombinedImuFactor` takes a bias per keyframe and embeds the random walk in its
trailing six rows, so it needs no such edge. Using both counts that
uncertainty twice. Weighting follows the same split: the shared-bias form uses
a 9×9 information built from measurement noise alone, the combined form the
full 15×15.

| | `ImuFactor` | `CombinedImuFactor` |
|---|---|---|
| `se23` | 9D, `(SE23, SE23, bias)` | 15D, `(SE23, bias, SE23, bias)` |
| `sgal3` | 10D, `(SGal3, SGal3, bias)` | 16D, `(SGal3, bias, SGal3, bias)` |

Both names exist in both modules, so they are addressed by their group:
`imu::se23::ImuFactor`, `imu::sgal3::CombinedImuFactor`.

A keyframe is a **single** state variable on the group, not separate pose and
velocity blocks: the optimizer's update is then a group right-plus, and the
pose/velocity coupling inertial integration produces is the group's job.
The practical consequence is that aiding measurements attach to that state —
an `SE3`-only factor (`PoseTranslationPrior`, `ProjectionFactor`, …) cannot
attach to an `SE23` variable. `PriorFactor<SE23>` and `MarginalPriorFactor`
are generic and do work; see `tests/factor_integration.rs` for a GNSS
position+velocity fix written against a state.

SGal(3)'s extra row is `(t_j − t_i) − Δt`, weighted by `1/σ_t`
(`with_time_sigma`, default 100 µs).

An initial bias prior (`EuclideanPriorFactor` on R⁶) is required for
observability in every configuration.

### Visual (`src/factors/visual/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `StereoFactor` | `GenericStereoFactor3D` | rectified `(uL, uR, v)`, cheirality penalty |
| `InverseDepthFactor<CAM>` | `InvDepthFactor3` (unstable) | anchor pixel + inverse depth; well-conditioned for distant landmarks |
| `EssentialMatrixFactor` | `EssentialMatrixFactor` | scalar epipolar residual over the relative pose (scale-free) |
| `EssentialMatrixConstraint` | `EssentialMatrixConstraint` | measured `(R_E, u_E)` as a 6D pose constraint |
| `ProjectionFactor<CAM, OP>` | `GenericProjectionFactor` | 9 camera models, pose/point/intrinsics flags, smooth cheirality penalty |
| `DepthFactor` / `OneSidedDepthFactor` | OKVIS `DepthErrorT` | 1D depth on a homogeneous landmark; one-sided variant penalizes only too-near points |
| `HomogeneousPointFactor` | `HomogeneousPointFactor` | unary prior on an R⁴ homogeneous landmark |
| `SmartProjectionFactor<CAM>` | `SmartProjectionPose3Factor` | pose-only, DLT + Gauss-Newton re-triangulation, exact implicit-Schur Jacobian; registers with `NoiseModel::null()` (internal whitening); degeneracy → bounded penalty |

### LiDAR (`src/factors/lidar/`)
| Factor | Analogue | Notes |
|---|---|---|
| `IcpFactor<F>` | distance-field ICP | `lidar/distance_field.rs`; generic over a `DistanceField` |
| `LidarEdgeFactor` | LOAM edge | `lidar/edge.rs`; point-to-line vector rejection |
| `LidarPlaneFactor` | LOAM plane | `lidar/plane.rs`; a type alias for `IcpFactor<PrecomputedPlane>` |
| `PoseToPointFactor` | `PoseToPointFactor` (unstable) | matched 3D correspondences |
| `PointToPlaneFactor` | LIO-SAM `PoseToPlane` fork | target plane `(n, d)`; no distance field needed |
| `GicpFactor` | Segal et al. GICP (external everywhere in GTSAM-land) | plane-to-plane via combined-covariance whitening, frozen at construction with a rotation hint |

### GNSS / navigation (`src/factors/navigation/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `GpsFactor`, `GpsAsyncFactor` | `GPSFactor` | `navigation/gps.rs`; async variant propagates through `ImuPreintegration` |
| `GpsVelocityFactor` | `GPSVelocityFactor` | R³ velocity |
| `PseudorangeFactor` | `PseudorangeFactor` | receiver position + clock bias (meters); degenerate geometry penalized |
| `DopplerFactor` | `DopplerFactor` | range-rate over position + velocity |
| `BarometricFactor` | `BarometricFactor` | `navigation/barometric.rs`; `[SE3 pose, R¹ bias]` |
| `AttitudeFactor` | `AttitudeFactor` / `MagFactor` | `navigation/attitude.rs`; gravity/magnetometer direction; add two (gravity + mag) for full AHRS |

### Range / bearing (`src/factors/ranging/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `PosePoseRangeFactor` | `RangeFactor<Pose3,Pose3>` | |
| `PosePointRangeFactor` | `RangeFactor<Pose3,Point3>` | |
| `BearingRangeFactor` | `BearingRangeFactor` | 4D residual: 3 bearing + 1 range |
| `BearingFactor` | `BearingFactor` | 2D, projected onto the tangent plane at the measurement |

### Marginalization (`src/factors/marginal/`)
| Factor | Analogue | Notes |
|---|---|---|
| `MarginalPriorFactor` | iSAM2 `LinearContainerFactor` | Gaussian prior over eliminated variables; manifold-agnostic via a caller-supplied local-log closure; container semantics (constant `S` Jacobian, rebuild on relinearization) |

The partial-pose priors now live in `pose/partial_prior.rs`: they anchor a pose
for loop-closure initialization rather than summarizing eliminated variables.

## Deliberately skipped (with reasons)

- **2D–2D / 3D–3D "matching factors"**: GTSAM has none — data association is
  upstream. The right analogues are `EssentialMatrixFactor` (2D–2D) and
  `PoseToPointFactor` (3D–3D), both implemented.
- `SmartProjectionRigFactor`, rolling-shutter variants, `SmartStereoProjection*`:
  niche/experimental in GTSAM itself (`gtsam_unstable`).
- `GeneralSFMFactor`: superseded here by `ProjectionFactor` + smart factors.
- Rotation-averaging family (`WahbaFactor`, `FrobeniusFactor`, `RotateFactor`,
  `KarcherMeanFactor`): initialization helpers, not measurement factors.
- `AntiFactor`, `BetweenFactorEM`, `RISAM*`: measurement rejection belongs to
  the robust-loss layer (`LossFunction` + Triggs corrector), which apex-solver
  already has.
- Carrier-phase GNSS: raw-ranging only for now.

## Test coverage

Every exported factor appears in at least one solved graph, enforced by
`every_exported_factor_is_exercised_by_an_integration_test` in
`tests/factor_coverage.rs`, which reads these modules' own `pub use` lines.

* `tests/factor_integration.rs` — VIO/SLAM scenarios on a synthetic trajectory.
* `tests/factor_coverage.rs` — the remaining factors, one scenario each.
* `tests/nclt_gnss_fusion.rs` — odometry + GNSS on the real
  [NCLT](https://robots.engin.umich.edu/nclt/) dataset (~40 MB of CSV, fetched
  on first run). Real recorded noise rather than measurements generated from
  the models under test: dead-reckoned odometry drifts to 44 m over 17 minutes,
  fusing GNSS holds it at the ~6 m receiver noise floor.

## Solver notes learned during integration

- **`fix_variable` caveat**: fixed indices are honored when applying steps,
  but the linear system still treats fixed coordinates as free; their step is
  discarded afterwards. When a fixed variable strongly couples with free ones
  through the same factors (e.g. fixed points in stereo BA), free variables
  are systematically under-corrected. Anchor with tight
  `EuclideanPriorFactor`/`PriorFactor` priors instead — see
  `tests/factor_integration.rs` (`anchor_rn`/`anchor_se3`).
- Fully-fixed variables can also surface as "structurally empty diagonal"
  errors in the sparse LM damping. Prefer prior anchoring there too.
- Planar/degenerate scenes: coplanar point clouds are homography-degenerate
  for projection/stereo/smart factors and in-plane-degenerate for point-to-plane
  factors. Integration tests deliberately use depth-distributed clouds.
