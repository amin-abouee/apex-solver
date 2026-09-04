# Factor Catalog

A critical audit of the GTSAM factor library mapped onto apex-solver: what is
implemented, what was deliberately skipped, and why. Each factor implements
the buffer-writing [`Factor`](../src/factors/mod.rs) trait with analytical
manifold Jacobians and finite-difference unit tests.

**Layout.** A factor lives in the folder named for its sensor modality, and is
addressed by that path — `factors::visual::StereoFactor`,
`factors::inertial::ImuFactor`. `apex_solver::prelude` re-exports only the
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

### Inertial (`src/factors/inertial/`) — shared `ImuPreintegration`

Two independent axes, following GTSAM. **Combined or not** decides how bias
evolution is modelled and therefore the residual dimension; **state
parameterization** decides how pose and velocity are stored.

Non-combined factors share one bias variable per edge and need a companion
bias random-walk edge — build one with `inertial::bias::{bias_random_walk,
bias_random_walk_noise}`. Combined factors embed that walk in their trailing
six rows and take a bias per frame, so adding the edge alongside one of them
would count the same uncertainty twice. The weighting follows the same split:
non-combined factors use a 9×9 information built from measurement noise alone,
combined factors the full 15×15.

| Factor | Residual | Blocks | GTSAM analogue |
|---|---|---|---|
| `ImuFactor` | 9D | `(SE3, vel, SE3, vel, bias)` | `ImuFactor` |
| `Se23ImuFactor` | 9D | `(SE23, SE23, bias)` | `ImuFactor2` |
| `CombinedImuFactor` | 15D | `(SE3, vel, bias) × 2` | `CombinedImuFactor` |
| `CombinedSe23ImuFactor` | 15D | `(SE23, bias) × 2` | — |
| `Sgal3ImuFactor` | 9D | `(SE3, vel, SE3, vel, bias)` | — |
| `Sgal3StateImuFactor` | **10D** | `(SGal3, SGal3, bias)` | — |
| `Sgal3CombinedImuFactor` | 15D | `(SE3, vel, bias) × 2` | — |
| `Sgal3CombinedStateImuFactor` | **16D** | `(SGal3, bias) × 2` | — |

SGal(3)'s tangent carries a **time** coordinate, giving a `(t_j − t_i) − Δt`
residual row — but only where time is estimated. The native-`SGal3` factors
take it as part of the state, so that row is a real constraint (sensor
time-offset and rolling-shutter calibration; weight it with
`with_time_sigma`). The split-block factors have no time variable, so the row
would be identically zero on both residual and Jacobian, and they drop it.

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
| `BarometricFactor` | `BarometricFactor` | `[SE3 pose, R¹ bias]` |
| `AttitudeFactor` | `AttitudeFactor` / `MagFactor` | gravity/magnetometer direction; add two (gravity + mag) for full AHRS |

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
