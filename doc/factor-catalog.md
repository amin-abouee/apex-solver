# Factor Catalog

A critical audit of the GTSAM factor library mapped onto apex-solver: what is
implemented, what was deliberately skipped, and why. Each factor implements
the buffer-writing [`Factor`](../src/factors/mod.rs) trait with analytical
manifold Jacobians and finite-difference unit tests.

## Implemented

### Pose / SLAM core (`src/factors/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `PriorFactor<T>` | `PriorFactor<T>` | tangent-space, any Lie group |
| `EuclideanPriorFactor` | `PriorFactor<Rn>` / `PartialPriorFactor` | plain Rⁿ anchoring |
| `BetweenFactor<T>` | `BetweenFactor<T>` | relative constraints; loop closure = between + robust loss |
| `ProjectionFactor<CAM, OP>` | `GenericProjectionFactor` | 9 camera models, pose/point/intrinsics flags, smooth cheirality penalty |
| `BearingFactor` | `BearingFactor` | |
| `HomogeneousPointFactor` | `HomogeneousPointFactor` | |

### IMU (`src/factors/imu/`) — shared `ImuPreintegration`
All five produce the identical 15D residual `[p, q, v, bg, ba]`; the
Gauss–Markov bias random walk is **embedded in the residual** (GTSAM
`CombinedImuFactor` / OKVIS convention) — no separate bias edge is needed.
An initial bias prior (`EuclideanPriorFactor` on R⁶) is required for
observability; if you prefer GTSAM's non-combined style (bias walk as a
separate edge), a `BetweenFactor<Rn<6>>` with `diag(σ_gw², σ_aw²)·dt` covers
it with zero new code.

| Factor | Layout | Formulation |
|---|---|---|
| `ImuFactor` | `(pose_i, sb_i, pose_j, sb_j)` | SE(2)₃ |
| `CombinedImuFactor` | `(pose, vel, bias) × 2` | SE(2)₃ |
| `CombinedSe23ImuFactor` | `(SE23 state, bias) × 2` | SE(2)₃ native states |
| `Sgal3ImuFactor` | `(pose_i, sb_i, pose_j, sb_j)` | SGal(3); FD tests currently `#[ignore]` (tangent Jacobian chain under investigation; the zero-residual/group-composition formulation tests pass) |
| `Sgal3CombinedImuFactor` | `(pose, vel, bias) × 2` | SGal(3) |

### Visual (`src/factors/visual/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `StereoFactor` | `GenericStereoFactor3D` | rectified `(uL, uR, v)`, cheirality penalty |
| `InverseDepthFactor<CAM>` | `InvDepthFactor3` (unstable) | anchor pixel + inverse depth; well-conditioned for distant landmarks |
| `EssentialMatrixFactor` | `EssentialMatrixFactor` | scalar epipolar residual over the relative pose (scale-free) |
| `EssentialMatrixConstraint` | `EssentialMatrixConstraint` | measured `(R_E, u_E)` as a 6D pose constraint |
| `SmartProjectionFactor<CAM>` | `SmartProjectionPose3Factor` | pose-only, DLT + Gauss-Newton re-triangulation, exact implicit-Schur Jacobian; registers with `NoiseModel::null()` (internal whitening); degeneracy → bounded penalty |

### LiDAR (`src/factors/lidar/`)
| Factor | Analogue | Notes |
|---|---|---|
| `IcpFactor<F>` (existing) | distance-field ICP | |
| `LidarEdgeFactor`, `LidarPlaneFactor` (existing) | LIO-SAM edge/plane | |
| `PoseToPointFactor` | `PoseToPointFactor` (unstable) | matched 3D correspondences |
| `PointToPlaneFactor` | LIO-SAM `PoseToPlane` fork | target plane `(n, d)`; no distance field needed |
| `GicpFactor` | Segal et al. GICP (external everywhere in GTSAM-land) | plane-to-plane via combined-covariance whitening, frozen at construction with a rotation hint |

### GNSS / navigation (`src/factors/navigation/`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `GpsFactor`, `GpsAsyncFactor` (existing) | `GPSFactor` | |
| `GpsVelocityFactor` | `GPSVelocityFactor` | R³ velocity |
| `PseudorangeFactor` | `PseudorangeFactor` | receiver position + clock bias (meters); degenerate geometry penalized |
| `DopplerFactor` | `DopplerFactor` | range-rate over position + velocity |
| `BarometricFactor` | `BarometricFactor` | `[SE3 pose, R¹ bias]` |
| `AttitudeFactor` | `AttitudeFactor` / `MagFactor` | gravity/magnetometer direction; add two (gravity + mag) for full AHRS |

### Range family (`src/factors/range_factor.rs`)
| Factor | GTSAM analogue | Notes |
|---|---|---|
| `PosePoseRangeFactor` | `RangeFactor<Pose3,Pose3>` | |
| `PosePointRangeFactor` | `RangeFactor<Pose3,Point3>` | |
| `BearingRangeFactor` | `BearingRangeFactor` | 4D residual: 3 bearing + 1 range |

### Marginalization (`src/factors/marginal/`)
| Factor | Analogue | Notes |
|---|---|---|
| `MarginalPriorFactor` | iSAM2 `LinearContainerFactor` | Gaussian prior over eliminated variables; manifold-agnostic via a caller-supplied local-log closure; container semantics (constant `S` Jacobian, rebuild on relinearization) |
| `PoseRotationPrior` | `PoseRotationPrior` | rotation-only loop-closure init |
| `PoseTranslationPrior` | `PoseTranslationPrior` | |

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
