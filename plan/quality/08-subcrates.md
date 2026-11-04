# apex-solver workspace — the three sub-crates

Cross-cutting items (defaults, features, panics, duplicated code) are in files
01-07; this file lists what is specific to each member crate. The factor
extensibility work in the apex-vio main checkout (`plan/factor/02` N1-N4) covers
the `LieGroup` trait shape and `CameraModel::jacobian_pose`; it is referenced,
not repeated.

---

## `apex-manifolds` (13,962 lines of source, 8 groups plus `Rn`)

**What is good.** Closed-form exp/log/Jacobians with documented small-angle
fallbacks, a capability layer (`ActsOnPoints3`, `ConstDoF`, `RigidPose`,
`HasVelocity`) that replaced runtime group checks, and finite-difference
oracles in tests.

### S-M1 · One generic law suite for all groups · P1 · M
`tests/subgroup_law.rs` and `tests/jacobian_identities.rs` exist, but each group
file also carries its own 600-1,100 lines of tests with different coverage
(Sim3 and SGal3 have FD oracles; others do not all). **Proposal:** a
`testing` module (behind `cfg(test)` and a `testing` feature for dependents)
with `fn law_suite<G: LieGroup>()` covering: `exp(log(g)) = g`,
`log(exp(τ)) = τ` inside the injectivity radius, `compose` associativity,
inverse laws, `adjoint` against `compose`/`inverse`, every
`jacobian_self/jacobian_other` against a central difference, `act` Jacobian
against FD, `right_plus`/`right_minus` round trip, `normalize` idempotence, and
near-singular tangent behaviour (θ → 0, θ → π). Run it for every group with
`proptest` strategies (the August review recommended this and it is still
open). The per-group files then keep only group-specific tests. This is the
guard that makes S-P5 (closed-form SGal3) safe.

### S-M2 · Constructors with ten parameters · P3 · S
`SE23::from_components` (`se23.rs:479`) and `SGal3::from_components`
(`sgal3.rs:605`) take 10 and 9 floats under `#[allow(too_many_arguments)]`.
Replace with `from_parts(&SO3-or-quaternion, Vector3, Vector3 [, f64])` (the
`from_parts` form already exists for SGal3 in `exp`). See S-R3.

### S-M3 · `generator(i)` and the `Rn` assertions · P1 · M
See S-C3. Both are panics reachable from safe code.

### S-M4 · Jacobian inverses return identity on failure · P1 · M
See S-C2 (Sim3, SGal3).

### S-M5 · SGal3 Jacobians by finite differences · P2 · L
See S-P5.

### S-M6 · File size · P3 · M
Test code is 38-58 % of the big group files (07, S-T5). Move group tests to
`src/<group>/tests.rs`; production halves are 700-1,000 lines each, which is a
reasonable size.

---

## `apex-camera-models` (7,811 lines, 10 models)

**What is good.** Convergence gates on iterative unprojection (the KB model
rejects an out-of-domain radius and a non-converged Newton iteration instead of
clamping), a single `CameraModelError` taxonomy, analytic projection and
point/intrinsic Jacobians for every model.

### S-K1 · The crate has no tests of its own · P1 · S
There is no `crates/apex-camera-models/tests/`. The model integration tests are
`tests/camera_{pinhole,radtan,kannala_brandt,double_sphere,eucm,fov,ucm}_integration.rs`
and `camera_test_utils.rs` in the **parent** `apex-solver` crate. A change to a
model crate is therefore verified only if you run the consumer, and the model
crate cannot be released on its own with confidence. **Proposal:** move those
files into `crates/apex-camera-models/tests/`; keep a thin set in the parent
for the factor-level checks that really need `Problem`.

### S-K2 · Analytic unprojection Jacobian · P2 · M
Add `CameraModel::jacobian_unproject(&self, pixel) -> Matrix3x2` and use it in
`InverseDepthFactor::unproject_jacobian` (S-C7) and apex-vio's bearing code.
Models with closed-form unprojection (pinhole, UCM, DS, EUCM) differentiate
directly; iterative ones (KB, FTheta, FOV, RadTan) use the implicit-function
theorem at the converged solution (`dθ/du = -f'^{-1} · ∂f/∂u`), which costs one
extra evaluation and is exact up to the solve tolerance.

### S-K3 · Calibration code inside the models · P2 · M
`linear_estimation` exists in seven model files (S-R6). It is a calibration
initializer, not a camera model. Move to a `calibration` module behind a
trait; this removes about 700 lines from the model files and the shared
validation block (40 windows between DS and UCM).

### S-K4 · `jacobian_pose` couples the camera crate to SE3 · P2 · M
Already planned (plan/factor N4): remove it from the trait and let the factor
compose `∂uv/∂p_cam` with the group Jacobian. The solver's `ProjectionFactor` no
longer calls it (it uses `act_point`), so the method is now unused in this
workspace and removal is a pure API change (a breaking release, S-E5).

### S-K5 · Naming noise · P3 · S
106 `clippy::similar_names` and 23 `many_single_char_names` hits come from the
literature's symbols (`fx, fy, cx, cy, xi, alpha`, ...). Do not rename; group
the four pinhole intrinsics in an `Intrinsics4 { fx, fy, cx, cy }` value passed
to the projection helpers, which also removes four-argument signatures.

### S-K6 · Dispatch enum lives downstream · P3 · S
`CameraModelEnum` (a ten-way enum with project/unproject/Jacobians) is defined
in apex-vio (`config/camera.rs`). Any other consumer must rewrite it. Consider
hosting it (feature `dispatch`) in the camera crate; apex-cv works generically
over `CameraModel` today, so only apex-vio would change.

---

## `apex-io` (19,797 lines of source plus 3,002 in `bin/`; the largest member)

### S-I1 · Features and binaries · P1 · M
See S-A8 and S-T1/T2: `default = []`, split `rosbag`, move the seven binaries
(3,002 lines) out.

### S-I2 · Untrusted-size allocations; no fuzzing · P1 · M
See S-E2 for the exact sites and the `bounded_len` helper.

### S-I3 · `utils.rs` is two modules · P3 · S
`utils.rs` (806 lines) holds the dataset registry (`DatasetRegistry`,
`OdometryEntry`, `BaEntry`, `SensorEntry`) and the download helpers
(`download_file`, `extract_tar_gz`, `decompress_bzip2`,
`ensure_*_dataset`). Split into `registry.rs` and `download.rs`; only the
second needs the `download` feature. Today the two share one file with ten
scattered `#[cfg(feature = "download")]` attributes.

### S-I4 · g2o and TORO parsing · P2 · S
See S-R5 (`field::<T>` helper, 58 shared windows, 103 index sites).

### S-I5 · Two TUM readers · P2 · S
`apex-io/src/trajectory/tum.rs` (387 lines, `parse`/`render` on `Trajectory`) and
`apex-cv/crates/apex-evaluation/src/io/tum.rs` (230 lines, `read_tum`/`associate`
on a different `Trajectory`) parse the same format into different types. apex-vio
imports the second. **Proposal:** one reader, in `apex-io`, returning a plain
`Vec<TimedPose>`; `apex-evaluation` consumes it. (Cross-repo item X-5 in
the apex-vio plan.)

### S-I6 · Storage backends · P3 · M
`ros2/storage/mcap.rs` (1,358 lines) and `sqlite.rs` (1,073): define the
`BagStorage` trait once and share message-range filtering (S-R8).

### S-I7 · Writer paths · P3 · M
`ros2/writer.rs` (976 lines) and `bag_convert` implement ROS1<->ROS2 conversion.
They are only exercised through the binary. Move conversion into a library
function with a round-trip test (write a synthetic ROS1 bag, convert, read back
and compare messages) so that the code that remains after S-A8 is covered.

### S-I8 · `apex-io` is four libraries in one crate · P2 · L
Source lines: `rosbag/` 12,283 (ROS1 and ROS2 readers and writers, CDR, MCAP,
SQLite, DDS), `graph/` 2,920 (g2o, TORO), `asl/` 1,325 (EuRoC format),
`trajectory/` 1,285, plus `bal.rs` 767, `utils.rs` 806 (downloads), `logger.rs`
286. Consumers want different slices: apex-solver wants `graph` and `bal`;
apex-vio wants `rosbag`, `asl`, `trajectory`; the CLI wants `utils`.
**Proposal (after S-T2 has split the features):** if the feature split still
leaves cross-dependencies tangled, split into `apex-bag` (rosbag+dds),
`apex-dataset` (asl, bal, registry, download) and keep `graph`/`trajectory` as
`apex-io`. Do it only if the feature flags prove insufficient: features are the
cheaper, reversible step, and a crate split is a semver event for three
repositories.
