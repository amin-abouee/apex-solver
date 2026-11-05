# apex-solver — code redundancy

Detector: lines are normalized (numbers and string literals blanked, comments,
attributes and bare braces dropped); an 8-line window repeated anywhere in
production code counts as duplicated. Excluded: inline `#[cfg(test)]` modules,
`tests/`, `benches/`, `examples/`, files named `tests.rs` or `*_tests.rs`, and
windows that are only rows of numbers (constant tables). Result: **5,391 of
32,558 significant lines (16.6 %)**. For comparison apex-cv is 3.9 % and
apex-vio 2.4 %, so this repository carries most of the duplication in the
ecosystem. Test-fixture duplication is listed separately as S-R4.

| Rank | Pair or file | Shared windows | Kind |
|---:|---|---:|---|
| 1 | `optimizer/dog_leg.rs` ~ `gauss_newton.rs` | 184 | loop + plumbing (S-R1) |
| 2 | `bin/pose_graph_g2o.rs` with itself | 168 | two dataset drivers (S-R2) |
| 3 | `dog_leg.rs` ~ `levenberg_marquardt.rs` | 92 | loop + plumbing (S-R1) |
| 4 | `gauss_newton.rs` ~ `levenberg_marquardt.rs` | 79 | loop + plumbing (S-R1) |
| 5 | `manifolds/se23.rs` ~ `sgal3.rs` | 66 | accessors, constructors, adjoint blocks (S-R3) |
| 6 | `observers/visualization.rs` with itself | 63 | repeated logging blocks (S-A7) |
| 7 | `io/graph/g2o.rs` ~ `toro.rs` | 58 | tokenize + field parsing (S-R5) |
| 8 | `camera-models/double_sphere.rs` ~ `ucm.rs` | 40 | validation + `linear_estimation` (S-R6) |
| 9 | `double_sphere.rs` ~ `eucm.rs` | 28 | same (S-R6) |
| 10 | `src/logger.rs` ~ `apex-io/src/logger.rs` | 28 | whole module (S-A6) |
| 11 | `se23.rs` ~ `se3.rs` | 28 | accessors (S-R3) |
| 12 | `ros2/storage/sqlite.rs` with itself | 27 | repeated SQL access (S-R8) |

## S-R1 · Optimizer plumbing · P1 · L
See S-A1. 355 windows across three files is the largest single block in the
repo. Removing it also removes the 13 `too_many_arguments` suppressions that
exist only to support the copies.

## S-R2 · `bin/pose_graph_g2o.rs`: two drivers, one shape · P2 · M
`test_se2_dataset` (`:395`, 388 lines) and `test_se3_dataset` (`:784`, 354 lines)
differ in the vertex type, the loader and the expected-cost table; the
configuration, the solver sweep, the table printing and the failure handling
are the same. **Proposal:** `fn run_dataset<G: LieGroup>(spec: &DatasetSpec,
load: impl Fn(&Path) -> Result<Graph<G>>)` and a `const DATASETS: &[DatasetSpec]`.
Expected saving: about 450 lines. Part of S-A8.

## S-R3 · `se23.rs` / `sgal3.rs` / `se3.rs` · P2 · M
The 66 shared windows are mostly structural code: parameter accessors
(`x()`, `y()`, `z()`, `rotation_so3`, `rotation_quaternion`, ...), `new`
constructors and `from_components` (10 parameters, `#[allow(too_many_arguments)]`
in both `se23.rs:479` and `sgal3.rs:605`), and the shared adjoint blocks.
**Proposal:** a small private `macro_rules!` (`core/marginalization.rs` already
uses one) or a `trait SemiDirectParts` with default
methods for the accessors, and builder-style constructors
(`SE23::from_parts(rotation, translation, velocity)`) in place of a 10-argument
constructor. Do **not** try to merge the group laws: SGal3's `act` is not an
action (documented in `capability.rs`) and its tangent order differs
(`[ρ,ν,θ,s]` versus `[ρ,θ,ν]`).
Gate: the property/law tests in `crates/apex-manifolds/tests/` plus unchanged
Jacobian tables.

## S-R4 · Duplicate test fixtures · P3 · S
Measured separately because the main detector skips test files:
`imu/se23/tests.rs` ~ `imu/sgal3/tests.rs` (60 windows) and
`motion/tests.rs` ~ `visual/extrinsic_tests.rs` (54) build the same
pose/extrinsic/IMU fixtures. Move them to `factors/common/test_utils.rs`
(exists) as named builders. Also gives `tests/factor_coverage.rs` (2,565 lines)
something to share.

## S-R5 · g2o / TORO parsing · P2 · S
`g2o.rs:304-336` and `toro.rs:130-162` both tokenize a line and then parse
each field with the same five-line `parts[i].parse::<T>().map_err(|_|
IoError::InvalidNumber { line, value: parts[i].to_string() })?`. `parse_edge_se3`
(`g2o.rs:488`, 127 lines) repeats it ten times for one edge.
**Proposal:**
```text
fn field<T: FromStr>(parts: &[&str], index: usize, line: usize) -> Result<T, IoError>
```
(which also replaces the 103 direct `parts[...]` index operations flagged by
`indexing_slicing` with a bounds-checked access that returns
`IoError::MissingFields`). Then `parse_edge_se3` becomes a struct literal.
Expected saving: about 250 lines across both files and no remaining index
panics on malformed input.

## S-R6 · Camera models · P2 · M
`double_sphere.rs`, `ucm.rs`, `eucm.rs` share parameter validation (`:96-135`
versus `:85-124`), `linear_estimation` scaffolding, and tail helpers.
`linear_estimation` is a calibration routine living inside a projection-model
crate: seven of the ten models implement it (double_sphere, eucm, fov, ftheta, kannala_brandt, rad_tan, ucm; `kannala_brandt.rs:65` is 117 lines).
**Proposal:** a `params::check_finite_positive(&[(&str, f64)])` helper and a
`calibration` module taking `&dyn LinearInit` (or a generic) so the model files
hold projection math only. This also halves the 106 `clippy::similar_names`
hits, which come from `fx/fy/cx/cy`-style locals repeated in every model: keep
the names (they are the literature's) and group them in a `Intrinsics4` struct
passed to the math helpers.

Further camera-model pairs measure 19-26 windows each (`eucm`~`ucm` 26,
`eucm`~`fov` 23, `fov`~`ucm` 21, `eucm`~`rad_tan` 19), i.e. the whole family
shares validation and calibration scaffolding, not one pair.

## S-R7 · Loggers · P2 · S
See S-A6.

## S-R8 · `sqlite.rs` and `mcap.rs` storage backends · P3 · M
`ros2/storage/sqlite.rs` repeats its own prepared-statement and row-decoding
blocks (27 windows); `mcap.rs` is 1,358 lines. Define the storage interface
once (`trait BagStorage { fn connections(); fn messages(filter) -> impl Iterator }`)
if not already in `ros2/storage/mod.rs`, and move shared message-range
filtering out of the two backends. Lower priority: the SQLite backend is
needed only for rosbag2 `.db3` files (see S-T2 for gating it behind its own
feature).

## S-R9 · Numeric tolerances · P2 · S
Not a code-block duplicate but a duplicated *constant*: production code
contains 104 bare tolerance literals (`1e-12` x37, `1e-6` x25, `1e-10` x17,
`1e-9` x10, `1e-8` x7). See S-C4 for the proposal (named tolerances with a
documented rationale).

## S-R10 · The typed marginal prior repeats the boxed prior's residual loop · P3 · S
Added with this audit's predecessor work: `MarginalPriorFactorT::linearize`
(`factors/marginal/marginal_prior.rs`) repeats the residual accumulation
`r = S (θ − b)` and the constant-Jacobian copy of `MarginalPriorFactor`
(20 shared windows inside the file). Both build `θ` differently (a boxed
closure versus `right_minus` against stored anchors) and then do the same
arithmetic. **Proposal:** one private `fn apply_sqrt_information(&self_parts,
delta, residual, jacobian)`; each factor keeps only its `θ` computation.
Bitwise neutral; the parity test between the two priors
(`the_typed_prior_matches_the_boxed_prior_and_the_full_solution`) is the gate.

## Tracking

Re-run the detector in CI as an advisory job and fail only if the duplicated
share rises more than one point above the committed baseline. The script is
about 60 lines (kept with the audit tooling; see `07`).
