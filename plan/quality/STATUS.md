# Status of the quality audit at 1.6.0

The audit in this folder was written on 2026-10-20 against `73c831d`
(`feature/factor-generic`). This page records what is still true at the 1.6.0
release (2026-11-04). It was produced by re-running the audit's own evidence
against the tree, not by trusting the findings: *Fixed* and *Open* below were
checked; *Not re-checked* findings are exactly that, and should be re-measured
with `tools/` before anyone relies on them.

The first table is what the release changed; the second is what remains.
When a finding ships, move it to *Fixed* with the commit, and when the last open
one is done, delete this folder (the sibling apex-cv audit was removed the same
way).

## Fixed or mostly fixed since the audit

| ID | Finding | Evidence at 1.6.0 |
|---|---|---|
| S-C3 | `Rn` panics inside trait methods that cannot fail | `rn.rs` has no `assert!`, `panic!` or `unreachable!` outside tests; `try_compose`, `try_right_plus`, `try_right_minus`, `try_interp`, `try_act` report a dimension mismatch (apex-manifolds 0.5.0) |
| S-T3 | Two tests red or flaky on the baseline | the allocation test now tolerates rayon's bookkeeping and the conformance tests no longer fail on rare random draws; the full suite passed on the release tree in release mode and on Rust 1.95 |
| S-E1 | Panic paths and lint suppressions | *partly*: there are **0** `#[allow(...)]` attributes in the sources and `scripts/check-conventions.sh` enforces it in CI; **not done**: `panic`, `unreachable`, `todo`, `unimplemented`, `allow_attributes` are still not in `[workspace.lints.clippy]` |
| S-E3 | Stringly-typed errors | *partly*: every module has hierarchical, `#[non_exhaustive]` errors with typed fields; name-keyed factories (`loss_from_name` and friends) remain |
| S-A2 | `Problem` is a god object | *partly*: `add_residual_block`, `fix_variable` and `set_variable_bounds` return `CoreResult` instead of panicking; `problem.rs` is still ~2,000 lines |
| S-A5 | Four overlapping "group acts on a point" helpers | *partly*: capability traits (`ActsOnPoints3`, `RigidPose`, `HasVelocity`) and `factors::common::group_point` are the one path; `skew` has one public home in apex-manifolds |
| S-C2 | Jacobian inverses fall back to identity | *partly*: the Sim(3) and SGal(3) fallbacks are now logged on the first hit and each power of ten; the fallback itself (`regularized_inverse`) remains |
| S-P9 | Allocation budgets for wrappers, not the solver loop | *partly*: `tests/allocation_free_wrappers.rs` and `tests/allocation_free_dense_solve.rs` exist; I found no budget test for the full solver loop |

## Still open (checked at 1.6.0)

| ID | Finding | Evidence |
|---|---|---|
| S-A1 | One optimizer loop, not three | `optimize_with_mode` still exists in `levenberg_marquardt.rs`, `dog_leg.rs` and `gauss_newton.rs` |
| S-A3 | Fixed-index and bounds state exist twice | `Problem.fixed_variable_indexes` and the variables' own `fixed_indices` are both still written |
| S-A4 | Bounds accepted, stored, then rejected at solve time | `optimizer/mod.rs` still rejects any non-trivial bound |
| S-A6 | Two logger modules | `src/logger.rs` and apex-io's `logger.rs` both exist |
| S-A7 | Visualization module is 1,950 lines | `observers/visualization.rs` is 1,953 lines |
| S-A8, S-T1 | CLI binaries in the library; `cli` and `download` default features | `default = ["cli", "download"]` |
| S-C1 | Marginalization documents "solved, not inverted" and inverts | `core/marginalization.rs` still calls `pseudo_inverse` and keeps the comment |
| S-C4 | Inline tolerance literals | 93 `1e-1x` literals in `src/optimizer` |
| S-E2 | Parsers lack fuzzing | no fuzz targets in the repository |
| S-E4 | The eight-arm `ManifoldType` match appears in several places | 4 `ManifoldType::SE3 =>` arms across `src` and `apex-io` |
| S-E6 | Dataset downloads: no integrity check, no atomic write | no hash, `.part` or `rename` in `apex-io`'s `utils.rs` |
| S-P3 | A rejected step is undone by applying the negative step | `apply_negative_parameter_step` is still used |
| S-P8 | Hash sets and maps on hot lookups | 40 `HashSet`/`HashMap` uses in `linalg` and `linearizer` |
| S-T4 | Duplicate dependency versions and no policy | `deny.toml` sets no `multiple-versions` policy |
| S-T6 | No debug-profile test run | `[profile.test]` still inherits `release` |
| S-T7 | rustdoc warnings, and the docs job does not gate on them | `cargo doc` emits ~50 warnings; `docs.yml` sets no `RUSTDOCFLAGS` |

## Not re-checked at 1.6.0

S-R1 to S-R10 (duplication and refactors), S-P1, S-P2, S-P4 to S-P7 (per-solve
setup, trial-step copies, preintegration appends, SGal3 central differences,
marginalization cost, column-scaling clones), S-C5 to S-C7 (IMU boundary
interpolation, `f64` timestamps, runtime finite differences), S-E5 (API surface
and semver), S-T2 (`apex-io` feature granularity and `image` codecs), S-T5
(tests inside the files they test) and S-T8 (repository hygiene). Several of
these are plausibly still true: the perf work in 1.6.0 touched the Schur and
normal-equation paths, so S-P1/S-P2/S-P7 in particular may have moved.

Cross-repository items (`X-n`) lived in apex-slam's `plan/quality/07-cross-repo.md`
and are not part of this folder.
