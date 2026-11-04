# apex-solver — quality audit: overview and roadmap

Scope: the `apex-solver` repository at branch `feature/factor-generic` (HEAD
`73c831d`), covering the four crates it holds:
`apex-solver` (the optimizer), `apex-manifolds`, `apex-camera-models`,
`apex-io`. This is a **plan only**. No code was changed to produce it.

How the audit was done, so the numbers can be reproduced:

| Method | What it measured |
|---|---|
| Own scanner (brace-matching, strings and comments stripped) | function length, branch points, parameter count, per file |
| `cargo clippy --lib --bins` with ~35 extra structural/perf lints | counts per lint, production code only |
| 8-line normalized-window duplicate detector (numbers and strings blanked) | share of production lines that repeat elsewhere |
| `cargo tree -d`, `cargo tree -e normal` | duplicate dependency versions, dependency weight |
| `cargo doc --workspace --no-deps` | rustdoc warnings (43 now, 44 at baseline `6e1e19d`) |
| Reading the code at every location cited | every finding below names a file and a function |

Items that span the three repositories are in
`apex-vio-slam/plan/quality/07-cross-repo.md` (IDs `X-n`); the measuring scripts
are in `plan/quality/tools/`.

Rules the repo already holds itself to, which every proposal here must keep:
no `unwrap`/`expect` (also in tests), no lint suppression, sparse comments,
tests everywhere, and **a change lands only if it keeps or improves the
baseline** (`cargo test`, solver clippy, and for numerics: golden costs on the
BAL/g2o benchmarks and byte-identical downstream VIO/LIO trajectories).

## Scorecard

| Measure | Value | Reading |
|---|---:|---|
| Rust lines (all `.rs`, incl. tests, benches, bins) | 108,870 | large for one repo; 10 of the 25 biggest files are in `apex-manifolds` and `apex-io` |
| Production functions | 2,518 | |
| ... longer than 100 lines | 44 | 8 above 200, 2 above 300 |
| Production lines inside a duplicated 8-line window | **16.6 %** | highest of the three repos (apex-cv 3.9 %, apex-vio 2.4 %) |
| Tests | 2,330 | strong; two known-bad (see 07) |
| `clippy::indexing_slicing` (production) | 1,281 | deliberate in numeric code; the risky subset is parsers (g2o 103) |
| Normal-dependency crates, default features | 296 | 228 with `--no-default-features` |
| Duplicate dependency families in the normal tree | 6 | `equator` x3, `getrandom`, `rand`/`rand_core`, `syn`, `thiserror`, `webpki-roots` |
| rustdoc warnings | 43 | all broken intra-doc links, none new |

## What is already good (so the plan does not re-fix it)

- Zero `unsafe`; `unwrap_used`/`expect_used` denied workspace-wide.
- The August 2026 review in the (git-ignored) `analysis/code-review/` folder
  has largely been executed: tangent-space prior factor, noise models,
  information matrices, MSRV job, `cargo-deny`, parallel cached normal
  equations (`linalg/sparse/normal_eq.rs`), workspace-reusing assembly. Those
  items are **not** repeated here. The ones that survive are listed in
  `07-build-deps-ci-tests-hygiene.md` under "Carried over".
- The factor-extensibility plan in the apex-vio main checkout
  (`plan/factor/00..06`) covers group-generic factors (P0-P3 done on this
  branch) and the cross-crate lifts U1-U9. This plan references it instead of
  duplicating it.
- Wrapper/aggregate allocation hygiene and the typed marginal prior landed in
  the last sessions (`tests/allocation_free_wrappers.rs`).

## Findings index

Severity: **P0** wrong results, data loss or a safety hole. **P1** high
leverage: removes a large class of problems or a measured cost. **P2** worth
doing in the normal course. **P3** polish. Effort: S under a day, M a few
days, L over a week.

| ID | Sev | Effort | Title | File |
|---|:--:|:--:|---|---|
| S-A1 | P1 | L | Three copies of the optimizer loop; 17-parameter helpers | 01 |
| S-A2 | P1 | L | `Problem` is a god object and carries panicking wrappers | 01 |
| S-A3 | P2 | M | Fixed-index/bounds state held twice, as hash containers in a public trait | 01 |
| S-A4 | P1 | M | Variable bounds are accepted, stored, then rejected at solve time | 01 |
| S-A5 | P2 | M | Four overlapping "group acts on a point" helpers and constants | 01 |
| S-A6 | P2 | S | Two logger modules; subscriber/chrono are mandatory library dependencies | 01 |
| S-A7 | P2 | M | 1,950-line visualization module with six 120+ line methods | 01 |
| S-A8 | P1 | M | CLI binaries live in the library repo and drag `cli`+`download` into defaults | 01 |
| S-L1..L9 | P1-P2 | M-L | Long functions (53 above 90 lines) with decomposition for each family | 02 |
| S-R1..R10 | P1-P3 | M | Redundancy: 16.6 % of production lines duplicated | 03 |
| S-P1 | P1 | L | Per-solve setup rebuilt and variables deep-cloned on every `optimize` | 04 |
| S-P2 | P1 | S | Trial-step cost copies the whole residual and discards it | 04 |
| S-P3 | P2 | S | Rejected steps are undone by a negative step | 04 |
| S-P4 | P3 | S | `ImuPreintegration::append` copies every sample on each append (not on a hot path today) | 04 |
| S-P5 | P2 | L | SGal3 Jacobians by central differences, with allocations | 04 |
| S-P6 | P2 | M | Marginalization forms a dense JᵀJ and an explicit pseudo-inverse | 04 |
| S-P7 | P2 | S | Column scaling clones the Jacobian every iteration | 04 |
| S-P8 | P2 | S | Hash sets and maps on hot lookups | 04 |
| S-P9 | P2 | M | No allocation budgets for the solver loop, only for wrappers | 04 |
| S-C1 | P1 | S | Marginalization says "solved, not inverted" and inverts | 05 |
| S-C2 | P1 | M | Jacobian-inverse failures return identity (Sim3, SGal3) | 05 |
| S-C3 | P1 | M | `Rn` dimension mismatch panics inside trait methods that cannot fail | 05 |
| S-C4 | P3 | S | 104 inline tolerance literals in production code, unnamed | 05 |
| S-C5 | P3 | S | IMU boundary interpolation possibly wrong when t0 and t1 share a sample pair (to verify) | 05 |
| S-C6 | P3 | M | Absolute `f64` epoch timestamps with absolute epsilons | 05 |
| S-C7 | P2 | M | Runtime finite differences inside the inverse-depth factor | 05 |
| S-E1 | P1 | S | Panic paths in the public API; lint set weaker than apex-cv/apex-vio | 06 |
| S-E2 | P1 | M | Untrusted-size allocations in bag/CDR/g2o readers | 06 |
| S-E3 | P2 | M | Stringly-typed errors and `loss_from_name(&str)` | 06 |
| S-E4 | P2 | M | `ManifoldType` match repeated in several places | 06 |
| S-E5 | P2 | M | API surface and semver discipline (this branch is already source-breaking) | 06 |
| S-E6 | P2 | S | Dataset downloads: no checksum, no atomic write, whole files in memory | 06 |
| S-T1..T8 | P1-P3 | S-M | Defaults, duplicate versions, flaky/failing tests, inline test bulk, hygiene | 07 |
| S-M*, S-K*, S-I* | P1-P3 | S-L | Manifolds, camera models, apex-io specifics (`apex-io` is the largest member, 19.8k lines) | 08 |

## Recommended order

Each phase ends with the repo's normal gate. Phases 0-1 are safe to interleave
with feature work.

**Phase 0, a week of small safe wins** (no numerics change):
S-E1 lints and panic removal, S-T3 red/flaky tests, S-P2 cost-only evaluation,
S-P7, S-A6, S-C1 comment fix, S-C5 test, S-T1/T2 default features and `image`
codecs, S-T6/T7 debug test run and rustdoc, S-E2 allocation caps, S-K1 move the
camera tests next to the crate.

**Phase 1, correctness and robustness**: S-C2, S-C3, S-E2 fuzz targets, S-A4
(decide implement vs remove bounds), S-C4 named tolerances.

**Phase 2, structure**: S-A1 (one optimizer loop), S-A2 (split `Problem`),
S-A5, S-A8, S-R1..R9, long-function families.

**Phase 3, performance program** (needs Phase 2's single loop): S-P1 warm-start
sessions, S-P4, S-P5, S-P6, S-P9. apex-vio's own profile says 93 % of a
keyframe solve is inside this crate's Levenberg-Marquardt, so this is the
largest runtime lever for the whole stack.

Every numerical change is gated by: all tests, clippy clean, golden final costs
on `ba_benchmark`, criterion within noise or better, and byte-identical
apex-vio EuRoC and Hilti trajectories.
