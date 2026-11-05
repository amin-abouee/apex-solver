# apex-solver — build, dependencies, CI, tests, repository hygiene

---

## S-T1 · Default features pull a CLI and an HTTP client into every consumer · P1 · S

See S-A8 for the change. Numbers: `cargo tree -p apex-solver -e normal` lists
**296** crates with default features and **228** with
`--no-default-features` (68 fewer: `clap`, `ureq`, `rustls`, `ring`, `bzip2`,
`flate2`, `tar`, `webpki-roots`, ...). In the apex-vio tree those come in
through `apex-io` -> `apex-solver`.

---

## S-T2 · `apex-io` features are too coarse, and `image` is built with every codec · P1 · M

**Evidence.**
- `rosbag` bundles `rusqlite` (compiles SQLite from C source), `mcap`, `zstd`,
  `lz4_flex`, `bzip2`, `byteorder`, `hex`, `serde_norway` together
  (`crates/apex-io/Cargo.toml` features). apex-vio enables `rosbag` to read MCAP
  and ROS1 bags and never opens a rosbag2 `.db3`, but pays for the C
  dependency and its `hashbrown` 0.14 duplicate.
- `image = "0.25"` in the workspace dependency table has default features, so
  `avif`, `exr`, `gif`, `tiff`, `webp` ... all build. The code uses
  `image::open`/`DynamicImage` for ASL (PNG) streaming. `exr` brings
  `rand 0.8` into the normal tree (`cargo tree -i rand@0.8.7`), one of the six
  duplicate families.

**Proposal.**
- Split `rosbag` into `rosbag-ros1`, `rosbag-mcap`, `rosbag-sqlite`
  (`rosbag` = the union, for compatibility). apex-vio depends on the first two.
- `image = { version = "0.25", default-features = false, features = ["png"] }`
  at the workspace level (apex-cv already does exactly this).
- Check that `dds`/`rustdds`/`tokio` stay behind `dds` and out of the default build.

**Gate.** `cargo tree` in apex-vio shows no `rusqlite`, `exr`, `avif`; build time
drops (record before/after with `cargo build --timings`).

---

## S-T3 · Two tests are red or flaky on the untouched baseline · P1 · S

**Evidence.** On the unmodified baseline commit `6e1e19d`:
- `tests/factor_coverage.rs:2560 every_exported_factor_is_exercised_by_an_integration_test`
  fails: "these exported factors have no integration coverage:
  `Relative4DofFactor`".
- `tests/allocation_free_dense_solve.rs a_warm_solve_of_the_same_shape_allocates_nothing`
  failed three runs out of three at the baseline and has passed on some later
  runs (flaky), consistent with dhat counting blocks allocated by other
  threads. See S-P9.

**Why it matters.** `cargo test --workspace` stops at the first failing binary
(use `--no-fail-fast`), a red suite trains people to ignore red, and the CI job
`cargo test --workspace --all-features --release` runs that first test, so
it should fail on it unless CI differs from the local run.

**Proposal.** Add the missing integration test for `Relative4DofFactor` (the
guard is doing its job); make the allocation test hermetic (single-threaded
rayon pool, `--test-threads=1` via a `[[test]]` setting or an in-test guard).
If neither can land this week, mark with `#[ignore = "<reason and issue>"]`
rather than leave the suite red.

---

## S-T4 · Duplicate dependency versions and no policy · P2 · S

**Evidence.** Normal tree, six families: `equator` 0.2/0.4/0.6 (via `faer`),
`getrandom` 0.2/0.3 (via `ring`/`rand`), `rand` 0.8/0.9 (via `exr`),
`thiserror` 1/2 (`sysctl` via `gemm-common`), `syn` 2/3 (`clap_derive`),
`webpki-roots` 0.26/1 (`ureq`). Dev tree: `faer` 0.20/0.22/0.24
(`factrs`, `tiny-solver` comparison benches). `deny.toml` uses the default
`multiple-versions = "warn"`.

**Proposal.** Fix what we control (S-T2 removes `rand 0.8`; S-T1 removes
`syn 3`, `webpki-roots` and `getrandom 0.2` from default consumers), and set
`multiple-versions = "deny"` with an explicit `skip` list and a reason for each
remaining pair (`equator`/`thiserror` come from `faer` and cannot be fixed
here). The sibling repos need the same line, and an `apex-*` duplicate check
(plan/factor U8): two copies of `apex-manifolds` in one graph is a silent
type-mismatch bug.

---

## S-T5 · Tests live inside the files they test; some files are mostly tests · P2 · M

**Evidence.** Share of each file that is its own `#[cfg(test)]` module
(measured from the first `#[cfg(test)]` line):

| File | Lines | Test lines | Share |
|---|---:|---:|---:|
| `apex-manifolds/src/so3.rs` | 1,935 | 1,129 | 58 % |
| `core/problem.rs` | 1,997 | 1,030 | 52 % |
| `linalg/sparse/schur/explicit.rs` | 2,991 | 1,321 | 44 % |
| `optimizer/levenberg_marquardt.rs` | 2,578 | 1,068 | 41 % |
| `core/loss_functions.rs` | 2,884 | 1,120 | 39 % |
| `apex-manifolds/src/sgal3.rs` | 1,864 | 965 | 52 % |
| `optimizer/mod.rs` | 2,007 | 763 | 38 % |

The biggest single test file is `tests/factor_coverage.rs` (2,565 lines).
Part of the "this file is huge" impression in the size ranking (see
`00-overview.md`) is therefore test code, which is the right place to look
first when sizing a refactor: `Problem`'s production half is under 1,000 lines.

**Why it matters.** Every edit to production code recompiles and re-reads a
file twice its real size, and review diffs mix the two. Inline `#[cfg(test)]`
is idiomatic for small unit tests; at 1,000+ lines it stops being useful.

**Proposal.** Move the test module of any file whose tests exceed 400 lines to
`src/<module>/tests.rs` (`#[cfg(test)] mod tests;`), which is how
`factors/imu/se23/tests.rs` is already organized. Pure move, no content
change; one commit per file; the diff is mechanical so review is quick.

---

## S-T6 · No debug-profile test run; `profile.test` is release · P2 · S

**Evidence.** `Cargo.toml [profile.test] inherits = "release"` (the comment
explains: debug builds are 50x slower on the large-graph tests). CI runs only
`--release` tests. Consequences, stated in the manifest itself: `debug_assert!`
and integer overflow checks are not exercised. The repo contains `debug_assert!`
in numerical code (`rn.rs:300`, factor `debug_assert_eq!` on block sizes).

**Proposal.** An additional CI job: `cargo test --workspace --lib` in the
default debug profile (lib unit tests only, which are fast) on one OS. Long
integration tests stay in release.

---

## S-T7 · rustdoc has 43 warnings and the docs job does not gate on them · P3 · S

**Evidence.** `cargo doc --workspace --no-deps`: 43 warnings (44 at baseline),
all unresolved or private intra-doc links (unit text such as `[m]` and `[px]`
read as links; moved items such as `ProjectionFactor` at `factors/mod.rs:28`,
`super::se23::Se23ImuFactor` at `imu/bias.rs:4`; `super::gpu` at
`linearizer/cpu/mod.rs:4`; a link to a private function in
`sim3.rs:654`).

**Proposal.** Escape the unit brackets (`\[m\]`), fix or remove the stale
links, and run `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
in `docs.yml`.

---

## S-T8 · Repository hygiene · P3 · S

- Ignored but valuable: `analysis/` (the August review, 181 KB), `codex/`
  (`FIX_REPORT.md`, issue numbers cited in code comments: ISSUE-0003,
  0004, 0005, 0007, 0010, 0011), `scripts/` (including `factor_census.sh`)
  are in `.gitignore`. Source comments point to documents a fresh clone does
  not have. **Decide:** track them under `doc/engineering/` (preferred for the
  issue documents the code cites), or stop citing them from code.
- `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` at the root are untracked copies of
  the same instructions. One canonical file plus symlinks.
- PDFs in `doc/references/` (about 2 MB) and PNG plots in `doc/plots/`: fine, but
  check the `exclude` list in `Cargo.toml` keeps them out of the published
  crate (it excludes `doc/**/*.pdf`, not the PNGs).
- `Cargo.lock` is tracked on purpose (documented); keep.

---

## Carried over from the August review (still open)

| Item | Status |
|---|---|
| Triple-duplicated optimizer plumbing | open -> S-A1 |
| SoA parameter storage replacing the deep clone | open -> S-P1 |
| Feature-gate `apex-io` heavy modules; bins to a CLI crate | open -> S-T1/T2, S-A8 |
| Property tests for manifolds | partly: `tests/subgroup_law.rs` exists; extend to all groups with `proptest` |
| Sparse selected-inversion covariance | out of scope here |
| CI cache key, MSRV job, `cargo-deny` | done |
| Tangent-space prior, noise models, information matrices | done |
