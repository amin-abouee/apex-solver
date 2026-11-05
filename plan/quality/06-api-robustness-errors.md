# apex-solver — API surface, robustness and error handling

---

## S-E1 · Panic paths and 62 lint suppressions; the lint set is weaker than the sibling repos · P1 · S

**Evidence.**
- Root `Cargo.toml` denies `unwrap_used`, `expect_used`, `print_stdout`,
  `print_stderr` only. apex-cv and apex-vio also deny `panic`, `todo`,
  `unimplemented`, `unreachable` and `allow_attributes`.
- Production panic sites in this repo: three `unwrap_or_else(|e| panic!(...))`
  in `core/problem.rs` (`add_residual_block_with_noise`, `fix_variable`,
  `set_variable_bounds`), ten `unreachable!()` in the manifolds
  (`Tangent::generator` of SE₂(3), Sim3, SGal3 and others), 19 `assert!`
  on the library path (mostly `Rn` dimension checks, S-C3).
- Lint suppressions in Rust sources: **62** (`#[allow(deprecated)]` 15,
  `too_many_arguments` 13, `unwrap_used`/`expect_used` 16, `print_*` 12,
  `type_complexity` 2, `dead_code`/`unused` 4). apex-vio has none; apex-cv has
  one `#![expect(dead_code)]` in a bench helper. The ecosystem rule is "findings
  are fixed, never silenced".

**Proposal.** In order, each its own commit:
1. Add `panic`, `unreachable`, `todo`, `unimplemented`, `allow_attributes` to
   `[workspace.lints.clippy]`. The build then lists the real work.
2. Remove the panicking wrappers (S-A2) and make `generator`/`Rn` ops total (S-C3).
3. Retire suppressions by cause:
   - `deprecated` (15): finish the deprecation (S-A1).
   - `too_many_arguments` (13): parameter structs (S-A1, S-L2, S-R3).
   - `unwrap_used`/`expect_used` (16, in `apex-io` sources and tests): tests
     return `Result<(), Box<dyn Error>>` and use `?`, as the rest of the repo
     does.
   - `print_*` (12): the CLI crate (S-A8) enables printing in its own
     `[lints.clippy]` table; configuring a lint is not suppressing one.
   - `type_complexity` (2): type aliases. `dead_code`/`unused` (4): delete.
4. Add `scripts/check.sh`-style grep for `#[expect(` (apex-vio's script closes
   that hole, because clippy cannot deny `expect` attributes).

**Gate.** `cargo clippy --workspace --all-targets` with the stricter set and no
`allow` attribute anywhere.

---

## S-E2 · Allocation sizes taken from untrusted input; parsers lack fuzzing · P1 · M

**Evidence.** The bag and graph readers parse files that may come from
downloads (`download` feature fetches datasets over HTTP). Sizes read from the
file drive allocations:
- `rosbag/ros1/record.rs:225,230`: `vec![0u8; header_len]` and
  `vec![0u8; data_len]` straight from record headers.
- `rosbag/ros1/reader.rs:428,436`: `Vec::with_capacity(uncompressed_size)`
  from a chunk header.
- `rosbag/ros2/messages.rs:667`: `vec![0u8; len as usize]` from a CDR length.
- `rosbag/ros1/deserialize.rs:141`: `Vec::with_capacity(len)` (the sibling at
  `:129` already caps with `len.min(1 << 16)`; this one does not).
A corrupt or hostile 4-byte length field requests up to 4 GiB and, when the
allocation fails, aborts the process instead of returning a `ReaderError`.
`indexing_slicing` flags 103 sites in `graph/g2o.rs` alone; most are
`parts[i]` after one `parts.len() < N` check, but the check and the index
are far apart.

**Proposal.**
1. A `fn bounded_len(len: u64, remaining: usize, what: &str) -> Result<usize>`
   used at every site above: reject a length larger than the bytes still in
   the buffer (or a configured ceiling), and read with `Read::take(len)` +
   `read_to_end` so memory grows with data actually present.
2. `cargo-fuzz` targets (`fuzz/` directory, nightly only, not in the normal
   build): `g2o::read`, `toro::read`, `ros1 record parser`, `cdr deserializer`
   for each message type, `mcap` message iterator, `bal::read`, `tum::read`.
   Seed the corpus from `tests/test_bags`.
3. The `field::<T>()` helper of S-R5 removes the g2o indexing class.

**Gate.** Each fuzz target runs 5 minutes in a nightly CI job without a crash,
and the new unit tests feed truncated and oversized inputs to every site.

---

## S-E3 · Stringly-typed errors and name-keyed factories · P2 · M

**Evidence.**
- `CoreError` variants carry `String` (`Variable(String)`, `DimensionMismatch(String)`,
  `InvalidConstraint(String)`, `ResidualBlock(String)` ... `core/mod.rs:30-60`).
- `Factor::validate_variables` returns `Result<(), String>`, so every factor
  author writes ad-hoc messages and callers cannot match on the cause.
- `loss_from_name(&str, Option<f64>)` (`core/loss_functions.rs:1703`) parses
  loss names at run time with a 36-line `match` on lower-cased strings; used by
  the CLI and examples.

**Proposal.**
- `FactorError` gains structured variants (`BlockCount { expected, got }`,
  `WrongManifold { block, expected, got }`, `Dimension { what, expected, got }`);
  `validate_variables` returns `Result<(), FactorError>`. The helpers in
  `factors/common/validate.rs` (`expect_block_sizes`, `expect_manifolds`) are
  the one place to change; call sites need no edit.
- `enum LossKind` with `FromStr` and `Display`, constructed once; keep
  `loss_from_name` as `LossKind::from_str(name)?.build(scale)`.
- Keep a `Display`-only `String` payload for genuinely free-form text.

**Gate.** Semver: `validate_variables`'s signature changes (a trait method
implemented by downstream users), so this is a major-version item (S-E5).

---

## S-E4 · The eight-arm `ManifoldType` match appears in several places · P2 · M

**Evidence.** `core/problem.rs:~700` (build a `Box<dyn ManifoldVariable>`),
`core/marginalization.rs:644` (the `log_for!` macro, one arm per group),
`apex-manifolds/src/lib.rs` (five matches). Adding or removing a group (the
`SGal3` and `Sim3` additions touched all of them) means finding each by grep.

**Proposal.** One macro in `apex-manifolds`:
`with_group!(manifold_type, G => expr)` expanding to the match once and calling
`expr` with `G` bound to the concrete type. Call sites become one line each, and
a missing group becomes a compile error at one location. This also gives
S-A2's `VariableStore` its constructor without a second match.

---

## S-E5 · API surface and semver discipline · P2 · M

**Evidence.**
- Internal helpers are `pub`: `pub use ...SymbolicStructure` from `problem.rs`,
  `Problem::apply_constraints_to_variables`, and in `optimizer/mod.rs`
  `pub fn` `initialize_optimization_state`, `check_convergence`,
  `iteration_preamble`, `notify_observers*`, `build_solver_result`,
  `create_optimizer_summary`, `pub struct ConvergenceParams`. They are
  implementation details of the three optimizers (S-A1), and each is now a
  commitment to downstream users.
- `release-plz.toml` has `semver_check = true`, which is right, but this
  branch (`feature/factor-generic`) already contains breaking changes:
  `BearingFactorSe23` removed, `InverseDepthSe23Factor`,
  `InverseDepthBearingSe23Factor`, `InverseRangeProjectionSe23Factor`,
  `PhotometricInverseRangeSe23Factor` renamed and made generic, new generic
  parameters on the LiDAR, range, navigation and motion factors, and
  `ProjectionFactor` gained a defaulted third parameter. Commit prefixes are
  `refactor:`, so `protect_breaking_commits` will not recognise them.

**Proposal.**
1. Decide the release shape for the branch now: a 2.0.0, or keep the old names
   as `#[deprecated]` type aliases for one minor release (aliases are not
   lint suppressions; they trigger `deprecated` warnings only for users).
2. `pub(crate)` for the optimizer plumbing after S-A1; `#[doc(hidden)]` where a
   public item is needed by a sibling crate.
3. Check in a `cargo public-api` snapshot and diff it in CI; keep the
   semver-checks run in release-plz.
4. A `CHANGELOG.md` entry per breaking rename with the migration line.

**Gate.** `cargo semver-checks` clean against the last release, or an explicit
major bump.

---

## S-E6 · Dataset downloads: no integrity check, no atomic write, whole files in memory · P2 · S

**Evidence.** `crates/apex-io/src/utils.rs`:
- `download_file` (`:373`) reads the entire HTTP body into a `Vec` and then
  writes it to the destination path; there is no size cap, no checksum, no
  timeout configuration beyond the HTTP client's defaults, and it writes
  straight to the final path.
- `ensure_odometry_dataset` (`:196`) and its siblings return the path as soon
  as `path.exists()` is true. A download interrupted after the file was
  created (or a full disk) therefore leaves a truncated file that is accepted
  as the dataset forever; the failure then shows up later as a parse error
  inside the optimizer benchmark.
- `decompress_bzip2` (`:414`) decompresses the whole archive into a `Vec<u8>`
  before writing: memory proportional to the decompressed size, unbounded for
  a hostile archive.
- The data directories are relative constants (`ODOMETRY_DATA_DIR`,
  `BUNDLE_ADJUSTMENT_DATA_DIR`), so running a binary from another working
  directory silently re-downloads into a new `data/` tree.

**Proposal.**
1. Stream to `<name>.part` with `std::io::copy` (bounded by a `Read::take`
   ceiling taken from the registry entry), `fsync`, then `rename` to the final
   name: the destination exists only when complete.
2. Registry entries carry `sha256` and `bytes`; verify after download and
   re-verify cheaply (size) on the `exists()` path. The registry is
   `DatasetRegistry` data, so this is a data change plus ten lines.
3. Stream the bzip2 decode through `io::copy` into the output file.
4. Resolve the data root from `APEX_DATA_DIR`, then
   `CARGO_MANIFEST_DIR`-relative for benches, then the current directory,
   documented in one place (this is configuration, so it belongs in the CLI
   crate of S-A8, not the library).
5. The `extract_tar_gz` call relies on `tar::Archive::unpack`, which refuses
   `..` and absolute paths in current versions; add a test with a crafted
   archive so an upgrade cannot silently change that.

**Gate.** Tests using a local HTTP server (`tiny_http` or a `file://`-style
test double behind a trait) for: truncated body, wrong hash, interrupted
write, oversized body.
