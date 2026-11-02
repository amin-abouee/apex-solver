# Contributing to Apex Solver

Thanks for helping with nonlinear least squares for SLAM and vision. This
repository is a Cargo workspace of four crates released together:
`apex-solver` (root), `apex-manifolds`, `apex-io` and `apex-camera-models`.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security problems privately, as described in [SECURITY.md](SECURITY.md), not in
a public issue.

## Quick start

```bash
cargo fmt --all -- --check
./scripts/check-conventions.sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features --release
```

Notes:

- Tests run under `[profile.test] inherits = "release"` (see root `Cargo.toml`);
  debug-only checks (`debug_assert!`, overflow) are **not** exercised by
  `cargo test` — use `cargo run`/`cargo check` for those paths.
- MSRV is 1.95 (enforced by the `msrv` CI job). Avoid newer std APIs.
- Integration tests download small public datasets on first run; later runs use
  the cache under `data/`.

## Code rules (CI enforces all of them)

- `unsafe_code = "forbid"` workspace-wide. No new `unsafe` without discussion.
- No `unwrap()`, `expect()`, `panic!` or lint suppression
  (`#[allow]`, `#[expect]`, `#![allow]`), in library code, tests, doc examples
  and CLI binaries alike. `scripts/check-conventions.sh` greps for them and
  clippy denies them. Fix the finding instead of silencing it.
- No `println!`/`eprintln!`. Libraries log through `tracing`; CLIs print with
  `writeln!(std::io::stdout().lock(), ...)?`.
- Errors are typed and layered (`src/error.rs` of each crate): a module returns
  its own error type, converts upward with `?`/`#[from]`, keeps causes as
  `source()`, and is logged once where it is built (`ErrorLogging`). Do not
  construct the top-level `ApexSolverError` in a deep module.
- Factors take their Jacobian blocks from `apex-manifolds`; do not re-derive a
  manifold's derivatives by hand inside a factor.
- Numerical changes need a regression test with golden values, not just
  thresholds. Never assert on wall-clock time. A flaky test is fixed, not retried.

## Feature flags

`apex-io` keeps heavy deps opt-in; keep it that way:

| Feature | Default | Pulls in |
|---|:---:|---|
| `rosbag` | off | `rusqlite` (bundled C), `mcap`, `zstd`, `lz4_flex`, … |
| `download` | **on** | `ureq`, `bzip2`, `flate2`, `tar` |
| `visualization` / `dds` / `asl-async` | off | `rerun` / `rustdds+tokio` / `tokio` |

Root `apex-solver`: `cli` (clap, on), `rosbag` pass-through (off),
`download` pass-through (on via `apex-io` default). New bins using `clap`
need `required-features = ["cli"]`; bins touching `apex_io::rosbag` need
`required-features = ["rosbag"]`; dataset-download bins need `"download"`.

Before opening a PR that adds a dependency, check `cargo tree -e no-dev`
impact and update the table in `crates/apex-io/doc/cookbook/src/appendix/features.md`.

## Collaboration workflow

1. **Discuss first** for anything larger than a bug fix: open an issue or a
   GitHub Discussion with the problem, the proposed approach and the expected
   effect on accuracy or speed.
2. **Branch** from `main`: `feature/<topic>`, `fix/<topic>`, `docs/<topic>` or
   `chore/<topic>`. Fork the repository if you do not have write access.
3. **One concern per pull request.** Keep diffs surgical: no unrelated
   refactors or formatting churn. Split large work into reviewable steps.
4. **Commit messages** are one line, `<type>: <description>`, with the type one
   of `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci` (the
   history also uses `add` and `update`). release-plz reads these to build
   changelogs and version bumps, so use `feat`/`fix` for user-visible changes.
5. **Open the PR against `main`** using the template. Link the issue, state
   whether the change is breaking, and tick the verification list only for
   checks you ran.
6. **Review.** The code owner reviews every PR; one approval and a green CI are
   required to merge. Address review comments with new commits, then rebase or
   merge `main` once the PR is approved. Maintainers squash only when the history
   is noise; focused commits are kept.
7. **Dependent crates.** A change to a public API of one workspace crate lists
   the other crates it affects in the PR description (and in each
   `CHANGELOG.md`).

## Pull requests

- Breaking changes (including feature-flag flips like `rosbag` gating) must
  be called out in `CHANGELOG.md` under `[Unreleased]` with a migration snippet.
- Update the docs you touch: rustdoc, the crate README and the cookbook.
- CI must pass on Linux, macOS and Windows: fmt, conventions, clippy
  (`-D warnings`), tests, the MSRV build, `cargo audit`, `cargo-deny` and
  `cargo-semver-checks`.
- Run every CI check locally before pushing, after your last edit or merge:

  ```bash
  cargo fmt --all -- --check
  ./scripts/check-conventions.sh
  cargo check --workspace --all-targets --all-features
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  cargo test --workspace --all-features --release
  cargo check --examples --features visualization
  cargo +1.95 check --workspace --all-targets --all-features
  cargo audit && cargo deny check      # when dependencies changed
  mdbook build doc/cookbook            # when the cookbook changed
  ```

## Reporting issues

Use the bug-report / feature-request templates. For wrong numerical results,
include: dataset (or minimal repro), solver config, initial vs final cost,
and the expected value with its source (paper, Ceres, GTSAM).

## License

Contributions are licensed under the Apache License, Version 2.0 (see
[LICENSE](LICENSE)), as stated in section 5 of that license.
