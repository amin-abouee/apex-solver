# Contributing to Apex Solver

Thanks for helping with nonlinear least squares for SLAM and vision.

## Quick start

```bash
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features --release
cargo fmt --all -- --check
```

Notes:

- Tests run under `[profile.test] inherits = "release"` (see root `Cargo.toml`);
  debug-only checks (`debug_assert!`, overflow) are **not** exercised by
  `cargo test` — use `cargo run`/`cargo check` for those paths.
- MSRV is 1.93 (enforced by the `msrv` CI job). Avoid newer std APIs.
- `unsafe_code = "forbid"` workspace-wide. No new `unsafe` without discussion.

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

## Pull requests

- Keep diffs surgical: no unrelated refactors or formatting churn.
- Conventional commits (`feat:`, `fix:`, `docs:`, …) — release-plz builds
  version bumps and changelog entries from them.
- Breaking changes (including feature-flag flips like `rosbag` gating) must
  be called out in `CHANGELOG.md` under `[Unreleased]` with a migration snippet.
- Numerical changes need a regression test with golden values, not just
  thresholds; never assert on wall-clock time.
- CI must pass: fmt, clippy (`-D warnings`), tests, `cargo-deny`,
  `cargo-semver-checks`.

## Reporting issues

Use the bug-report / feature-request templates. For wrong numerical results,
include: dataset (or minimal repro), solver config, initial vs final cost,
and the expected value with its source (paper, Ceres, GTSAM).
