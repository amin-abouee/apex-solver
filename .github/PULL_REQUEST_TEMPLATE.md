## What and why

<!-- Link the issue. One or two sentences on the user-visible change. -->

## Breaking?

- [ ] No
- [ ] Yes — migration note added to `CHANGELOG.md` under `[Unreleased]`

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features --release`
- [ ] New/changed numerics covered by golden-value tests (no wall-clock asserts)
- [ ] Feature impact checked (`cargo tree -e no-dev`, bins carry `required-features`)
