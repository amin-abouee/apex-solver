## What and why

<!-- Link the issue (Closes #123). One or two sentences on the user-visible change and the reason. -->

## Scope

- Crates touched: <!-- apex-solver / apex-manifolds / apex-io / apex-camera-models -->
- Other Apex crates affected (apex-cv, apex-vio, ...): <!-- none, or what must change there -->

## Breaking?

- [ ] No
- [ ] Yes — migration note added to `CHANGELOG.md` under `[Unreleased]` (and the crate's own changelog)

## Verification

Tick only what you ran on the final tree (after the last edit or merge).

- [ ] `cargo fmt --all -- --check`
- [ ] `./scripts/check-conventions.sh` (no `unwrap`/`expect`/lint suppression, bins included)
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features --release`
- [ ] `cargo +1.95 check --workspace --all-targets --all-features` (MSRV)
- [ ] `cargo audit` and `cargo deny check` (when dependencies changed)
- [ ] New/changed numerics covered by golden-value tests (no wall-clock asserts)
- [ ] Feature impact checked (`cargo tree -e no-dev`, bins carry `required-features`)
- [ ] Docs updated (rustdoc, README, cookbook) where behavior changed

## Notes for reviewers

<!-- Design choices, trade-offs, follow-ups, anything unusual. Benchmarks for performance-sensitive changes. -->
