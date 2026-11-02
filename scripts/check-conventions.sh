#!/usr/bin/env bash
# Conventions gate: no unwrap/expect and no lint suppression in any tracked
# Rust file, tests and doc examples included.
#
# clippy's deny set covers unwrap_used, expect_used and allow_attributes, but
# `#[expect(...)]` silences a lint exactly as `#[allow(...)]` does and clippy
# has no lint for it, and doc comments are not linted at all. This grep closes
# both holes. It reads `git ls-files`, so build output and data are skipped.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

hits=$(git ls-files -z '*.rs' | xargs -0 grep -nH \
    -e '\.unwrap()' -e '\.expect(' -e '\.unwrap_err()' \
    -e '#!\?\[allow' -e '#!\?\[expect' \
    -e 'cfg_attr(.*allow' -e 'cfg_attr(.*expect' \
    | grep -v '\.unwrap_or' || true)

if [[ -n "$hits" ]]; then
    printf '%s\n' "$hits"
    printf '\nFAIL: unwrap/expect or a lint-suppression attribute (see above)\n' >&2
    exit 1
fi
printf 'ok: no unwrap/expect/allow/expect-attr\n'
