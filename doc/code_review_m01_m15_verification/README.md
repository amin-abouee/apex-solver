# Verification of the m01–m15 Code Review

Independent sanity check of every finding in [`../code_review_m01_m15/`](../code_review_m01_m15/),
re-verified against the code as it exists on **`main` @ `9e193a0`**.

**Headline: the audit was largely accurate, but it is out of date.** Three of its five HIGH
performance findings had already been fixed before this check began. Four claims are wrong
or mis-scoped, one proposed remedy was already measured as harmful and reverted, and every
line number in it is stale. Five real issues went unreported, one of them on the bundle-
adjustment hot path.

**Do not execute `../code_review_m01_m15/fix-roadmap.md` as written.** Use
[05-revised-roadmap.md](05-revised-roadmap.md) instead.

---

## Why the review went stale

The audit is dated **26 Aug 2026**. Three rounds of work landed after it:

| Date | Round | Record |
|---|---|---|
| 26 Aug | m01–m15 audit written | `../code_review_m01_m15/` |
| 28 Aug | Performance round 1 | `../perf-round1-results.md` |
| 29 Aug | Architecture round | `../arch-round-results.md` |
| 31 Aug | Noise-model round | `../noise-round-results.md` |
| 31 Aug | **This verification** | here |

`main` @ `9e193a0` is the merge of PR #52; its tree is byte-identical to `d041499`, the
`feature/noise_model` tip.

---

## Documents

| File | Read it for |
|---|---|
| [00-disposition-table.md](00-disposition-table.md) | **Start here.** All 86 finding IDs, verdict, current `file:line` |
| [01-confirmed.md](01-confirmed.md) | The 48 findings that still hold, re-rated, with fixes |
| [02-already-fixed.md](02-already-fixed.md) | The 13 closed by later rounds, mapped to commits |
| [03-corrections.md](03-corrections.md) | Six claims that are wrong, mis-scoped, or mis-severitied |
| [04-new-findings.md](04-new-findings.md) | Five issues the audit did not report |
| [05-revised-roadmap.md](05-revised-roadmap.md) | Replacement remediation plan |
| [06-fix-round-results.md](06-fix-round-results.md) | **What was actually fixed, and measured** — including four estimates that measurement overturned |

## Method

1. Every finding was re-read against current source; nothing was accepted on the review's
   word. Locations are quoted from the file as it stands, not carried over.
2. The two correctness claims were **executed**, not merely inspected — a throwaway
   integration test drove `BetweenFactor::<Rn>::linearize` and the residual-block
   registration guard. Verbatim output appears in [01-confirmed.md](01-confirmed.md#c-4-)
   and [04-new-findings.md](04-new-findings.md). The test file was deleted afterwards; no
   source file was modified by this review.
3. Global claims (zero `unsafe`, zero non-test `unwrap`/`expect`, zero TODO markers,
   dependency-version duplication) were re-swept independently.
4. Severities were re-assigned from what the code does today, not from the review's labels.

## Verdict legend

`CONFIRMED` · `CONFIRMED↑` (under-rated) · `CONFIRMED↓` (over-rated) · `FIXED` · `PARTIAL` ·
`CORRECTED` · `OBSOLETE` · `DUPLICATE`. Full definitions in
[00-disposition-table.md](00-disposition-table.md).

## Results by skill file

| Skill | Confirmed | Fixed | Partial | Corrected | Verdict on the original |
|---|---|---|---|---|---|
| m01 ownership | 12 | 1 | 1 | — | Accurate; H4 under-stated, H5 overtaken |
| m02 resource | 2 | — | — | — | Accurate |
| m03 mutability | 1 | — | — | — | Accurate |
| m04 zero-cost | 2 | 3 | — | 1 | Half of it shipped |
| m05 type-driven | 5 | — | 2 | 1 | H1's remedy is harmful |
| m06 errors | 4 | — | — | — | M06-1 over-rated; L4 obsolete |
| m07 concurrency | 1 | — | — | — | Accurate |
| m08 unsafe | — | — | — | — | Accurate — still zero `unsafe` |
| m09 domain | 5 | — | — | — | H1 real and **reproduced**; root cause narrower |
| m10 performance | 7 | 5 | 1 | 1 | Strongest section, most overtaken |
| m11 ecosystem | 5 | — | — | 1 | `serde` dev-dep claim wrong |
| m12 lifecycle | 5 | 3 | — | — | Accurate |
| m13 domain errors | 3 | 1 | — | — | Accurate |
| m14 mental models | — | — | — | — | Correctly filed as N/A |
| m15 anti-patterns | 6 | — | 1 | 1 | Loops have grown; damping now 2-way not 4-way |

## Status

A fix round has been applied — see [06-fix-round-results.md](06-fix-round-results.md).
All five HIGH items, every numbered MEDIUM, and the hygiene tail are closed; BA is ~4%
faster end to end and 45% faster on the explicit-Schur drain at Ladybug scale, with solver
status, iteration counts and final costs bit-identical throughout. One item is deliberately
left open for a decision: [NF-6](04-new-findings.md#nf6), the dead matrix-free Schur solver.

## What to do first

1. **Wave 1** of [05-revised-roadmap.md](05-revised-roadmap.md) — four deletions, no API
   change, immediately measurable.
2. **[NF-1](04-new-findings.md)** — ~14 million small heap allocations per Ladybug solve,
   from inside the parallel assembly loop. The largest single item left.
3. **[C-4](01-confirmed.md)** — `BetweenFactor<Rn>` panics for any dimension ≠ 3, behind a
   documented API claim, with no test covering it.
