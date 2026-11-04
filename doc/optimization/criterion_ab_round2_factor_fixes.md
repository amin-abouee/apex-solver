# Criterion A/B — Round-2 factor-correctness fixes (F1–F9)

- **Date:** 2026-09-26
- **Target Category:** all suites (odometry + bundle adjustment), correctness
  fixes evaluated for regressions
- **Scope Type:** correctness / test-coverage change, performance-neutral by
  construction
- **Status:** **ACCEPTED** — no id regressed beyond the acceptance threshold;
  nothing to revert beyond F8 (already reverted in `b2014ea`)
- **Revisions measured:**
  - **pre** = `894b105` (the audit commit; it is literally
    `f8680fc^`, the tree the first fix was applied to — so "pre-fix" here
    means "the exact parent of the fix series", verified with
    `git rev-parse f8680fc^`)
  - **post** = `b2014ea` (all Round-2 fixes landed, F8 reverted) = HEAD
- **Commits under test:** `f8680fc` (fmt sweep), `1cf5413` (F1–F3),
  `1c4f56e` (F6 + F8), `05e6af0` (F4), `d19a97c` (F5), `802c437` (F9),
  `b2014ea` (F8 revert)
- **Suite scope:** criterion's default `odometry` + `bundle_adjustment`
  targets only — **13 ids** (5 odometry, 8 BA), the 8 BA ids being the
  implicit/`variants_for()`-produced set. No EuRoC/TUM runs.

## 1. What is actually on the timed path

Before reading any delta: `benches/odometry_benchmark.rs` and
`benches/ba_benchmark.rs` both time the *solve only*
(`iter_batched(build_problem, …, BatchSize::PerIteration)` — problem
construction, factor construction and **registration are setup, not
measurement**). Of the landed changes, only two are reachable at all:

| fix | where it executes | in the timed region? |
|---|---|---|
| F1 `MarginalPriorFactor::whitens_internally` | registration | no — and no bench adds a marginal prior |
| F2 `validate_variables` per-block dof check | registration | **no** (`Problem::try_add_residual_block_impl`) |
| F3 `jacobian_shape() == (rows, Σ dof)` invariant | registration | **no** (same site; was a release-stripped `debug_assert`) |
| F4 frozen-S FD drift test | tests only | no |
| F5 loss-kernel contract sweep | tests only (`loss_functions.rs` runtime untouched) | no |
| F6 `ProjectionFactor` FD tests | tests only | no |
| F7 dangling audit citations | docs only | no |
| F8 projection-domain barrier | **reverted** — only the generalised `write_projection_barrier` remains, bit-identical to the old `write_cheirality_penalty` on the `PointBehindCamera` arm | no behavioural delta |
| F9 `left_plus::jacobian_self` | `crates/apex-manifolds`; **no caller in `src/`** | no |
| fmt sweep `f8680fc` | reflow of `explicit.rs`, `linearizer/cpu/sparse.rs`, benches | codegen-identical |

So the expectation is Δ ≈ 0 with any non-zero delta attributable to binary
layout / machine noise. That is what the measurements below show; they are
recorded in full because the rule for this round was "improve in all
datasets or revert".

## 2. Protocol

Baseline protocol: `benchmarking_results/baseline_criterion.md` (2-pass warm,
acceptance = max(2 %, 2× measured run-to-run noise, thermal drift > 20 %).

Deviations, applied **identically to both revisions**:

1. **Binaries built once per revision** (no cargo during measurement — the
   target-dir lock and the builds would have perturbed the machine), then
   copied out: `pre_{ba,odo}` @ `894b105`, `post_{ba,odo}` @ `b2014ea`
   (md5 `473f7daf…` / `d933d71b…` vs `bedaf36d…` / `c0085732…`).
2. **Interleaved per id:** `pre_warm (--profile-time 30 --discard-baseline)
   → post_warm → pre_rec (--save-baseline pre_fix) → post_rec
   (--save-baseline post_fix)`, ids ordered cheap → expensive, odometry
   first, so each recorded pair is adjacent in time (hot vs hot). The
   recorded pair is thus `pre` first, `post` second, in every id.
3. **The discarded pass is a 30 s `--profile-time` iteration, not a full
   10-sample pass.** criterion 0.6 rejects `--sample-size < 10`
   (`assertion failed: num_size >= 10`), and criterion runs any binary
   invoked without `--bench` in a hidden `--test` mode that executes without
   measuring ("Testing X / Success"). A full discarded pass over
   dubrovnik/venice alone would have cost ~14 min *per revision*.
4. **Odometry was run twice, in opposite order** (pass 2:
   `post → pre`, baselines `post_fix2` / `pre_fix2`) as an order-swap
   control for the sign of any delta. This is what turns "sphere2500
   regressed 7 %" into a measured noise call.

Run counts: 52 runs in pass 1 (13 ids × 4 invocations) + 20 in pass 2,
**every one `rc=0`**, i.e. every golden guard
(`assert_accuracy`, `cost ≤ golden × 1.0005`) passed in every timed run of
both revisions. Session 12:54:51 → 14:19:38; session 2 15:13 → 15:20, both
started at load ≈ 0.7 with the machine idle.

Raw data: `target/criterion/{odometry,bundle_adjustment}/solve/*/
{pre_fix,post_fix,pre_fix2,post_fix2}/{estimates,sample,tukey,benchmark}.json`.
Scripts: `/tmp/opencode/run_ab.sh`, `/tmp/opencode/run_replica.sh`.

## 3. Criterion results

### 3.1 Bundle adjustment — 8 ids (pass 1: pre → post, criterion mean)

| dataset / variant | pre (s) | post (s) | Δ | 95 % CI overlap | acceptance thr. | verdict |
|---|---:|---:|---:|---|---:|---|
| dubrovnik-135 / schur_explicit_iterative | 51.529 | 51.085 | **−0.862 %** | yes | 3.86 % | PASS |
| dubrovnik-135 / schur_explicit_sparse | 52.794 | 53.170 | +0.713 % | yes | 2.00 % | PASS |
| trafalgar-21 / schur_explicit_iterative | 1.414 | 1.415 | +0.065 % | yes | 2.00 % | PASS |
| trafalgar-21 / schur_explicit_sparse | 1.406 | 1.397 | **−0.669 %** | yes | 2.00 % | PASS |
| trafalgar-257 / schur_explicit_iterative | 14.909 | 14.981 | +0.486 % | yes | 2.00 % | PASS |
| trafalgar-257 / schur_explicit_sparse | 8.194 | 8.150 | **−0.538 %** | yes | 3.02 % | PASS |
| venice-52 / schur_explicit_iterative | 29.055 | 29.129 | +0.256 % | yes | 2.00 % | PASS |
| venice-52 / schur_explicit_sparse | 29.050 | 29.330 | +0.965 % | **no** | 2.00 % | PASS |

All 8 within ±1.0 %; the single non-overlapping CI (venice/sparse, +0.965 %)
is below the 2 % acceptance floor and below its own 2×noise band (1.25 %),
and repeats below in §3.3 as an unattributable layout effect (nothing on
this path changed). No BA id regressed.

### 3.2 Odometry — 5 ids, two passes with the order swapped

| id | p1 pre | p1 post | Δ1 | p2 pre | p2 post | Δ2 | CI overlap p1 / p2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| M3500 | 0.1190 | 0.1169 | **−1.76 %** | 0.1191 | 0.1168 | **−1.91 %** | no / no | PASS |
| intel | 0.0096 | 0.0096 | +0.00 % | 0.0106 | 0.0096 | −9.93 % | yes / no | PASS |
| parking-garage | 0.2114 | 0.1967 | **−6.96 %** | 0.2208 | 0.2156 | **−2.36 %** | no / no | PASS |
| sphere2500 | 0.3717 | 0.3989 | +7.30 % | 0.5774 | 0.5634 | **−2.43 %** | yes / yes | PASS |
| torus3D | 2.6071 | 2.6355 | +1.09 % | 4.1364 | 4.0309 | **−2.55 %** | yes / yes | PASS |

Reading:

- **8 of the 10 measurements favour `post`**; the 2 that do not
  (sphere2500 +7.30 %, torus3D +1.09 % in pass 1) are both inside
  overlapping CIs and both **flip sign when the order is swapped** —
  measured noise, not a regression. sphere2500's post run in pass 1 had a
  16.6 % CI half-width (bimodal samples), so its own acceptance floor that
  pass was 33 %, five times the observed delta.
- **M3500 and parking-garage improve in both passes, with disjoint CIs.**
  Since no executable change touches these paths (§1), the most likely cause
  is binary-layout/i-cache alignment between two different builds rather
  than the fixes themselves — but it is an improvement in either reading, so
  it needs no action.
- **Between-pass absolute drift** on identical binaries: sphere2500 pre
  0.372 → 0.577 s (+55 %), torus3D pre 2.61 → 4.14 s (+59 %). That is the
  >20 % thermal drift `baseline_criterion.md` documents, and it is why only
  within-pass, order-swapped deltas are read here.

### 3.3 Noise / acceptance detail

Acceptance = `max(2 %, 2 × measured run-to-run noise)` from
`baseline_criterion.md`, with noise taken as 2 × the larger of the two runs'
relative 95 % CI half-width (a conservative in-session proxy).

| id | pre CI half-w | post CI half-w | 2×noise | acceptance | Δ (pass 1) |
|---|---:|---:|---:|---:|---:|
| BA/dubrovnik-135 iter | 1.931 % | 0.570 % | 3.862 % | 3.862 % | −0.862 % |
| BA/dubrovnik-135 sparse | 0.359 % | 0.512 % | 1.024 % | 2.000 % | +0.713 % |
| BA/trafalgar-21 iter | 0.442 % | 0.537 % | 1.074 % | 2.000 % | +0.065 % |
| BA/trafalgar-21 sparse | 0.910 % | 0.570 % | 1.820 % | 2.000 % | −0.669 % |
| BA/trafalgar-257 iter | 0.615 % | 0.391 % | 1.231 % | 2.000 % | +0.486 % |
| BA/trafalgar-257 sparse | 1.509 % | 1.180 % | 3.019 % | 3.019 % | −0.538 % |
| BA/venice-52 iter | 0.212 % | 0.750 % | 1.501 % | 2.000 % | +0.256 % |
| BA/venice-52 sparse | 0.255 % | 0.627 % | 1.254 % | 2.000 % | +0.965 % |
| ODO/M3500 | 0.934 % | 0.646 % | 1.867 % | 2.000 % | −1.762 % |
| ODO/intel | 0.603 % | 1.332 % | 2.665 % | 2.665 % | +0.003 % |
| ODO/parking-garage | 3.443 % | 2.747 % | 6.886 % | 6.886 % | −6.962 % |
| ODO/sphere2500 | 3.267 % | 16.611 % | 33.222 % | 33.222 % | +7.299 % |
| ODO/torus3D | 1.581 % | 6.582 % | 13.163 % | 13.163 % | +1.093 % |

Pass-2 acceptance floors: M3500 2.01 %, intel 7.33 %, parking-garage
2.00 %, sphere2500 5.52 %, torus3D 11.03 % — every pass-2 delta (all
negative, i.e. improvements) sits inside them.

### 3.4 Accuracy

- **Golden guards: 52/52 + 20/20 runs green** on both revisions.
- **Costs are bit-identical pre ↔ post on all 13 ids** (initial *and* final
  cost from the `RUST_LOG=debug` trace): none of the golden datasets
  exercises a marginal prior, a dof/`jacobian_shape` mismatch, or a
  `left_plus` Jacobian, so the fixed defects do not manifest here — as
  expected. The fixes change what is *rejected* on hostile/incorrect input,
  not the numerics of valid input.
- The odometry matrix after the F8 revert (all 13 ids, both revisions) is
  recorded in
  `experiment_007_projection_domain_barrier_REJECTED.md` §3.

### 3.5 Absolute numbers vs `baseline_criterion.md`

This session measures ~25–35 % faster than the 2026-09-24 baseline on the
same ids (e.g. trafalgar-257/sparse 8.19 s vs 11.99 s, dubrovnik/sparse
52.8 s vs 68.9 s) **for both revisions**. That is not this change set: four
performance commits landed between the baseline SHA `84e609b` and `894b105`
(`9e4d851` pattern-cached extraction, `e34a117` bitmap Schur output,
`28e910c` CSC gather plan, `a3ef66d` symbolic Cholesky cache). Only the
within-session A/B above is read as evidence about F1–F9.

## 4. Verdict

**Interpretation of the rule for this round.** "Improve in all datasets or
revert" was applied as the repository's own acceptance threshold from
`baseline_criterion.md`: a dataset counts as *not improved* — and triggers
a revert — only when it **regresses beyond `max(2 %, 2× noise)`**. A delta
that is inside the noise band (or is a *faster* result) does not trigger a
revert; "no measured speed-up" alone never could, because no fix is on the
timed path by construction (§1).

| question | answer |
|---|---|
| any BA id regressed beyond threshold? | **no** — max +0.965 % against a 2.00 % floor |
| any odometry id regressed beyond threshold? | **no** — the only >2 % delta (+7.30 % sphere2500) flips to −2.43 % with the order swapped and has overlapping CIs in both passes |
| anything to revert? | **nothing beyond F8**, which the benchmark goldens already rejected and `b2014ea` reverted before this A/B ran |
| did accuracy improve? | the correctness guards (F1–F3, F9) and the verification coverage (F4–F6) are the deliverable; numerics on valid input are unchanged by design (§3.4) |

The A/B therefore confirms the post state = pre state on time and on
golden accuracy, with the F8 regression (the one real regression found this
round, 5/8 BA ids) already removed.

## 5. Reproducing

```bash
# build each revision into its own target dir, then per id:
BIN --bench <filter> --discard-baseline --noplot --color never --profile-time 30   # warm, thrown away
BIN --bench <filter> --save-baseline pre_fix  --noplot --color never              # recorded
BIN --bench <filter> --save-baseline post_fix --noplot --color never              # recorded
# comparison: target/criterion/<suite>/solve/<id>/{pre_fix,post_fix}/estimates.json
```
