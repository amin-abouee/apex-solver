# Verification of the commits after `a3ef66d` (Round-2 audit series)

- **Date:** 2026-09-27
- **Host:** Apple Mac mini M4 (10 cores, 32 GB), macOS, rust 1.98.0
- **Question:** `a3ef66d` (Exp 006, symbolic-Cholesky cache) is the last
  commit the optimization program validated end to end. Are the 11 commits
  after it (`4f0bbbf` … `e37ee4c`) correct, and do they leave every dataset
  neutral or better? The earlier A/B
  (`criterion_ab_round2_factor_fixes.md`) covered only the 13 criterion ids
  and explicitly skipped EuRoC/TUM.
- **Verdict:** **keep all 11.** None changes solver numerics on any dataset
  (odometry bit-identical, BA identical). One of them (`802c437`) broke
  `clippy -D warnings`; fixed in `d3c9817`. Two pre-existing harness defects
  surfaced along the way and are fixed (`5a912b8`, `14d3c82`).

## 1. Code review of the runtime delta `a3ef66d..e37ee4c`

| commit | what reaches runtime | finding |
|---|---|---|
| `4f0bbbf`, `894b105`, `e37ee4c` | docs only | — |
| `6c85269` | new `benches/implicit_ba_benchmark.rs` | bench only |
| `f8680fc` | `cargo fmt` reflow of `explicit.rs`, `linearizer/cpu/sparse.rs` | codegen-neutral (confirmed by §2) |
| `1cf5413` (F1–F3) | registration-time checks in `Problem::try_add_residual_block_impl` | rejects bad input only; not on the solve path |
| `1c4f56e` + `b2014ea` (F6/F8, net) | `write_cheirality_penalty` → `write_projection_barrier` | new `deficit ≤ 0 ⇒ no Jacobian` early return is unreachable: every model that raises `PointBehindCamera` (pinhole, KB, RadTan, FTheta, UCM, EUCM) does so only for `z < min_z`, so the deficit is strictly positive. The 3-term gradient sum equals the old 1-term product bit-for-bit (the other two terms are exact zeros). `projection_deficit` is unused at runtime. |
| `05e6af0`, `d19a97c` | tests only | — |
| `802c437` (F9) | `LieGroup::left_plus` `jacobian_self`: `Ad(g)` → `I` | mathematically right: `exp(φ)·(g·exp(δ)) = (exp(φ)·g)·exp(δ)`, so the right-local Jacobian w.r.t. `g` is `I`. No runtime caller passes `jacobian_self`. **But** the test file failed `clippy --all-targets -D warnings` on rust 1.98 (`neg_cmp_op_on_partial_ord`, `!(err <= tol)`), contradicting the round's "clippy green" claim → fixed in `d3c9817` (`err.is_nan() \|\| err > tol`, same semantics). |

## 2. Accuracy parity, `a3ef66d` vs HEAD

Both revisions built from separate worktrees; each run on each revision,
sequentially.

**Odometry** — `pose_graph_g2o --save-output` on all 16 g2o graphs in
`data/odometry/{2d,3d}` × {LM, GN, DL} = 48 runs per revision. The written
graphs were compared numerically token by token: **all 48 pairs identical
(max |Δ| = 0)**. (A byte `cmp` differs only in the `# Timestamp:` header
line; the solver is deterministic run-to-run.)

**Bundle adjustment** — `bundle_adjustment` on trafalgar-21, trafalgar-257,
dubrovnik-356, ladybug-1723, venice-1778 × {implicit, explicit,
explicit-iterative, chunked} = 20 runs per revision: iterations, initial
cost, final cost and final RMSE **identical on all 20**.

**Tests** — `cargo test --release --workspace --all-features`: 2281 passed,
0 failed, 1 ignored. `cargo fmt --check` clean;
`cargo clippy --workspace --all-targets --all-features -D warnings` clean
after `d3c9817`.

## 3. EuRoC / TUM VI

apex-vio does not build against `feature/criterion`: it is written against
`feature/window_optimization` (e.g. `factors::imu::PreintegrationError`),
and the two branches diverged at `27ee933`. The criterion series was
therefore merged into a local integration branch
(`integ/window-opt+criterion`, `c955332`, not pushed) and A/B'd against
`feature/window_optimization` itself.

Merge notes: one textual conflict (`linearizer/cpu/sparse.rs` — keep the
CSC gather plan, call window_optimization's evaluation-aware
`compute_block_into_with`) and two semantic ones git did not flag (both
branches added `MarginalPriorFactor::whitens_internally`; criterion dropped
`assemble_sparse`'s use of `variable_index_map`). Integration branch:
2350 tests passed, 0 failed, clippy clean.

*Results: see §5 (filled when the sweep completes).*

## 4. Defects found in the benchmark harness (pre-existing, fixed)

1. **`APEX_BENCH_SCHUR` explicit variants ran the implicit solver**
   (`5a912b8`). Since `3609401`, `apex_runs()` derived every arm from
   `library_default()` (= `ImplicitSparseSchur`, which ignores
   `schur_variant`). `sparse`, `chunked` and `explicit-iterative` all ran
   implicit under explicit labels, as did the dense row's sparse reference.
   Verified fixed from the bench's own config line.
2. **GTSAM harness built against the wrong Eigen** (`14d3c82`). The CMake
   config pins eigen@3 (3.4) for GTSAM; Homebrew GTSAM 4.3.0 is built with
   `GTSAM_USE_SYSTEM_EIGEN` on eigen 5.0.1. The ABI mismatch builds cleanly
   and silently breaks linearization: LM rejected every step (0 iterations
   on all 8 pose graphs, cost *rising*, e.g. sphere2500 1.28e5 → 2.09e5; BA
   RMSE ~1500 px). With the fix GTSAM 4.3 converges normally (M3500
   1.510939 / 6 iterations, sphere2500 21.29, cubicle 5.38).

## 5. EuRoC / TUM results

`scripts/evaluate_all.sh` (apex-vio, frozen snapshot of its working tree),
`JOBS=1`, seeds 42 / 7 / 123, all 11 EuRoC sequences + TUM VI rooms 1–6:
`feature/window_optimization` (`9ca3feb`) vs the integration branch
(`c955332`). 102 runs, all `rc=0`.

| set | mean ATE (m) | median ATE (m) | n |
|---|---|---|---|
| EuRoC, window_optimization | 0.1537 | 0.1460 | 11 |
| EuRoC, + criterion series | 0.1537 | 0.1460 | 11 |
| TUM VI, window_optimization | 0.0670 | 0.0602 | 6 |
| TUM VI, + criterion series | 0.0670 | 0.0602 | 6 |

Per sequence: Δ = +0.0000 on all 17, scale identical. Across all 51 runs the
logged RMSE / scale values match exactly, and the saved TUM trajectory
files are **byte-identical** — the whole `a3ef66d`-and-earlier optimization
program plus the Round-2 audit fixes are VIO-neutral.

## 6. Also found: the doc's BA timings predate a convergence fix

The first full cross-solver run on `a3ef66d` measured apex implicit BA at
21 iterations (cap) on Trafalgar / Dubrovnik — `doc/performance.md` says 9 /
17 — and Dubrovnik at 122 s vs 31.5 s. `git bisect` over
`3609401..a3ef66d` (probe: default BA bench, Rust only) lands on
**`f078711` "fix: eliminate fixed tangent columns from the linear solve"**:

| rev | Ladybug RMSE / it | Trafalgar RMSE / it | Dubrovnik RMSE / it | Venice RMSE / it |
|---|---|---|---|---|
| `2e31695` (before) | 0.8765 / 21 | 0.7981 / 9 | 0.7686 / 17 | 0.7521 / 2 |
| `f078711` (after) | 0.8747 / 21 | 0.7728 / 21 | 0.7432 / 21 | 0.7451 / 2 |

This is a correctness fix, not a regression: before it, BA's gauge-fixed
camera 0 was solved as a free variable and its step zeroed afterwards
(ISSUE-0003), so LM's step-quality ratio compared a model reduction for one
step against the cost change of another and stopped early at a worse
optimum. Every dataset now reaches a **lower** RMSE; the cost is iterations
(to the cap) and, on Dubrovnik, time per iteration. The latter is the next
thing to profile (implicit-Schur PCG work per LM iteration).
