# Revised Roadmap — replaces `../code_review_m01_m15/fix-roadmap.md`

Re-ordered by payoff ÷ risk against `main` @ `9e193a0`. Items already shipped are removed;
severities are the re-rated ones from [01-confirmed.md](01-confirmed.md); the two remedies
the original roadmap got wrong are rewritten.

**Baseline discipline.** Capture `benches/tools/run_repeated.sh` output before Wave 1 —
`doc/perf-round1-results.md` documents the methodology and the current numbers. Waves 1–2
must leave final χ² bit-identical on the eight odometry datasets, as the perf round's did.

---

## Wave 1 — One-line hot-path deletions (hours, near-zero risk)

Every item is a removal. No API changes, no new abstractions.

| # | Fix | Location | Ref |
|---|---|---|---|
| 1.1 | Delete `residuals.as_ref().to_owned()`; pass `residuals` | LM `:892`, GN `:526`, DL `:1071` | [C-1](01-confirmed.md) |
| 1.2 | Delete the dead `gradient.clone()`; sink `self.hessian = Some(hessian)` below the last use of the local | `explicit_schur.rs:1104,1107,1159,1162` | [C-2](01-confirmed.md) |
| 1.3 | Reorder `solve_with_system` before the field moves to drop one of two `JᵀJ` clones | `implicit_schur.rs:1019–1020` | [NF-3](04-new-findings.md) |
| 1.4 | Move `jacobian_values` into `AssemblyWorkspace` | `cpu/sparse.rs:119` | [NF-4](04-new-findings.md) |

**Acceptance:** bit-identical χ² on all eight odometry datasets and unchanged iteration
counts on the four BAL datasets; wall-clock improvement visible in the benchmark CSV;
`cargo clippy --workspace --all-targets` clean.

---

## Wave 2 — Bundle-adjustment hot path (the biggest remaining win)

| # | Fix | Location | Ref |
|---|---|---|---|
| 2.1 | Make `evaluate_internal` take `&[f64]` instead of building a `Matrix3xX` per call; keep the matrix path only for `fixed_landmarks` | `projection_factor.rs:431–435`, `:447` | [NF-1](04-new-findings.md) |
| 2.2 | Damp H_cc by editing cached diagonal positions instead of the triplet `find` loop — reuse the `diag_pos` approach from `normal_eq.rs`, precomputing H_cc's diagonal offsets in `initialize_structure` | `explicit_schur.rs:1188–1205` | [C-3](01-confirmed.md) |
| 2.3 | Precompute H_cc/H_cp extraction index maps once; `Vec::with_capacity` for the S drain; swap the drain loop nesting to match `s_dense`'s row-major layout | `explicit_schur.rs:476,509,778–932` | [C-8](01-confirmed.md) |
| 2.4 | Build the visibility index once in `initialize_structure`; store flat CSR arrays instead of `Vec<Vec<usize>>` | `implicit_schur.rs:783,860` | [C-10](01-confirmed.md) |
| 2.5 | Build `initialize_optimization_state` once on the LM Schur path and thread it through | `levenberg_marquardt.rs:1251`, `:1007` | [C-9](01-confirmed.md) |
| 2.6 | Own the CSC pattern in solver state to avoid the per-assemble clone | `cpu/sparse.rs:134` | [C-11](01-confirmed.md) |
| 2.7 | Construct the residual `Mat` from the workspace buffer instead of `Mat::from_fn` | `cpu/sparse.rs:131`, `cpu/dense.rs:71`, `problem.rs:335` | M10-L1 |

**Acceptance:** allocation count per iteration measurably down (dhat or Instruments) on
Ladybug; BAL iteration counts and convergence statuses unchanged; RMSE deltas no larger than
the ≤1e-3 floating-point summation-order effects already documented for the perf round.

---

## Wave 3 — Correctness (behaviour-changing; tests first)

| # | Fix | Location | Ref |
|---|---|---|---|
| 3.1 | **`BetweenFactor<Rn>`**: make the Jacobian dimension-aware. The fault is `jacobian_identity()` being an associated function with no `self` — fix at that signature (add a `&self` trait method, or restrict `BetweenFactor`'s bound to fixed-DOF groups). Patching `zero_jacobian` alone changes nothing. Fix `Rn::apply_to_vector` (`rn.rs:290`) the same way. Regression tests at `Rn(2)` and `Rn(4)` — **there is currently no `BetweenFactor<Rn>` test at all**. | `rn.rs:124,290,321`; `apex-manifolds/src/lib.rs:449` | [C-4](01-confirmed.md), [X-2](03-corrections.md#c2) |
| 3.2 | Make LM reject unsupported `JacobianMode × LinearSolverType` pairs, copying the GN/DogLeg arms verbatim | `levenberg_marquardt.rs:1240,1270` | [C-6](01-confirmed.md) |
| 3.3 | **Schur classification — rewritten remedy.** Do *not* auto-classify unmarked DOF-3 variables; the arch round measured that as corrupting self-calibration. Instead, when a Schur solver is selected and `schur_landmark_keys` is non-empty, `warn!` with the count of unmarked DOF-3 variables so a partial marking is visible without changing behaviour. | `problem.rs:62`, `implicit_schur.rs:959–1002` | [X-3](03-corrections.md#c3) |
| 3.4 | Guard cross-`Problem` key mixing: give `Problem` an id and check it in `try_add_residual_block_impl` | `problem.rs:141–172` | [NF-2](04-new-findings.md) |
| 3.5 | Surface intrinsics decode failure instead of `.ok()`-swallowing it (warn-once minimum) | `projection_factor.rs:443–445` | M06-2 |
| 3.6 | `set_variable_bounds` returns `CoreResult<()>` on inverted ranges | `problem.rs:197–212` | M05-M2 |
| 3.7 | Turn the gather-loop skip into an error — **schedule before the sliding-window work**, which makes it reachable | `linearizer/mod.rs:207–214` | [C-14](01-confirmed.md#m06-1) |

**Acceptance:** a test reproducing each trap fails before and passes after. 3.1 must fail
with `Gemv: dimensions mismatch` on today's `main`.

---

## Wave 4 — Hygiene (one small PR)

| # | Fix | Ref |
|---|---|---|
| 4.1 | Remove unused root deps: `colored`, `serde_json`, `ureq`, `bzip2`, `num_cpus`, `rand`, and the `[dependencies]` `serde`. **Keep the `[dev-dependencies]` `serde`** — the benches use it. Drop workspace `memmap2`. | [X-1](03-corrections.md#c1) |
| 4.2 | Delete dead `enable_visualization` field + setter from GN/DogLeg | M11-M1 |
| 4.3 | Delete the commented-out `IterativeState` | M15-M3 |
| 4.4 | Delete or deprecate `create_jacobi_scaling`, `sparse_to_dense`, `dense_to_sparse` — all test-only | [X-4](03-corrections.md), M01-L4 |
| 4.5 | `ManifoldError` → `#[derive(thiserror::Error)]` | M13-1 |
| 4.6 | `clap::ValueEnum` in `bin/pose_graph_g2o.rs` (removes `unreachable!()` at `:323`); guard the div-by-zero metrics in `bin/bundle_adjustment.rs:409–431` | M15-M2, M06-L2 |
| 4.7 | Store `Box<dyn Factor + Send + Sync>`, or drop `Sync` from the trait | M02-1 |
| 4.8 | Doc fixes: the `RefCell` multi-thread comment (`visualization.rs:1853`) and the ROS1 reader's memory-mapping claim (`ros1/reader.rs:3`) | M03-1, M12-M1 |

---

## Wave 5 — Lifecycle & I/O

| # | Fix | Ref |
|---|---|---|
| 5.1 | Bound the ROS1 chunk cache to an LRU of depth 1–2 | [C-5](01-confirmed.md) |
| 5.2 | Add `Drop for Ros1Reader` matching its three sibling bag types | M12-M1 |
| 5.3 | Carry DDS startup outcome on a side channel (oneshot with the init `Result`) | M07-1 |
| 5.4 | sqlite: true streaming iterator, or rename to a batch API | M02-2 / M12-M3 |
| 5.5 | `DdsError::{RuntimeCreation, ThreadSpawn}` instead of reusing `ThreadJoin` | M13-4 |
| 5.6 | Carry error sources structurally in `log_with_source` so `chain()` can recover them; de-duplicate the copy in `apex-io` | M13-2 |
| 5.7 | Smaller clones: DogLeg steps, `try_inverse_mut`, stats clones, `set_covariance` | M01-M2, M01-L1, M01-L2, M01-L3 |

---

## Wave 6 — Structural (last, after the perf waves are measured)

| # | Fix | Ref |
|---|---|---|
| 6.1 | Unify the three optimizer loops behind a `StepStrategy` driver. **They have grown since the audit** — 235 / 227 / 195 lines, up from 218 / 203 / 184 — so the cost of deferring this is rising. | M15-H1, M15-M1 |
| 6.2 | Fold explicit Schur's damping into the shared `NormalEquationsCache` path, closing the last of the four divergent implementations | [X-5](03-corrections.md), M15-H2 |
| 6.3 | Split `Problem`: move logging out, evaluation toward the linearizer, make `Variable` constraint fields private | M09-M1, M09-M2, M05-M3 |
| 6.4 | Typestate constructors for `ProjectionFactor<CAM, OnlyIntrinsics>` requiring fixed state | M05-M1 |
| 6.5 | Row-space compaction on `remove_residual_block` — sequence with the sliding-window work | M09-L1 |

---

## Suggested PR slicing

1. **PR A** = Wave 1. Pure deletions; safest and most visible.
2. **PR B** = Wave 2. Rebase after A.
3. **PR C** = Wave 3, one commit + test per item. 3.1 and 3.7 are the ones that matter.
4. **PR D** = Wave 4. Trivial.
5. **PR E** = Wave 5.
6. **PR F/G** = Wave 6, needs design sign-off on the `StepStrategy` shape.

## Definition of done

- [ ] Wave 1–2 benchmark CSV regenerated, before/after, with χ² parity demonstrated
- [ ] Every Wave 3 item has a test that fails on `9e193a0`
- [ ] `cargo clippy --workspace --all-targets` clean
- [ ] Finding IDs in [00-disposition-table.md](00-disposition-table.md) annotated ✅ as they close
