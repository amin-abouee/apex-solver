# Disposition Table — every m01–m15 finding, verified against `main` @ `9e193a0`

All 86 finding IDs appearing in `../code_review_m01_m15/`, each with a verdict and its
**current** location. Line numbers in the original review are from 26 Aug and no longer
resolve; the "Location on `main`" column is authoritative.

## Verdict legend

| Verdict | Meaning |
|---|---|
| **CONFIRMED** | Still present, described accurately, severity stands |
| **CONFIRMED↓** | Still present, but severity over-rated — downgraded here |
| **CONFIRMED↑** | Still present and **under**-rated, or narrower/broader than described |
| **FIXED** | Closed by the perf/arch/noise rounds; no action needed |
| **PARTIAL** | Half closed; the named remainder is still real |
| **CORRECTED** | Real issue, but the review's description or proposed fix is wrong |
| **OBSOLETE** | The concern no longer applies because the surrounding design changed |
| **DUPLICATE** | Cross-reference to a primary entry; no independent action |

---

## m01 — Ownership & Borrowing

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M01-H1 | **CONFIRMED** | `optimizer/levenberg_marquardt.rs:892` | `residuals.as_ref().to_owned()`; callee takes `&Mat<f64>` (`linalg/mod.rs:259`) |
| M01-H2 | **CONFIRMED** | `optimizer/gauss_newton.rs:526` | same pattern |
| M01-H3 | **CONFIRMED** | `optimizer/dog_leg.rs:1071` | same pattern |
| M01-H4 | **CONFIRMED↑** | `linalg/sparse/explicit_schur.rs:1104,1107,1159,1162` | quoted `jt.mul()` code is gone, but the clones remain; the **`gradient` clone is dead outright** — review only proposed reordering |
| M01-H5 | **PARTIAL** | `linalg/sparse/implicit_schur.rs:1019–1020` | E0502 workaround gone (`ne_cache`); two full-CSC clones remain → MED |
| M01-H6 | **CONFIRMED** | `linearizer/cpu/sparse.rs:134` | `symbolic_structure.pattern.clone()` per assemble |
| M01-M1 | **FIXED** | `linearizer/mod.rs:339–370` | in-place parallel column scaling |
| M01-M2 | **CONFIRMED** | `optimizer/dog_leg.rs:886,1052,1126` | step clones |
| M01-M3 | **CONFIRMED** | `core/residual_block.rs:282–299` | legacy owned-value API double-buffers |
| M01-M4 | **CONFIRMED↓** | `optimizer/mod.rs:357–367` | `Mat::zeros` per rejected step; total_dof-sized, once per rejection — not per-inner-loop |
| M01-M5 | **CONFIRMED** | `factors/between_factor.rs:183–196` | four `Some(&mut …)` passed unconditionally |
| M01-M6 | **CONFIRMED↑** | `factors/projection_factor.rs:428,437,447` | review missed the **hot** branch — see [04-new-findings](04-new-findings.md#nf1) |
| M01-L1 | **CONFIRMED** | `implicit_schur.rs:408,579` | `block.clone().try_inverse()` |
| M01-L2 | **CONFIRMED** | `levenberg_marquardt.rs:1112,1189` | debug-gated stats clones |
| M01-L3 | **CONFIRMED** | `core/problem.rs:476` | `set_covariance(cov.clone())` |
| M01-L4 | **CONFIRMED** | `linalg/utils.rs:14,41` | `sparse_to_dense`/`dense_to_sparse` — no non-test callers |

## m02 — Smart Pointers

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M02-1 | **CONFIRMED** | `core/residual_block.rs:125`; trait `factors/mod.rs:178` | trait is `Send + Sync`, storage erases `Sync` |
| M02-2 | **CONFIRMED** | `apex-io/…/storage/sqlite.rs:225–235` | iterator facade over `let mut all_messages = Vec::new()` |

## m03 — Mutability

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M03-1 | **CONFIRMED** | `observers/visualization.rs:1853` | comment claims `RefCell` allows "access from multiple threads" |

## m04 — Zero-Cost Abstractions

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M04-1 | **FIXED** | `linearizer/mod.rs:135–175` | `AssemblyWorkspace` caches order/offsets/arena/buffer |
| M04-2 | **DUPLICATE** | → M01-H1..H3 | |
| M04-3 | **FIXED** | `linearizer/mod.rs:339–370` | = M01-M1 |
| M04-4 | **CORRECTED** | `optimizer/mod.rs:484–503` | real, but **dead public API** (test-only callers), not a perf finding |
| M04-L1 | **FIXED** | `linalg/sparse/normal_eq.rs` | `Jᵀ` built once and cached |
| M04-L2 | **CONFIRMED** | `optimizer/mod.rs:329–339`, `core/variable.rs:359–371` | step copied twice per variable |
| M04-L3 | **DUPLICATE** | → M01-M4 | |
| M04-L4 | **DUPLICATE** | → M10-10 | |

## m05 — Type-Driven Design

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M05-H1 | **CORRECTED** | `core/problem.rs:62`, `implicit_schur.rs:959–1002` | gap is real; the **proposed fix is harmful** — see [03-corrections](03-corrections.md#c3) |
| M05-H2 | **PARTIAL** | `levenberg_marquardt.rs:1240,1270` | GN (`gauss_newton.rs:818`) and DogLeg (`dog_leg.rs:1459`) now error; **LM still coerces in both arms** |
| M05-M1 | **CONFIRMED** | `factors/projection_factor.rs:428,437` | `unwrap_or_else(SE3::identity)` / `Matrix3xX::zeros(0)` |
| M05-M2 | **CONFIRMED** | `core/problem.rs:197–212` | invalid bounds warn-and-drop, no `Result` |
| M05-M3 | **CONFIRMED** | `core/problem.rs:34–35` vs `core/variable.rs` | constraints have two owners |
| M05-L1 | **CONFIRMED** | `core/problem.rs:183–191` | `fix_variable` accepts idx ≥ DOF, silently no-ops |
| M05-L2 | **PARTIAL** | `apex-manifolds/src/lib.rs:493–496` | sentinel remains but is now behind an `is_dynamic()` predicate (commit `112f3ae`) |

## m06 — Error Handling

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M06-1 | **CONFIRMED↓** | `linearizer/mod.rs:207–214` | **proved unreachable** through the public API — see [01-confirmed](01-confirmed.md#m06-1) |
| M06-2 | **CONFIRMED** | `factors/projection_factor.rs:443–445` | `.ok().unwrap_or_else(|| self.camera.clone())` |
| M06-L1 | **CONFIRMED** | `core/variable.rs:365,374` | `step[..dof]` unguarded |
| M06-L2 | **CONFIRMED** | `bin/bundle_adjustment.rs:409,410,411,421,431` | division by `num_obs`, `initial_cost`, `iterations` |
| M06-L3 | **CONFIRMED** | `apex-io/bin/bag_convert.rs` | 21 `-> Option<…>` decode helpers erase the failure reason |
| M06-L4 | **OBSOLETE** | `factors/projection_factor.rs:451–458,482–510` | the assert is `debug_assert_eq!` (compiled out in release) and `validate_variables` now rejects the mismatch at registration with a rich message |

## m07 — Concurrency

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M07-1 | **CONFIRMED** | `apex-io/…/dds/subscriber.rs:79–94` | `listen()` returns `Ok(rx)` before the thread initialises |

## m08 — Unsafe

| — | **CONFIRMED** | workspace-wide | zero `unsafe` outside `target/`; re-swept, still zero |

## m09 — Domain Modeling

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M09-H1 | **CONFIRMED↑** | `apex-manifolds/src/rn.rs:124,321`; `lib.rs:449` | **reproduced — real panic.** Root cause narrower than stated; `apply_to_vector` (`rn.rs:290`) also affected |
| M09-M1 | **CONFIRMED** | `core/problem.rs` (whole type) | logging/eval/storage/covariance in one aggregate |
| M09-M2 | **CONFIRMED** | `core/problem.rs:69–72`; `se3.rs:420–425` | `debug_assert_eq!(s.len(), 7)` only |
| M09-M3 | **CONFIRMED↓** | → M06-1 | same downgrade |
| M09-L1 | **CONFIRMED** | `core/problem.rs:174–181` | `remove_residual_block` leaves permanent row-space gaps |
| M09-L2 | **CONFIRMED** | `examples/loss_function_comparison.rs:465` | HashSet→Vec→sort; cosmetic |

## m10 — Performance

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M10-P1 | **CONFIRMED** | → M01-H1..H3 | the single best payoff/risk item left |
| M10-P2 | **CONFIRMED** | `explicit_schur.rs:1188–1205` | full triplet rebuild + linear `find` per camera index |
| M10-P3 | **FIXED** | `cholesky.rs:117`, `qr.rs:116` | `ne_cache.damped_hessian()` diagonal edit |
| M10-P4 | **PARTIAL** | → M01-H5 | HIGH → MED |
| M10-P5 | **FIXED** | `linearizer/mod.rs:135–175` | = M04-1 |
| M10-6 | **CORRECTED** | → M04-4 | dead API, not perf |
| M10-7 | **FIXED** | → M01-M1 | |
| M10-8 | **CONFIRMED** | `explicit_schur.rs:778–932` | row-major store / column-major read, `Vec::new()` triplets, dense cam² accumulator |
| M10-9 | **FIXED** | `normal_eq.rs` | `Jᵀ` cached |
| M10-10 | **CONFIRMED** | `levenberg_marquardt.rs:1251` and `:1007` | Schur path initialises twice; each pass clones all variables + full residual/cost eval |
| M10-11 | **CONFIRMED** | `implicit_schur.rs:860` calling `:783` | `Vec<Vec<usize>>` visibility index rebuilt inside every solve |
| M10-L1 | **CONFIRMED** | `cpu/sparse.rs:131`, `cpu/dense.rs:71`, `problem.rs:335` | `Mat::from_fn` element-wise residual copy |
| M10-L2 | **FIXED** | `explicit_schur.rs:711` | `ap` hoisted; faer SIMD kernels |

## m11 — Ecosystem

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M11-H1 | **CORRECTED** | `Cargo.toml:143–154`, `:33` | six unused deps + `memmap2` confirmed; the **`serde` dev-dependency claim is wrong** — see [03-corrections](03-corrections.md#c1) |
| M11-M1 | **CONFIRMED↓** | `gauss_newton.rs:244,382`; `dog_leg.rs:373,593` | field still dead, but setters now documented `Deprecated: no-op` → LOW |
| M11-M2 | **CONFIRMED** | `Cargo.lock:3629,3654,3680`; `:7945,7956,7966` | 3× faer (0.20.2/0.22.6/0.24.4), 3× rand (0.8.7/0.9.5/0.10.2) |
| M11-L1 | **CONFIRMED** | `Cargo.lock:6151` | nalgebra resolves uniformly to 0.33.3 — positive finding |
| M11-L2 | **CONFIRMED** | — | deprecations well-formed — positive finding |
| M11-L3 | **CONFIRMED** | — | `visualization` gating correct — positive finding |

## m12 — Lifecycle & Buffer Reuse

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M12-H1 | **CONFIRMED** | `apex-io/…/ros1/reader.rs:41,359–363` | unbounded `HashMap<u64, Vec<u8>>` chunk cache |
| M12-M1 | **CONFIRMED** | `ros1/reader.rs:3`; Drop impls at `ros1/writer.rs:390`, `ros2/reader.rs:297`, `ros2/writer.rs:514` | no `Drop for Ros1Reader`; module doc still claims memory-mapping |
| M12-M2 | **FIXED** | → M04-1 | |
| M12-M3 | **CONFIRMED** | → M02-2 | |
| M12-L1 | **CONFIRMED** | `explicit_schur.rs:476–541` | H_cc/H_cp triplet rebuild + re-sort per solve |
| M12-L2 | **CONFIRMED** | → M10-8 | |
| M12-L3 | **FIXED** | → M10-P3 | |
| M12-L4 | **FIXED** | → M04-1 | |

## m13 — Domain Error Design

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M13-1 | **CONFIRMED** | `apex-manifolds/src/lib.rs:74–131` | hand-written `Display`, no thiserror |
| M13-2 | **CONFIRMED** | `src/error.rs:197–200` | `log_with_source` logs `{:?}` and drops the source; `source()` stays `None` |
| M13-3 | **FIXED** | `apex-manifolds/src/sim3.rs:68–103` | `regularized_inverse_3/_7` (Tikhonov + warn/error); no `// SAFETY:` labels remain anywhere |
| M13-4 | **CONFIRMED** | `dds/subscriber.rs:91,161` | `DdsError::ThreadJoin` reused for spawn and runtime-creation failures |

## m14 — Mental Models

| — | **N/A** | — | Learning skill; the review's own "no findings possible" framing is correct |

## m15 — Anti-Patterns

| ID | Verdict | Location on `main` | Note |
|---|---|---|---|
| M15-H1 | **CONFIRMED↑** | LM `:993`, DL `:1201`, GN `:591` | loops have **grown** since the review: 235 / 227 / 195 lines (was 218 / 203 / 184) |
| M15-H2 | **PARTIAL** | `explicit_schur.rs:1188` vs `ne_cache.damped_hessian` | formation now shared; damping is **two** ways, not four |
| M15-M1 | **CONFIRMED** | as M15-H1, + `explicit_schur.rs:778` (155 ln) | |
| M15-M2 | **CONFIRMED** | `bin/pose_graph_g2o.rs:323` | `unreachable!()` bridging two stringly-typed tables |
| M15-M3 | **CONFIRMED** | `optimizer/mod.rs:146–159` | commented-out `IterativeState` |
| M15-M4 | **CONFIRMED↓** | → M11-M1 | |
| M15-L1 | **CONFIRMED** | `pose_graph_g2o.rs:323` | subsumed by M15-M2 |
| M15-L2 | **CONFIRMED** | `examples/loss_function_comparison.rs:465` | = M09-L2 |

---

## Tally

| Verdict | Count |
|---|---|
| CONFIRMED (incl. ↑/↓) | 48 |
| FIXED | 13 |
| PARTIAL | 5 |
| CORRECTED | 4 |
| OBSOLETE | 1 |
| DUPLICATE / cross-ref | 8 |
| Positive findings restated | 6 |
| N/A (m14) | 1 |
| **Total IDs** | **86** |

Plus **5 issues the review did not find** — see [04-new-findings.md](04-new-findings.md).
