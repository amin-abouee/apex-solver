# apex-solver — long functions and how to break them

Method: every function in production code (test modules excluded) measured by
the scanner described in `00-overview.md`. "Branch points" counts `if`,
`match`, `for`, `while`, `loop`, `?;`, `&&`, `||` in the body, a crude
cyclomatic proxy. Counts: 44 functions above 100 lines, 17 above 150, 8 above
200, 2 above 300.

Rule of thumb used for the proposals: a function should read as a list of
named steps. Where a function already has labelled phases in its comments
(most of the big ones do), those comments are the cut lines. The
decomposition must not change the floating-point operation order of numeric
code; each proposal says where that matters.

## Inventory (53 functions of 90 lines or more)

| Lines | Branch pts | Params | Function | Location |
|---:|---:|---:|---|---|
| 388 | 45 | 2 | `test_se2_dataset` | `bin/pose_graph_g2o.rs:395` |
| 354 | 42 | 2 | `test_se3_dataset` | `bin/pose_graph_g2o.rs:784` |
| 270 | 29 | 3 | `compute` | `src/core/marginalization.rs:120` |
| 256 | 19 | 6 | `run_bundle_adjustment` | `bin/bundle_adjustment.rs:286` |
| 255 | 14 | 3 | `integrate_from` | `src/factors/imu/preintegration.rs:199` |
| 239 | 30 | 5 | `eliminate` | `src/linalg/sparse/schur/chunk_eliminator.rs:312` |
| 233 | 13 | 4 | `optimize_with_mode` | `src/optimizer/levenberg_marquardt.rs:1141` |
| 226 | 11 | 4 | `optimize_with_mode` | `src/optimizer/dog_leg.rs:1320` |
| 195 | 30 | 4 | `compute_schur_complement` | `src/linalg/sparse/schur/explicit.rs:855` |
| 193 | 9 | 4 | `optimize_with_mode` | `src/optimizer/gauss_newton.rs:691` |
| 178 | 0 | 5 | `evaluate` | `src/factors/imu/sgal3/factors.rs:77` |
| 171 | 35 | 1 | `open` | `crates/apex-io/src/rosbag/ros1/reader.rs:80` |
| 159 | 16 | 4 | `linearize` | `src/factors/visual/inverse_depth.rs:94` |
| 154 | 20 | 3 | `log_final_state` | `src/observers/visualization.rs:1019` |
| 151 | 23 | 0 | `main` | `crates/apex-io/bin/download_datasets.rs:369` |
| 151 | 20 | 4 | `compute_optimization_step_generic` | `src/optimizer/dog_leg.rs:1105` |
| 151 | 6 | 4 | `linearize` | `src/factors/visual/depth.rs:81` |
| 149 | 0 | 0 | `get_all_message_types` | `crates/apex-io/bin/write_dummy_bag.rs:117` |
| 146 | 19 | 1 | `from_cdr` | `crates/apex-io/src/rosbag/ros2/messages.rs:641` |
| 143 | 31 | 1 | `convert_ros1_to_ros2` | `crates/apex-io/bin/bag_convert.rs:184` |
| 137 | 26 | 0 | `main` | `crates/apex-io/bin/bag_filter.rs:102` |
| 134 | 15 | 6 | `evaluate_internal` | `src/factors/visual/projection.rs:235` |
| 133 | 17 | 2 | `write` | `crates/apex-io/src/graph/g2o.rs:19` |
| 131 | 19 | 3 | `log_manifolds` | `src/observers/visualization.rs:1533` |
| 130 | 3 | 4 | `linearize` | `src/factors/navigation/gps.rs:253` |
| 127 | 11 | 2 | `parse_edge_se3` | `crates/apex-io/src/graph/g2o.rs:488` |
| 125 | 16 | 4 | `linearize` | `src/factors/visual/smart_projection.rs:329` |
| 124 | 19 | 2 | `log_initial_state` | `src/observers/visualization.rs:1408` |
| 122 | 20 | 0 | `main` | `bin/pose_graph_g2o.rs:1139` |
| 122 | 13 | 2 | `log_initial_ba_state` | `src/observers/visualization.rs:886` |
| 121 | 9 | 3 | `run_reader_loop` | `crates/apex-io/src/rosbag/ros2/dds/subscriber.rs:97` |
| 120 | 18 | 2 | `build` | `src/linalg/sparse/schur/chunk_eliminator.rs:81` |
| 118 | 15 | 4 | `solve_with_cholesky` | `src/linalg/sparse/schur/explicit.rs:651` |
| 118 | 8 | 2 | `optimize` | `src/optimizer/levenberg_marquardt.rs:1386` |
| 117 | 16 | 3 | `linear_estimation` | `crates/apex-camera-models/src/kannala_brandt.rs:65` |
| 116 | 23 | 1 | `convert_ros2_to_ros1` | `crates/apex-io/bin/bag_convert.rs:330` |
| 116 | 8 | 2 | `optimize` | `src/optimizer/gauss_newton.rs:892` |
| 116 | 8 | 2 | `optimize` | `src/optimizer/dog_leg.rs:1558` |
| 116 | 10 | 6 | `propagation` | `src/factors/imu/preintegration.rs:459` |
| 115 | 12 | 3 | `evaluate` | `src/factors/visual/photometric_inverse_range.rs:303` |
| 114 | 23 | 1 | `from_cdr` | `crates/apex-io/src/rosbag/ros2/messages.rs:364` |
| 112 | 24 | 4 | `solve_with_hessian` | `src/linalg/dense/schur/explicit.rs:114` |
| 111 | 19 | 1 | `open` | `crates/apex-io/src/rosbag/ros2/reader.rs:46` |
| 111 | 9 | 10 | `apply_schur_operator` | `src/linalg/sparse/schur/implicit.rs:498` |
| 100 | 13 | 4 | `solve_from_jacobian` | `src/linalg/sparse/schur/implicit.rs:728` |
| 99 | 11 | 4 | `linearize` | `src/factors/visual/stereo.rs:130` |
| 97 | 9 | 2 | `unproject` | `crates/apex-camera-models/src/kannala_brandt.rs:331` |
| 95 | 6 | 4 | `linearize` | `src/factors/visual/inverse_depth_plane.rs:130` |
| 94 | 16 | 2 | `parse` | `crates/apex-io/src/asl/ground_truth.rs:91` |
| 93 | 17 | 2 | `ensure_structure` | `src/linalg/sparse/schur/implicit.rs:398` |
| 93 | 6 | 4 | `linearize` | `src/factors/visual/inverse_depth_bearing.rs:178` |
| 91 | 13 | 2 | `extract_message_to_csv` | `crates/apex-io/bin/extract_topic_data.rs:377` |
| 90 | 27 | 1 | `validate` | `crates/apex-camera-models/src/lib.rs:182` |

(Test helpers and the two dataset runners in `bin/` are included because they
ship in the repository; see S-A8 for the binaries.)

## Families and proposed cuts

### S-L1 · `core::marginalization::Marginalizer::compute` (270 lines) · P1 · M
`core/marginalization.rs:120`. The function's own banner comments are the cut
lines:
1. `absorbed_blocks(problem, dropped) -> Vec<FactorKey>` (Markov-blanket set).
2. `ColumnLayout::new(problem, blanket)`: kept first, eliminated second, with
   `dims`, `offsets`, `kept_dof`, `total_dof`.
3. `dense_jacobian(problem, absorbed, layout) -> (J, r)` through
   `compute_block_into` (keeps whitening and the robust-loss corrector).
4. `schur_marginal(J, r, layout) -> (Λ_p, g_p)`; see S-P6 and S-C1 for what
   this step should do instead.
5. `first_estimate_offset(problem, kept, ...)` (the frozen-block correction).
6. `Marginal::assemble(...)`.
Each step becomes unit-testable on a 3-variable problem; today only the whole
function is. Gate: the existing tests in `core/marginalization.rs` and the
typed-prior parity test pass; marginals bit-identical on the linear chain.

### S-L2 · `chunk_eliminator::eliminate` (239) and `explicit::compute_schur_complement` (195) · P2 · M
`linalg/sparse/schur/chunk_eliminator.rs:312`, `explicit.rs:855`. Both are
hand-optimised, bandwidth-bound loops with the two-phase structure already
named in comments ("Phase A", "Phase B").
- `eliminate`: `accumulate_chunk_rows` (FᵀF, Fᵀr, EᵀE, EᵀF, Eᵀr per chunk),
  `apply_damping`, `schur_correct_chunk` (invert block, subtract rank-dof
  update), `leading_rows` / `trailing_rows` (rows outside any chunk).
- `compute_schur_complement`: `init_from_kept`, `gather_block_strip`,
  `multiply_by_inverse`, `rank_update`, `extract_sparse_pattern`.
Constraint: these are the BAL hot path. Cut only where the compiler will still
inline (`#[inline]` on the small helpers is allowed; suppressions are not).
Gate: `ba_benchmark` golden costs identical and criterion no slower than
noise on trafalgar-257 and dubrovnik-135.
Also: `parameters >= 9` on `gather_strip` and `sweep_rows`
(`chunk_eliminator.rs:561`, `:636`) and `apply_schur_operator` (10,
`implicit.rs:498`) carry `#[allow(clippy::too_many_arguments)]`. Replace the
argument lists with a `StripCursor` / `SweepContext` struct.

### S-L3 · IMU preintegration: `integrate_from` (255) and `propagation` (116) · P1 · M
`factors/imu/preintegration.rs:199` and `:459`. `integrate_from` interleaves
(a) clipping the first and last sample pair to `[t0, t1]` with linear
interpolation, (b) the mean-state update, (c) covariance propagation,
(d) bias-Jacobian chains (`dalpha_db_g`, `dv_db_g`, `dp_db_g`, `dp_dsigma`),
(e) the double-integral accumulators. Proposed cut:
`clip_interval(pair, t0, t1) -> Segment`, `advance_state(&mut State, &Segment)`,
`advance_covariance(&mut Cov, &Segment)`, `advance_bias_jacobians(...)`.
This is also where S-C5 (boundary case) and S-P4 (`append`) get fixed.
Gate: `factors/imu/{se23,sgal3}/tests.rs` plus a new golden test that
integrates a recorded 200 Hz segment and compares the final state, covariance
and bias Jacobians **bitwise** before/after.

### S-L4 · Optimizer drivers (`optimize_with_mode` x3, `optimize` x3, `compute_optimization_step_generic`) · P1 · L
Covered by S-A1. The three `optimize` entry points (116-118 lines each) are a
second triplication: each matches `(JacobianMode, LinearSolverType)` and builds
the linear solver. One `fn select_solver(mode, solver_type) -> Result<SolverChoice>`
removes about 250 lines.

### S-L5 · Factor `linearize`/`evaluate` bodies · P2 · M
`sgal3/factors.rs:77 evaluate` (178, zero branches: straight-line algebra),
`visual/inverse_depth.rs:94 linearize` (159), `visual/depth.rs:81` (151),
`navigation/gps.rs:253` (130), `visual/projection.rs:235 evaluate_internal`
(134), `visual/smart_projection.rs:329` (125),
`visual/photometric_inverse_range.rs:303 evaluate` (115).
Pattern to apply: `residual(...)`, `jacobian_blocks(...)`, `invalid_path(...)`
(the cheirality/penalty branch that several of these repeat). Share the
penalty branch through `factors/common/cheirality.rs`, which exists. For
`sgal3` versus `se23` see S-R3: they are not line-for-line copies (SGal3 has
a time coordinate and a coupling term), so share only the helpers
(`imu/common.rs` already has `split_bias`, `write_jacobian`).
Gate: `factors/common/conformance.rs` finite-difference parity must stay green
per factor; no change to the operation order inside each kernel.

### S-L6 · Readers and writers (`apex-io`) · P2 · M
`ros1/reader.rs:80 open` (171, 35 branch points), `ros2/messages.rs:641
from_cdr` (146, depth 7), `graph/g2o.rs:19 write` (133), `parse_edge_se3`
(127), `ros2/dds/subscriber.rs:97 run_reader_loop` (121, depth 8),
`bag_convert::convert_ros1_to_ros2` (143) and `convert_ros2_to_ros1` (116).
- `open`: split `read_header`, `read_index_data`, `build_connection_table`,
  `build_chunk_index`.
- `from_cdr`: a dispatch table from message type to a decoder function
  (`fn(&mut CdrDeserializer) -> Result<Message>`), one function per message.
- g2o/toro: see S-R5 (a `field::<T>(parts, i, line)` helper removes the
  repeated 5-line `parse().map_err(...)` blocks).
- `run_reader_loop`: depth 8 is `select!` inside `loop` inside a `match`; lift the
  per-message handler out.

### S-L7 · Binaries (`bin/`, `crates/apex-io/bin/`) · P2 · M
`test_se2_dataset` (388) and `test_se3_dataset` (354) are two instances of the
same driver; `run_bundle_adjustment` (256); `download_datasets::main` (151);
`bag_filter::main` (137); `get_all_message_types` (149, a literal table).
Resolved by S-A8 (move to `apex-cli` and make the runners table-driven).

### S-L8 · Observers · P2 · M
`visualization.rs` four methods of 120-154 lines. See S-A7.

### S-L9 · `Problem::try_add_residual_block_with_noise` and friends · P3 · S
`core/problem.rs:142-322` is 180 lines across two functions that validate shape,
variables, noise dimension, row offset and register. Split validation
(`ValidatedBlock::new`) from registration (`insert_block`). This is the
natural seam for S-A2's `BlockStore`.
