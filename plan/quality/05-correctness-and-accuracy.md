# apex-solver — correctness and numerical accuracy

Severity here is about *wrong or degraded results*, not style. The honest
summary: nothing found changes a published benchmark number today (the golden
costs and apex-vio parity runs are all green), but four places can return a
wrong answer silently, which is the class of defect that is expensive to find
later.

---

## S-C1 · Marginalization documents "solved, not inverted" and inverts · P1 · S

**Evidence.** `core/marginalization.rs:284`: "Solved, not inverted:
`O(n_m²·n_k)` instead of `O(n_m³ + n_m²·n_k)`, and better conditioned."
`:297`: `let lambda_mm_inv = pseudo_inverse(&lambda_mm, 1e-12)`, then
`Λ_kk − Λ_km · Λ_mm⁻¹ · Λ_mk` (`:302`). The tolerance argument is relative
(`pseudo_inverse(matrix, relative_tolerance)`, `:476`), which is good.

**Why it matters.** The comment states a complexity and a conditioning
property the code does not have; the next reader will trust it. An explicit
inverse of an ill-conditioned `Λ_mm` squares the conditioning error that a
solve would only incur once.

**Proposal.** Factor `Λ_mm` once (symmetric eigen, which the pseudo-inverse
already computes; or LDLT with a pivot threshold) and solve for
`Λ_mm⁻¹ Λ_mk` and `Λ_mm⁻¹ g_m` as right-hand sides. Keep the eigen path for the
rank-deficient (gauge) case, because that case is real and tested
(`a_relative_only_constraint_leaves_no_absolute_information`). Fix or delete the
comment either way. Combined with S-P6.

**Gate.** Exact-equivalence tests on the linear chain at 1e-6 hold; add a
badly scaled chain (information spanning 1e-8 to 1e8) that shows the solve is
no worse than the inverse.

---

## S-C2 · Jacobian inverses fall back to identity · P1 · M

**Evidence.**
- `crates/apex-manifolds/src/sim3.rs:75-105 regularized_inverse_3 / _7`: try
  inverse, then a Tikhonov-regularized inverse with `eps = 1e-9 * (trace+1)/n`,
  then `Matrix::identity()` with a `tracing::error!`.
- `sgal3.rs:716, :724 right_jacobian_inv / left_jacobian_inv`: `try_inverse()`
  or `Matrix10::identity()` with a log line.

**Why it matters.** `LieGroup` methods return plain matrices, so the failure
cannot be reported through the type; the log line is the only signal. A
solver that reaches this state continues with a wrong Jacobian and may
converge to a wrong answer or merely stall, with no error in the result. (The
Sim3 version improved on the original silent identity; the last-resort branch
is the same defect, smaller.)

**Proposal.**
1. Add `try_right_jacobian_inv`/`try_left_jacobian_inv` returning
   `Result<_, ManifoldError>` (the error enum exists in `lib.rs`), implement the
   infallible ones on top, documented "returns identity on singular input, use
   `try_*` to detect".
2. In the optimizer-facing paths (`plus`/`minus` with Jacobians used by
   factors) call the `try_` form and propagate it as the factor error type's Jacobian-failure variant, so
   a singular block rejects the step instead of corrupting it.
3. With S-P5 (closed-form SGal3) the singular case for SGal3 reduces to a
   threshold check on the closed form; delete the matrix inverse.

**Gate.** New test: construct a tangent at `σ → −∞` (Sim3) and at the SGal3
degeneracy and require `Err`, not identity.

---

## S-C3 · `Rn` panics inside trait methods that cannot fail · P1 · M

**Evidence.** `crates/apex-manifolds/src/rn.rs:241, 418, 455, 491, 751`
(`compose`, `right_plus`, `right_minus`, `left_plus`, `interp`): `assert_eq!` on
the two dimensions, in the `LieGroup` implementation for the dynamic-size
group. Also `unreachable!()` in the public `generator(i)` of SE₂(3), Sim3 and
SGal3 (`se23.rs:800, :808`, `sim3.rs:823, :832`, `sgal3.rs:886, :893`):
`generator(9)` panics, and it is reachable from safe external code.

**Why it matters.** Registration validates variable sizes
(`Factor::validate_variables`), so ordinary use is protected, but a panic in a
rayon worker poisons the pool (the August review's point) and `unreachable!` is
the wrong tool for an out-of-range argument of a public method.

**Proposal.**
- `generator(i)` returns `Option<LieAlgebra>` (or takes a `GeneratorIndex`
  newtype constructed from `0..DOF` by `TryFrom`).
- For `Rn`, dimension equality becomes an invariant of the *variable* (checked
  at `add_variable` and `set_variable_params`), and `compose` etc. use
  `zip` over the shorter length with a `debug_assert_eq!`. The type-level
  alternative, a const-generic `Rn<const N: usize>` for fixed-size cases, is in
  `plan/factor` (N2, `ConstDoF`); `Rn` keeps the dynamic form only for
  landmarks and parameters whose size is data-dependent.
- Add `panic`, `unreachable`, `todo`, `unimplemented` to the deny list (S-E1).

---

## S-C4 · Inline tolerance literals without names · P3 · S

**Evidence.** Production code only (inline `#[cfg(test)]` modules, `tests.rs`
files, `tests/`, `benches/`, `examples/` excluded) holds 104 bare tolerance
literals across the four crates: `1e-12` x37, `1e-6` x25, `1e-10` x17, `1e-9`
x10, `1e-8` x7, `1e-15` x4. (Tests legitimately use many more.) Several of
the 104 mean different things in different places (absolute versus relative, angle versus squared
angle, singular-value cut-off versus step guard). The crate does have named
constants for some (`SMALL_ANGLE_THRESHOLD = 1e-10`), which shows the intent.

**Why it matters.** Changing the meaning of "singular" or "small angle" in one
module cannot be done without reading all of them; and two nearly equal
literals (`1e-12`, `1e-10`) in adjacent functions invite an inconsistent fix.

**Proposal.** A `tolerances` module per crate (`apex_manifolds::tol`,
`apex_solver::tol`) with named constants and one-line justification each:
`SMALL_ANGLE`, `SINGULAR_PIVOT`, `RANK_RELATIVE`, `STEP_NORM_FLOOR`,
`DT_FLOOR`. Replace by mechanical search-and-replace in a dedicated commit
that is **provably a no-op** (same literal values, so byte-identical output).
Only afterwards revisit the values one at a time with measurements.

---

## S-C5 · IMU boundary interpolation when both ends lie in one sample pair · P3 · S (to verify)

**Evidence.** `factors/imu/preintegration.rs:199-260`. At the start boundary
the first pair's `omega_s_0`/`acc_s_0` are replaced by their values interpolated
to `t0`. At the end boundary (`next_time > end`) the code interpolates again,
`omega_s_1 = (1-r)·omega_s_0 + r·omega_s_1` with `r = (end - t0_meas)/interval`.
If `t0` and `t1` both fall inside the *same* measurement pair, `omega_s_0` is
already the value at `t0` while `r` is still measured from `t0_meas`, which
looks inconsistent.

**Status.** Not reproduced: it needs an interval shorter than one IMU period,
which none of the current datasets (200 Hz IMU, 20 Hz camera, 10 Hz LiDAR)
produces. It is a candidate bug found by reading, hence "to verify".

**Proposal.** Write the test first: a constant-acceleration IMU stream,
`[t0, t1]` inside one pair, compare against the analytic integral. If it
fails, fix by interpolating both ends from the *original* samples. Becomes
trivial once `integrate_from` is split (S-L3: `clip_interval` returns the two
boundary values once).

---

## S-C6 · Absolute timestamps in `f64` seconds with absolute epsilons · P3 · M

**Evidence.** Sample and frame times are `f64` seconds since the epoch
(EuRoC stamps are about 1.4e9 s), and the code compares with absolute epsilons:
`dt < 1e-12` and `m.timestamp > self.t1 + 1e-12` in
`preintegration.rs`. At 1.4e9 the spacing between adjacent `f64` values is
about 2.4e-7 s, so an epsilon of 1e-12 is an exact comparison, and the
interpolation weights carry up to 240 ns of quantization (about 5e-5 of a
5 ms IMU period). apex-vio documents this (`io/euroc.rs:51`) and judges it
negligible; the magnitude agrees: a velocity error on the order of
3e-5 m/s per second of integration at 10 m/s², well under the IMU noise floor.

**Proposal (low urgency).** Let the sensor-facing crates carry a
`Timestamp(u64 ns)` and let the estimator work in `f64` seconds *relative to
the sequence origin*, which removes the quantization and makes the epsilons
mean what they say. Do not do this before S-A1/S-L3; it touches every
interface and gains little accuracy.

---

## S-C7 · Runtime finite differences inside factors · P2 · M

**Evidence.** `factors/visual/inverse_depth.rs:70-91 unproject_jacobian`
differentiates `camera.unproject` by central differences with
`NUMERICAL_DERIVATIVE_EPS` on every `linearize`, four extra unprojections per
call, because `CameraModel` has no analytic unprojection Jacobian. The result
has finite-difference error (about 1e-8 relative) in a Jacobian whose other
blocks are analytic.

**Proposal.** `CameraModel::jacobian_unproject(&self, pixel) -> Matrix3x2`,
analytic per model (most models already invert a closed form or run a
Newton iteration whose Jacobian is available). See S-K2 in `08`. Keep FD in
tests as the oracle. The same helper serves apex-vio's bearing code.

**Gate.** `tests/camera_*_integration.rs` FD comparison per model; the
`InverseDepthFactor` conformance test; no solver-output change beyond 1e-8
(this one is not bitwise).

---

## What was checked and found sound

- Triggs corrector and the loss roster: regression-tested including the
  ISSUE-0005/0010/0011 cases.
- Fixed-DOF handling (ISSUE-0003): fixed components are excluded from the
  column layout and from the covariance; tests exist.
- LM acceptance and damping: gain ratio against the model's predicted
  reduction, separate accept/adapt decisions (the Ceres structure).
- Deterministic reductions: per-block cost is summed in block order, so the
  result is independent of thread count.
