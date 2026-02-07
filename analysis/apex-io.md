# apex-io Crate Analysis

## Overview

The `apex-io` crate provides file format support for pose graphs and datasets: G2O (SLAM standard), TORO (legacy 2D), and BAL (Bundle Adjustment in the Large). It defines the `Graph` container with `VertexSE2/SE3` and `EdgeSE2/SE3` types, and the `GraphLoader` trait.

**Files analyzed:** `lib.rs`, `g2o.rs`, `toro.rs`, `bal.rs`

---

## Performance Issues

### P1. Unnecessary String Allocations in Error Paths

Throughout all parsers, `format!()` and `.to_string()` are used in error paths even for fixed messages:

```rust
// lib.rs — Fixed string unnecessarily allocated
IoError::UnsupportedFormat("No file extension".to_string())
```

**Affected:** `lib.rs`, `g2o.rs` (lines 94, 180, 185+), `toro.rs` (lines 45, 69-71), `bal.rs` (lines 218-220)

**Fix:** Use `Cow<'static, str>` or string literals for fixed error messages.

### P2. Hardcoded Parallel Threshold

`g2o.rs` (line 78) — `let minimum_lines_for_parallel = 1000;` is hardcoded with no justification. Modern CPUs benefit from parallelization with smaller workloads. Consider lowering to ~200 lines or making it configurable.

### P3. Vec Instead of Array for Fixed-Size Data

`bal.rs` (lines 274-275, 296-297) — Camera parameters are parsed into `Vec::with_capacity(9)` and point coordinates into `Vec::with_capacity(3)`. Since sizes are known at compile time, `[f64; 9]` and `[f64; 3]` would avoid heap allocation entirely.

### P4. Intermediate Vec Collection in Parallel Path

`g2o.rs` (lines 130-131) — Parallel parsing collects all results into `Vec<ParsedItem>` before inserting into the Graph. For very large files, this doubles memory usage. A custom rayon reducer could insert directly.

### P5. Quadratic Index Access in Information Matrix Construction

`g2o.rs` (lines 389-406) — SE3 information matrix is built with 21 individual index accesses from a flat vector. This is fragile and could be more efficient with an upper-triangular iterator.

---

## Code Quality Issues

### Q1. Unsafe `memmap2` Without SAFETY Comments

Three files use `unsafe { memmap2::Mmap::map(&file) }` without documenting why it's safe:
- `g2o.rs` (lines 13-14)
- `toro.rs` (lines 13-18)
- `bal.rs` (lines 162-166)

**Required comment:** `// SAFETY: File handle is valid and exclusive; no concurrent writes occur during parsing.`

### Q2. Inconsistent Error Handling

Some errors are logged with `.log()` or `.log_with_source()`, others are not:
- `lib.rs` line 309: No logging context on `G2oLoader::load()`
- `lib.rs` line 321: Logging added with `map_err(|e| e.log())`
- This inconsistency makes debugging harder when errors occur in production.

### Q3. Silent Skip of Unknown G2O Types

`g2o.rs` (lines 208-209) — Unknown vertex/edge types are silently dropped:
```rust
_ => {
    // Skip unknown types silently for compatibility
}
```
This masks corrupt files and makes debugging difficult. Should at least emit a `tracing::warn!`.

### Q4. Inconsistent Quaternion Validation

- `g2o.rs` (lines 337-343) — SE3 **vertex** quaternions are validated (norm check)
- SE3 **edge** quaternions have **no** validation
- Validation tolerance is 0.01 (1%), which is extremely loose — should be ~1e-6

### Q5. `EdgeSE3` Boxed Inconsistently in `ParsedItem`

`g2o.rs` (lines 412-418) — Only `EdgeSE3` is `Box`-ed in the `ParsedItem` enum:
```rust
enum ParsedItem {
    VertexSE2(VertexSE2),
    VertexSE3(VertexSE3),
    EdgeSE2(EdgeSE2),
    EdgeSE3(Box<EdgeSE3>),  // Why boxed? Others aren't
}
```
Likely because `EdgeSE3` is larger (contains `Matrix6`), but the rationale should be documented.

### Q6. Unused `serde` Dependency

`Cargo.toml` lists `serde` and `serde_json` as dependencies, but neither is used in the crate's source code. These are dead dependencies that increase compile time.

### Q7. Public Fields Without Encapsulation

`lib.rs` — `VertexSE2`, `VertexSE3`, `EdgeSE2`, `EdgeSE3` all have public fields. If internal representation changes (e.g., storing normalized angles), all downstream code breaks. Consider private fields with accessors.

### Q8. Inconsistent Vertex Accessor API

- `VertexSE2` has `.x()`, `.y()`, `.theta()` shorthand accessors
- `VertexSE3` has `.x()`, `.y()`, `.z()` but requires `.pose.rotation()` for rotation
- API should be symmetric across SE2/SE3

---

## Readability & Maintainability

### R1. Complex Nested Error Handling in BAL Parser

`bal.rs` (lines 166-207) — 10+ sequential `map_err()` calls with repeated error context patterns:
```rust
value.trim().parse::<f64>()
    .map_err(|_| IoError::InvalidNumber { line: line_num, value: value.to_string() })?
```

A helper function would cut 60+ lines of repetition:
```rust
fn parse_number(value: &str, line_num: usize) -> Result<f64, IoError> { ... }
```

### R2. Magic Numbers Without Documentation

- `g2o.rs` (line 78): `1000` lines for parallel threshold — why?
- `g2o.rs` (line 331): `0.01` quaternion tolerance — why 1%?
- `bal.rs` (line 48): `DEFAULT_FOCAL_LENGTH = 500.0` — based on what?

### R3. Unclear Loop Variable Names

`bal.rs` (lines 273-276) — `param_idx` in the camera parameter loop is never used as an index:
```rust
for param_idx in 0..9 {  // param_idx could be _
```

### R4. Confusing Error Messages

`bal.rs` (lines 186-188) — Uses 0-indexed `param_idx` in user-facing error messages, but users expect 1-indexed parameter names.

---

## Redundancy / DRY Violations

### D1. 60+ Repeated Parse Error Patterns (CRITICAL)

Every numeric parse across all files follows:
```rust
parts[i].parse::<T>()
    .map_err(|_| IoError::InvalidNumber {
        line: line_num,
        value: parts[i].to_string(),
    })?
```

This pattern appears **60+ times** across `g2o.rs`, `toro.rs`, and `bal.rs`.

**Fix:** Extract into a shared helper:
```rust
fn parse_field<T: FromStr>(value: &str, line: usize) -> Result<T, IoError> { ... }
```

### D2. Identical File Loading Boilerplate (3 files)

All three loaders open a file and memory-map it with identical code:
```rust
let file = File::open(path_ref).map_err(|e| { ... })?;
let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| { ... })? };
let content = std::str::from_utf8(&mmap).map_err(|e| { ... })?;
```

**Fix:** Extract into `fn load_file_mmap(path: &Path) -> Result<String, IoError>`.

### D3. Duplicate Vertex Parsing Between G2O and TORO

`g2o.rs` (lines 152-189) and `toro.rs` (lines 45-72) have nearly identical `parse_vertex_se2` implementations with the same field extraction logic.

### D4. Identical Information Matrix Serialization

Upper-triangular extraction/construction for symmetric matrices is repeated:
- `g2o.rs` (lines 328-333): SE2 edge writing
- `g2o.rs` (lines 356-360): SE2 edge reading
- `toro.rs` (lines 53-58): Same pattern

**Fix:** Create `fn upper_triangular_to_symmetric<const N: usize>(values: &[f64]) -> SMatrix<f64, N, N>`.

---

## Safety & Numerical Stability

### S1. Unsafe Code Without Safety Documentation

Three instances of `unsafe { memmap2::Mmap::map(...) }` with no safety invariant documentation. While likely safe (file is opened exclusively and not modified during parsing), the Rust convention requires explicit documentation of why the unsafe block is sound.

### S2. Unbounded Vec Allocation

All parsers load the entire file content into a `Vec<&str>` of lines before processing:
- `g2o.rs` (line 76-77)
- `toro.rs` (line 43)
- `bal.rs` (line 184)

For very large files (100MB+), this requires loading all line references into memory before parsing begins. A streaming parser would be more memory-efficient.

### S3. No Bounds Checking Before Index Access

`g2o.rs` (lines 389-406) — Accesses `info_values[0..21]` for a 6x6 upper-triangular matrix without first validating `info_values.len() == 21`. If the input has fewer fields, this panics.

`bal.rs` (lines 273-284) — Creates a 9-element Vec, then accesses indices 0-8 without length verification.

### S4. Loose Quaternion Tolerance

`g2o.rs` (lines 337-343) — Quaternion norm validation uses `0.01` (1%) tolerance:
```rust
if (quat_norm - 1.0).abs() > 0.01 {
    return Err(IoError::InvalidQuaternion { ... });
}
```

A quaternion with norm 1.01 or 0.99 is accepted and silently normalized. This hides data quality issues. Tolerance should be ~`1e-6`.

---

## API Design

### A1. BAL Loader Doesn't Implement `GraphLoader` Trait

`lib.rs` (lines 349-356) — `GraphLoader` trait defines `load()` and `write()`. `BalLoader` has a `load()` method but does **not** implement the `GraphLoader` trait and has no `write()`. This inconsistency breaks the unified loader interface.

### A2. No Builder Pattern for Graph Construction

`Graph` has public fields but no builder. Creating a Graph requires manual HashMap insertion. A `GraphBuilder` would improve ergonomics and validate invariants.

### A3. Visualization Methods Scattered Across Types

- `VertexSE2` has `to_rerun_position_2d()`, `to_rerun_position_3d()`
- `VertexSE3` has `to_rerun_transform()`

Visualization is a cross-cutting concern and should be in a separate trait (e.g., `impl RerunConvertible for VertexSE2`) or module, not scattered across vertex types.

### A4. No Serialization Despite Dependencies

`serde` and `serde_json` are listed in `Cargo.toml` but not used. Either implement JSON export (useful for debugging graphs) or remove the dependencies.

---

## Testing Gaps

### T1. BAL Loader Has ZERO Tests (CRITICAL)

`bal.rs` has no `#[cfg(test)]` module at all. This is the most critical testing gap — the BAL loader handles complex multi-section file parsing with camera parameters, observations, and 3D points.

**Missing tests:**
- Valid BAL file parsing
- Invalid focal lengths (negative, zero, NaN, Inf)
- Malformed observation/camera/point blocks
- Count mismatches (header says 100 cameras, file has 50)
- Whitespace variations
- Empty file handling

### T2. No Round-Trip Tests

No tests verify: load G2O -> write G2O -> load again -> compare. This would catch serialization bugs and ensure data integrity.

### T3. No Parallel Parsing Correctness Tests

`g2o.rs` has a parallel parsing path (>1000 lines) and a sequential path. No test verifies both paths produce identical results. Subtle ordering or race condition bugs could hide in the parallel version.

### T4. Missing Edge Cases in G2O Tests

- Empty file handling
- Very large coordinates (overflow risk)
- Duplicate vertex IDs (is this valid?)
- Duplicate edges (allowed?)
- Non-integer IDs
- Files with only vertices (no edges)

### T5. Incomplete Quaternion Validation Tests

No tests verify:
- That non-normalized quaternions are correctly normalized after tolerance check
- That edges don't validate quaternions (potential bug)
- Behavior at the tolerance boundary (norm = 1.01 exactly)

---

## Prioritized Recommendations

### Critical
1. **Add BAL loader tests** — completely untested parser handling complex file formats
2. **Add SAFETY comments** to all 3 `unsafe` memmap blocks
3. **Extract shared parse helper** — eliminate 60+ repeated error patterns

### High
4. **Add bounds checking** before indexed access to parsed arrays
5. **Tighten quaternion validation tolerance** from 0.01 to 1e-6
6. **Add `tracing::warn!`** for unknown G2O types instead of silent skip
7. **Add round-trip tests** (load -> write -> load -> compare)

### Medium
8. **Extract `load_file_mmap()` helper** to deduplicate file loading boilerplate
9. **Implement `GraphLoader` for `BalLoader`** or document why it's intentionally excluded
10. **Remove unused `serde`/`serde_json` dependencies** from Cargo.toml
11. **Add parallel vs sequential correctness tests** for G2O parser

### Low
12. **Use arrays instead of Vecs** for fixed-size camera/point data in BAL parser
13. **Make parallel threshold configurable** or lower from 1000 to ~200
14. **Add builder pattern** for `Graph` construction
15. **Move visualization methods** to a separate trait or module
