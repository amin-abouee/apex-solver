---
name: hierarchical-error-handling
description: "CRITICAL: Use when creating, modifying, or reviewing error types, Result propagation, or logging across workspace crates (solver, cv, vio). Enforces hierarchical thiserror propagation, log-at-origin with ErrorLogging (.log() / .log_with_source()), and structured tracing (no println!/print!)."
---

# Hierarchical Error Propagation & Logging Rules

This document defines the mandatory error handling and logging architecture across the multi-crate workspace (`solver` → `cv` → `vio`), combining the patterns in [`src/error.rs`](file:///Volumes/External/Workspace/rust/apex-solver/src/error.rs), `rust-skills` (`m06-error-handling`, `m13-domain-error`, `coding-guidelines`), and the [Google Rust Style Guide](https://google.github.io/styleguide/rustguide.html).

---

## 1. Terminology: What Is This Pattern Called in Rust?

When function `A` (in `vio`) calls `B` (in `cv`), and `B` calls `C` (in `solver`), and an error occurs in `C`:

1. **Hierarchical Error Wrapping (Typed Bubble-Up Propagation)**:
   Each module and crate defines its own strongly-typed error enum using `thiserror::Error`. Higher-level crates wrap lower-level crate errors using `#[from]` (generating `From<ChildError> for ParentError`), allowing the `?` (`Try`) operator to propagate errors up the call stack (`C → B → A`) automatically without manual `.map_err()` boilerplate at intermediate layers.
2. **Transparent Delegation (`#[error(transparent)]`)**:
   Top-level crate error enums (`SolverError`, `CvError`, `VioError`) use `#[error(transparent)]` on variants that wrap lower-level errors. When formatted with `{}` (`Display`), the outer error delegates directly to the inner error's message in `C` instead of prepending redundant wrapper text at every hop.
3. **Log-at-Origin with Silent Bubble-Up**:
   Using the [`ErrorLogging`](file:///Volumes/External/Workspace/rust/apex-solver/src/error.rs#L156-L205) extension trait (`.log()` and `.log_with_source(src)`), the error is logged **exactly once** at the leaf site (`C`) where the domain error is first constructed. Intermediate callers (`B` and `A`) propagate the error silently with `?`. This prevents the classic **"log-and-rethrow" duplicate logging anti-pattern**.

---

## 2. Multi-Crate Hierarchy (`solver` → `cv` → `vio`)

Dependency and error flow strictly bottom-up (no circular dependencies):

```text
Layer C (Deep / Math)          Layer B (Mid / Vision)          Layer A (Top / System)
┌──────────────────────┐       ┌──────────────────────┐        ┌──────────────────────┐
│       solver         │       │          cv          │        │         vio          │
│                      │       │                      │        │                      │
│  LinAlgError    ─┐   │       │  FeatureError   ─┐   │        │  ImuError       ─┐   │
│  ManifoldError  ─┼──►│──────►│  GeometryError  ─┼──►│───────►│  FrontendError  ─┼──►│
│  CoreError      ─┤   │ #[from]  Solver(#[from])─┘   │ #[from]│  BackendError   ─┤   │
│  OptimizerError ─┘   │       │         │            │        │  Cv(#[from])    ─┤   │
│         │            │       │         ▼            │        │  Solver(#[from])─┘   │
│         ▼            │       │      CvError         │        │         │            │
│    SolverError       │       └──────────────────────┘        │         ▼            │
└──────────────────────┘                                       │      VioError        │
                                                               └──────────────────────┘
```

### Rules for Cross-Crate and Intra-Crate Layers

1. **Mandatory Per-Module Error Structure Across All Crates (`solver`, `cv`, and `vio`)**:
   - **Every module** in `cv` and `vio` (just like in `solver`) **must** define its own dedicated module-level error enum (e.g., `FeatureError`, `TrackingError`, `TriangulationError`, `GeometryError` in `cv`; `ImuError`, `FrontendError`, `BackendError`, `InitializationError`, `MarginalizationError` in `vio`) alongside a module `Result` alias (e.g., `pub type FeatureResult<T> = Result<T, FeatureError>;`).
   - Deep/internal modules within `cv` and `vio` must **never** return the top-level crate error (`CvError` or `VioError`) directly, nor use generic string errors. They must return their module-specific error type and bubble up into the crate's top-level error (`CvError` / `VioError`) via `#[from]`.
   - If a submodule in `cv` or `vio` calls a lower-level crate (`solver` or `cv`), its module error enum must wrap the relevant lower-level error via `#[from]` so `?` works seamlessly at the module level before reaching `CvError` or `VioError`.
2. **Leaf functions (`C` in `solver`, or leaf modules in `cv`/`vio`) construct and log the error once**:
   - Return module-specific errors (e.g., `Result<T, LinAlgError>`, `Result<T, TriangulationError>`, `Result<T, ImuError>`), **never** the top-level crate error directly from deep modules.
   - Attach `.log()` when constructing a domain error from a failed condition, or `.log_with_source(err)` when translating an external/third-party error that does not implement `#[from]`.
3. **Intermediate functions (`B` in `cv`) propagate with bare `?`**:
   - Do **not** call `.log()` again when propagating an already-logged error from `solver` or a deeper module.
   - Use `#[from]` in the calling module's error and in `CvError` so `solver_fn()?` and submodule calls convert automatically.
4. **Top-level functions (`A` in `vio`) propagate with `?` or handle recovery**:
   - `VioError` wraps all `vio` module errors (`#[from] ImuError`, `#[from] FrontendError`, `#[from] BackendError`, etc.) as well as `#[from] CvError` and `#[from] SolverError` (for direct `vio → cv` or `vio → solver` calls).
   - If `vio` inspects or reports the full error cause hierarchy at an application boundary, use `.chain()` or `.chain_compact()`.

---

## 3. Reference Implementation Across `solver`, `cv`, and `vio`

### 3.1 Shared `ErrorLogging` Trait (in `solver` and re-exported)

```rust
use std::error::Error as StdError;

/// Blanket trait providing single-site structured logging on error creation.
pub trait ErrorLogging: Sized + std::fmt::Display {
    /// Log the error via `tracing::error!` at the point of creation and return `self`.
    #[track_caller]
    fn log(self) -> Self {
        tracing::error!("{}", self);
        self
    }

    /// Log the error along with a third-party/underlying source error (`Debug`) and return `self`.
    #[track_caller]
    fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        tracing::error!("{} | Source: {:?}", self, source_error);
        self
    }
}

impl<T: std::fmt::Display> ErrorLogging for T {}
```

### 3.2 Crate 1: `solver` (Function `C` — Leaf Origin)

```rust
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum LinAlgError {
    #[error("Singular matrix detected: {0}")]
    SingularMatrix(String),
}

#[derive(Debug, Error)]
pub enum SolverError {
    #[error(transparent)]
    LinearAlgebra(#[from] LinAlgError),
}

pub type SolverResult<T> = Result<T, SolverError>;

/// Function C (in `solver`): Creates the error and logs it ONCE.
pub fn function_c(det: f64) -> Result<f64, LinAlgError> {
    if det.abs() < 1e-12 {
        // Logged ONCE here at the origin in C:
        return Err(LinAlgError::SingularMatrix(format!("det = {det:.2e}")).log());
    }
    Ok(1.0 / det)
}
```

### 3.3 Crate 2: `cv` (Function `B` — Module + Crate Error Structure)

Every module in `cv` defines its own error enum and converts into `CvError`:

```rust
use solver::{ErrorLogging, LinAlgError, SolverError};
use thiserror::Error;

/// Module-level error in `cv::triangulation` (never returns `CvError` directly)
#[derive(Debug, Error)]
pub enum TriangulationError {
    #[error("Insufficient feature matches: found {found}, minimum required {min}")]
    InsufficientMatches { found: usize, min: usize },

    #[error("Triangulation linear solve failed: {0}")]
    LinAlg(#[from] LinAlgError),
}

pub type TriangulationResult<T> = Result<T, TriangulationError>;

/// Top-level crate error in `cv::error`
#[derive(Debug, Error)]
pub enum CvError {
    #[error(transparent)]
    Triangulation(#[from] TriangulationError),

    /// Transparently wraps `SolverError` so `Display` prints C's exact message
    #[error(transparent)]
    Solver(#[from] SolverError),

    /// Direct conversion from `LinAlgError` if `cv` calls `solver::linalg` directly
    #[error(transparent)]
    LinAlg(#[from] LinAlgError),
}

pub type CvResult<T> = Result<T, CvError>;

/// Function B (in `cv::triangulation`): Calls C with `?`. Does NOT log again.
pub fn function_b(matches: usize, det: f64) -> TriangulationResult<f64> {
    if matches < 8 {
        // Originating in B -> log here:
        return Err(TriangulationError::InsufficientMatches { found: matches, min: 8 }.log());
    }
    // Originating in C -> propagate silently with `?` (already logged in C):
    let inv = solver::function_c(det)?;
    Ok(inv)
}
```

### 3.4 Crate 3: `vio` (Function `A` — Module + Crate Error Structure)

Every module in `vio` defines its own error enum and converts into `VioError`:

```rust
use cv::{CvError, TriangulationError};
use solver::{ErrorLogging, SolverError};
use thiserror::Error;

/// Module-level error in `vio::frontend` (never returns `VioError` directly)
#[derive(Debug, Error)]
pub enum FrontendError {
    #[error("IMU preintegration delta time invalid: {dt}s")]
    InvalidImuDt { dt: f64 },

    #[error(transparent)]
    Triangulation(#[from] TriangulationError),

    #[error(transparent)]
    Cv(#[from] CvError),
}

pub type FrontendResult<T> = Result<T, FrontendError>;

/// Top-level crate error in `vio::error`
#[derive(Debug, Error)]
pub enum VioError {
    #[error(transparent)]
    Frontend(#[from] FrontendError),

    #[error(transparent)]
    Cv(#[from] CvError),

    #[error(transparent)]
    Solver(#[from] SolverError),
}

pub type VioResult<T> = Result<T, VioError>;

/// Module-level function in `vio::frontend`: Calls B with bare `?`
pub fn track_frame(matches: usize, det: f64) -> FrontendResult<f64> {
    let val = cv::function_b(matches, det)?;
    Ok(val)
}

/// Top-level API function A (in `vio`): Calls `vio::frontend` with bare `?`. Only C's message was logged!
pub fn function_a(matches: usize, det: f64) -> VioResult<f64> {
    let val = track_frame(matches, det)?;
    Ok(val)
}
```

---

## 4. Core Error Rules (Google Rust Style Guide & `rust-skills`)

1. **No `unwrap()` or ` panic!()` in Library Code (`solver`, `cv`, `vio`)**:
   - Return `Result<T, E>` for all recoverable or input-dependent failures (singular matrices, failed convergence, insufficient features, bad sensor data).
   - Use `.expect("detailed invariant explanation")` **only** when a logical invariant guarantees `Some`/`Ok` and failure represents an internal bug.
   - Tests may use `Result<(), Box<dyn std::error::Error>>` with `?` instead of `.unwrap()`.
2. **Use `thiserror` for Library Crates, Never Stringly-Typed Errors**:
   - Every error type must derive `Debug` and `thiserror::Error`.
   - Prefer structured enum variants (e.g., `InvalidDimension { expected: usize, actual: usize }`) over raw `String` payloads where numeric context aids debugging.
3. **Preserve Error Chains (`source()`)**:
   - Always use `#[from]` or `#[source]` when wrapping lower-level errors so `std::error::Error::source()` traverses the entire `VioError → CvError → SolverError` chain.
   - Provide `.chain()` (multi-line) and `.chain_compact()` (single-line `A → B → C`) helper methods on top-level crate error types.
4. **Dual-Path Context Preservation**:
   - When a deep error (e.g., `LinAlgError`) occurs inside a mid-level subsystem (e.g., `OptimizerError`), wrap it in `OptimizerError::LinAlg(#[from] LinAlgError)` using `#[error("Linear algebra error: {0}")]` so `.chain()` preserves the subsystem context, while top-level API wrappers (`SolverError`, `CvError`, `VioError`) use `#[error(transparent)]`.

---

## 5. Logging & Observability Policy (Strictly No `println!` / `print!`)

Standard output macros (`println!`, `print!`, `eprintln!`, `eprint!`, `dbg!`) are **strictly forbidden** in library code (`solver`, `cv`, `vio`). All diagnostic output must go through the `tracing` crate.

### 5.1 Log Level Guidelines

| Macro | When to Use |
|-------|-------------|
| `tracing::error!` | Unrecoverable operation failure at the leaf origin (automatically invoked via `.log()` / `.log_with_source()`). |
| `tracing::warn!` | Recoverable anomaly, fallback taken, degraded accuracy (e.g., step rejected in LM, high reprojection residual, dropped frame). |
| `tracing::info!` | High-level lifecycle milestones (e.g., solver initialized, convergence summary, VIO tracking initialized). Never inside tight inner loops. |
| `tracing::debug!` | Per-iteration or per-frame algorithm diagnostics (e.g., iteration cost, damping $\lambda$, gradient norm, feature counts). |
| `tracing::trace!` | Fine-grained matrix dimensions, individual residual/Jacobian values, inner-loop step details. |

### 5.2 Enforce via Clippy Lints

Add the following workspace-level lints to `Cargo.toml` (or crate roots) so `cargo clippy --all-targets -- -D warnings` automatically rejects `print!` / `println!` / `dbg!` / `unwrap()`:

```toml
[workspace.lints.clippy]
print_stdout = "deny"
print_stderr = "deny"
dbg_macro = "deny"
unwrap_used = "deny"
```
*(In `tests/`, `examples/`, or `benches/`, allow only where explicitly needed via `#![allow(clippy::unwrap_used, clippy::print_stdout)]`.)*

---

## 6. How to Install This as a Workspace Skill

To make AI agents automatically follow these rules across your codebase:

1. **Project Skill (Recommended)**:
   Place this file at `.agents/skills/hierarchical-error-handling/SKILL.md` (or `.gemini/skills/hierarchical-error-handling/SKILL.md`) in your workspace root.
2. **Always-On Workspace Rule**:
   Reference or include the core mandates in your root `AGENTS.md` / `GEMINI.md` under **Error Handling & Logging**:
   - *"Follow `error_rules.md`: use `thiserror` hierarchical propagation (`solver` → `cv` → `vio`), log once at leaf error creation via `.log()` / `.log_with_source()`, propagate silently with `?`, and never use `println!`/`print!`/`eprintln!`/`dbg!`—always use `tracing::{info, debug, warn, error, trace}`."*
