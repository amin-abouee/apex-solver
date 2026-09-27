//! # Implicit Sparse Schur Complement Solver
//!
//! Matrix-free Schur complement solved with Preconditioned Conjugate
//! Gradients — Ceres's `ITERATIVE_SCHUR`, and matrix-free in the same sense
//! Ceres means it: **neither `S` nor `JᵀJ` is ever formed**.
//!
//! ## What "implicit" has to mean
//!
//! Writing `J = [E | F]` for the eliminated and retained column sets, the
//! reduced system is
//!
//! ```text
//! S = FᵀF − FᵀE·(EᵀE)⁻¹·EᵀF
//! ```
//!
//! and PCG only ever needs its action on a vector. Expanding that action so it
//! reads `J` and nothing else gives the whole solver:
//!
//! ```text
//! y = F·v                       scatter over the retained columns
//! t = Eᵀ·y                      gather over the eliminated columns
//! u = (EᵀE + λD_e)⁻¹·t          block-diagonal, one small solve per block
//! y ← y − E·u                   scatter over the eliminated columns
//! S·v = Fᵀ·y + λD_k·v           gather over the retained columns
//! ```
//!
//! Four passes over `J`'s nonzeros, no intermediate matrix. Ceres's own
//! documentation states the same cost model: "the cost of this evaluation
//! scales with the number of non-zeros in the Jacobian".
//!
//! That distinction is the reason this solver exists. `JᵀJ` for a bundle
//! adjustment problem carries a dense block for **every pair of cameras
//! sharing a landmark** — fill-in `J` does not have — so forming it costs far
//! more memory than `J`, and walking it costs far more time than walking `J`.
//! A solver that built `JᵀJ` and only skipped `S` would avoid the smaller of
//! the two costs while paying the larger one.
//!
//! ## When to use it
//!
//! When `JᵀJ` or `S` will not fit, or barely fits. The explicit solvers
//! factorize an exact `S` and are faster whenever they fit in memory, so this
//! is the large-problem fallback rather than the default —
//! `ExplicitSparseSchur` remains what
//! [`LevenbergMarquardtConfig::for_bundle_adjustment`](crate::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::for_bundle_adjustment)
//! selects.
//!
//! ## Preconditioning
//!
//! [`SchurPreconditioner`] selects between the three Ceres offers for
//! `ITERATIVE_SCHUR`, all built from `J`:
//!
//! | This crate | Ceres | Built from |
//! |---|---|---|
//! | [`SchurPreconditioner::None`] | `IDENTITY` | — |
//! | [`SchurPreconditioner::BlockDiagonal`] | `JACOBI` | diagonal blocks of `FᵀF` |
//! | [`SchurPreconditioner::SchurJacobi`] | `SCHUR_JACOBI` | diagonal blocks of `S` |
//!
//! `SchurJacobi` is the default and generally the best of the three; it costs
//! one pass over the observations to build.
//!
//! ## Automatic group recognition
//!
//! Like every other Schur solver, the eliminated ("group 0") and retained
//! ("group 1") variable sets come from [`SchurPartition`] — the same
//! structure [`ExplicitSparseSchur`](super::explicit::ExplicitSparseSchur) and
//! [`ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur)
//! use. This solver places no restriction on the eliminated variables' DOF or
//! column layout: 3-D points, 1-DOF inverse-depth landmarks, or a mix of both
//! in one problem all work identically, and the retained and eliminated
//! columns may interleave arbitrarily.
//!
//! ## Usage Example
//!
//! ```no_run
//! # use apex_solver::linalg::{ImplicitSparseSchur, SchurPreconditioner};
//! # use apex_solver::linalg::StructureAware;
//! # use apex_solver::core::VarKey;
//! # use apex_solver::core::variable::ManifoldVariable;
//! # use slotmap::{SlotMap, SecondaryMap};
//! # use std::collections::HashSet;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
//! # let variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
//! # let landmark_keys: HashSet<VarKey> = HashSet::new();
//! use apex_solver::linalg::{ImplicitSparseSchur, SchurPreconditioner};
//! use apex_solver::linalg::StructureAware;
//!
//! let mut solver = ImplicitSparseSchur::new()
//!     .with_preconditioner(SchurPreconditioner::SchurJacobi)
//!     .with_cg_config(500, 1e-9);
//! solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
//! # Ok(())
//! # }
//! ```

use crate::core::VarKey;
use crate::core::variable::ManifoldVariable;
use crate::error::ErrorLogging;
use crate::linalg::regularization::invert_with_retry_dyn;
use crate::linalg::schur::jacobian_ops::{column_dot, diag_jt_j};
use crate::linalg::schur::{
    BlockSpan, EliminatedBlocks, PcgParams, SchurOrdering, SchurPartition, SchurPreconditioner,
    effective_landmark_keys, jt_j_vec_product, jt_vec, solve_pcg,
};
use crate::linalg::sparse::pattern;
use crate::linalg::{Damping, LinAlgError, LinAlgResult, LinearSolver, SparseMode, StructureAware};
use faer::Mat;
use faer::sparse::SparseColMat;
use nalgebra::DMatrix;
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};
use tracing::debug;

/// Structural facts about `J` that depend only on its sparsity, so they are
/// rebuilt when the pattern changes and reused across every solve in between.
#[derive(Debug, Clone, Default)]
struct StructureCache {
    /// Global column of each retained local index, and of each eliminated
    /// local index.
    ///
    /// The operator's two gather passes write one output entry per column, so
    /// with this mapping they parallelize over the output slice directly —
    /// every entry written by exactly one task, no aliasing to reason about.
    kept_cols: Vec<usize>,
    eliminated_cols: Vec<usize>,
    /// Sorted rows touched by each eliminated block.
    eliminated_rows: Vec<Vec<usize>>,
    /// Eliminated blocks visible from each kept block — they share a row.
    visibility: Vec<Vec<usize>>,
    /// Residual rows per chunk of the two row-space passes, `F·v` and `E·u`.
    chunk_rows: usize,
    /// Per row chunk, every retained column's nonzeros that fall inside it.
    ///
    /// The two row-space passes are scatters when run column by column, so
    /// they cannot split over columns without two tasks writing one row.
    /// Split over row chunks instead: each chunk owns its slice of `y`, and
    /// replays the columns in the column-wise scatter's order, so every row
    /// receives the same additions in the same order — bit-identical results.
    kept_spans: Vec<Vec<ColumnSpan>>,
    /// Same, over the eliminated columns; `local` is the eliminated index.
    eliminated_spans: Vec<Vec<ColumnSpan>>,
    /// Eliminated block owning each eliminated local index.
    eliminated_block_of: Vec<u32>,
    /// Pattern the cache was built from.
    fingerprint: Option<pattern::PatternFingerprint>,
}

/// One column's nonzeros inside one row chunk: `J`'s value range `lo..hi`,
/// and the column's local index in the vector the pass reads.
///
/// `u32` keeps the per-chunk lists cache-friendly; `ensure_structure` rejects
/// a Jacobian whose nonzero count or dimensions do not fit.
#[derive(Debug, Clone, Copy)]
struct ColumnSpan {
    lo: u32,
    hi: u32,
    local: u32,
}

/// Rows per chunk: enough chunks for load balance across the pool, few
/// enough that each carries real work.
fn row_chunk_len(nrows: usize) -> usize {
    const MIN_CHUNK_ROWS: usize = 4096;
    let chunks = rayon::current_num_threads().max(1) * 8;
    nrows.div_ceil(chunks).max(MIN_CHUNK_ROWS)
}

/// The row-chunk span lists of [`StructureCache`] for one chunk length.
struct RowSpans {
    kept: Vec<Vec<ColumnSpan>>,
    eliminated: Vec<Vec<ColumnSpan>>,
    block_of: Vec<u32>,
}

impl RowSpans {
    /// Push every column in exactly the order the column-wise scatter visits
    /// them: retained blocks then offsets, eliminated blocks then offsets.
    fn build(
        jacobian: &SparseColMat<usize, f64>,
        partition: &SchurPartition,
        chunk_rows: usize,
    ) -> LinAlgResult<Self> {
        let chunks = jacobian.nrows().div_ceil(chunk_rows);
        let mut kept = vec![Vec::new(); chunks];
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                if let Some(local) = partition.kept_local(col) {
                    push_column_spans(jacobian, col, local, chunk_rows, &mut kept)?;
                }
            }
        }
        let mut eliminated = vec![Vec::new(); chunks];
        let mut block_of = vec![0u32; partition.eliminated_dof()];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let owner = u32::try_from(block_idx).map_err(|_| {
                LinAlgError::InvalidInput(format!(
                    "{block_idx} eliminated blocks exceed the implicit Schur operator's u32 range"
                ))
                .log()
            })?;
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                block_of[base + offset] = owner;
                push_column_spans(
                    jacobian,
                    block.col_start + offset,
                    base + offset,
                    chunk_rows,
                    &mut eliminated,
                )?;
            }
        }
        Ok(Self {
            kept,
            eliminated,
            block_of,
        })
    }
}

/// Append `col`'s nonzeros, split at chunk boundaries, to the chunk lists.
fn push_column_spans(
    jacobian: &SparseColMat<usize, f64>,
    col: usize,
    local: usize,
    chunk_rows: usize,
    spans: &mut [Vec<ColumnSpan>],
) -> LinAlgResult<()> {
    let symbolic = jacobian.symbolic();
    let start = symbolic.col_range(col).start;
    let rows = symbolic.row_idx_of_col_raw(col);
    let narrow = |x: usize| {
        u32::try_from(x).map_err(|_| {
            LinAlgError::InvalidInput(format!(
                "Jacobian index {x} exceeds the implicit Schur operator's u32 range"
            ))
            .log()
        })
    };
    let local = narrow(local)?;
    let mut k = 0;
    while let Some(&first_row) = rows.get(k) {
        let chunk = first_row / chunk_rows;
        let chunk_end = (chunk + 1) * chunk_rows;
        let hi = k + rows[k..].partition_point(|&r| r < chunk_end);
        let slot = spans.get_mut(chunk).ok_or_else(|| {
            LinAlgError::InvalidInput(format!("row {first_row} outside the Jacobian")).log()
        })?;
        slot.push(ColumnSpan {
            lo: narrow(start + k)?,
            hi: narrow(start + hi)?,
            local,
        });
        k = hi;
    }
    Ok(())
}

/// Implicit (matrix-free) Schur complement solver using Preconditioned
/// Conjugate Gradients. Equivalent to Ceres's `ITERATIVE_SCHUR`.
///
/// Forms neither `S` nor `JᵀJ`; see the module documentation for the operator.
#[derive(Debug, Clone)]
pub struct ImplicitSparseSchur {
    partition: Option<SchurPartition>,
    /// `(EᵀE + λD_e)⁻¹` per eliminated block, rebuilt once per solve.
    eliminated: EliminatedBlocks,
    /// Largest eliminated block's DOF, for sizing the block-apply scratch.
    max_eliminated_dof: usize,
    /// Automatic eliminated/retained ("group 0"/"group 1") classification,
    /// combined with manual marks in `initialize_structure`.
    ordering: SchurOrdering,

    // CG parameters
    max_cg_iterations: usize,
    cg_tolerance: f64,
    /// Forcing-sequence parameter η; `0.0` disables the quadratic-model rule.
    cg_q_tolerance: f64,

    // Preconditioner type
    preconditioner_type: SchurPreconditioner,

    /// `J` of the last successful solve.
    ///
    /// Held so [`LinearSolver::hessian_vec_product`] can evaluate `Jᵀ(J·v)`
    /// for the optimizers' quadratic model — the same arrangement
    /// `ExplicitSparseSchur`'s chunked variant uses. One copy of `J`, against
    /// the `JᵀJ` this solver exists to avoid.
    jacobian: Option<SparseColMat<usize, f64>>,
    /// Pattern the retained `jacobian` was built from, so its storage can be
    /// reused across solves instead of reallocated.
    jacobian_pattern: Option<pattern::PatternFingerprint>,
    /// `+Jᵀr`, published through [`LinearSolver::get_gradient`].
    gradient: Option<Mat<f64>>,

    // Workspace buffers for the Schur operator (avoid repeated allocations).
    workspace_rows: Vec<f64>, // residual-row sized buffer
    workspace_lm: Vec<f64>,   // eliminated-DOF sized buffer
    workspace_cam: Vec<f64>,  // kept-DOF sized buffer (parallel gather output)
    block_scratch: Vec<f64>,  // eliminated-DOF sized: u = (EᵀE+λD_e)⁻¹·t

    structure: StructureCache,
}

impl ImplicitSparseSchur {
    /// Default: Schur-Jacobi preconditioner, 500 max iterations, 1e-9 relative
    /// tolerance — matching Ceres's `ITERATIVE_SCHUR` defaults.
    pub fn new() -> Self {
        Self {
            partition: None,
            eliminated: EliminatedBlocks::default(),
            max_eliminated_dof: 1,
            ordering: SchurOrdering::default(),
            max_cg_iterations: 500,
            cg_tolerance: 1e-9,
            cg_q_tolerance: crate::linalg::schur::DEFAULT_ETA,
            preconditioner_type: SchurPreconditioner::default(),
            jacobian: None,
            jacobian_pattern: None,
            gradient: None,
            workspace_rows: Vec::new(),
            workspace_lm: Vec::new(),
            workspace_cam: Vec::new(),
            block_scratch: Vec::new(),
            structure: StructureCache::default(),
        }
    }

    /// Set the PCG iteration cap and relative residual tolerance.
    pub fn with_cg_config(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_cg_iterations = max_iterations;
        self.cg_tolerance = tolerance;
        self
    }

    /// Select the preconditioner — see the table in the module documentation.
    ///
    /// Mirrors
    /// [`ExplicitSparseSchur::with_preconditioner`](super::explicit::ExplicitSparseSchur::with_preconditioner),
    /// so the two PCG-based solvers are configured the same way.
    pub fn with_preconditioner(mut self, preconditioner: SchurPreconditioner) -> Self {
        self.preconditioner_type = preconditioner;
        self
    }

    /// Set the PCG forcing-sequence parameter η (`0.0` disables the
    /// quadratic-model stopping rule).
    pub fn with_cg_q_tolerance(mut self, q_tolerance: f64) -> Self {
        self.cg_q_tolerance = q_tolerance;
        self
    }

    /// Set how eliminated variables are recognized.
    pub fn with_ordering(mut self, ordering: SchurOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Construct with explicit CG parameters, keeping the default
    /// preconditioner.
    pub fn with_cg_params(max_iterations: usize, tolerance: f64) -> Self {
        Self::new().with_cg_config(max_iterations, tolerance)
    }

    /// Construct with explicit CG parameters and preconditioner.
    pub fn with_config(
        max_iterations: usize,
        tolerance: f64,
        preconditioner: SchurPreconditioner,
    ) -> Self {
        Self::new()
            .with_cg_config(max_iterations, tolerance)
            .with_preconditioner(preconditioner)
    }

    /// The partition, once [`StructureAware::initialize_structure`] has run.
    pub fn partition(&self) -> Option<&SchurPartition> {
        self.partition.as_ref()
    }

    fn require_partition(&self) -> LinAlgResult<&SchurPartition> {
        self.partition.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput(
                "Schur solver used before initialize_structure; the partition is unknown".into(),
            )
            .log()
        })
    }

    /// Rebuild the structural caches if `J`'s sparsity changed.
    ///
    /// Also enforces the elimination precondition against `J` directly: a
    /// residual row touching two eliminated variables means `EᵀE` is not
    /// block-diagonal, so inverting it blockwise would silently produce a wrong
    /// step. `SchurPartition::verify_block_diagonal` makes the same check
    /// against `JᵀJ`, which this solver never forms.
    fn ensure_structure(&mut self, jacobian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let fingerprint = pattern::PatternFingerprint::of(jacobian);
        if self.structure.fingerprint == Some(fingerprint) {
            return Ok(());
        }

        let partition = self.require_partition()?;
        let symbolic = jacobian.symbolic();

        // Row -> owning eliminated block, and each block's sorted row set.
        let mut row_owner = vec![usize::MAX; jacobian.nrows()];
        let mut eliminated_rows = vec![Vec::new(); partition.eliminated_blocks().len()];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            for offset in 0..block.dof {
                for &row in symbolic.row_idx_of_col_raw(block.col_start + offset) {
                    match row_owner[row] {
                        usize::MAX => {
                            row_owner[row] = block_idx;
                            eliminated_rows[block_idx].push(row);
                        }
                        owned if owned == block_idx => {}
                        owned => {
                            let other = partition.eliminated_blocks()[owned];
                            return Err(LinAlgError::InvalidInput(format!(
                                "variables {:?} and {:?} are both marked for elimination but \
                                 share residual row {row}, so EᵀE is not block-diagonal and \
                                 the elimination would give a wrong step; eliminate only \
                                 mutually unconnected variables",
                                block.key, other.key
                            ))
                            .log());
                        }
                    }
                }
            }
        }
        for rows in &mut eliminated_rows {
            rows.sort_unstable();
        }

        // Eliminated blocks visible from each kept block, via shared rows.
        let visibility: Vec<Vec<usize>> = partition
            .kept_blocks()
            .par_iter()
            .map(|block| {
                let mut seen = Vec::new();
                for offset in 0..block.dof {
                    for &row in symbolic.row_idx_of_col_raw(block.col_start + offset) {
                        let owner = row_owner[row];
                        if owner != usize::MAX {
                            seen.push(owner);
                        }
                    }
                }
                seen.sort_unstable();
                seen.dedup();
                seen
            })
            .collect();

        let mut kept_cols = vec![0usize; partition.kept_dof()];
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                if let Some(local) = partition.kept_local(col) {
                    kept_cols[local] = col;
                }
            }
        }
        let mut eliminated_cols = vec![0usize; partition.eliminated_dof()];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                eliminated_cols[base + offset] = block.col_start + offset;
            }
        }

        let chunk_rows = row_chunk_len(jacobian.nrows());
        let spans = RowSpans::build(jacobian, partition, chunk_rows)?;

        self.structure = StructureCache {
            kept_cols,
            eliminated_cols,
            eliminated_rows,
            visibility,
            chunk_rows,
            kept_spans: spans.kept,
            eliminated_spans: spans.eliminated,
            eliminated_block_of: spans.block_of,
            fingerprint: Some(fingerprint),
        };
        Ok(())
    }

    /// `S·v = Fᵀ(F·v − E·(EᵀE+λD_e)⁻¹·Eᵀ·F·v) + λD_k·v`, reading only `J`.
    ///
    /// `damp_kept` carries `λ·D_k` per retained local column, already clamped;
    /// it is empty for an undamped solve. The eliminated side's damping is
    /// baked into the inverted blocks before this is ever called.
    #[allow(clippy::too_many_arguments)]
    fn apply_schur_operator(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        damp_kept: &[f64],
        v: &Mat<f64>,
        out: &mut Mat<f64>,
        out_buf: &mut [f64],
        rows: &mut [f64],
        temp_lm: &mut [f64],
        scratch: &mut [f64],
    ) {
        let symbolic = jacobian.symbolic();
        let row_idx = symbolic.row_idx();
        let values = jacobian.val();
        let chunk_rows = self.structure.chunk_rows.max(1);

        // y = F·v — per row chunk, replaying the column-wise scatter's order.
        rows.par_chunks_mut(chunk_rows)
            .zip(self.structure.kept_spans.par_iter())
            .enumerate()
            .for_each(|(chunk, (y, spans))| {
                y.fill(0.0);
                let first = chunk * chunk_rows;
                for span in spans {
                    let x = v[(span.local as usize, 0)];
                    if x == 0.0 {
                        continue;
                    }
                    for k in span.lo as usize..span.hi as usize {
                        y[row_idx[k] - first] += values[k] * x;
                    }
                }
            });

        // t = Eᵀ·y — a gather, so it parallelizes over the output entries.
        let rows_ref: &[f64] = rows;
        temp_lm
            .par_iter_mut()
            .zip(self.structure.eliminated_cols.par_iter())
            .for_each(|(t, &col)| {
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                *t = idx
                    .iter()
                    .zip(vals)
                    .map(|(&row, val)| val * rows_ref[row])
                    .sum();
            });

        // u = (EᵀE + λD_e)⁻¹·t — one output entry per task: entry `base + r`
        // of block `b` is row `r` of that block's inverse times its slice of t.
        let t: &[f64] = temp_lm;
        let eliminated = &self.eliminated;
        let block_of = &self.structure.eliminated_block_of;
        scratch.par_iter_mut().enumerate().for_each(|(i, u)| {
            let block_idx = block_of[i] as usize;
            let dof = eliminated.dof(block_idx);
            let base = partition.eliminated_offset(block_idx);
            let inv = eliminated.block(block_idx);
            let r = i - base;
            let mut acc = 0.0;
            for c in 0..dof {
                acc += inv[c * dof + r] * t[base + c];
            }
            *u = acc;
        });

        // y ← y − E·u, again per row chunk in the scatter's order.
        let u: &[f64] = scratch;
        rows.par_chunks_mut(chunk_rows)
            .zip(self.structure.eliminated_spans.par_iter())
            .enumerate()
            .for_each(|(chunk, (y, spans))| {
                let first = chunk * chunk_rows;
                for span in spans {
                    let u = u[span.local as usize];
                    if u == 0.0 {
                        continue;
                    }
                    for k in span.lo as usize..span.hi as usize {
                        y[row_idx[k] - first] -= values[k] * u;
                    }
                }
            });

        // S·v = Fᵀ·y + λD_k·v — also a gather, also parallel over outputs.
        // The result lands in `out_buf` and is copied into `out` afterwards:
        // `Mat` does not hand out a mutable slice rayon can split.
        let rows_ref: &[f64] = rows;
        out_buf
            .par_iter_mut()
            .enumerate()
            .zip(self.structure.kept_cols.par_iter())
            .for_each(|((local, o), &col)| {
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc: f64 = idx
                    .iter()
                    .zip(vals)
                    .map(|(&row, val)| val * rows_ref[row])
                    .sum();
                if let Some(d) = damp_kept.get(local) {
                    acc += d * v[(local, 0)];
                }
                *o = acc;
            });
        for (local, value) in out_buf.iter().enumerate() {
            out[(local, 0)] = *value;
        }
    }

    /// Dense `|rows| × dof` strip of `J` over `rows`, for the columns of
    /// `block`.
    ///
    /// `rows` is sorted and each CSC column's row indices are sorted, so each
    /// entry is one binary search. Rows the column does not touch stay zero.
    fn gather_strip(
        jacobian: &SparseColMat<usize, f64>,
        block: &BlockSpan,
        rows: &[usize],
    ) -> DMatrix<f64> {
        let symbolic = jacobian.symbolic();
        let mut strip = DMatrix::zeros(rows.len(), block.dof);
        for offset in 0..block.dof {
            let col = block.col_start + offset;
            let idx = symbolic.row_idx_of_col_raw(col);
            let vals = jacobian.val_of_col(col);
            for (local_row, &row) in rows.iter().enumerate() {
                if let Ok(k) = idx.binary_search(&row) {
                    strip[(local_row, offset)] = vals[k];
                }
            }
        }
        strip
    }

    /// Build the PCG preconditioner blocks from `J`.
    ///
    /// See the table in the module documentation for what each option is. All
    /// of them produce one dense `dof × dof` inverse per retained variable,
    /// or `None` for [`SchurPreconditioner::None`]. A block that will not
    /// invert falls back to identity: a preconditioner is a convergence aid,
    /// so a singular block must not fail the solve.
    fn build_preconditioner(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        damp_kept: &[f64],
    ) -> Option<Vec<DMatrix<f64>>> {
        if self.preconditioner_type == SchurPreconditioner::None {
            return None;
        }

        let blocks = partition
            .kept_blocks()
            .par_iter()
            .enumerate()
            .map(|(kept_idx, block)| {
                let dof = block.dof;
                let base = partition.kept_offset(kept_idx);

                // (FᵀF + λD_k) restricted to this block.
                let mut s_ii = DMatrix::zeros(dof, dof);
                for c in 0..dof {
                    for r in 0..dof {
                        s_ii[(r, c)] =
                            column_dot(jacobian, block.col_start + r, block.col_start + c);
                    }
                    if let Some(d) = damp_kept.get(base + c) {
                        s_ii[(c, c)] += d;
                    }
                }

                // Schur-Jacobi additionally subtracts the elimination
                // correction, which is what makes it a preconditioner for `S`
                // rather than for the unreduced retained block.
                if self.preconditioner_type == SchurPreconditioner::SchurJacobi {
                    let visible = self
                        .structure
                        .visibility
                        .get(kept_idx)
                        .map_or(&[][..], Vec::as_slice);
                    for &elim_idx in visible {
                        let elim_block = partition.eliminated_blocks()[elim_idx];
                        let rows = &self.structure.eliminated_rows[elim_idx];
                        let f_strip = Self::gather_strip(jacobian, block, rows);
                        let e_strip = Self::gather_strip(jacobian, &elim_block, rows);

                        // m = F_iᵀ·E_j, the (i, j) coupling block of `FᵀE`.
                        let m = f_strip.transpose() * e_strip;
                        let elim_dof = elim_block.dof;
                        let inv = DMatrix::from_column_slice(
                            elim_dof,
                            elim_dof,
                            self.eliminated.block(elim_idx),
                        );
                        s_ii -= &m * inv * m.transpose();
                    }
                }

                invert_with_retry_dyn(&s_ii).unwrap_or_else(|| DMatrix::identity(dof, dof))
            })
            .collect();
        Some(blocks)
    }

    /// `z = M⁻¹·r` for the block-diagonal preconditioner, or `z = r` without one.
    fn apply_preconditioner(blocks: Option<&[DMatrix<f64>]>, r: &Mat<f64>, z: &mut Mat<f64>) {
        let Some(blocks) = blocks else {
            faer::zip!(z, r).for_each(|faer::unzip!(z, r)| *z = *r);
            return;
        };
        let mut offset = 0usize;
        for inv in blocks {
            let dof = inv.nrows();
            for row in 0..dof {
                let mut acc = 0.0;
                for col in 0..dof {
                    acc += inv[(row, col)] * r[(offset + col, 0)];
                }
                z[(offset + row, 0)] = acc;
            }
            offset += dof;
        }
    }

    /// The whole solve, from `J` and `r` to the full-length update.
    ///
    /// `damping` is `None` for the plain normal equations.
    fn solve_from_jacobian(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: Option<&Damping>,
    ) -> LinAlgResult<Mat<f64>> {
        self.ensure_structure(jacobian)?;
        let partition = self.require_partition()?.clone();

        if jacobian.ncols() != partition.total_dof() {
            return Err(LinAlgError::InvalidInput(format!(
                "Jacobian has {} columns but the partition covers {}",
                jacobian.ncols(),
                partition.total_dof()
            ))
            .log());
        }

        // g = Jᵀr; the reduced system is built from −g, as in every other
        // Schur solver here.
        let gradient = jt_vec(jacobian, residuals);
        let kept_dof = partition.kept_dof();
        let eliminated_dof = partition.eliminated_dof();

        // λ·D per column, from diag(JᵀJ) — the only part of `JᵀJ` needed.
        let damp_kept: Vec<f64> = match damping {
            None => Vec::new(),
            Some(d) => {
                let diag = diag_jt_j(jacobian);
                let mut per_kept = vec![0.0; kept_dof];
                for block in partition.kept_blocks() {
                    for offset in 0..block.dof {
                        let col = block.col_start + offset;
                        if let Some(local) = partition.kept_local(col) {
                            per_kept[local] = d.diagonal_term(diag[col]);
                        }
                    }
                }
                per_kept
            }
        };

        // (EᵀE + λD_e)⁻¹, blockwise.
        let mut eliminated = std::mem::take(&mut self.eliminated);
        eliminated.gather_from_jacobian(jacobian, &partition);
        if let Some(d) = damping {
            eliminated.damp(d);
        }
        let inversion = eliminated.invert_in_place(&partition);
        self.eliminated = eliminated;
        inversion?;

        // g_reduced = −g_k + H_ke·H_ee⁻¹·g_e, assembled with `neg_gradient`
        // playing the role of `g` (so the reduced right-hand side matches the
        // explicit solvers' `g_k − H_ke·H_ee⁻¹·g_e` convention).
        let mut g_k = Mat::<f64>::zeros(kept_dof, 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                if let Some(local) = partition.kept_local(col) {
                    g_k[(local, 0)] = -gradient[(col, 0)];
                }
            }
        }
        let mut g_e = Mat::<f64>::zeros(eliminated_dof, 1);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                g_e[(base + offset, 0)] = -gradient[(block.col_start + offset, 0)];
            }
        }

        // correction = H_ke·H_ee⁻¹·g_e = Fᵀ·E·H_ee⁻¹·g_e, again through `J`.
        let mut hee_ge = Mat::<f64>::zeros(eliminated_dof, 1);
        self.apply_eliminated_inverse(&partition, &g_e, &mut hee_ge);
        let correction = self.coupling_times(&partition, jacobian, &hee_ge);

        let g_reduced = Mat::from_fn(kept_dof, 1, |i, _| g_k[(i, 0)] - correction[(i, 0)]);

        let precond = self.build_preconditioner(&partition, jacobian, &damp_kept);
        let delta_k = self.solve_pcg_block(
            &partition,
            jacobian,
            &damp_kept,
            &g_reduced,
            precond.as_deref(),
        );

        // δ_e = H_ee⁻¹·(g_e − H_keᵀ·δ_k) = H_ee⁻¹·(g_e − Eᵀ·F·δ_k)
        let coupling_t = self.coupling_transpose_times(&partition, jacobian, &delta_k);
        let rhs_e = Mat::from_fn(eliminated_dof, 1, |i, _| g_e[(i, 0)] - coupling_t[(i, 0)]);
        let mut delta_e = Mat::<f64>::zeros(eliminated_dof, 1);
        self.apply_eliminated_inverse(&partition, &rhs_e, &mut delta_e);

        let delta = self.combine_updates(&partition, &delta_k, &delta_e);

        self.gradient = Some(gradient);
        self.retain_jacobian(jacobian);
        Ok(delta)
    }

    /// Keep `J` for [`LinearSolver::hessian_vec_product`], reusing the existing
    /// allocation when the sparsity has not changed.
    ///
    /// Only the values move in that case — on the largest BAL problem a fresh
    /// clone would allocate and free ~2 GB per optimizer iteration. The
    /// fingerprint guards the reuse: identical nonzero counts alone would not
    /// prove the patterns match.
    fn retain_jacobian(&mut self, jacobian: &SparseColMat<usize, f64>) {
        let fingerprint = self.structure.fingerprint;
        let reusable = self.jacobian_pattern == fingerprint
            && self
                .jacobian
                .as_ref()
                .is_some_and(|held| held.val().len() == jacobian.val().len());

        match self.jacobian.as_mut() {
            Some(held) if reusable => held.val_mut().copy_from_slice(jacobian.val()),
            _ => {
                self.jacobian = Some(jacobian.clone());
                self.jacobian_pattern = fingerprint;
            }
        }
    }

    /// `x = H_ee⁻¹·b`, blockwise.
    fn apply_eliminated_inverse(&self, partition: &SchurPartition, b: &Mat<f64>, x: &mut Mat<f64>) {
        for block_idx in 0..partition.eliminated_blocks().len() {
            let dof = self.eliminated.dof(block_idx);
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * b[(base + c, 0)];
                }
                x[(base + r, 0)] = acc;
            }
        }
    }

    /// `H_ke·u = Fᵀ·(E·u)`, from `J`.
    fn coupling_times(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        u: &Mat<f64>,
    ) -> Mat<f64> {
        let symbolic = jacobian.symbolic();
        let mut rows = vec![0.0; jacobian.nrows()];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let x = u[(base + offset, 0)];
                if x == 0.0 {
                    continue;
                }
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] += vals[k] * x;
                }
            }
        }

        let mut out = Mat::<f64>::zeros(partition.kept_dof(), 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (k, &row) in idx.iter().enumerate() {
                    acc += vals[k] * rows[row];
                }
                out[(local, 0)] = acc;
            }
        }
        out
    }

    /// `H_keᵀ·v = Eᵀ·(F·v)`, from `J`.
    fn coupling_transpose_times(
        &self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        v: &Mat<f64>,
    ) -> Mat<f64> {
        let symbolic = jacobian.symbolic();
        let mut rows = vec![0.0; jacobian.nrows()];
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let x = v[(local, 0)];
                if x == 0.0 {
                    continue;
                }
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] += vals[k] * x;
                }
            }
        }

        let mut out = Mat::<f64>::zeros(partition.eliminated_dof(), 1);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (k, &row) in idx.iter().enumerate() {
                    acc += vals[k] * rows[row];
                }
                out[(base + offset, 0)] = acc;
            }
        }
        out
    }

    /// Solve `S·x = b` with the shared PCG primitive and the matrix-free operator.
    fn solve_pcg_block(
        &mut self,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        damp_kept: &[f64],
        b: &Mat<f64>,
        precond: Option<&[DMatrix<f64>]>,
    ) -> Mat<f64> {
        // Workspaces come out of `self` so the operator closure can borrow
        // `self` immutably alongside them.
        let mut rows = std::mem::take(&mut self.workspace_rows);
        let mut temp_lm = std::mem::take(&mut self.workspace_lm);
        let mut out_buf = std::mem::take(&mut self.workspace_cam);
        let mut scratch = std::mem::take(&mut self.block_scratch);
        out_buf.clear();
        out_buf.resize(partition.kept_dof(), 0.0);
        rows.clear();
        rows.resize(jacobian.nrows(), 0.0);
        temp_lm.clear();
        temp_lm.resize(partition.eliminated_dof(), 0.0);
        scratch.clear();
        scratch.resize(partition.eliminated_dof(), 0.0);

        let result = solve_pcg(
            b,
            &PcgParams::new(self.max_cg_iterations, self.cg_tolerance)
                .with_q_tolerance(self.cg_q_tolerance),
            |p, ap| {
                self.apply_schur_operator(
                    partition,
                    jacobian,
                    damp_kept,
                    p,
                    ap,
                    &mut out_buf,
                    &mut rows,
                    &mut temp_lm,
                    &mut scratch,
                );
            },
            |r, z| Self::apply_preconditioner(precond, r, z),
        );

        self.workspace_rows = rows;
        self.workspace_lm = temp_lm;
        self.workspace_cam = out_buf;
        self.block_scratch = scratch;

        // The PCG iteration count is the whole cost of this solver — each one
        // is four passes over `nnz(J)` — so it is the first number to look at
        // when tuning `cg_tolerance` or comparing preconditioners.
        debug!(
            "PCG: {} iterations, residual {:.3e}, {:?} ({:?})",
            result.iterations, result.final_residual, result.termination, self.preconditioner_type
        );
        result.x
    }

    /// Scatter the two solution halves back into one full-length update.
    fn combine_updates(
        &self,
        partition: &SchurPartition,
        delta_k: &Mat<f64>,
        delta_e: &Mat<f64>,
    ) -> Mat<f64> {
        let mut delta = Mat::zeros(partition.total_dof(), 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let global = block.col_start + offset;
                if let Some(local) = partition.kept_local(global) {
                    delta[(global, 0)] = delta_k[(local, 0)];
                }
            }
        }
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                delta[(block.col_start + offset, 0)] = delta_e[(base + offset, 0)];
            }
        }
        delta
    }
}

impl Default for ImplicitSparseSchur {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSolver<SparseMode> for ImplicitSparseSchur {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_from_jacobian(residuals, jacobian, None)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_from_jacobian(residuals, jacobian, Some(damping))
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        // `JᵀJ` was never formed, so evaluate its action from `J`.
        Some(jt_j_vec_product(self.jacobian.as_ref()?, v))
    }

    /// Always `None`: this solver never materializes `JᵀJ`.
    ///
    /// That is the documented degradation for a matrix-free backend — callers
    /// use [`LinearSolver::hessian_vec_product`] instead, which is served
    /// exactly from `J`.
    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        None
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

impl StructureAware for ImplicitSparseSchur {
    fn initialize_structure(
        &mut self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        schur_landmark_keys: &std::collections::HashSet<VarKey>,
    ) -> LinAlgResult<()> {
        let effective_keys =
            effective_landmark_keys(variables, schur_landmark_keys, &self.ordering);

        let mut kept = Vec::new();
        let mut eliminated = Vec::new();

        for (key, variable) in variables {
            let col_start = *variable_index_map.get(key).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("VarKey {:?} not in index map", key))
            })?;
            let span = BlockSpan {
                key,
                col_start,
                // Free columns only — fixed tangent coordinates own no
                // column in the linear solve (ISSUE-0003).
                dof: variable.free_dof(),
            };
            if effective_keys.contains(&key) {
                eliminated.push(span);
            } else {
                kept.push(span);
            }
        }

        let partition = SchurPartition::new(kept, eliminated)?;
        self.max_eliminated_dof = partition
            .eliminated_blocks()
            .iter()
            .map(|b| b.dof)
            .max()
            .unwrap_or(0)
            .max(1);
        self.eliminated = EliminatedBlocks::new(&partition);
        self.workspace_lm = vec![0.0; partition.eliminated_dof()];
        self.workspace_cam = vec![0.0; partition.kept_dof()];
        self.block_scratch = vec![0.0; self.max_eliminated_dof];
        self.workspace_rows.clear();
        self.structure = StructureCache::default();
        self.partition = Some(partition);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::VarKey;
    use crate::core::variable::Variable;
    use apex_manifolds::{LieGroup, rn, se3};
    use faer::sparse::Triplet;
    use nalgebra::DVector;
    use slotmap::{SecondaryMap, SlotMap};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    type TestSetup = (
        SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        SparseColMat<usize, f64>,
        faer::Mat<f64>,
        std::collections::HashSet<VarKey>,
    );

    /// Build the same 2-camera + 3-landmark test setup used in explicit tests.
    /// Jacobian: 36 rows × 21 cols  (H_cc = 3·I₁₂, H_pp = 4·I₃ — positive definite)
    fn create_schur_test_setup() -> Result<TestSetup, Box<dyn std::error::Error>> {
        let se3_id = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let pt_zero = DVector::from_vec(vec![0.0, 0.0, 0.0]);

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let cam0 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_id.as_slice(),
        ))));
        let cam1 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_id.as_slice(),
        ))));
        let pt0 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));
        let pt1 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));
        let pt2 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));

        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(cam0, 0);
        variable_index_map.insert(cam1, 6);
        variable_index_map.insert(pt0, 12);
        variable_index_map.insert(pt1, 15);
        variable_index_map.insert(pt2, 18);

        let cam_cols = [0usize, 6];
        let lm_cols = [12usize, 15, 18];
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for (ci, &cam_col) in cam_cols.iter().enumerate() {
            for (li, &lm_col) in lm_cols.iter().enumerate() {
                let row_base = (ci * 3 + li) * 6;
                for k in 0..6 {
                    triplets.push(Triplet::new(row_base + k, cam_col + k, 1.0));
                    triplets.push(Triplet::new(row_base + k, lm_col + (k % 3), 1.0));
                }
            }
        }

        let jacobian = SparseColMat::try_new_from_triplets(36, 21, &triplets)?;
        let residuals = faer::Mat::from_fn(36, 1, |i, _| (i % 5) as f64 * 0.1);

        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(pt0);
        landmark_keys.insert(pt1);
        landmark_keys.insert(pt2);

        Ok((
            variables,
            variable_index_map,
            jacobian,
            residuals,
            landmark_keys,
        ))
    }

    #[test]
    fn test_implicit_schur_creation() {
        let solver = ImplicitSparseSchur::new();
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_custom_params() {
        let solver = ImplicitSparseSchur::with_cg_params(100, 1e-8);
        assert_eq!(solver.max_cg_iterations, 100);
        assert_eq!(solver.cg_tolerance, 1e-8);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_full_config() {
        let solver =
            ImplicitSparseSchur::with_config(200, 1e-10, SchurPreconditioner::BlockDiagonal);
        assert_eq!(solver.max_cg_iterations, 200);
        assert_eq!(solver.cg_tolerance, 1e-10);
        assert_eq!(
            solver.preconditioner_type,
            SchurPreconditioner::BlockDiagonal
        );
    }

    #[test]
    fn test_implicit_schur_default() {
        let solver = ImplicitSparseSchur::default();
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert!(solver.partition.is_none());
        assert!(solver.jacobian.is_none());
    }

    #[test]
    fn test_implicit_schur_initialize_structure() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let partition = solver.partition.as_ref().ok_or("partition is None")?;
        assert_eq!(partition.kept_blocks().len(), 2);
        assert_eq!(partition.eliminated_blocks().len(), 3);
        assert_eq!(partition.kept_dof(), 12);
        assert_eq!(partition.eliminated_dof(), 9);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_normal_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        assert_eq!(delta.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_augmented_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(0.1),
        )?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_get_hessian_gradient() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_none());
        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_none());

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // `JᵀJ` is never formed, so `get_hessian` stays `None` by design and
        // the quadratic model is served through `hessian_vec_product`.
        assert!(
            LinearSolver::<SparseMode>::get_hessian(&solver).is_none(),
            "matrix-free solver must not publish a Hessian"
        );
        let g = LinearSolver::<SparseMode>::get_gradient(&solver).ok_or("gradient is None")?;
        assert_eq!(g.nrows(), 21);

        let v = faer::Mat::from_fn(21, 1, |i, _| ((i % 3) as f64) - 1.0);
        let hv = LinearSolver::<SparseMode>::hessian_vec_product(&solver, &v)
            .ok_or("hessian_vec_product is None")?;
        assert_eq!(hv.nrows(), 21);
        Ok(())
    }

    /// A preconditioner changes how fast PCG converges, never what it
    /// converges to, so all three must produce the same step on the same
    /// system. This is what makes each option's implementation testable: a
    /// preconditioner that was silently ignored, or one built wrongly, shows
    /// up here as a disagreement or a failure to converge.
    #[test]
    fn every_preconditioner_reaches_the_same_step() -> TestResult {
        let mut steps = Vec::new();
        for preconditioner in [
            SchurPreconditioner::None,
            SchurPreconditioner::BlockDiagonal,
            SchurPreconditioner::SchurJacobi,
        ] {
            let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
                create_schur_test_setup()?;
            let mut solver = ImplicitSparseSchur::new()
                .with_cg_config(1000, 1e-12)
                .with_preconditioner(preconditioner);
            solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
            steps.push(LinearSolver::<SparseMode>::solve_normal_equation(
                &mut solver,
                &residuals,
                &jacobian,
            )?);
        }

        let reference = &steps[0];
        for (idx, step) in steps.iter().enumerate().skip(1) {
            for row in 0..reference.nrows() {
                let scale = reference[(row, 0)].abs().max(1.0);
                assert!(
                    (reference[(row, 0)] - step[(row, 0)]).abs() / scale < 1e-6,
                    "preconditioner {idx} diverged at row {row}: {} vs {}",
                    step[(row, 0)],
                    reference[(row, 0)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_implicit_schur_block_diagonal_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver =
            ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::BlockDiagonal);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_schur_jacobi_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver =
            ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::SchurJacobi);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_no_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::None);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_augmented_lambda_effect() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;

        let mut s1 = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        s1.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let d1 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut s1,
            &residuals,
            &jacobian,
            &Damping::identity(0.001),
        )?;

        let mut s2 = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        s2.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let d2 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut s2,
            &residuals,
            &jacobian,
            &Damping::identity(100.0),
        )?;

        let norm_diff: f64 = (0..21).map(|i| (d1[(i, 0)] - d2[(i, 0)]).powi(2)).sum();
        assert!(
            norm_diff > 1e-10,
            "Different λ should yield different updates"
        );
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_without_init_returns_error() -> TestResult {
        let triplets: Vec<Triplet<usize, usize, f64>> = vec![Triplet::new(0, 0, 1.0)];
        let jacobian =
            SparseColMat::try_new_from_triplets(1, 1, &triplets).map_err(|e| format!("{e:?}"))?;
        let residuals = faer::Mat::from_fn(1, 1, |_, _| 1.0);
        let mut solver = ImplicitSparseSchur::new();

        let result =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_implicit_initialize_structure_no_cameras_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::rn;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(rn::Rn::new(DVector::zeros(3)))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(k);

        let mut solver = ImplicitSparseSchur::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no retained variables present"
        );
    }

    #[test]
    fn test_implicit_initialize_structure_no_landmarks_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::se3;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let landmark_keys = std::collections::HashSet::<VarKey>::new();

        let mut solver = ImplicitSparseSchur::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no eliminated variables present"
        );
    }

    /// A mixed-DOF, non-contiguous partition (1-DOF inverse depth interleaved
    /// with retained poses) must work end to end — the whole point of moving
    /// this solver onto [`SchurPartition`].
    #[test]
    fn test_implicit_schur_supports_non_contiguous_mixed_dof_partition() -> TestResult {
        use crate::linalg::schur::BlockSpan;

        // Layout: kept A [0..6), eliminated P [6..7) (1-DOF), kept B [7..13),
        // eliminated Q [13..14) (1-DOF). Neither side is contiguous.
        let a = Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]));
        let b = Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]));
        let p = Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0])));
        let q = Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0])));

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let key_a = variables.insert(Box::new(a));
        let key_p = variables.insert(Box::new(p));
        let key_b = variables.insert(Box::new(b));
        let key_q = variables.insert(Box::new(q));

        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(key_a, 0);
        variable_index_map.insert(key_p, 6);
        variable_index_map.insert(key_b, 7);
        variable_index_map.insert(key_q, 13);

        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(key_p);
        landmark_keys.insert(key_q);

        // Observations: A-P, B-P, A-Q, B-Q, plus a small prior on every
        // column so the system is SPD.
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut row = 0usize;
        for &cam_col in &[0usize, 7] {
            for &lm_col in &[6usize, 13] {
                for k in 0..1 {
                    triplets.push(Triplet::new(row + k, cam_col + k, 1.0));
                    triplets.push(Triplet::new(row + k, lm_col, 1.0));
                }
                row += 1;
            }
        }
        for col in 0..14 {
            triplets.push(Triplet::new(row, col, 0.5));
            row += 1;
        }
        let jacobian = SparseColMat::try_new_from_triplets(row, 14, &triplets)?;
        let residuals = faer::Mat::from_fn(row, 1, |i, _| 0.1 * (i as f64 + 1.0));

        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-10);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 14);
        for i in 0..14 {
            assert!(delta[(i, 0)].is_finite());
        }

        // A block spanning a variable never claimed by anything must still
        // round-trip through BlockSpan without panicking (sanity check on the
        // type used to build the partition above).
        let _ = BlockSpan {
            key: key_a,
            col_start: 0,
            dof: 6,
        };
        Ok(())
    }

    /// The column-by-column operator the row-chunked passes replaced, kept
    /// verbatim as the reference they must reproduce bit for bit.
    fn column_scatter_reference(
        solver: &ImplicitSparseSchur,
        partition: &SchurPartition,
        jacobian: &SparseColMat<usize, f64>,
        v: &Mat<f64>,
    ) -> Vec<f64> {
        let symbolic = jacobian.symbolic();
        let mut rows = vec![0.0; jacobian.nrows()];
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local) = partition.kept_local(col) else {
                    continue;
                };
                let x = v[(local, 0)];
                if x == 0.0 {
                    continue;
                }
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] += vals[k] * x;
                }
            }
        }
        let mut t = vec![0.0; partition.eliminated_dof()];
        for (slot, &col) in t.iter_mut().zip(&solver.structure.eliminated_cols) {
            let idx = symbolic.row_idx_of_col_raw(col);
            let vals = jacobian.val_of_col(col);
            *slot = idx
                .iter()
                .zip(vals)
                .map(|(&row, val)| val * rows[row])
                .sum();
        }
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let dof = solver.eliminated.dof(block_idx);
            let base = partition.eliminated_offset(block_idx);
            let inv = solver.eliminated.block(block_idx);
            let mut u = vec![0.0; dof];
            for (r, out) in u.iter_mut().enumerate() {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * t[base + c];
                }
                *out = acc;
            }
            for (offset, &u) in u.iter().enumerate() {
                if u == 0.0 {
                    continue;
                }
                let col = block.col_start + offset;
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                for (k, &row) in idx.iter().enumerate() {
                    rows[row] -= vals[k] * u;
                }
            }
        }
        solver
            .structure
            .kept_cols
            .iter()
            .map(|&col| {
                let idx = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                idx.iter()
                    .zip(vals)
                    .map(|(&row, val)| val * rows[row])
                    .sum()
            })
            .collect()
    }

    /// Row-chunked `F·v` and `E·u` must equal the column-wise scatter exactly,
    /// for every chunk length — including ones that split a column's rows
    /// and rows that two retained blocks share, where addition order shows.
    #[test]
    fn row_chunked_operator_is_bit_identical_to_column_scatter() -> TestResult {
        let cam = || -> Box<dyn ManifoldVariable> {
            Box::new(Variable::new(se3::SE3::from_param_slice(&[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])))
        };
        let point = || -> Box<dyn ManifoldVariable> {
            Box::new(Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0; 3]))))
        };
        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let cams: Vec<VarKey> = (0..3).map(|_| variables.insert(cam())).collect();
        let points: Vec<VarKey> = (0..4).map(|_| variables.insert(point())).collect();
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        for (i, &k) in cams.iter().enumerate() {
            variable_index_map.insert(k, i * 6);
        }
        for (i, &k) in points.iter().enumerate() {
            variable_index_map.insert(k, 18 + i * 3);
        }
        let landmark_keys: std::collections::HashSet<VarKey> = points.iter().copied().collect();

        // Every (camera, point) pair observed, two rows each; odd rows also
        // touch the next camera, so rows carry two retained blocks. Values are
        // irrational-ish so a reordered sum would change low bits.
        let value = |r: usize, c: usize| ((r * 31 + c * 17) as f64 * 0.618_033_988_7).sin() + 1.5;
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut row = 0usize;
        for p in 0..4 {
            for c in 0..3 {
                for _ in 0..2 {
                    for k in 0..6 {
                        triplets.push(Triplet::new(row, c * 6 + k, value(row, c * 6 + k)));
                        if row % 2 == 1 {
                            let other = ((c + 1) % 3) * 6 + k;
                            triplets.push(Triplet::new(row, other, value(row, other)));
                        }
                    }
                    for k in 0..3 {
                        let col = 18 + p * 3 + k;
                        triplets.push(Triplet::new(row, col, value(row, col)));
                    }
                    row += 1;
                }
            }
        }
        let jacobian = SparseColMat::try_new_from_triplets(row, 30, &triplets)?;

        let mut solver = ImplicitSparseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        solver.ensure_structure(&jacobian)?;
        let partition = solver.require_partition()?.clone();
        solver
            .eliminated
            .gather_from_jacobian(&jacobian, &partition);
        solver.eliminated.invert_in_place(&partition)?;

        let v = Mat::from_fn(partition.kept_dof(), 1, |i, _| {
            (i as f64 * 0.37).cos() - 0.2
        });
        let want = column_scatter_reference(&solver, &partition, &jacobian, &v);

        for chunk_rows in [1usize, 2, 3, 5, 7, row, row + 4] {
            let spans = RowSpans::build(&jacobian, &partition, chunk_rows)?;
            solver.structure.chunk_rows = chunk_rows;
            solver.structure.kept_spans = spans.kept;
            solver.structure.eliminated_spans = spans.eliminated;
            solver.structure.eliminated_block_of = spans.block_of;

            let mut out = Mat::zeros(partition.kept_dof(), 1);
            let mut out_buf = vec![0.0; partition.kept_dof()];
            let mut rows = vec![0.0; jacobian.nrows()];
            let mut temp_lm = vec![0.0; partition.eliminated_dof()];
            let mut scratch = vec![0.0; partition.eliminated_dof()];
            solver.apply_schur_operator(
                &partition,
                &jacobian,
                &[],
                &v,
                &mut out,
                &mut out_buf,
                &mut rows,
                &mut temp_lm,
                &mut scratch,
            );
            for (i, w) in want.iter().enumerate() {
                assert_eq!(
                    out[(i, 0)].to_bits(),
                    w.to_bits(),
                    "chunk_rows={chunk_rows}, entry {i}: {} vs {w}",
                    out[(i, 0)]
                );
            }
        }
        Ok(())
    }
}
