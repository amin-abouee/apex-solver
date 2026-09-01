//! Chunk-wise Schur elimination: from `J` straight to the reduced system.
//!
//! The other Schur path forms `H = JᵀJ` and then partitions it. That costs
//! `Jᵀ`, a value permutation and `JᵀJ` itself before elimination even starts —
//! about 32 GB on the largest BAL problem, against 8.4 GB for `J` alone.
//!
//! This eliminator never forms `JᵀJ`. Following Ceres's `SchurEliminator`, it
//! walks *chunks* — the rows sharing one eliminated variable — and accumulates
//! only small dense quantities per chunk:
//!
//! ```text
//! per chunk c, with E = J[rows_c, e-cols] and F = J[rows_c, kept-cols]:
//!     ete   = Eᵀ E          (dof × dof)
//!     etf   = Eᵀ F          (dof × |F_c|)
//!     etr   = Eᵀ r_c        (dof)
//!
//!     S     -= etfᵀ · ete⁻¹ · etf
//!     g_red -= etfᵀ · ete⁻¹ · etr
//! ```
//!
//! then the scratch is reused for the next chunk. `ete⁻¹` is retained because
//! back-substitution needs it; everything else is transient.
//!
//! # Why this can read a column-major `J`
//!
//! A chunk is a set of *rows*, and `J` arrives as CSC. Reading rows out of a
//! column-major matrix would normally mean building a CSR copy — which would
//! cost as much as `J` and undo the saving. Two facts make it unnecessary:
//!
//! 1. Each residual row touches at most one eliminated variable, so chunks
//!    **partition** the rows. This is the same precondition that
//!    [`SchurPartition::verify_block_diagonal`] enforces.
//! 2. Row indices within a CSC column are sorted.
//!
//! So when the chunks are swept in increasing row order, each kept column can
//! carry a cursor into its `row_idx` array that only ever moves forward. Every
//! entry of `J` is visited once across the whole sweep — O(nnz) total, with one
//! cursor per kept column as the only extra memory.
//!
//! That ordering is a **precondition**: chunk row ranges must be contiguous and
//! increasing. [`ChunkLayout::build`] checks it rather than assuming it.

use faer::Mat;
use faer::sparse::SparseColMat;

use super::schur_partition::{EliminatedBlocks, SchurPartition};
use crate::error::ErrorLogging;
use crate::linalg::{Damping, LinAlgError, LinAlgResult};

/// Which rows belong to which eliminated variable.
///
/// Built once per sparsity pattern: the row ranges follow the Jacobian's
/// structure, not its values.
#[derive(Debug, Clone, Default)]
pub struct ChunkLayout {
    /// `(row_start, row_end)` per eliminated block, in increasing row order.
    ranges: Vec<(usize, usize)>,
    /// Retained *local* columns touched by each chunk, ascending.
    ///
    /// Flattened into one buffer with `col_spans` delimiting each chunk.
    /// Without this the sweep would test every retained column against every
    /// chunk — 1.6 billion checks on Ladybug — which dominates everything else.
    chunk_cols: Vec<u32>,
    /// `(start, len)` into `chunk_cols` per chunk.
    col_spans: Vec<(usize, usize)>,
    /// Total rows of the Jacobian the layout was built from.
    nrows: usize,
}

impl ChunkLayout {
    /// Derive the chunk row ranges from `J`'s pattern.
    ///
    /// Errors when a chunk's rows are not contiguous, or when chunks are not in
    /// increasing row order — the sweep below depends on both.
    pub fn build(
        jacobian: &SparseColMat<usize, f64>,
        partition: &SchurPartition,
    ) -> LinAlgResult<Self> {
        let symbolic = jacobian.symbolic();
        let mut ranges = Vec::with_capacity(partition.eliminated_blocks().len());

        for block in partition.eliminated_blocks() {
            // Union of rows over the block's columns. Each column's indices are
            // sorted, so min/max over them bounds the chunk.
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            let mut count = 0usize;
            for offset in 0..block.dof {
                let rows = symbolic.row_idx_of_col_raw(block.col_start + offset);
                if let (Some(&first), Some(&last)) = (rows.first(), rows.last()) {
                    lo = lo.min(first);
                    hi = hi.max(last + 1);
                }
                count += rows.len();
            }

            if count == 0 {
                return Err(LinAlgError::InvalidInput(format!(
                    "variable {:?} is marked for elimination but appears in no residual; \
                     it cannot be eliminated",
                    block.key
                ))
                .log());
            }
            ranges.push((lo, hi));
        }

        // The sweep needs disjoint ranges in increasing order.
        let mut previous_end = 0usize;
        for (idx, &(lo, hi)) in ranges.iter().enumerate() {
            if lo < previous_end {
                return Err(LinAlgError::InvalidInput(format!(
                    "chunk {idx} starts at row {lo}, before the previous chunk ended at \
                     {previous_end}; chunk-wise elimination needs each eliminated variable's \
                     rows to form a contiguous, increasing range"
                ))
                .log());
            }
            previous_end = hi;
        }

        // Which retained columns each chunk touches. One pass over the
        // retained columns, bucketing their rows into chunks by binary search.
        let mut per_chunk: Vec<Vec<u32>> = vec![Vec::new(); ranges.len()];
        let mut local = 0u32;
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let rows = symbolic.row_idx_of_col_raw(block.col_start + offset);
                let mut last: Option<usize> = None;
                for &row in rows {
                    // Chunks are disjoint and increasing, so a row maps to at
                    // most one of them.
                    let found = ranges.partition_point(|&(_, hi)| hi <= row);
                    if found < ranges.len() && row >= ranges[found].0 && last != Some(found) {
                        per_chunk[found].push(local);
                        last = Some(found);
                    }
                }
                local += 1;
            }
        }

        let mut chunk_cols = Vec::new();
        let mut col_spans = Vec::with_capacity(ranges.len());
        for cols in &per_chunk {
            col_spans.push((chunk_cols.len(), cols.len()));
            chunk_cols.extend_from_slice(cols);
        }

        Ok(Self {
            ranges,
            chunk_cols,
            col_spans,
            nrows: jacobian.nrows(),
        })
    }

    /// Retained local columns touched by chunk `idx`.
    #[inline]
    pub fn cols(&self, idx: usize) -> &[u32] {
        let (start, len) = self.col_spans[idx];
        &self.chunk_cols[start..start + len]
    }

    /// Number of chunks.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether there are no chunks.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// `(row_start, row_end)` of chunk `idx`.
    #[inline]
    pub fn range(&self, idx: usize) -> (usize, usize) {
        self.ranges[idx]
    }

    /// Rows of the Jacobian this layout describes.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Whether this layout still describes `jacobian`.
    pub fn matches(&self, jacobian: &SparseColMat<usize, f64>) -> bool {
        self.nrows == jacobian.nrows()
    }
}

/// The reduced system produced by elimination.
pub struct ReducedSystem {
    /// `S = H_kk − H_ke·H_ee⁻¹·H_keᵀ`, dense in the retained variables.
    ///
    /// Kept dense here for the same reason the other explicit path does: `S` is
    /// typically dense for bundle adjustment. Callers that factorize it convert
    /// to sparse.
    pub s: Vec<f64>,
    /// `g_red = g_k − H_ke·H_ee⁻¹·g_e`.
    pub g_reduced: Mat<f64>,
    /// `H_ee⁻¹` per chunk, retained for back-substitution.
    pub eliminated_inverse: EliminatedBlocks,
    /// `H_ee⁻¹·g_e`, retained for back-substitution.
    pub eliminated_rhs: Mat<f64>,
    /// Side length of `s`.
    pub kept_dof: usize,
}

/// Accumulates the reduced system chunk by chunk, reusing its scratch.
#[derive(Debug, Clone, Default)]
pub struct ChunkedSchurEliminator {
    layout: ChunkLayout,
    /// Monotonic cursor into each retained column's `row_idx`, one per column.
    ///
    /// Advancing rather than searching is what keeps the sweep linear: every
    /// entry of `J` is passed exactly once across all chunks.
    cursors: Vec<usize>,
}

impl ChunkedSchurEliminator {
    /// Prepare for a Jacobian with this pattern. Call once per structure.
    pub fn new(
        jacobian: &SparseColMat<usize, f64>,
        partition: &SchurPartition,
    ) -> LinAlgResult<Self> {
        Ok(Self {
            layout: ChunkLayout::build(jacobian, partition)?,
            cursors: vec![0; partition.kept_dof()],
        })
    }

    /// Whether the cached layout still applies.
    pub fn matches(&self, jacobian: &SparseColMat<usize, f64>) -> bool {
        self.layout.matches(jacobian)
    }

    /// Eliminate every chunk, returning the reduced system.
    ///
    /// One sweep does everything: within each chunk's row range it gathers the
    /// retained block `F`, accumulates `FᵀF` and `Fᵀr` into the reduced system,
    /// then subtracts that chunk's rank-`dof` correction. Rows belonging to no
    /// chunk — priors on retained variables — contribute `FᵀF` only.
    ///
    /// Accumulating `H_kk` inside the sweep is what keeps this linear: forming
    /// it by iterating pairs of retained columns would be O(kept_dof²) dot
    /// products, which is 107 million of them on Ladybug alone.
    ///
    /// `damping` is applied to `H_kk` and to each `H_ee` block *before* the
    /// corrections are subtracted, matching the direct paths, which damp the
    /// full system and then partition it.
    pub fn eliminate(
        &mut self,
        jacobian: &SparseColMat<usize, f64>,
        residuals: &Mat<f64>,
        partition: &SchurPartition,
        damping: Option<&Damping>,
    ) -> LinAlgResult<ReducedSystem> {
        let kept_dof = partition.kept_dof();
        let symbolic = jacobian.symbolic();

        let mut s = vec![0.0f64; kept_dof * kept_dof];
        let mut g_k = Mat::<f64>::zeros(kept_dof, 1);
        let mut eliminated_inverse = EliminatedBlocks::new(partition);
        let mut eliminated_rhs = Mat::<f64>::zeros(partition.eliminated_dof(), 1);

        // Global column of each retained local index, hoisted out of the sweep.
        let kept_global: Vec<usize> = partition
            .kept_blocks()
            .iter()
            .flat_map(|b| (0..b.dof).map(move |o| b.col_start + o))
            .collect();

        self.cursors.clear();
        self.cursors.resize(kept_dof, 0);

        // Rows before the first chunk belong to no eliminated variable.
        let leading_end = if self.layout.is_empty() {
            self.layout.nrows()
        } else {
            self.layout.range(0).0
        };
        if leading_end > 0 {
            self.sweep_rows(
                jacobian,
                residuals,
                &kept_global,
                0,
                leading_end,
                &mut s,
                &mut g_k,
                kept_dof,
            );
        }

        // λ·D on H_kk. Applied here so `D_jj` clamps against the un-eliminated
        // diagonal; damping `S` afterwards would clamp against a different
        // matrix and give a different, wrong step.
        //
        // H_kk is not complete until every chunk's FᵀF has been added, so the
        // diagonal contributions are collected first and damping applied after
        // the FᵀF accumulation but before any correction is subtracted. That is
        // handled by running the sweep in two phases below.
        let mut chunk_spans = Vec::with_capacity(self.layout.len());
        for chunk in 0..self.layout.len() {
            let (row_start, row_end) = self.layout.range(chunk);
            chunk_spans.push((chunk, row_start, row_end));
        }

        // Phase A: FᵀF and Fᵀr for every chunk's rows, plus EᵀE, EᵀF, Eᵀr.
        let mut chunk_data: Vec<ChunkData> = Vec::with_capacity(chunk_spans.len());
        for &(chunk, row_start, row_end) in &chunk_spans {
            let dof = eliminated_inverse.dof(chunk);
            if dof == 0 || row_start >= row_end {
                chunk_data.push(ChunkData::default());
                continue;
            }
            let block = partition.eliminated_blocks()[chunk];

            // Gather this chunk's retained strip and accumulate FᵀF / Fᵀr.
            let chunk_cols = self.layout.cols(chunk).to_vec();
            let (cols, strip) = self.gather_strip(
                jacobian,
                residuals,
                &kept_global,
                row_start,
                row_end,
                Some(&chunk_cols),
                &mut s,
                &mut g_k,
                kept_dof,
            );

            // EᵀE, Eᵀr and EᵀF from the same row range.
            let ete = eliminated_inverse.block_mut(chunk);
            ete.fill(0.0);
            let mut etr = vec![0.0f64; dof];
            let mut etf = vec![0.0f64; cols.len() * dof];

            for a in 0..dof {
                let col_a = block.col_start + a;
                let rows_a = symbolic.row_idx_of_col_raw(col_a);
                let vals_a = jacobian.val_of_col(col_a);
                for (i, &row) in rows_a.iter().enumerate() {
                    if row < row_start || row >= row_end {
                        continue;
                    }
                    let e = vals_a[i];
                    etr[a] += e * residuals[(row, 0)];

                    // EᵀF for this row, against the strip's dense row.
                    let local_row = row - row_start;
                    let strip_row = &strip[local_row * cols.len()..(local_row + 1) * cols.len()];
                    for (c, &f) in strip_row.iter().enumerate() {
                        etf[c * dof + a] += e * f;
                    }

                    // EᵀE
                    for b in 0..dof {
                        let col_b = block.col_start + b;
                        let rows_b = symbolic.row_idx_of_col_raw(col_b);
                        if let Ok(pos) = rows_b.binary_search(&row) {
                            ete[b * dof + a] += e * jacobian.val_of_col(col_b)[pos];
                        }
                    }
                }
            }

            chunk_data.push(ChunkData {
                cols,
                etf,
                etr,
                dof,
            });
        }

        // Rows after the last chunk, if any.
        if let Some(&(_, _, last_end)) = chunk_spans.last()
            && last_end < self.layout.nrows()
        {
            self.sweep_rows(
                jacobian,
                residuals,
                &kept_global,
                last_end,
                self.layout.nrows(),
                &mut s,
                &mut g_k,
                kept_dof,
            );
        }

        // H_kk is complete: damp it, and each H_ee block, before eliminating.
        if let Some(damping) = damping {
            for i in 0..kept_dof {
                let pos = i * kept_dof + i;
                s[pos] += damping.diagonal_term(s[pos]);
            }
            for chunk in 0..eliminated_inverse.len() {
                let dof = eliminated_inverse.dof(chunk);
                let ete = eliminated_inverse.block_mut(chunk);
                for k in 0..dof {
                    let pos = k * dof + k;
                    ete[pos] += damping.diagonal_term(ete[pos]);
                }
            }
        }

        // Phase B: invert each H_ee block and subtract its rank-dof correction.
        let mut g_reduced = g_k;
        for (chunk, data) in chunk_data.iter().enumerate() {
            let dof = data.dof;
            if dof == 0 || data.cols.is_empty() {
                continue;
            }
            let block = partition.eliminated_blocks()[chunk];
            eliminated_inverse.invert_one(chunk, block.key)?;
            let inv = eliminated_inverse.block(chunk);
            let base = partition.eliminated_offset(chunk);

            // w = H_ee⁻¹·Eᵀr, retained for back-substitution.
            for r in 0..dof {
                let mut acc = 0.0;
                for (c, &etr_c) in data.etr.iter().enumerate() {
                    acc += inv[c * dof + r] * etr_c;
                }
                eliminated_rhs[(base + r, 0)] = acc;
            }

            for (i, &col_i) in data.cols.iter().enumerate() {
                let etf_i = &data.etf[i * dof..(i + 1) * dof];
                let mut contrib = vec![0.0f64; dof];
                for (c, slot) in contrib.iter_mut().enumerate() {
                    let inv_col = &inv[c * dof..(c + 1) * dof];
                    let mut acc = 0.0;
                    for k in 0..dof {
                        acc += etf_i[k] * inv_col[k];
                    }
                    *slot = acc;
                }

                let g_term: f64 = contrib
                    .iter()
                    .zip(data.etr.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                g_reduced[(col_i, 0)] -= g_term;

                let row_base = col_i * kept_dof;
                for (j, &col_j) in data.cols.iter().enumerate() {
                    let etf_j = &data.etf[j * dof..(j + 1) * dof];
                    let mut acc = 0.0;
                    for k in 0..dof {
                        acc += contrib[k] * etf_j[k];
                    }
                    s[row_base + col_j] -= acc;
                }
            }
        }

        Ok(ReducedSystem {
            s,
            g_reduced,
            eliminated_inverse,
            eliminated_rhs,
            kept_dof,
        })
    }

    /// Gather the retained entries of `[row_start, row_end)` into a dense strip,
    /// accumulating `FᵀF` and `Fᵀr` on the way.
    ///
    /// Returns the touched retained columns and the `rows × cols` strip.
    #[allow(clippy::too_many_arguments)]
    fn gather_strip(
        &mut self,
        jacobian: &SparseColMat<usize, f64>,
        residuals: &Mat<f64>,
        kept_global: &[usize],
        row_start: usize,
        row_end: usize,
        candidate_cols: Option<&[u32]>,
        s: &mut [f64],
        g_k: &mut Mat<f64>,
        kept_dof: usize,
    ) -> (Vec<usize>, Vec<f64>) {
        let symbolic = jacobian.symbolic();
        let n_rows = row_end - row_start;

        // Candidate columns: the chunk's cached list when we have one,
        // otherwise every retained column (only for the few rows outside any
        // chunk). Cursors still advance monotonically, so each column's entries
        // are visited once across the whole sweep.
        let owned_all: Vec<u32>;
        let candidates: &[u32] = match candidate_cols {
            Some(c) => c,
            None => {
                owned_all = (0..kept_global.len() as u32).collect();
                &owned_all
            }
        };

        let mut cols = Vec::with_capacity(candidates.len());
        let mut spans = Vec::with_capacity(candidates.len());
        for &local_u32 in candidates {
            let local = local_u32 as usize;
            let cursor = &mut self.cursors[local];
            let rows = symbolic.row_idx_of_col_raw(kept_global[local]);
            while *cursor < rows.len() && rows[*cursor] < row_start {
                *cursor += 1;
            }
            let begin = *cursor;
            let mut probe = begin;
            while probe < rows.len() && rows[probe] < row_end {
                probe += 1;
            }
            if probe > begin {
                cols.push(local);
                spans.push((begin, probe));
                *cursor = probe;
            }
        }

        let width = cols.len();
        let mut strip = vec![0.0f64; n_rows * width];
        for (c, (&local, &(begin, end))) in cols.iter().zip(spans.iter()).enumerate() {
            let global = kept_global[local];
            let rows = symbolic.row_idx_of_col_raw(global);
            let vals = jacobian.val_of_col(global);
            for k in begin..end {
                strip[(rows[k] - row_start) * width + c] = vals[k];
            }
        }

        // FᵀF and Fᵀr over this row range only.
        for r in 0..n_rows {
            let row = &strip[r * width..(r + 1) * width];
            let residual = residuals[(row_start + r, 0)];
            for (i, &vi) in row.iter().enumerate() {
                if vi == 0.0 {
                    continue;
                }
                g_k[(cols[i], 0)] += vi * residual;
                let base = cols[i] * kept_dof;
                for (j, &vj) in row.iter().enumerate() {
                    if vj != 0.0 {
                        s[base + cols[j]] += vi * vj;
                    }
                }
            }
        }

        (cols, strip)
    }

    /// `FᵀF` and `Fᵀr` for a row range that belongs to no chunk.
    #[allow(clippy::too_many_arguments)]
    fn sweep_rows(
        &mut self,
        jacobian: &SparseColMat<usize, f64>,
        residuals: &Mat<f64>,
        kept_global: &[usize],
        row_start: usize,
        row_end: usize,
        s: &mut [f64],
        g_k: &mut Mat<f64>,
        kept_dof: usize,
    ) {
        let _ = self.gather_strip(
            jacobian, residuals, kept_global, row_start, row_end, None, s, g_k, kept_dof,
        );
    }
}

/// Per-chunk quantities carried from the gather phase to the correction phase.
#[derive(Debug, Default)]
struct ChunkData {
    /// Retained columns this chunk touches.
    cols: Vec<usize>,
    /// `Eᵀ F`, column-major over `cols`.
    etf: Vec<f64>,
    /// `Eᵀ r`.
    etr: Vec<f64>,
    dof: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse::schur_partition::BlockSpan;
    use faer::sparse::Triplet;
    use slotmap::KeyData;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn key(i: u64) -> crate::core::VarKey {
        crate::core::VarKey::from(KeyData::from_ffi((1u64 << 32) | i))
    }

    /// Two kept 2-DOF variables, two eliminated 1-DOF variables. Rows are
    /// grouped by eliminated variable, as chunking requires.
    ///
    /// rows 0..2 -> eliminated 0, rows 2..4 -> eliminated 1, rows 4..8 priors
    /// `(J, r, partition)` for the shared fixture.
    type Fixture = (SparseColMat<usize, f64>, Mat<f64>, SchurPartition);

    fn tiny_system() -> Result<Fixture, Box<dyn std::error::Error>> {
        // columns: 0,1 kept A | 2,3 kept B | 4 elim P | 5 elim Q
        let t = vec![
            // chunk 0 (rows 0,1): kept A + elim P
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(0, 1, 0.5),
            Triplet::new(0, 4, 2.0),
            Triplet::new(1, 0, 0.3),
            Triplet::new(1, 2, 1.1),
            Triplet::new(1, 4, 1.5),
            // chunk 1 (rows 2,3): kept B + elim Q
            Triplet::new(2, 2, 0.9),
            Triplet::new(2, 3, 1.3),
            Triplet::new(2, 5, 1.7),
            Triplet::new(3, 0, 0.4),
            Triplet::new(3, 3, 0.8),
            Triplet::new(3, 5, 2.1),
            // priors on every column so JᵀJ is SPD
            Triplet::new(4, 0, 0.7),
            Triplet::new(5, 1, 0.6),
            Triplet::new(6, 2, 0.9),
            Triplet::new(7, 3, 1.2),
        ];
        let j = SparseColMat::try_new_from_triplets(8, 6, &t)?;
        let r = Mat::from_fn(8, 1, |i, _| 0.2 + (i % 3) as f64 * 0.4);

        let partition = SchurPartition::new(
            vec![
                BlockSpan { key: key(0), col_start: 0, dof: 2 },
                BlockSpan { key: key(1), col_start: 2, dof: 2 },
            ],
            vec![
                BlockSpan { key: key(2), col_start: 4, dof: 1 },
                BlockSpan { key: key(3), col_start: 5, dof: 1 },
            ],
        )?;
        Ok((j, r, partition))
    }

    /// Dense reference: build JᵀJ and Jᵀr outright, then form the Schur
    /// complement by direct linear algebra. Chunking is only a reordering of
    /// that same sum, so the two must agree to round-off.
    fn dense_reference(
        j: &SparseColMat<usize, f64>,
        r: &Mat<f64>,
        partition: &SchurPartition,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = j.ncols();
        let m = j.nrows();
        let mut dense = vec![vec![0.0f64; n]; m];
        for (col, _) in (0..n).map(|c| (c, ())) {
            let rows = j.symbolic().row_idx_of_col_raw(col);
            let vals = j.val_of_col(col);
            for (i, &row) in rows.iter().enumerate() {
                dense[row][col] = vals[i];
            }
        }
        let mut h = vec![vec![0.0f64; n]; n];
        let mut g = vec![0.0f64; n];
        for a in 0..n {
            for b in 0..n {
                let mut acc = 0.0;
                for row in dense.iter() {
                    acc += row[a] * row[b];
                }
                h[a][b] = acc;
            }
            let mut acc = 0.0;
            for (row_idx, row) in dense.iter().enumerate() {
                acc += row[a] * r[(row_idx, 0)];
            }
            g[a] = acc;
        }

        // S = H_kk - H_ke H_ee^-1 H_ek  (all blocks 1-DOF eliminated here)
        let kept: Vec<usize> = partition
            .kept_blocks()
            .iter()
            .flat_map(|b| b.col_start..b.col_start + b.dof)
            .collect();
        let elim: Vec<usize> = partition
            .eliminated_blocks()
            .iter()
            .flat_map(|b| b.col_start..b.col_start + b.dof)
            .collect();

        let k = kept.len();
        let mut s = vec![0.0f64; k * k];
        let mut g_red = vec![0.0f64; k];
        for (li, &ci) in kept.iter().enumerate() {
            g_red[li] = g[ci];
            for (lj, &cj) in kept.iter().enumerate() {
                s[li * k + lj] = h[ci][cj];
            }
        }
        for &e in &elim {
            let inv = 1.0 / h[e][e];
            for (li, &ci) in kept.iter().enumerate() {
                g_red[li] -= h[ci][e] * inv * g[e];
                for (lj, &cj) in kept.iter().enumerate() {
                    s[li * k + lj] -= h[ci][e] * inv * h[e][cj];
                }
            }
        }
        (s, g_red)
    }

    /// The whole point: chunk-wise elimination reproduces the reduced system
    /// that forming JᵀJ and partitioning it would give.
    #[test]
    fn chunked_elimination_matches_dense_reference() -> TestResult {
        let (j, r, partition) = tiny_system()?;
        let (want_s, want_g) = dense_reference(&j, &r, &partition);

        let mut elim = ChunkedSchurEliminator::new(&j, &partition)?;
        let got = elim.eliminate(&j, &r, &partition, None)?;

        assert_eq!(got.kept_dof, 4);
        for i in 0..4 {
            assert!(
                (got.g_reduced[(i, 0)] - want_g[i]).abs() < 1e-10,
                "g_reduced[{i}]: got {}, want {}",
                got.g_reduced[(i, 0)],
                want_g[i]
            );
            for k in 0..4 {
                assert!(
                    (got.s[i * 4 + k] - want_s[i * 4 + k]).abs() < 1e-10,
                    "S[{i},{k}]: got {}, want {}",
                    got.s[i * 4 + k],
                    want_s[i * 4 + k]
                );
            }
        }
        Ok(())
    }

    /// Damping must land on both sides' diagonals before elimination, exactly
    /// as the direct paths apply it to the full system.
    #[test]
    fn damping_reaches_both_sides() -> TestResult {
        let (j, r, partition) = tiny_system()?;
        let mut elim = ChunkedSchurEliminator::new(&j, &partition)?;

        let plain = elim.eliminate(&j, &r, &partition, None)?;
        let damping = Damping::new(0.5, 1e-6, 1e32)?;
        let damped = elim.eliminate(&j, &r, &partition, Some(&damping))?;

        for i in 0..plain.kept_dof {
            let d = i * plain.kept_dof + i;
            assert!(
                damped.s[d] > plain.s[d],
                "damping must raise S's diagonal at {i}"
            );
        }
        // Off-diagonals differ too, because H_ee was damped before inversion.
        assert!(
            (damped.s[1] - plain.s[1]).abs() > 1e-12,
            "damping H_ee must change the Schur complement off-diagonal"
        );
        Ok(())
    }

    /// The sweep depends on chunks being contiguous and increasing; a layout
    /// that violates it must be rejected, not silently mis-swept.
    #[test]
    fn interleaved_chunks_are_rejected() -> TestResult {
        // elim P occupies rows 0 and 2; elim Q occupies row 1 — they interleave.
        let t = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 2, 1.5),
            Triplet::new(2, 0, 1.0),
            Triplet::new(2, 1, 1.1),
        ];
        let j = SparseColMat::try_new_from_triplets(3, 3, &t)?;
        let partition = SchurPartition::new(
            vec![BlockSpan { key: key(0), col_start: 0, dof: 1 }],
            vec![
                BlockSpan { key: key(1), col_start: 1, dof: 1 },
                BlockSpan { key: key(2), col_start: 2, dof: 1 },
            ],
        )?;

        let Err(err) = ChunkedSchurEliminator::new(&j, &partition) else {
            panic!("interleaved chunk rows must be rejected");
        };
        assert!(err.to_string().contains("contiguous"), "{err}");
        Ok(())
    }

    /// An eliminated variable observed by nothing cannot be eliminated; saying
    /// so beats inverting a zero block.
    #[test]
    fn unobserved_eliminated_variable_is_reported() -> TestResult {
        let t = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 0.5),
        ];
        let j = SparseColMat::try_new_from_triplets(2, 2, &t)?;
        let partition = SchurPartition::new(
            vec![BlockSpan { key: key(0), col_start: 0, dof: 1 }],
            vec![BlockSpan { key: key(1), col_start: 1, dof: 1 }],
        )?;

        let Err(err) = ChunkedSchurEliminator::new(&j, &partition) else {
            panic!("an unobserved eliminated variable must be reported");
        };
        assert!(err.to_string().contains("no residual"), "{err}");
        Ok(())
    }
}
