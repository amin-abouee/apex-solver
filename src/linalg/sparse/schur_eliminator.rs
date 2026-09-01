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

        Ok(Self {
            ranges,
            nrows: jacobian.nrows(),
        })
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
#[derive(Debug, Default)]
pub struct ChunkedSchurEliminator {
    layout: ChunkLayout,
    // Per-chunk scratch, sized for the widest chunk and reused.
    /// `Eᵀ F` for the current chunk, row-major `dof × |F_c|`.
    etf: Vec<f64>,
    /// Kept columns touched by the current chunk, in increasing order.
    kept_cols: Vec<usize>,
    /// Monotonic cursor into each kept column's `row_idx`, one per kept column.
    cursors: Vec<usize>,
    /// `Eᵀ r` for the current chunk.
    etr: Vec<f64>,
}

impl ChunkedSchurEliminator {
    /// Prepare for a Jacobian with this pattern. Call once per structure.
    pub fn new(
        jacobian: &SparseColMat<usize, f64>,
        partition: &SchurPartition,
    ) -> LinAlgResult<Self> {
        Ok(Self {
            layout: ChunkLayout::build(jacobian, partition)?,
            etf: Vec::new(),
            kept_cols: Vec::new(),
            cursors: vec![0; partition.kept_dof()],
            etr: Vec::new(),
        })
    }

    /// Whether the cached layout still applies.
    pub fn matches(&self, jacobian: &SparseColMat<usize, f64>) -> bool {
        self.layout.matches(jacobian)
    }

    /// Eliminate every chunk, returning the reduced system.
    ///
    /// `damping` is applied to `H_ee`'s diagonal before inversion and to `S`'s
    /// diagonal afterwards, which is the same `λ·D` the direct paths apply to
    /// the full system before partitioning.
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
        let mut g_reduced = Mat::<f64>::zeros(kept_dof, 1);
        let mut eliminated_inverse = EliminatedBlocks::new(partition);
        let mut eliminated_rhs = Mat::<f64>::zeros(partition.eliminated_dof(), 1);

        // Cursors restart at the head of every column for each sweep.
        self.cursors.clear();
        self.cursors.resize(kept_dof, 0);

        // Pass 1: the retained-retained part, H_kk = Fᵀ F over all rows, and
        // g_k = Fᵀ r. Accumulated column-wise, which CSC gives directly.
        accumulate_kept_normal_equations(
            jacobian,
            residuals,
            partition,
            kept_dof,
            &mut s,
            &mut g_reduced,
        );

        // Pass 2: one chunk at a time, subtract that chunk's rank-`dof`
        // contribution. Nothing here is proportional to nnz(JᵀJ).
        for chunk in 0..self.layout.len() {
            let (row_start, row_end) = self.layout.range(chunk);
            let dof = eliminated_inverse.dof(chunk);
            if dof == 0 || row_start >= row_end {
                continue;
            }
            let block = partition.eliminated_blocks()[chunk];

            // --- ete = Eᵀ E and etr = Eᵀ r, straight from the block's columns.
            let ete = eliminated_inverse.block_mut(chunk);
            ete.fill(0.0);
            self.etr.clear();
            self.etr.resize(dof, 0.0);
            for a in 0..dof {
                let rows_a = symbolic.row_idx_of_col_raw(block.col_start + a);
                let vals_a = jacobian.val_of_col(block.col_start + a);
                for (i, &row) in rows_a.iter().enumerate() {
                    self.etr[a] += vals_a[i] * residuals[(row, 0)];
                }
                for b in 0..dof {
                    let rows_b = symbolic.row_idx_of_col_raw(block.col_start + b);
                    let vals_b = jacobian.val_of_col(block.col_start + b);
                    ete[b * dof + a] += dot_on_shared_rows(rows_a, vals_a, rows_b, vals_b);
                }
            }

            // --- etf = Eᵀ F, gathered by advancing each kept column's cursor
            //     through this chunk's row range.
            self.kept_cols.clear();
            self.etf.clear();
            for (local_col, cursor) in self.cursors.iter_mut().enumerate() {
                let global_col = kept_global_col(partition, local_col);
                let rows = symbolic.row_idx_of_col_raw(global_col);
                let vals = jacobian.val_of_col(global_col);

                // Skip anything before this chunk (already consumed).
                while *cursor < rows.len() && rows[*cursor] < row_start {
                    *cursor += 1;
                }
                let entry_start = *cursor;
                let mut probe = *cursor;
                while probe < rows.len() && rows[probe] < row_end {
                    probe += 1;
                }
                if probe == entry_start {
                    continue;
                }

                // This column participates: accumulate its dof-vector of Eᵀ F.
                let slot = self.etf.len();
                self.etf.resize(slot + dof, 0.0);
                for k in entry_start..probe {
                    let row = rows[k];
                    let f = vals[k];
                    for a in 0..dof {
                        let rows_a = symbolic.row_idx_of_col_raw(block.col_start + a);
                        let vals_a = jacobian.val_of_col(block.col_start + a);
                        if let Ok(pos) = rows_a.binary_search(&row) {
                            self.etf[slot + a] += vals_a[pos] * f;
                        }
                    }
                }
                self.kept_cols.push(local_col);
                *cursor = probe;
            }

            // --- damp and invert this chunk's ete
            if let Some(damping) = damping {
                let ete = eliminated_inverse.block_mut(chunk);
                for k in 0..dof {
                    let pos = k * dof + k;
                    ete[pos] += damping.diagonal_term(ete[pos]);
                }
            }
            eliminated_inverse.invert_one(chunk, block.key)?;

            // --- apply the rank-dof update
            let inv = eliminated_inverse.block(chunk);
            let base = partition.eliminated_offset(chunk);

            // w = ete⁻¹ · etr, retained for back-substitution
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * self.etr[c];
                }
                eliminated_rhs[(base + r, 0)] = acc;
            }

            for (i, &col_i) in self.kept_cols.iter().enumerate() {
                let etf_i = &self.etf[i * dof..(i + 1) * dof];
                // contrib_i = etf_iᵀ · ete⁻¹
                let mut contrib = [0.0f64; 16];
                let contrib = &mut contrib[..dof.min(16)];
                for (c, slot) in contrib.iter_mut().enumerate() {
                    let inv_col = &inv[c * dof..(c + 1) * dof];
                    let mut acc = 0.0;
                    for k in 0..dof {
                        acc += etf_i[k] * inv_col[k];
                    }
                    *slot = acc;
                }

                // g_red -= etf_iᵀ · ete⁻¹ · etr
                let g_term: f64 = contrib
                    .iter()
                    .zip(self.etr.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                g_reduced[(col_i, 0)] -= g_term;

                // S -= etf_iᵀ · ete⁻¹ · etf_j
                let row_base = col_i * kept_dof;
                for (j, &col_j) in self.kept_cols.iter().enumerate() {
                    let etf_j = &self.etf[j * dof..(j + 1) * dof];
                    let mut acc = 0.0;
                    for k in 0..dof {
                        acc += contrib[k] * etf_j[k];
                    }
                    s[row_base + col_j] -= acc;
                }
            }
        }

        // Damping on the retained diagonal, matching the eliminated side.
        if let Some(damping) = damping {
            for i in 0..kept_dof {
                let pos = i * kept_dof + i;
                s[pos] += damping.diagonal_term(s[pos]);
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
}

/// `H_kk = Fᵀ F` and `g_k = Fᵀ r`, accumulated column-wise from CSC.
fn accumulate_kept_normal_equations(
    jacobian: &SparseColMat<usize, f64>,
    residuals: &Mat<f64>,
    partition: &SchurPartition,
    kept_dof: usize,
    s: &mut [f64],
    g_k: &mut Mat<f64>,
) {
    let symbolic = jacobian.symbolic();
    for local_a in 0..kept_dof {
        let col_a = kept_global_col(partition, local_a);
        let rows_a = symbolic.row_idx_of_col_raw(col_a);
        let vals_a = jacobian.val_of_col(col_a);

        for (i, &row) in rows_a.iter().enumerate() {
            g_k[(local_a, 0)] += vals_a[i] * residuals[(row, 0)];
        }
        for local_b in 0..kept_dof {
            let col_b = kept_global_col(partition, local_b);
            let rows_b = symbolic.row_idx_of_col_raw(col_b);
            let vals_b = jacobian.val_of_col(col_b);
            let v = dot_on_shared_rows(rows_a, vals_a, rows_b, vals_b);
            if v != 0.0 {
                s[local_a * kept_dof + local_b] += v;
            }
        }
    }
}

/// Global column of the retained variable at local index `local`.
///
/// Linear in the number of retained blocks; callers hoist it out of inner loops.
fn kept_global_col(partition: &SchurPartition, local: usize) -> usize {
    let mut seen = 0usize;
    for block in partition.kept_blocks() {
        if local < seen + block.dof {
            return block.col_start + (local - seen);
        }
        seen += block.dof;
    }
    unreachable_col(local)
}

fn unreachable_col(local: usize) -> usize {
    debug_assert!(false, "local kept column {local} is out of range");
    0
}

/// Dot product of two sorted sparse columns over the rows they share.
fn dot_on_shared_rows(rows_a: &[usize], vals_a: &[f64], rows_b: &[usize], vals_b: &[f64]) -> f64 {
    let (mut i, mut j, mut acc) = (0usize, 0usize, 0.0f64);
    while i < rows_a.len() && j < rows_b.len() {
        match rows_a[i].cmp(&rows_b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                acc += vals_a[i] * vals_b[j];
                i += 1;
                j += 1;
            }
        }
    }
    acc
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
