//! Generalized variable partition for Schur complement elimination.
//!
//! Schur elimination splits the variables of `H·δ = −g` into two sets and
//! eliminates one of them:
//!
//! ```text
//! ⎡ H_kk  H_ke ⎤ ⎡ δ_k ⎤   ⎡ g_k ⎤        S = H_kk − H_ke·H_ee⁻¹·H_keᵀ
//! ⎢            ⎥ ⎢     ⎥ = ⎢     ⎥   ⟹    S·δ_k = g_k − H_ke·H_ee⁻¹·g_e
//! ⎣ H_keᵀ H_ee ⎦ ⎣ δ_e ⎦   ⎣ g_e ⎦        δ_e = H_ee⁻¹·(g_e − H_keᵀ·δ_k)
//! ```
//!
//! Nothing in that derivation mentions cameras or 3-D points. Following Ceres's
//! naming, the eliminated set is the **e-blocks** and the retained set the
//! **f-blocks** (here: *kept*). This module carries the partition in a form that
//! is agnostic to what the variables mean, so the same solver covers classic
//! bundle adjustment, inverse-depth parameterizations, LiDAR features, and
//! sliding-window marginalization.
//!
//! # What this generalizes
//!
//! | Property | Previously | Here |
//! |---|---|---|
//! | Eliminated block size | hardcoded 3 | any DOF, mixed within one problem |
//! | Column layout | kept and eliminated each assumed contiguous | arbitrary interleaving |
//! | `H_ee` block-diagonality | assumed, unchecked | verified, typed error |
//!
//! # Preconditions
//!
//! Elimination is only valid when the eliminated variables are **mutually
//! unconnected** — no factor may touch two of them. That is what makes `H_ee`
//! block-diagonal, and hence cheap to invert. [`SchurPartition::verify_block_diagonal`]
//! checks this against the actual Hessian pattern rather than trusting the caller.

use faer::sparse::SparseColMat;
use nalgebra::{DMatrix, Matrix3};

use crate::core::VarKey;
use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult};

/// Sentinel for a column that is not retained.
const NOT_KEPT: u32 = u32::MAX;
/// Sentinel for a column that is not eliminated.
const NOT_ELIMINATED: u32 = u32::MAX;

fn too_many_columns(total: usize) -> LinAlgError {
    LinAlgError::InvalidInput(format!(
        "system has {total} columns, more than the {} this partition can index",
        u32::MAX
    ))
    .log()
}

/// One variable's span of tangent-space columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSpan {
    /// The variable this block belongs to.
    pub key: VarKey,
    /// First global column of the block.
    pub col_start: usize,
    /// Number of columns (the variable's DOF).
    pub dof: usize,
}

/// Which side of the partition a global column falls on, and where it sits in
/// that side's local ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColSlot {
    /// Retained: local index within the reduced system.
    Kept(usize),
    /// Eliminated: `(block index, local column within that block)`.
    Eliminated(usize, usize),
}

/// A variable partition ready for Schur elimination.
///
/// Local orderings are assigned by ascending global column, so the reduced
/// system's variable order is deterministic and independent of slotmap
/// iteration order.
#[derive(Debug, Clone, Default)]
pub struct SchurPartition {
    kept_blocks: Vec<BlockSpan>,
    eliminated_blocks: Vec<BlockSpan>,
    kept_dof: usize,
    eliminated_dof: usize,
    /// Global column → local index in the reduced system, or [`NOT_KEPT`].
    kept_local: Vec<u32>,
    /// Global column → owning eliminated block, or [`NOT_ELIMINATED`].
    elim_block: Vec<u32>,
    /// Global column → local column within its eliminated block.
    elim_offset: Vec<u32>,
    /// Start of each eliminated block within the eliminated-local column space.
    eliminated_offsets: Vec<usize>,
}

impl SchurPartition {
    /// Build a partition from per-variable spans.
    ///
    /// `spans` may arrive in any order and the two sides may interleave
    /// arbitrarily in global column space — sliding-window marginalization
    /// eliminates a variable sitting in the middle of the ordering.
    pub fn new(mut kept: Vec<BlockSpan>, mut eliminated: Vec<BlockSpan>) -> LinAlgResult<Self> {
        if kept.is_empty() {
            return Err(LinAlgError::InvalidInput(
                "Schur elimination needs at least one retained variable".into(),
            )
            .log());
        }
        if eliminated.is_empty() {
            return Err(LinAlgError::InvalidInput(
                "Schur elimination needs at least one variable to eliminate".into(),
            )
            .log());
        }

        kept.sort_by_key(|b| b.col_start);
        eliminated.sort_by_key(|b| b.col_start);

        let kept_dof: usize = kept.iter().map(|b| b.dof).sum();
        let eliminated_dof: usize = eliminated.iter().map(|b| b.dof).sum();
        let total = kept_dof + eliminated_dof;

        // Every column of the system must be claimed exactly once. A gap or an
        // overlap means the caller's index map disagrees with the partition,
        // which would silently mis-address the Hessian.
        let mut kept_local = vec![NOT_KEPT; total];
        let mut elim_block = vec![NOT_ELIMINATED; total];
        let mut elim_offset = vec![0u32; total];
        let mut claimed = vec![false; total];

        let mut local = 0usize;
        for block in &kept {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                Self::claim(&mut claimed, col, total, block.key)?;
                kept_local[col] = u32::try_from(local).map_err(|_| too_many_columns(total))?;
                local += 1;
            }
        }

        let mut eliminated_offsets = Vec::with_capacity(eliminated.len());
        let mut local = 0usize;
        for (block_idx, block) in eliminated.iter().enumerate() {
            eliminated_offsets.push(local);
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                Self::claim(&mut claimed, col, total, block.key)?;
                elim_block[col] =
                    u32::try_from(block_idx).map_err(|_| too_many_columns(total))?;
                elim_offset[col] = u32::try_from(offset).map_err(|_| too_many_columns(total))?;
            }
            local += block.dof;
        }

        if let Some(col) = claimed.iter().position(|c| !c) {
            return Err(LinAlgError::InvalidInput(format!(
                "column {col} of the {total}-column system belongs to no variable; the \
                 partition does not cover the problem"
            ))
            .log());
        }

        Ok(Self {
            kept_blocks: kept,
            eliminated_blocks: eliminated,
            kept_dof,
            eliminated_dof,
            kept_local,
            elim_block,
            elim_offset,
            eliminated_offsets,
        })
    }

    fn claim(claimed: &mut [bool], col: usize, total: usize, key: VarKey) -> LinAlgResult<()> {
        let slot = claimed.get_mut(col).ok_or_else(|| {
            LinAlgError::InvalidInput(format!(
                "variable {key:?} spans column {col}, past the {total}-column system"
            ))
            .log()
        })?;
        if *slot {
            return Err(LinAlgError::InvalidInput(format!(
                "column {col} is claimed by more than one variable (at {key:?}); variable \
                 column ranges must not overlap"
            ))
            .log());
        }
        *slot = true;
        Ok(())
    }

    /// Retained variables, ordered by global column.
    pub fn kept_blocks(&self) -> &[BlockSpan] {
        &self.kept_blocks
    }

    /// Eliminated variables, ordered by global column.
    pub fn eliminated_blocks(&self) -> &[BlockSpan] {
        &self.eliminated_blocks
    }

    /// Size of the reduced system.
    pub fn kept_dof(&self) -> usize {
        self.kept_dof
    }

    /// Combined DOF of everything being eliminated.
    pub fn eliminated_dof(&self) -> usize {
        self.eliminated_dof
    }

    /// Total columns of the full system.
    pub fn total_dof(&self) -> usize {
        self.kept_dof + self.eliminated_dof
    }

    /// Where a global column sits, or `None` if out of range.
    pub fn slot(&self, global_col: usize) -> Option<ColSlot> {
        if let Some(local) = self.kept_local(global_col) {
            return Some(ColSlot::Kept(local));
        }
        self.eliminated_local(global_col)
            .map(|(b, o)| ColSlot::Eliminated(b, o))
    }

    /// Local index of `global_col` in the reduced system, if it is retained.
    ///
    /// On the hot path for every Hessian nonzero, so it is a single indexed
    /// load and a sentinel compare.
    #[inline]
    pub fn kept_local(&self, global_col: usize) -> Option<usize> {
        match self.kept_local.get(global_col) {
            Some(&local) if local != NOT_KEPT => Some(local as usize),
            _ => None,
        }
    }

    /// `(block index, local column)` of `global_col`, if it is eliminated.
    #[inline]
    pub fn eliminated_local(&self, global_col: usize) -> Option<(usize, usize)> {
        match self.elim_block.get(global_col) {
            Some(&block) if block != NOT_ELIMINATED => {
                Some((block as usize, self.elim_offset[global_col] as usize))
            }
            _ => None,
        }
    }

    /// First eliminated-local column of block `block_idx`.
    #[inline]
    pub fn eliminated_offset(&self, block_idx: usize) -> usize {
        self.eliminated_offsets[block_idx]
    }

    /// Verify that no factor couples two eliminated variables.
    ///
    /// Schur elimination inverts `H_ee` blockwise, which is only the true
    /// inverse when `H_ee` is block-diagonal — i.e. when the eliminated
    /// variables are mutually unconnected. Violating it yields a wrong step
    /// with no other symptom, so this is checked against the Hessian's actual
    /// sparsity rather than assumed.
    ///
    /// Cost: one pass over the eliminated columns' nonzeros.
    pub fn verify_block_diagonal(&self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let symbolic = hessian.symbolic();
        for block in &self.eliminated_blocks {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some((this_block, _)) = self.eliminated_local(col) else {
                    continue;
                };
                for &row in symbolic.row_idx_of_col_raw(col) {
                    if let Some((other_block, _)) = self.eliminated_local(row)
                        && other_block != this_block
                    {
                        let other = self.eliminated_blocks[other_block];
                        return Err(LinAlgError::InvalidInput(format!(
                            "variables {:?} and {:?} are both marked for elimination but are \
                             connected by a factor, so H_ee is not block-diagonal and Schur \
                             elimination would give a wrong step; eliminate only mutually \
                             unconnected variables",
                            block.key, other.key
                        ))
                        .log());
                    }
                }
            }
        }
        Ok(())
    }
}

/// The dense diagonal blocks of `H_ee`, packed into one arena.
///
/// Block sizes vary per problem (1 for inverse depth, 3 for a point, 6 for a
/// marginalized pose) and may be mixed within one problem, so a fixed-size
/// matrix type will not do. A `Vec<DMatrix>` would heap-allocate once per block
/// per iteration — 156 502 allocations per iteration on Ladybug — so the blocks
/// share a single flat buffer that is allocated once and reused, mirroring
/// `AssemblyWorkspace::jac_arena`.
///
/// Each block is stored column-major, matching nalgebra and faer.
#[derive(Debug, Clone, Default)]
pub struct EliminatedBlocks {
    values: Vec<f64>,
    /// `(start, dof)` into `values` per block.
    spans: Vec<(usize, usize)>,
}

impl EliminatedBlocks {
    /// Allocate storage sized for `partition`'s eliminated blocks.
    pub fn new(partition: &SchurPartition) -> Self {
        let mut spans = Vec::with_capacity(partition.eliminated_blocks().len());
        let mut total = 0usize;
        for block in partition.eliminated_blocks() {
            spans.push((total, block.dof));
            total += block.dof * block.dof;
        }
        Self {
            values: vec![0.0; total],
            spans,
        }
    }

    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether there are no blocks.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// DOF of block `idx`.
    #[inline]
    pub fn dof(&self, idx: usize) -> usize {
        self.spans[idx].1
    }

    /// Column-major values of block `idx`.
    #[inline]
    pub fn block(&self, idx: usize) -> &[f64] {
        let (start, dof) = self.spans[idx];
        &self.values[start..start + dof * dof]
    }

    /// Mutable column-major values of block `idx`.
    #[inline]
    pub fn block_mut(&mut self, idx: usize) -> &mut [f64] {
        let (start, dof) = self.spans[idx];
        &mut self.values[start..start + dof * dof]
    }

    /// Read entry `(row, col)` of block `idx`.
    #[inline]
    pub fn at(&self, idx: usize, row: usize, col: usize) -> f64 {
        let (start, dof) = self.spans[idx];
        self.values[start + col * dof + row]
    }

    /// Zero every block, keeping the allocation.
    pub fn clear(&mut self) {
        self.values.fill(0.0);
    }

    /// Gather the diagonal blocks of `H_ee` out of the full Hessian.
    ///
    /// Only entries whose row and column land in the *same* eliminated block are
    /// taken; cross-block entries are the block-diagonality violation that
    /// [`SchurPartition::verify_block_diagonal`] reports.
    pub fn gather(&mut self, hessian: &SparseColMat<usize, f64>, partition: &SchurPartition) {
        self.clear();
        let symbolic = hessian.symbolic();
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let (start, dof) = self.spans[block_idx];
            for local_col in 0..dof {
                let global_col = block.col_start + local_col;
                let rows = symbolic.row_idx_of_col_raw(global_col);
                let vals = hessian.val_of_col(global_col);
                for (idx, &global_row) in rows.iter().enumerate() {
                    if let Some((row_block, local_row)) = partition.eliminated_local(global_row)
                        && row_block == block_idx
                    {
                        self.values[start + local_col * dof + local_row] = vals[idx];
                    }
                }
            }
        }
    }

    /// Add `λ·D` to every block's diagonal, with `D_jj = clamp(H_jj, …)`.
    pub fn damp(&mut self, damping: &crate::linalg::Damping) {
        for &(start, dof) in &self.spans {
            for k in 0..dof {
                let pos = start + k * dof + k;
                self.values[pos] += damping.diagonal_term(self.values[pos]);
            }
        }
    }

    /// Invert every block in place.
    ///
    /// Dispatches on DOF so the common sizes stay on stack-allocated types —
    /// 3×3 keeps the exact path used before this was generalized. Singular
    /// blocks get a scaled-identity retry before failing, matching the previous
    /// behaviour for landmark blocks.
    pub fn invert_in_place(&mut self, partition: &SchurPartition) -> LinAlgResult<()> {
        for idx in 0..self.spans.len() {
            let dof = self.spans[idx].1;
            let key = partition.eliminated_blocks()[idx].key;
            match dof {
                0 => continue,
                1 => {
                    let block = self.block_mut(idx);
                    let v = block[0];
                    block[0] = if v.abs() > f64::EPSILON {
                        1.0 / v
                    } else {
                        return Err(singular(key, dof));
                    };
                }
                3 => {
                    let block = self.block_mut(idx);
                    let m = Matrix3::from_column_slice(block);
                    let inv = invert_with_retry_3(&m).ok_or_else(|| singular(key, dof))?;
                    block.copy_from_slice(inv.as_slice());
                }
                n => {
                    let block = self.block_mut(idx);
                    let m = DMatrix::from_column_slice(n, n, block);
                    let inv = invert_with_retry_dyn(&m).ok_or_else(|| singular(key, dof))?;
                    block.copy_from_slice(inv.as_slice());
                }
            }
        }
        Ok(())
    }
}

fn singular(key: VarKey, dof: usize) -> LinAlgError {
    LinAlgError::SingularMatrix(format!(
        "H_ee block for variable {key:?} ({dof}×{dof}) is singular even after regularization; \
         the variable is unobserved or its observations are degenerate"
    ))
    .log()
}

/// Invert a 3×3 block, retrying with Tikhonov regularization scaled to its trace.
fn invert_with_retry_3(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    if let Some(inv) = m.try_inverse() {
        return Some(inv);
    }
    let reg = (1e-6 * m.diagonal().iter().sum::<f64>().abs() / 3.0).max(1e-8);
    (m + Matrix3::identity() * reg).try_inverse()
}

/// Dimension-generic counterpart of [`invert_with_retry_3`].
fn invert_with_retry_dyn(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if let Some(inv) = m.clone().try_inverse() {
        return Some(inv);
    }
    let n = m.nrows();
    let reg = (1e-6 * m.diagonal().iter().sum::<f64>().abs() / n as f64).max(1e-8);
    (m + DMatrix::identity(n, n) * reg).try_inverse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Damping;
    use faer::sparse::Triplet;
    use slotmap::KeyData;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn key(i: u64) -> VarKey {
        VarKey::from(KeyData::from_ffi((1u64 << 32) | i))
    }

    fn span(k: u64, col_start: usize, dof: usize) -> BlockSpan {
        BlockSpan {
            key: key(k),
            col_start,
            dof,
        }
    }

    /// Classic BA shape: kept poses first, eliminated points after.
    #[test]
    fn partition_contiguous_bundle_adjustment_layout() -> TestResult {
        let p = SchurPartition::new(
            vec![span(0, 0, 6), span(1, 6, 6)],
            vec![span(2, 12, 3), span(3, 15, 3)],
        )?;
        assert_eq!(p.kept_dof(), 12);
        assert_eq!(p.eliminated_dof(), 6);
        assert_eq!(p.total_dof(), 18);
        assert_eq!(p.kept_local(7), Some(7));
        assert_eq!(p.eliminated_local(16), Some((1, 1)));
        assert_eq!(p.eliminated_offset(1), 3);
        Ok(())
    }

    /// Marginalization eliminates a variable sitting *between* retained ones,
    /// so neither side is a contiguous column range.
    #[test]
    fn partition_handles_non_contiguous_interleaving() -> TestResult {
        // columns: [kept 0..6) [eliminated 6..9) [kept 9..15)
        let p = SchurPartition::new(
            vec![span(0, 0, 6), span(2, 9, 6)],
            vec![span(1, 6, 3)],
        )?;
        assert_eq!(p.kept_dof(), 12);
        assert_eq!(p.eliminated_dof(), 3);

        // Local indices are assigned by ascending global column, skipping the
        // eliminated span entirely.
        assert_eq!(p.kept_local(5), Some(5));
        assert_eq!(p.kept_local(9), Some(6));
        assert_eq!(p.kept_local(14), Some(11));
        assert_eq!(p.kept_local(7), None, "column 7 is eliminated, not kept");
        assert_eq!(p.eliminated_local(7), Some((0, 1)));
        Ok(())
    }

    /// Inverse depth (1 DOF) and 3-D points (3 DOF) in the same problem.
    #[test]
    fn partition_supports_mixed_eliminated_sizes() -> TestResult {
        let p = SchurPartition::new(
            vec![span(0, 0, 6)],
            vec![span(1, 6, 1), span(2, 7, 3), span(3, 10, 1)],
        )?;
        assert_eq!(p.eliminated_dof(), 5);
        assert_eq!(p.eliminated_offset(0), 0);
        assert_eq!(p.eliminated_offset(1), 1);
        assert_eq!(p.eliminated_offset(2), 4);
        assert_eq!(p.eliminated_local(6), Some((0, 0)));
        assert_eq!(p.eliminated_local(9), Some((1, 2)));
        assert_eq!(p.eliminated_local(10), Some((2, 0)));
        Ok(())
    }

    #[test]
    fn partition_rejects_degenerate_and_inconsistent_input() {
        assert!(SchurPartition::new(vec![], vec![span(0, 0, 3)]).is_err());
        assert!(SchurPartition::new(vec![span(0, 0, 6)], vec![]).is_err());

        // Overlapping column ranges.
        let overlap = SchurPartition::new(vec![span(0, 0, 6)], vec![span(1, 4, 3)]);
        assert!(overlap.is_err(), "overlapping ranges must be rejected");

        // A gap: columns 6..9 belong to nobody.
        let gap = SchurPartition::new(vec![span(0, 0, 6)], vec![span(1, 9, 3)]);
        assert!(gap.is_err(), "uncovered columns must be rejected");
    }

    fn hessian_from(
        n: usize,
        entries: &[(usize, usize, f64)],
    ) -> Result<SparseColMat<usize, f64>, Box<dyn std::error::Error>> {
        let triplets: Vec<Triplet<usize, usize, f64>> = entries
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();
        Ok(SparseColMat::try_new_from_triplets(n, n, &triplets)?)
    }

    /// Two eliminated variables sharing a factor make H_ee non-block-diagonal,
    /// which silently invalidates the elimination.
    #[test]
    fn verify_block_diagonal_rejects_coupled_eliminated_variables() -> TestResult {
        let p = SchurPartition::new(vec![span(0, 0, 1)], vec![span(1, 1, 1), span(2, 2, 1)])?;

        let ok =
            hessian_from(3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (0, 1, 0.5), (1, 0, 0.5)])?;
        p.verify_block_diagonal(&ok)?;

        // A (1,2) entry couples the two eliminated variables.
        let bad = hessian_from(
            3,
            &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (1, 2, 0.5), (2, 1, 0.5)],
        )?;
        let Err(err) = p.verify_block_diagonal(&bad) else {
            panic!("coupled eliminated variables must be rejected");
        };
        assert!(err.to_string().contains("block-diagonal"), "{err}");
        Ok(())
    }

    /// Gathering must pick up each block's own entries and ignore coupling to
    /// retained variables.
    #[test]
    fn eliminated_blocks_gather_mixed_sizes() -> TestResult {
        let p = SchurPartition::new(vec![span(0, 0, 1)], vec![span(1, 1, 1), span(2, 2, 3)])?;
        let h = hessian_from(
            5,
            &[
                (0, 0, 9.0),
                (1, 1, 2.0),
                (0, 1, 7.0), // kept↔eliminated: must be ignored here
                (2, 2, 4.0),
                (3, 3, 5.0),
                (4, 4, 6.0),
                (2, 3, 1.0),
                (3, 2, 1.0),
            ],
        )?;

        let mut blocks = EliminatedBlocks::new(&p);
        blocks.gather(&h, &p);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.dof(0), 1);
        assert_eq!(blocks.dof(1), 3);
        assert!((blocks.at(0, 0, 0) - 2.0).abs() < 1e-12);
        assert!((blocks.at(1, 0, 0) - 4.0).abs() < 1e-12);
        assert!((blocks.at(1, 1, 1) - 5.0).abs() < 1e-12);
        assert!((blocks.at(1, 2, 2) - 6.0).abs() < 1e-12);
        assert!((blocks.at(1, 0, 1) - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// Inversion must be correct at every dispatched size, including the
    /// dynamic fallback.
    #[test]
    fn eliminated_blocks_invert_every_size_class() -> TestResult {
        for dof in [1usize, 2, 3, 4, 6, 9] {
            let p = SchurPartition::new(vec![span(0, 0, 1)], vec![span(1, 1, dof)])?;
            let mut entries = vec![(0usize, 0usize, 1.0f64)];
            // Diagonal block with distinct positive entries plus light coupling.
            for k in 0..dof {
                entries.push((1 + k, 1 + k, (k + 2) as f64));
            }
            if dof > 1 {
                entries.push((1, 2, 0.5));
                entries.push((2, 1, 0.5));
            }
            let h = hessian_from(1 + dof, &entries)?;

            let mut blocks = EliminatedBlocks::new(&p);
            blocks.gather(&h, &p);
            let original = DMatrix::from_column_slice(dof, dof, blocks.block(0));
            blocks.invert_in_place(&p)?;
            let inverted = DMatrix::from_column_slice(dof, dof, blocks.block(0));

            let identity = original * inverted;
            for r in 0..dof {
                for c in 0..dof {
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (identity[(r, c)] - want).abs() < 1e-9,
                        "dof={dof}: (H·H⁻¹)[{r},{c}] = {}, want {want}",
                        identity[(r, c)]
                    );
                }
            }
        }
        Ok(())
    }

    /// Damping adds λ·clamp(H_jj) to the diagonal of every block, at any size.
    #[test]
    fn eliminated_blocks_damp_diagonal_only() -> TestResult {
        let p = SchurPartition::new(vec![span(0, 0, 1)], vec![span(1, 1, 1), span(2, 2, 2)])?;
        let h = hessian_from(
            4,
            &[
                (0, 0, 1.0),
                (1, 1, 4.0),
                (2, 2, 8.0),
                (3, 3, 2.0),
                (2, 3, 3.0),
                (3, 2, 3.0),
            ],
        )?;
        let mut blocks = EliminatedBlocks::new(&p);
        blocks.gather(&h, &p);
        blocks.damp(&Damping::new(0.5, 1e-6, 1e32)?);

        assert!((blocks.at(0, 0, 0) - (4.0 + 0.5 * 4.0)).abs() < 1e-12);
        assert!((blocks.at(1, 0, 0) - (8.0 + 0.5 * 8.0)).abs() < 1e-12);
        assert!((blocks.at(1, 1, 1) - (2.0 + 0.5 * 2.0)).abs() < 1e-12);
        assert!(
            (blocks.at(1, 0, 1) - 3.0).abs() < 1e-12,
            "damping must not touch off-diagonal entries"
        );
        Ok(())
    }
}
