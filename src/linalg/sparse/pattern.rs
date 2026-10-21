//! Structural fingerprints for sparse-matrix caches.
//!
//! Several solvers cache work keyed on "the sparsity pattern hasn't changed":
//! the normal-equations symbolic product, the chunk-wise Schur layout, the
//! `H_ee` block-diagonality verdict and the implicit-Schur visibility index.
//! All of them used to key on the `(nrows, ncols, nnz)` triple, which aliases
//! patterns that permute entries at equal nonzero count — reusing a stale
//! cache for such a pattern silently produces a wrong step.
//!
//! [`PatternFingerprint`] keeps the triple as a fast reject but adds a
//! deterministic FNV-1a-style hash over the full column-row structure, so
//! equality means "same pattern" and not just "same shape". The hash costs one
//! O(nnz) integer pass per check, run in independent lanes so it is limited by
//! multiply throughput and not by the latency of one dependent chain. (FNV, not
//! `std`'s `DefaultHasher`: the latter is randomly seeded per instance and
//! cannot be compared across caches.)

use faer::sparse::{SparseColMat, SymbolicSparseColMatRef};

/// Fold one value into an FNV-1a hash.
///
/// Values are pre-mixed with a golden-ratio multiply because raw structural
/// indices (0, 1, 2, …) carry entropy only in their low bits; without the mix
/// consecutive columns would cancel each other out of the hash.
#[inline]
fn fold(mut hash: u64, value: usize) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    const MIX: u64 = 0x9E3779B97F4A7C15;
    hash ^= (value as u64).wrapping_mul(MIX);
    hash.wrapping_mul(PRIME)
}

/// Independent hash lanes of [`fold_slice`].
const LANES: usize = 4;

/// Fold a run of structural indices into `seed`.
///
/// One FNV chain is a dependent multiply per value, about a cycle count of the
/// multiply's latency each. Values are dealt round-robin to [`LANES`] chains
/// that run side by side, then the chains are folded together in order, so the
/// result still depends on every value and on its position.
#[inline]
fn fold_slice(seed: u64, values: &[usize]) -> u64 {
    let mut lanes = [seed; LANES];
    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane = fold(*lane, value);
        }
    }
    for (lane, &value) in lanes.iter_mut().zip(remainder) {
        *lane = fold(*lane, value);
    }
    lanes
        .iter()
        .fold(seed, |hash, &lane| fold(hash, lane as usize))
}

/// Structural identity of a sparse matrix: dimensions, nonzero count, and a
/// hash of the full sparsity pattern.
///
/// Compare with `==`: the three cheap fields reject first, so the common
/// "same pattern" case costs three integer compares and the "different shape"
/// case never touches the hash.
///
/// `Default` is the all-zero placeholder (never equal to a real fingerprint);
/// caches holding it must overwrite it at build time, as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PatternFingerprint {
    nrows: usize,
    ncols: usize,
    nnz: usize,
    hash: u64,
}

impl PatternFingerprint {
    /// Fingerprint the sparsity pattern of `matrix` (values ignored).
    pub fn of(matrix: &SparseColMat<usize, f64>) -> Self {
        Self::of_symbolic(matrix.symbolic())
    }

    /// Fingerprint a sparsity pattern given without values.
    ///
    /// Equal to [`Self::of`] for the matrix that owns the pattern, so a cache
    /// that holds the pattern can fingerprint it once and hand the result to
    /// whoever would otherwise hash the matrix again.
    pub fn of_symbolic(symbolic: SymbolicSparseColMatRef<'_, usize>) -> Self {
        let (nrows, ncols) = (symbolic.nrows(), symbolic.ncols());
        let mut hash: u64 = 0xcbf29ce484222325;
        hash = fold(hash, nrows);
        hash = fold(hash, ncols);

        // Compressed storage, the usual case: the column pointers and the row
        // indices are two flat runs.
        if symbolic.col_nnz().is_none() {
            let col_ptr = symbolic.col_ptr();
            if let (Some(&start), Some(&end)) = (col_ptr.first(), col_ptr.last())
                && let Some(row_idx) = symbolic.row_idx().get(start..end)
            {
                hash = fold_slice(hash, col_ptr);
                hash = fold_slice(hash, row_idx);
                return Self {
                    nrows,
                    ncols,
                    nnz: row_idx.len(),
                    hash,
                };
            }
        }

        // Columns that do not fill their range: walk them one by one.
        let mut nnz = 0usize;
        for col in 0..ncols {
            let rows = symbolic.row_idx_of_col_raw(col);
            hash = fold(hash, rows.len());
            hash = fold_slice(hash, rows);
            nnz += rows.len();
        }
        Self {
            nrows,
            ncols,
            nnz,
            hash,
        }
    }

    /// Rows of the fingerprinted matrix.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Columns of the fingerprinted matrix.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Nonzero count of the fingerprinted matrix.
    pub fn nnz(&self) -> usize {
        self.nnz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn equal_patterns_agree() -> TestResult {
        let t = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 3.0),
        ];
        let a = SparseColMat::try_new_from_triplets(2, 2, &t)?;
        let b = SparseColMat::try_new_from_triplets(2, 2, &t)?;
        assert_eq!(PatternFingerprint::of(&a), PatternFingerprint::of(&b));
        // Values are ignored: same pattern, different values still match.
        let t2 = vec![
            Triplet::new(0usize, 0usize, 9.0f64),
            Triplet::new(1, 0, 8.0),
            Triplet::new(1, 1, 7.0),
        ];
        let c = SparseColMat::try_new_from_triplets(2, 2, &t2)?;
        assert_eq!(PatternFingerprint::of(&a), PatternFingerprint::of(&c));
        Ok(())
    }

    /// A pattern wider than the lane count, so every lane and the remainder
    /// hold entries: moving any single entry to another row must show.
    #[test]
    fn a_moved_entry_changes_the_fingerprint_in_every_lane() -> TestResult {
        let rows = 11usize;
        let cols = 9usize;
        let build = |moved: Option<(usize, usize)>| -> Result<_, Box<dyn std::error::Error>> {
            let mut triplets = Vec::new();
            for col in 0..cols {
                for k in 0..3 {
                    let mut row = (col * 2 + k * 3) % rows;
                    if moved == Some((col, k)) {
                        row = (row + 1) % rows;
                    }
                    triplets.push(Triplet::new(row, col, 1.0f64));
                }
            }
            Ok(SparseColMat::try_new_from_triplets(rows, cols, &triplets)?)
        };
        let base = PatternFingerprint::of(&build(None)?);
        assert_eq!(base, PatternFingerprint::of(&build(None)?));
        for col in 0..cols {
            for k in 0..3 {
                let moved = build(Some((col, k)))?;
                assert_ne!(
                    base,
                    PatternFingerprint::of(&moved),
                    "moving entry {k} of column {col} must change the fingerprint"
                );
            }
        }
        Ok(())
    }

    /// The symbolic form is what a cache that holds only the pattern calls.
    #[test]
    fn the_symbolic_form_agrees_with_the_matrix_form() -> TestResult {
        let t = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(3, 0, 2.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 2, 3.0),
            Triplet::new(3, 3, 3.0),
        ];
        let a = SparseColMat::try_new_from_triplets(4, 4, &t)?;
        assert_eq!(
            PatternFingerprint::of(&a),
            PatternFingerprint::of_symbolic(a.symbolic())
        );
        Ok(())
    }

    #[test]
    fn permuted_pattern_at_equal_nnz_disagrees() -> TestResult {
        let t = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 3.0),
        ];
        let a = SparseColMat::try_new_from_triplets(2, 2, &t)?;
        // Same shape (2x2) and same nnz (3), but column 1 holds row 0
        // instead of row 1: a genuinely different pattern.
        let t2 = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 2.0),
            Triplet::new(0, 1, 3.0),
        ];
        let b = SparseColMat::try_new_from_triplets(2, 2, &t2)?;
        assert_eq!((a.nrows(), a.ncols(), 3), (b.nrows(), b.ncols(), 3));
        assert_ne!(
            PatternFingerprint::of(&a),
            PatternFingerprint::of(&b),
            "equal-nnz permutation must invalidate the fingerprint"
        );
        Ok(())
    }
}
