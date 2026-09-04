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
//! deterministic FNV-1a hash over the full column-row structure, so equality
//! means "same pattern" and not just "same shape". The hash costs one O(nnz)
//! integer pass per check — negligible next to the floating-point assembly it
//! guards. (FNV, not `std`'s `DefaultHasher`: the latter is randomly seeded
//! per instance and cannot be compared across caches.)

use faer::sparse::SparseColMat;

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
        let symbolic = matrix.symbolic();
        // One xor-multiply per structural integer — the same pass `compute_nnz`
        // style checks already pay, against O(nnz) floating-point assembly
        // plus factorization downstream.
        let mut hash: u64 = 0xcbf29ce484222325;
        hash = fold(hash, matrix.nrows());
        hash = fold(hash, matrix.ncols());
        let mut nnz = 0usize;
        for col in 0..matrix.ncols() {
            let rows = symbolic.row_idx_of_col_raw(col);
            hash = fold(hash, rows.len());
            for &row in rows {
                hash = fold(hash, row);
                nnz += 1;
            }
        }
        Self {
            nrows: matrix.nrows(),
            ncols: matrix.ncols(),
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
