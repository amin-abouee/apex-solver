use faer::{
    Mat,
    sparse::{SparseColMat, Triplet},
};

use crate::linalg::{LinAlgError, LinAlgResult};

/// Convert a sparse CSC matrix to a dense matrix.
///
/// This is O(nnz) for reading entries and O(nrows × ncols) for the output.
/// Used by dense solvers that accept sparse input from `Problem`.
#[inline]
pub fn sparse_to_dense(sparse: &SparseColMat<usize, f64>) -> Mat<f64> {
    let nrows = sparse.nrows();
    let ncols = sparse.ncols();
    let mut dense = Mat::zeros(nrows, ncols);

    // Iterate over each column of the CSC matrix
    let sparse_ref = sparse.as_ref();
    for col in 0..ncols {
        for (row, val) in sparse_ref
            .row_idx_of_col(col)
            .zip(sparse_ref.val_of_col(col))
        {
            dense[(row, col)] = *val;
        }
    }

    dense
}

/// Convert a dense matrix to a sparse CSC matrix, keeping only non-zero entries.
///
/// Used by dense solvers that need to return a sparse Hessian for compatibility
/// with the optimizer's observer system.
///
/// # Arguments
/// * `dense` — Dense matrix to convert
/// * `threshold` — Values with absolute value below this are treated as zero (default: 1e-15)
pub fn dense_to_sparse(dense: &Mat<f64>, threshold: f64) -> LinAlgResult<SparseColMat<usize, f64>> {
    let nrows = dense.nrows();
    let ncols = dense.ncols();

    // Collect triplets (row, col, value) for non-zero entries
    let mut triplets = Vec::new();
    for col in 0..ncols {
        for row in 0..nrows {
            let val = dense[(row, col)];
            if val.abs() > threshold {
                triplets.push(Triplet::new(row, col, val));
            }
        }
    }

    // Build sparse CSC matrix from triplets
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|e| {
        LinAlgError::SparseMatrixCreation(format!("dense_to_sparse failed: {e:?}")).log()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_sparse_to_dense_roundtrip() -> TestResult {
        // Create a small sparse matrix
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, 3.0),
            Triplet::new(1, 1, 4.0),
        ];
        let sparse = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;

        let dense = sparse_to_dense(&sparse);
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 1)], 2.0);
        assert_eq!(dense[(1, 0)], 3.0);
        assert_eq!(dense[(1, 1)], 4.0);
        Ok(())
    }

    #[test]
    fn test_dense_to_sparse_roundtrip() -> TestResult {
        let mut dense = Mat::zeros(3, 3);
        dense[(0, 0)] = 5.0;
        dense[(1, 1)] = 3.0;
        dense[(2, 2)] = 1.0;
        dense[(0, 2)] = 0.5;

        let sparse = dense_to_sparse(&dense, 1e-15)?;
        let back = sparse_to_dense(&sparse);

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (dense[(i, j)] - back[(i, j)]).abs() < 1e-14,
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    dense[(i, j)],
                    back[(i, j)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_sparse_to_dense_empty_columns() -> TestResult {
        // 3×3 matrix with only diagonal entries — middle column empty
        let triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(2, 2, 9.0)];
        let sparse = SparseColMat::try_new_from_triplets(3, 3, &triplets)?;

        let dense = sparse_to_dense(&sparse);
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(1, 1)], 0.0);
        assert_eq!(dense[(2, 2)], 9.0);
        assert_eq!(dense[(0, 1)], 0.0);
        Ok(())
    }
}
