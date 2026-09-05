//! Products with `J` that never form `JᵀJ`.
//!
//! Every Schur solver that eliminates straight from the Jacobian needs the same
//! handful of primitives: the gradient `Jᵀr`, the diagonal of `JᵀJ` for
//! damping, dot products between two columns (which are entries of `JᵀJ`
//! computed one at a time), and the action `Jᵀ(J·v)` the optimizers use for
//! their quadratic model.
//!
//! Each is `O(nnz(J))` and none allocates a matrix. That is the whole point:
//! `JᵀJ` for a large bundle-adjustment problem carries a dense 9×9 block for
//! every pair of cameras sharing a landmark, fill-in that `J` itself does not
//! have, so anything expressed against `J` is cheaper to evaluate *and* to
//! store than the same thing expressed against `JᵀJ`.
//!
//! Row indices within a CSC column are sorted, which [`column_dot`] relies on.

use faer::Mat;
use faer::sparse::SparseColMat;
use rayon::prelude::*;

/// Dot product of columns `a` and `b` of `j` — that is, the `(a, b)` entry of
/// `JᵀJ`, computed without forming any of the rest of it.
///
/// A sorted merge over the two columns' row indices, so the cost is the sum of
/// their nonzero counts rather than their product.
pub fn column_dot(j: &SparseColMat<usize, f64>, a: usize, b: usize) -> f64 {
    let symbolic = j.symbolic();
    let rows_a = symbolic.row_idx_of_col_raw(a);
    let rows_b = symbolic.row_idx_of_col_raw(b);
    let vals_a = j.val_of_col(a);
    let vals_b = j.val_of_col(b);

    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut acc = 0.0;
    while ia < rows_a.len() && ib < rows_b.len() {
        match rows_a[ia].cmp(&rows_b[ib]) {
            std::cmp::Ordering::Less => ia += 1,
            std::cmp::Ordering::Greater => ib += 1,
            std::cmp::Ordering::Equal => {
                acc += vals_a[ia] * vals_b[ib];
                ia += 1;
                ib += 1;
            }
        }
    }
    acc
}

/// `Jᵀ·v` for a column vector `v` of length `J.nrows()`.
///
/// One independent gather per column, so it parallelizes without contention:
/// each output entry is written by exactly one task.
pub fn jt_vec(j: &SparseColMat<usize, f64>, v: &Mat<f64>) -> Mat<f64> {
    let symbolic = j.symbolic();
    let out: Vec<f64> = (0..j.ncols())
        .into_par_iter()
        .map(|col| {
            let rows = symbolic.row_idx_of_col_raw(col);
            let vals = j.val_of_col(col);
            rows.iter()
                .zip(vals)
                .map(|(&row, val)| val * v[(row, 0)])
                .sum()
        })
        .collect();
    Mat::from_fn(j.ncols(), 1, |i, _| out[i])
}

/// Diagonal of `JᵀJ`: the squared norm of each column.
///
/// This is what Levenberg-Marquardt's `D = clamp(diag(JᵀJ), …)` needs, and the
/// only part of `JᵀJ` a matrix-free solver has to know.
pub fn diag_jt_j(j: &SparseColMat<usize, f64>) -> Vec<f64> {
    (0..j.ncols())
        .into_par_iter()
        .map(|col| j.val_of_col(col).iter().map(|v| v * v).sum())
        .collect()
}

/// `Jᵀ(J·v)` — the action of the un-damped `JᵀJ` on `v`, from `J` alone.
///
/// Used by solvers that never materialize `JᵀJ` to serve
/// [`LinearSolver::hessian_vec_product`](crate::linalg::LinearSolver::hessian_vec_product),
/// which the optimizers evaluate their quadratic model through.
///
/// The forward product scatters into rows, so it is left serial: a parallel
/// scatter would need either per-thread row buffers (one full residual vector
/// per thread) or atomics, and this is called once per optimizer iteration
/// rather than once per PCG iteration.
pub fn jt_j_vec_product(j: &SparseColMat<usize, f64>, v: &Mat<f64>) -> Mat<f64> {
    let symbolic = j.symbolic();

    let mut jv = Mat::<f64>::zeros(j.nrows(), 1);
    for col in 0..j.ncols() {
        let x = v[(col, 0)];
        if x == 0.0 {
            continue;
        }
        let rows = symbolic.row_idx_of_col_raw(col);
        let vals = j.val_of_col(col);
        for (idx, &row) in rows.iter().enumerate() {
            jv[(row, 0)] += vals[idx] * x;
        }
    }

    jt_vec(j, &jv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// J = [[1, 2], [3, 4], [0, 5]]
    fn sample() -> Result<SparseColMat<usize, f64>, Box<dyn std::error::Error>> {
        let triplets = vec![
            Triplet::new(0usize, 0usize, 1.0f64),
            Triplet::new(1, 0, 3.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 1, 4.0),
            Triplet::new(2, 1, 5.0),
        ];
        Ok(SparseColMat::try_new_from_triplets(3, 2, &triplets).map_err(|e| format!("{e:?}"))?)
    }

    #[test]
    fn column_dot_matches_jt_j_entries() -> TestResult {
        let j = sample()?;
        // JᵀJ = [[10, 14], [14, 45]]
        assert!((column_dot(&j, 0, 0) - 10.0).abs() < 1e-12);
        assert!((column_dot(&j, 0, 1) - 14.0).abs() < 1e-12);
        assert!((column_dot(&j, 1, 1) - 45.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn diag_matches_column_dots() -> TestResult {
        let j = sample()?;
        let diag = diag_jt_j(&j);
        assert!((diag[0] - 10.0).abs() < 1e-12);
        assert!((diag[1] - 45.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn jt_vec_matches_hand_computation() -> TestResult {
        let j = sample()?;
        let r = Mat::from_fn(3, 1, |i, _| (i + 1) as f64); // [1,2,3]
        let g = jt_vec(&j, &r);
        // Jᵀr = [1*1+3*2, 2*1+4*2+5*3] = [7, 25]
        assert!((g[(0, 0)] - 7.0).abs() < 1e-12);
        assert!((g[(1, 0)] - 25.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn jt_j_vec_product_matches_dense_jt_j() -> TestResult {
        let j = sample()?;
        let v = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { -1.0 });
        let out = jt_j_vec_product(&j, &v);
        // JᵀJ·v = [10*2 + 14*(-1), 14*2 + 45*(-1)] = [6, -17]
        assert!((out[(0, 0)] - 6.0).abs() < 1e-12);
        assert!((out[(1, 0)] + 17.0).abs() < 1e-12);
        Ok(())
    }
}
