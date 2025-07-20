use std::fmt::Debug;
use std::ops::Mul;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers;

use super::sparse::SparseLinearSolver;

#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    symbolic_pattern: Option<solvers::SymbolicLlt<usize>>,
}

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver {
            symbolic_pattern: None,
        }
    }
}
impl Default for SparseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseLinearSolver for SparseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>> {
        let hessian = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobians.as_ref());
        let gradient = jacobians.as_ref().transpose().mul(-residuals);

        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicLlt::try_new(hessian.symbolic(), faer::Side::Lower).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym.clone(), hessian.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(gradient);

            Some(dx)
        } else {
            None
        }
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
        lambda: f64,
    ) -> Option<faer::Mat<f64>> {
        let m = jacobians.nrows();
        let n = jacobians.ncols();

        // Create a sparse diagonal matrix from weights
        let mut w_triplets = Vec::with_capacity(m);
        for i in 0..m {
            w_triplets.push((i, i, weights.read(i, 0)));
        }
        let weights_diag =
            faer::sparse::SparseColMat::try_new_from_triplets(m, m, &w_triplets).unwrap();

        // H = J^T * W * J
        let hessian = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(weights_diag.as_ref().mul(jacobians.as_ref()));

        // g = J^T * W * -r
        let gradient = jacobians
            .as_ref()
            .transpose()
            .mul(weights_diag.as_ref().mul(-residuals));

        // H_aug = H + lambda * I
        let mut lambda_i_triplets = Vec::with_capacity(n);
        for i in 0..n {
            lambda_i_triplets.push((i, i, lambda));
        }
        let lambda_i =
            faer::sparse::SparseColMat::try_new_from_triplets(n, n, &lambda_i_triplets).unwrap();

        let augmented_hessian = hessian + lambda_i;

        // The sparsity pattern of the augmented Hessian might change between iterations
        // if weights or lambda change, so we don't cache the symbolic factorization.
        let sym =
            solvers::SymbolicLlt::try_new(augmented_hessian.symbolic(), faer::Side::Lower).unwrap();
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym, augmented_hessian.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(gradient);

            Some(dx)
        } else {
            None
        }
    }
}
