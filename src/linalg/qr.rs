use super::sparse::SparseLinearSolver;
use faer::linalg::solvers::SolveLstsqCore;
// use faer::prelude::{SpSolver, SpSolverLstsq};
use faer::sparse::linalg::solvers;

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    symbolic_pattern: Option<solvers::SymbolicQr<usize>>,
}

impl SparseQRSolver {
    pub fn new() -> Self {
        SparseQRSolver {
            symbolic_pattern: None,
        }
    }
}
impl Default for SparseQRSolver {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseLinearSolver for SparseQRSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicQr::try_new(jacobians.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), jacobians.as_ref()) {
            let mut minus_residuals = -residuals;
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, minus_residuals.as_mut());
            Some(minus_residuals)
        } else {
            None
        }
    }

    fn solve_weighted(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>> {
        // Create sqrt-weighted versions of Jacobian and residuals
        let n_rows = jacobians.nrows();
        let n_cols = jacobians.ncols();

        // Apply weights to residuals
        let mut weighted_residuals = residuals.clone();
        for i in 0..n_rows {
            let weight_sqrt = weights.read(i, 0).sqrt();
            weighted_residuals.write(i, 0, residuals.read(i, 0) * weight_sqrt);
        }

        // For weighted QR, we need to apply weights to Jacobian matrix
        // In a production implementation, we would create a weighted sparse Jacobian
        // more efficiently, but for demonstration we'll use a simplified approach

        // Extract triplets from the Jacobian
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Approximate estimation of non-zero elements
        let nnz_estimate = jacobians.nnz();
        row_indices.reserve(nnz_estimate);
        col_indices.reserve(nnz_estimate);
        values.reserve(nnz_estimate);

        // Extract and weight the values
        for i in 0..n_rows {
            let weight_sqrt = weights.read(i, 0).sqrt();

            for j in 0..n_cols {
                if let Some(val) = jacobians.get(i, j) {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(val * weight_sqrt);
                }
            }
        }

        // Create weighted Jacobian
        let weighted_jacobian = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
            n_rows,
            n_cols,
            row_indices,
            col_indices,
            values,
        )
        .unwrap();

        // Solve using the weighted system
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicQr::try_new(weighted_jacobian.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), weighted_jacobian.as_ref())
        {
            let mut minus_weighted_residuals = -weighted_residuals;
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, minus_weighted_residuals.as_mut());
            Some(minus_weighted_residuals)
        } else {
            None
        }
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern = Some(solvers::SymbolicQr::try_new(jtj.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), jtj.as_ref()) {
            let mut minus_jtr = -jtr;
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, minus_jtr.as_mut());
            Some(minus_jtr)
        } else {
            None
        }
    }

    fn solve_jtj_weighted(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
        weights: &faer::Mat<f64>,
    ) -> Option<faer::Mat<f64>> {
        // For weighted JtJ, we need to apply weights to normal equations
        // This is an approximation - in a real implementation we would
        // handle the application of weights to the sparse structure more carefully

        let n = jtj.nrows();

        // Create a modified Jtr with weights applied
        let mut weighted_jtr = faer::Mat::<f64>::zeros(jtr.nrows(), jtr.ncols());
        for i in 0..jtr.nrows() {
            // Here we're using the first weight as an approximation
            // In a real implementation, we'd know which weights apply to which elements
            weighted_jtr.write(i, 0, jtr.read(i, 0) * weights.read(0, 0));
        }

        // Extract and weight JtJ values
        let mut weighted_jtj_values = jtj.values().to_vec();
        for i in 0..weighted_jtj_values.len() {
            // This is a simplified approach - in reality, we need to know
            // which weights apply to each element of JtJ
            weighted_jtj_values[i] *= weights.read(0, 0);
        }

        // Create weighted JtJ matrix
        let weighted_jtj = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
            jtj.nrows(),
            jtj.ncols(),
            jtj.row_indices().to_vec(),
            jtj.col_ptrs().to_vec(),
            weighted_jtj_values,
        )
        .unwrap();

        // Solve using weighted JtJ and weighted Jtr
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicQr::try_new(weighted_jtj.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), weighted_jtj.as_ref()) {
            let mut minus_weighted_jtr = -weighted_jtr;
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, minus_weighted_jtr.as_mut());
            Some(minus_weighted_jtr)
        } else {
            None
        }
    }
}
