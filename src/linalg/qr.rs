use std::ops::Mul;

use super::sparse::SparseLinearSolver;
use faer::{
    Mat,
    linalg::solvers::SolveLstsqCore,
    sparse::{SparseColMat, linalg::solvers},
};

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    // Symbolic factorization can be cached if the sparsity pattern of the matrix is constant.
    // For augmented systems where lambda changes, this might not be safe to reuse.
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
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        weights: &Mat<f64>,
    ) -> Option<Mat<f64>> {
        let m = jacobians.nrows();

        // Create a sparse diagonal matrix from the weights vector.
        let mut w_triplets = Vec::with_capacity(m);
        for i in 0..m {
            w_triplets.push((i, i, weights.read(i, 0)));
        }
        let weights_diag = SparseColMat::try_new_from_triplets(m, m, &w_triplets).unwrap();

        // Form the normal equations explicitly: H = J^T * W * J
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

        // Solve the square system H * dx = g using QR decomposition.
        // The symbolic factorization can be cached since the sparsity of H is constant.
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern = Some(solvers::SymbolicQr::try_new(hessian.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), hessian.as_ref()) {
            let mut solution = gradient;
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, solution.as_mut());
            Some(solution)
        } else {
            None
        }
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        weights: &Mat<f64>,
        lambda: f64,
    ) -> Option<Mat<f64>> {
        let m = jacobians.nrows();
        let n = jacobians.ncols();

        // This problem (J'*W*J + lambda*I)x = -J'*W*r can be solved by constructing
        // an augmented least squares system.

        // 1. Create the sqrt of the weights diagonal matrix: sqrt(W)
        let mut sqrt_w_triplets = Vec::with_capacity(m);
        for i in 0..m {
            sqrt_w_triplets.push((i, i, weights.read(i, 0).sqrt()));
        }
        let sqrt_weights_diag =
            SparseColMat::try_new_from_triplets(m, m, &sqrt_w_triplets).unwrap();

        // 2. Form the top part of the augmented Jacobian: J_top = sqrt(W) * J
        let weighted_jacobians = sqrt_weights_diag.as_ref() * jacobians.as_ref();

        // 3. Form the bottom part of the augmented Jacobian: J_bottom = sqrt(lambda) * I
        let mut lambda_i_triplets = Vec::with_capacity(n);
        let sqrt_lambda = lambda.sqrt();
        for i in 0..n {
            lambda_i_triplets.push((i, i, sqrt_lambda));
        }
        let lambda_i = SparseColMat::try_new_from_triplets(n, n, &lambda_i_triplets).unwrap();

        // 4. Stack J_top and J_bottom to create the augmented Jacobian J_aug
        let j_aug = SparseColMat::from_blocks(&[
            [Some(weighted_jacobians.as_ref())],
            [Some(lambda_i.as_ref())],
        ])
        .unwrap();

        // 5. Form the augmented residual vector: r_aug = [ -sqrt(W)*r; 0 ]
        let weighted_residuals = sqrt_weights_diag.as_ref() * residuals.as_ref();
        let mut r_aug = Mat::<f64>::zeros(m + n, 1);
        r_aug
            .as_mut()
            .submatrix_mut(0, 0, m, 1)
            .copy_from(&(-weighted_residuals));

        // 6. Solve the least squares problem: J_aug * dx = r_aug
        // We don't cache the symbolic factorization because the structure of J_aug
        // depends on lambda and the weights, which can change.
        if let Ok(qr) = solvers::Qr::try_new(j_aug.as_ref()) {
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, r_aug.as_mut());
            // The solution dx is in the top n rows of the result.
            Some(r_aug.submatrix(0, 0, n, 1).to_owned())
        } else {
            None
        }
    }
}
