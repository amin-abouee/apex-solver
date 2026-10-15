use faer::{
    Accum, Conj, Mat, Par, Side,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    get_global_parallelism,
    linalg::{
        cholesky::llt::{factor, solve},
        matmul::matmul,
        solvers::{LltError, Solve},
    },
};

use crate::error::ErrorLogging;
use crate::linalg::{Damping, DenseMode, LinAlgError, LinAlgResult, LinearSolver};

/// Factorization scratch memory. A clone starts empty and regrows on first use.
struct Scratch(MemBuffer);

impl std::fmt::Debug for Scratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratch")
            .field("bytes", &self.0.len())
            .finish()
    }
}

impl Clone for Scratch {
    fn clone(&self) -> Self {
        Self(MemBuffer::new(StackReq::EMPTY))
    }
}

/// Dense Cholesky (LLT) linear solver for CPU.
///
/// Optimal for small-to-medium problems (< 500 DOF) where the Hessian is
/// moderately dense. Avoids sparse data structure overhead and benefits
/// from dense BLAS routines.
///
/// The normal equations, the factor and the solve live in buffers the solver
/// keeps between calls, so a solve of an unchanged shape allocates only the
/// step it returns.
#[derive(Debug, Clone)]
pub struct DenseCholeskySolver {
    /// Dense Hessian H = J^T · J
    hessian: Option<Mat<f64>>,

    /// Dense gradient g = J^T · r
    gradient: Option<Mat<f64>>,

    /// Lower factor, reused so a solve does not allocate it.
    factor: Mat<f64>,

    /// Right-hand side `-gradient`, solved in place.
    rhs: Mat<f64>,

    /// Scratch memory for the in-place factorization and solve.
    scratch: Scratch,
}

impl DenseCholeskySolver {
    pub fn new() -> Self {
        Self {
            hessian: None,
            gradient: None,
            factor: Mat::zeros(0, 0),
            rhs: Mat::zeros(0, 0),
            scratch: Scratch(MemBuffer::new(StackReq::EMPTY)),
        }
    }

    /// Solve with dense Jacobian directly (the core dense implementation).
    /// `hessian = JᵀJ` and `gradient = Jᵀr` into the retained buffers, with
    /// the exact kernel and parallelism the `*` operator uses.
    fn form_normal_equations(&mut self, residuals: &Mat<f64>, jacobian: &Mat<f64>) {
        let n = jacobian.ncols();
        let par = get_global_parallelism();
        let hessian = self.hessian.get_or_insert_with(|| Mat::zeros(n, n));
        if hessian.nrows() != n || hessian.ncols() != n {
            *hessian = Mat::zeros(n, n);
        }
        matmul(
            hessian.as_mut(),
            Accum::Replace,
            jacobian.transpose(),
            jacobian,
            1.0,
            par,
        );
        let gradient = self
            .gradient
            .get_or_insert_with(|| Mat::zeros(n, residuals.ncols()));
        if gradient.nrows() != n || gradient.ncols() != residuals.ncols() {
            *gradient = Mat::zeros(n, residuals.ncols());
        }
        matmul(
            gradient.as_mut(),
            Accum::Replace,
            jacobian.transpose(),
            residuals,
            1.0,
            par,
        );
    }

    /// Solve `(H + damping)·x = -g` from the retained `H` and `g`, factoring
    /// into reused buffers. The arithmetic is `Llt::new` + `Llt::solve`'s.
    fn solve_retained(
        &mut self,
        damping: Option<&Damping>,
        context: &str,
    ) -> LinAlgResult<Mat<f64>> {
        let par: Par = get_global_parallelism();
        let (Some(hessian), Some(gradient)) = (self.hessian.as_ref(), self.gradient.as_ref())
        else {
            return Err(LinAlgError::FactorizationFailed(format!("{context}: no system")).log());
        };
        let n = hessian.nrows();
        let k = gradient.ncols();

        if self.factor.nrows() != n {
            self.factor = Mat::zeros(n, n);
        }
        self.factor.fill(0.0);
        self.factor.copy_from_triangular_lower(hessian);
        if let Some(damping) = damping {
            for i in 0..n {
                let diagonal = hessian[(i, i)];
                self.factor[(i, i)] = diagonal + damping.diagonal_term(diagonal);
            }
        }

        if self.rhs.nrows() != n || self.rhs.ncols() != k {
            self.rhs = Mat::zeros(n, k);
        }
        for column in 0..k {
            for row in 0..n {
                self.rhs[(row, column)] = -gradient[(row, column)];
            }
        }

        let required = factor::cholesky_in_place_scratch::<f64>(n, par, Default::default())
            .or(solve::solve_in_place_scratch::<f64>(n, k, par));
        if self.scratch.0.len() < required.unaligned_bytes_required() {
            self.scratch = Scratch(MemBuffer::new(required));
        }
        let stack = MemStack::new(&mut self.scratch.0);
        factor::cholesky_in_place(
            self.factor.as_mut(),
            Default::default(),
            par,
            stack,
            Default::default(),
        )
        .map_err(|e: LltError| map_llt_error(e, context))?;
        for column in 0..n {
            for row in 0..column {
                self.factor[(row, column)] = 0.0;
            }
        }
        let stack = MemStack::new(&mut self.scratch.0);
        solve::solve_in_place_with_conj(
            self.factor.as_ref(),
            Conj::No,
            self.rhs.as_mut(),
            par,
            stack,
        );
        Ok(self.rhs.clone())
    }

    fn solve_dense_normal(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.form_normal_equations(residuals, jacobian);
        self.solve_retained(None, "Dense Cholesky factorization failed")
    }

    /// Solve with dense Jacobian and damping (the core dense augmented implementation).
    ///
    /// `H_aug = H + λ·D` with `D_jj = clamp(H_jj, min_diagonal, max_diagonal)`;
    /// the un-augmented Hessian is what stays cached, since DogLeg and LM need
    /// the true quadratic model.
    fn solve_dense_augmented(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.form_normal_equations(residuals, jacobian);
        self.solve_retained(
            Some(damping),
            "Augmented dense Cholesky factorization failed",
        )
    }
}

/// Solve `a · x = b` for SPD `a` via dense Cholesky (LLT).
///
/// Shared by every dense solver that reduces to one SPD system —
/// `ExplicitDenseSchur`'s reduced camera system — so the factorization and
/// error-mapping logic exists in one place.
pub(crate) fn solve_spd(a: &Mat<f64>, b: &Mat<f64>, context: &str) -> LinAlgResult<Mat<f64>> {
    let llt = a
        .as_ref()
        .llt(Side::Lower)
        .map_err(|e| map_llt_error(e, context))?;
    Ok(llt.solve(b))
}

impl Default for DenseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LinearSolver<DenseMode>
// ============================================================================

impl LinearSolver<DenseMode> for DenseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_dense_normal(residuals, jacobian)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.solve_dense_augmented(residuals, jacobian, damping)
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        Some(
            <DenseMode as crate::linearizer::AssemblyBackend>::hessian_vec_product(
                self.hessian.as_ref()?,
                v,
            ),
        )
    }

    fn get_hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

/// Map faer's LLT error to our LinAlgError
fn map_llt_error(e: LltError, context: &str) -> LinAlgError {
    LinAlgError::FactorizationFailed(format!("{context}: {e:?}")).log()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn create_test_data() -> (Mat<f64>, Mat<f64>) {
        // 4×2 Jacobian (overdetermined)
        let mut j = Mat::zeros(4, 2);
        j[(0, 0)] = 2.0;
        j[(0, 1)] = 1.0;
        j[(1, 0)] = 1.0;
        j[(1, 1)] = 3.0;
        j[(2, 0)] = 1.0;
        j[(2, 1)] = 1.0;
        j[(3, 0)] = 0.5;
        j[(3, 1)] = 2.0;

        let mut r = Mat::zeros(4, 1);
        r[(0, 0)] = 1.0;
        r[(1, 0)] = 2.0;
        r[(2, 0)] = 0.5;
        r[(3, 0)] = 1.5;

        (j, r)
    }

    #[test]
    fn test_dense_cholesky_solve_normal() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver = DenseCholeskySolver::new();

        let dx = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j)?;

        // Verify: J^T·J·dx ≈ -J^T·r
        let jtj = j.transpose() * &j;
        let jtr = j.transpose() * &r;
        let residual = &jtj * &dx + &jtr;

        for i in 0..dx.nrows() {
            assert!(
                residual[(i, 0)].abs() < TOLERANCE,
                "Residual at index {}: {}",
                i,
                residual[(i, 0)]
            );
        }

        assert!(solver.hessian.is_some());
        assert!(solver.gradient.is_some());

        Ok(())
    }

    #[test]
    fn test_dense_cholesky_solve_augmented() -> TestResult {
        let (j, r) = create_test_data();
        let lambda = 0.1;
        let mut solver = DenseCholeskySolver::new();

        let dx = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver,
            &r,
            &j,
            &Damping::identity(lambda),
        )?;

        // Verify: (J^T·J + λI)·dx ≈ -J^T·r
        let mut jtj = j.transpose() * &j;
        let jtr = j.transpose() * &r;
        for i in 0..jtj.nrows() {
            jtj[(i, i)] += lambda;
        }
        let residual = &jtj * &dx + &jtr;

        for i in 0..dx.nrows() {
            assert!(
                residual[(i, 0)].abs() < TOLERANCE,
                "Residual at index {}: {}",
                i,
                residual[(i, 0)]
            );
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // New tests for previously uncovered code paths
    // -------------------------------------------------------------------------

    /// get_hessian() and get_gradient() should return None before any solve.
    #[test]
    fn test_accessors_before_solve() {
        let solver = DenseCholeskySolver::new();
        assert!(
            LinearSolver::<DenseMode>::get_hessian(&solver).is_none(),
            "hessian should be None before solve"
        );
        assert!(
            LinearSolver::<DenseMode>::get_gradient(&solver).is_none(),
            "gradient should be None before solve"
        );
    }

    /// DenseCholeskySolver::default() should behave identically to new().
    #[test]
    fn test_default_equals_new() {
        let solver = DenseCholeskySolver::default();
        assert!(solver.hessian.is_none());
        assert!(solver.gradient.is_none());
    }

    /// A rank-deficient (singular) Jacobian should return an Err from solve_normal_equation.
    #[test]
    fn test_singular_jacobian_returns_error() {
        let mut solver = DenseCholeskySolver::new();

        // Jacobian with an all-zero column → H = J^T J is singular
        let mut j = Mat::zeros(3, 2);
        j[(0, 0)] = 1.0;
        j[(1, 0)] = 2.0;
        j[(2, 0)] = 3.0;
        // column 1 is all zeros

        let mut r = Mat::zeros(3, 1);
        r[(0, 0)] = 1.0;

        let result = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &r, &j);
        assert!(
            result.is_err(),
            "Singular Jacobian must produce a factorization error"
        );
    }

    /// solve_augmented_equation with lambda=0 should produce the same result as solve_normal_equation.
    #[test]
    fn test_augmented_with_zero_lambda_matches_normal() -> TestResult {
        let (j, r) = create_test_data();
        let mut solver_n = DenseCholeskySolver::new();
        let mut solver_a = DenseCholeskySolver::new();

        let dx_normal = LinearSolver::<DenseMode>::solve_normal_equation(&mut solver_n, &r, &j)?;
        let dx_augmented = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver_a,
            &r,
            &j,
            &Damping::identity(0.0),
        )?;

        for i in 0..dx_normal.nrows() {
            assert!(
                (dx_normal[(i, 0)] - dx_augmented[(i, 0)]).abs() < TOLERANCE,
                "Element {} differs: normal={}, augmented(λ=0)={}",
                i,
                dx_normal[(i, 0)],
                dx_augmented[(i, 0)]
            );
        }
        Ok(())
    }

    #[test]
    fn retained_buffers_match_the_allocating_arithmetic_bit_for_bit() -> TestResult {
        let mut state = 17u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let damping = Damping::identity(1e-10);
        let mut solver = DenseCholeskySolver::new();
        for rows in [52usize, 52, 9, 52] {
            let jacobian = Mat::<f64>::from_fn(rows, 3, |_, _| next() * 3.0);
            let residual = Mat::<f64>::from_fn(rows, 1, |_, _| next() * 2.0);

            let hessian = jacobian.transpose() * &jacobian;
            let gradient = jacobian.transpose() * &residual;
            let mut augmented = hessian.clone();
            for i in 0..3 {
                augmented[(i, i)] += damping.diagonal_term(hessian[(i, i)]);
            }
            let expected = solve_spd(&augmented, &(-&gradient), "reference")?;

            let step = solver.solve_augmented_equation(&residual, &jacobian, &damping)?;
            assert_eq!(step, expected);
            assert_eq!(solver.get_hessian(), Some(&hessian));
            assert_eq!(solver.get_gradient(), Some(&gradient));
        }
        Ok(())
    }

    #[test]
    fn retained_buffers_report_a_non_positive_definite_system() {
        let mut solver = DenseCholeskySolver::new();
        let jacobian = Mat::<f64>::zeros(4, 2);
        let residual = Mat::<f64>::from_fn(4, 1, |_, _| 1.0);
        assert!(solver.solve_normal_equation(&residual, &jacobian).is_err());
    }
}
