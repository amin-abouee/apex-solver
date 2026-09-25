//! Gaussian marginal prior over a set of variables (GTSAM iSAM2
//! `LinearContainerFactor` analogue).
//!
//! When a sliding window marginalizes old poses, the eliminated joint is
//! summarized as a Gaussian over the *remaining* variables. This factor
//! represents that Gaussian in **linear container** form:
//!
//! ```text
//! r(x) = S·( θ(x ⊟ x₀) − b )
//! J(x) = S                (constant — container semantics)
//! ```
//!
//! where `θ(x ⊟ x₀)` is the concatenated local tangent of the connected
//! variables relative to the marginal's linearization point `x₀` (computed
//! by a caller-supplied `local_log` closure, since the factor itself is
//! manifold-agnostic), `S` is the square-root information of the marginal,
//! and `b` encodes the information vector (`b = Λ⁻¹·g` for marginal
//! gradient `g`).
//!
//! Like GTSAM's linear container, the Jacobian is exact only at the
//! linearization point; rebuild the factor (re-marginalize) when the
//! estimate drifts far from `x₀`.

use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, DVector};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Compute the concatenated local tangent `θ(x ⊟ x₀)` of the connected
/// variables (one `&[f64]` param slice per block, in block order) into the
/// output buffer (length = sum of block tangent dims, same order).
pub type LocalLogFn = Box<dyn Fn(&[&[f64]], &mut [f64]) + Send + Sync>;

/// Gaussian marginal prior over one or more variables.
pub struct MarginalPriorFactor {
    /// Square-root information of the marginal (rows × total tangent dim).
    sqrt_info: DMatrix<f64>,
    /// Offset `b` in the tangent space of the linearization point.
    offset: DVector<f64>,
    /// Tangent dimension per connected block.
    dims: Vec<usize>,
    /// Caller-supplied local-tangent computation `θ(x ⊟ x₀)`.
    local_log: LocalLogFn,
}

impl MarginalPriorFactor {
    /// Create the marginal prior.
    ///
    /// * `dims` — tangent dimension of each connected block.
    /// * `sqrt_info` — square-root information `S` (rows × Σdims); the
    ///   residual is `S·(θ − b)` so the implied information is `SᵀS`.
    /// * `offset` — `b`, length Σdims (pass zeros for a plain marginal).
    /// * `local_log` — computes the concatenated local tangent per block.
    pub fn new(
        dims: Vec<usize>,
        sqrt_info: DMatrix<f64>,
        offset: DVector<f64>,
        local_log: LocalLogFn,
    ) -> Result<Self, String> {
        let total: usize = dims.iter().sum();
        if sqrt_info.nrows() != total {
            return Err(format!(
                "sqrt_info has {} rows, expected Σdims = {total}",
                sqrt_info.nrows()
            ));
        }
        if sqrt_info.ncols() != total {
            return Err(format!(
                "sqrt_info has {} columns, expected Σdims = {total}",
                sqrt_info.ncols()
            ));
        }
        if offset.len() != total {
            return Err(format!(
                "offset has length {}, expected {total}",
                offset.len()
            ));
        }
        Ok(Self {
            sqrt_info,
            offset,
            dims,
            local_log,
        })
    }

    fn total_dim(&self) -> usize {
        self.dims.iter().sum()
    }
}

impl Factor for MarginalPriorFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            self.dims.len(),
            "MarginalPriorFactor connected block count mismatch"
        );

        let total = self.total_dim();
        let mut delta = vec![0.0f64; total];
        (self.local_log)(params, &mut delta);

        let rows = self.sqrt_info.nrows();
        for (i, r) in residual.iter_mut().enumerate().take(rows) {
            let mut acc = 0.0;
            for (j, d) in delta.iter().enumerate() {
                acc += self.sqrt_info[(i, j)] * (d - self.offset[j]);
            }
            *r = acc;
        }

        let Some(mut jac) = jacobian else { return };
        // Container semantics: dθ/dδ ≈ I at the linearization point, so the
        // Jacobian is the constant square-root information.
        for i in 0..rows {
            for j in 0..total {
                *jac.rb_mut().get_mut(i, j) = self.sqrt_info[(i, j)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        self.sqrt_info.nrows()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.sqrt_info.nrows(), self.total_dim())
    }

    /// The residual `S·(θ − b)` and the Jacobian are pre-multiplied by the
    /// marginal's square-root information inside [`Self::linearize`], so an
    /// external noise model would whiten twice.
    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != self.dims.len() {
            return Err(format!(
                "MarginalPriorFactor expects {} variables, got {}",
                self.dims.len(),
                variables.len()
            ));
        }
        // Each block's tangent dim must match the variable it is registered
        // with: `sqrt_info` is indexed by `dims`, while the assembly scatters
        // Jacobian columns by the variable's own dof. A mismatch would land
        // the columns in the wrong place without any error.
        for (i, (variable, &dim)) in variables.iter().zip(&self.dims).enumerate() {
            let dof = variable.dof();
            if dof != dim {
                return Err(format!(
                    "MarginalPriorFactor block {i} declares tangent dim {dim} but the \
                     registered variable has dof {dof}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::common::test_utils::assert_close;
    use apex_manifolds::LieGroup;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::{SE3, SE3Tangent};

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    /// Local log over two SE3 blocks: right-minus against the linearization.
    fn se2_local_log(x0_a: &SE3, x0_b: &SE3) -> LocalLogFn {
        let x0_a = x0_a.clone();
        let x0_b = x0_b.clone();
        Box::new(move |params: &[&[f64]], out: &mut [f64]| {
            let a = SE3::from_param_slice(params[0]);
            let b = SE3::from_param_slice(params[1]);
            let ta = a.right_minus(&x0_a, None, None);
            let tb = b.right_minus(&x0_b, None, None);
            out[0..6].copy_from_slice(ta.as_slice());
            out[6..12].copy_from_slice(tb.as_slice());
        })
    }

    fn sample_pose(t: [f64; 3], r: [f64; 3]) -> SE3 {
        SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(t[0], t[1], t[2]),
            nalgebra::UnitQuaternion::from_euler_angles(r[0], r[1], r[2]),
        ))
    }

    #[test]
    fn zero_residual_at_linearization_with_zero_offset() -> TestResult<()> {
        let a = sample_pose([0.1, 0.2, 0.3], [0.01, 0.02, 0.03]);
        let b = sample_pose([1.0, -0.5, 0.2], [-0.02, 0.01, 0.0]);
        let sqrt_info = DMatrix::identity(12, 12);
        let offset = DVector::zeros(12);
        let factor =
            MarginalPriorFactor::new(vec![6, 6], sqrt_info, offset, se2_local_log(&a, &b))?;

        let mut residual = vec![0.0; 12];
        factor.linearize(
            &[a.as_param_slice(), b.as_param_slice()],
            &mut residual,
            None,
        );
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-12, "residual[{i}] = {r}");
        }
        Ok(())
    }

    #[test]
    fn residual_matches_sqrt_info_times_tangent_offset() -> TestResult<()> {
        let a = sample_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let b = sample_pose([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let sqrt_info = DMatrix::identity(12, 12);
        // Offset: pose b pulled by 0.1 m in x.
        let mut offset = DVector::zeros(12);
        offset[6] = 0.1;
        let factor =
            MarginalPriorFactor::new(vec![6, 6], sqrt_info, offset, se2_local_log(&a, &b))?;

        // Evaluating at x0 gives residual −offset (S = I).
        let mut residual = vec![0.0; 12];
        factor.linearize(
            &[a.as_param_slice(), b.as_param_slice()],
            &mut residual,
            None,
        );
        assert!((residual[6] - (-0.1)).abs() < 1e-12);

        // Evaluating at x0 shifted by +0.1 in x gives zero.
        let shifted = sample_pose([1.1, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let mut residual2 = vec![0.0; 12];
        factor.linearize(
            &[a.as_param_slice(), shifted.as_param_slice()],
            &mut residual2,
            None,
        );
        assert!(residual2[6].abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn jacobian_is_constant_sqrt_info_and_fd_consistent_near_linearization() -> TestResult<()> {
        let a = sample_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let b = sample_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let mut sqrt_info = DMatrix::identity(12, 12);
        sqrt_info[(0, 0)] = 2.0;
        sqrt_info[(6, 6)] = 3.0;
        let offset = DVector::zeros(12);
        let factor =
            MarginalPriorFactor::new(vec![6, 6], sqrt_info.clone(), offset, se2_local_log(&a, &b))?;

        let a_v: Vec<f64> = a.as_param_slice().to_vec();
        let b_v: Vec<f64> = b.as_param_slice().to_vec();

        let mut residual = vec![0.0; 12];
        let mut jac_buf = vec![0.0; 144];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 12, 12);
        factor.linearize(&[&a_v, &b_v], &mut residual, Some(jac_mut));

        // Jacobian must be exactly S (column-major layout).
        for row in 0..12 {
            for col in 0..12 {
                assert_close(
                    jac_buf[col * 12 + row],
                    sqrt_info[(row, col)],
                    1e-12,
                    "J vs S",
                );
            }
        }

        // FD near the linearization point: the container Jacobian is the
        // first-order derivative, so a small perturbation must match.
        const EPS: f64 = 1e-7;
        let mut tan = [0.0f64; 6];
        tan[0] = EPS;
        let perturbed: Vec<f64> = a
            .right_plus(&SE3Tangent::from_slice(&tan), None, None)
            .as_param_slice()
            .to_vec();
        let mut r_pert = vec![0.0; 12];
        factor.linearize(&[&perturbed, &b_v], &mut r_pert, None);
        for row in 0..12 {
            let fd = (r_pert[row] - residual[row]) / EPS;
            let ana = jac_buf[row];
            assert_close(ana, fd, 1e-3, "FD vs container Jacobian");
        }
        Ok(())
    }

    /// FD the factor's full Jacobian at `(a, b)` by central differences on
    /// the right-plus tangent of each connected block, column by column.
    fn fd_jacobian(factor: &MarginalPriorFactor, a: &SE3, b: &SE3, eps: f64) -> DMatrix<f64> {
        let eval = |a: &SE3, b: &SE3| -> Vec<f64> {
            let mut r = vec![0.0; factor.residual_dim()];
            factor.linearize(&[a.as_param_slice(), b.as_param_slice()], &mut r, None);
            r
        };

        let rows = factor.residual_dim();
        let mut jac = DMatrix::zeros(rows, factor.total_dim());
        for col in 0..factor.total_dim() {
            let (first_block, k) = if col < 6 {
                (true, col)
            } else {
                (false, col - 6)
            };
            let mut tan = [0.0f64; 6];
            tan[k] = eps;
            let plus = SE3Tangent::from_slice(&tan);
            tan[k] = -eps;
            let minus = SE3Tangent::from_slice(&tan);

            let (a_plus, b_plus) = if first_block {
                (a.right_plus(&plus, None, None), b.clone())
            } else {
                (a.clone(), b.right_plus(&plus, None, None))
            };
            let (a_minus, b_minus) = if first_block {
                (a.right_plus(&minus, None, None), b.clone())
            } else {
                (a.clone(), b.right_plus(&minus, None, None))
            };

            let r_plus = eval(&a_plus, &b_plus);
            let r_minus = eval(&a_minus, &b_minus);
            for (row, (p, m)) in r_plus.iter().zip(&r_minus).enumerate() {
                jac[(row, col)] = (p - m) / (2.0 * eps);
            }
        }
        jac
    }

    /// Evaluate the factor's analytic Jacobian — the frozen `S` — as a matrix.
    fn analytic_jacobian(factor: &MarginalPriorFactor, a: &SE3, b: &SE3) -> DMatrix<f64> {
        let rows = factor.residual_dim();
        let cols = factor.total_dim();
        let mut residual = vec![0.0; rows];
        let mut buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, rows, cols);
        factor.linearize(
            &[a.as_param_slice(), b.as_param_slice()],
            &mut residual,
            Some(jac_mut),
        );
        DMatrix::from_column_slice(rows, cols, &buf)
    }

    /// Relative Frobenius distance between two Jacobians, i.e. how far the
    /// factor's Jacobian is from the reference.
    fn rel_err(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a - b).norm() / b.norm()
    }

    /// Cosine of the angle between the two Jacobians flattened as vectors —
    /// 1.0 means they point the same way, whatever their magnitude.
    fn cosine(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let (fa, fb) = (a.as_slice(), b.as_slice());
        let dot: f64 = fa.iter().zip(fb).map(|(x, y)| x * y).sum();
        dot / (a.norm() * b.norm())
    }

    /// The container contract, pinned numerically rather than trusted from
    /// the doc comment.
    ///
    /// `MarginalPriorFactor` deliberately holds its Jacobian at the constant
    /// square-root information `S` while recomputing the residual from
    /// `θ(x ⊟ x₀)` on every evaluation — exactly GTSAM's
    /// `LinearContainerFactor` semantics, and for the same reason: the
    /// linearization is the model. `∂θ/∂δ = I` only at `x₀`, so away from it
    /// the frozen Jacobian is *not* the derivative of the residual it is
    /// paired with. This test quantifies that gap instead of leaving it as
    /// "rebuild when the estimate drifts":
    ///
    /// 1. at `x₀` the frozen `S` equals central-difference FD over **all**
    ///    columns, to solver precision (the pre-existing test perturbed one
    ///    column with a one-sided difference);
    /// 2. the gap grows monotonically with displacement, so the rebuild
    ///    rule is measurable rather than a matter of taste;
    /// 3. over the drift a windowed estimator actually sees, the frozen
    ///    Jacobian still points the same way as the true one (cosine ≈ 1),
    ///    so the Gauss–Newton step stays a descent step while the residual
    ///    itself remains exact — only the step direction is approximated.
    ///
    /// If any of the three breaks, either `local_log` has stopped being a
    /// consistent tangent parametrization or `S` has stopped being written
    /// into the Jacobian, and a marginal prior would be silently steering
    /// the estimate with a stale model.
    #[test]
    fn frozen_jacobian_matches_fd_at_x0_and_drifts_monotonically() -> TestResult<()> {
        // Non-identity, non-diagonal S: a scalar multiple of the identity
        // would hide a scaling error, and a diagonal one would hide a
        // column permutation.
        let mut sqrt_info = DMatrix::identity(12, 12);
        sqrt_info[(0, 0)] = 2.0;
        sqrt_info[(6, 6)] = 3.0;
        sqrt_info[(0, 6)] = 0.5;
        sqrt_info[(7, 1)] = -0.25;
        let x0_a = sample_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let x0_b = sample_pose([1.0, -0.5, 0.2], [-0.02, 0.01, 0.0]);
        let factor = MarginalPriorFactor::new(
            vec![6, 6],
            sqrt_info.clone(),
            DVector::zeros(12),
            se2_local_log(&x0_a, &x0_b),
        )?;

        // (1) Exactness at the linearization point, every column.
        let analytic = analytic_jacobian(&factor, &x0_a, &x0_b);
        assert_close(
            rel_err(&analytic, &sqrt_info),
            0.0,
            1e-12,
            "frozen J vs S at x0",
        );
        let fd0 = fd_jacobian(&factor, &x0_a, &x0_b, 1e-6);
        assert!(
            rel_err(&fd0, &analytic) < 1e-6,
            "frozen Jacobian must be the exact derivative at x0: FD differs by {}",
            rel_err(&fd0, &analytic)
        );

        // (2) + (3) Drift the *first* block away from x₀ along an all-six
        // tangent (translation and rotation together — a pure translation
        // would barely exercise the rotational coupling) and measure.
        let drifts = [0.0, 0.02, 0.05, 0.15];
        let mut errors: Vec<f64> = Vec::with_capacity(drifts.len());
        let mut cosines: Vec<f64> = Vec::with_capacity(drifts.len());
        for &d in &drifts {
            let tan = [d; 6];
            let a = x0_a.right_plus(&SE3Tangent::from_slice(&tan), None, None);
            let fd = fd_jacobian(&factor, &a, &x0_b, 1e-6);
            let err = rel_err(&fd, &analytic);
            let cos = cosine(&fd, &analytic);
            errors.push(err);
            cosines.push(cos);
        }

        // Monotone growth: every larger displacement is strictly worse.
        for w in errors.windows(2) {
            assert!(
                w[1] > w[0],
                "container drift must grow with displacement, got {:?}",
                errors
            );
        }

        // Measured on this `S` and scene: the error is essentially *linear*
        // in the displacement (0.0114 / 0.0285 / 0.0856 at drifts
        // 0.02 / 0.05 / 0.15 — a constant ≈0.57 per unit), not quadratic.
        // That is the expected shape: `S` is exact for the residual, and
        // only `∂θ/∂δ` departs from `I`, linearly in `θ`. So the residual
        // *value* the optimizer sees is always right and only the step
        // direction is approximated — which is what LM's
        // actual-vs-predicted reduction check is there to absorb.
        //
        // Budget: a 0.05-unit drift (a large step for a windowed rig) stays
        // under 6% Jacobian error; a 0.02-unit drift under 2.5%. Those are
        // the numbers that turn "rebuild when the estimate drifts" into
        // something a caller can check.
        assert!(
            errors[1] < 0.025,
            "at drift 0.02 the frozen Jacobian is {:.1}% off the true \
             derivative — over the 2.5% container budget",
            errors[1] * 100.0
        );
        assert!(
            errors[2] < 0.06,
            "at drift 0.05 the frozen Jacobian is {:.1}% off the true \
             derivative — over the 6% container budget",
            errors[2] * 100.0
        );

        // Descent direction preserved across the whole range: the frozen
        // Jacobian may be inaccurate, but it must never point the other way.
        for (d, cos) in drifts.iter().zip(&cosines) {
            assert!(
                *cos > 0.99,
                "at drift {d} the frozen Jacobian is no longer aligned with the \
                 true derivative (cos = {cos})"
            );
        }

        Ok(())
    }

    #[test]
    fn rejects_dimension_mismatch() -> TestResult<()> {
        let a = sample_pose([0.0; 3], [0.0; 3]);
        let b = sample_pose([0.0; 3], [0.0; 3]);
        assert!(
            MarginalPriorFactor::new(
                vec![6, 6],
                DMatrix::identity(11, 12),
                DVector::zeros(12),
                se2_local_log(&a, &b),
            )
            .is_err()
        );
        assert!(
            MarginalPriorFactor::new(
                vec![6, 6],
                DMatrix::identity(12, 12),
                DVector::zeros(11),
                se2_local_log(&a, &b),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn whitens_internally_flag_is_set() {
        let a = sample_pose([0.0; 3], [0.0; 3]);
        let b = sample_pose([0.0; 3], [0.0; 3]);
        let factor = MarginalPriorFactor::new(
            vec![6, 6],
            DMatrix::identity(12, 12),
            DVector::zeros(12),
            se2_local_log(&a, &b),
        )
        .unwrap_or_else(|e| panic!("construction: {e}"));
        assert!(
            factor.whitens_internally(),
            "sqrt_info is applied inside linearize, so the factor must claim to whiten \
             internally — otherwise an attached noise model whitens a second time"
        );
    }

    /// `sqrt_info` is indexed by the declared `dims`, while assembly scatters
    /// Jacobian columns by the variables' dofs. Registering a block whose dof
    /// differs from `dims[i]` would land the columns in the wrong place — it
    /// must be rejected at registration.
    #[test]
    fn rejects_registration_with_variable_of_wrong_dof() -> TestResult<()> {
        let a = sample_pose([0.0; 3], [0.0; 3]);
        let b = sample_pose([0.0; 3], [0.0; 3]);
        let factor = MarginalPriorFactor::new(
            vec![6, 6],
            DMatrix::identity(12, 12),
            DVector::zeros(12),
            se2_local_log(&a, &b),
        )?;

        // Two blocks of dof 6 are expected; an Rn(1) block has dof 1.
        let se3_var: &dyn ManifoldVariable = &crate::core::variable::Variable::new(a.clone());
        let rn_var: &dyn ManifoldVariable = &crate::core::variable::Variable::new(
            apex_manifolds::rn::Rn::new(nalgebra::DVector::zeros(1)),
        );

        assert!(factor.validate_variables(&[se3_var, se3_var]).is_ok());
        let err = factor
            .validate_variables(&[se3_var, rn_var])
            .err()
            .ok_or("dof mismatch must be rejected")?;
        assert!(err.contains("dof 1"), "{err}");
        Ok(())
    }
}
