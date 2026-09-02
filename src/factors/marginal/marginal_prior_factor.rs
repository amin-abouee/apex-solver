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
            return Err(format!("offset has length {}, expected {total}", offset.len()));
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
        for i in 0..rows {
            let mut acc = 0.0;
            for j in 0..total {
                acc += self.sqrt_info[(i, j)] * (delta[j] - self.offset[j]);
            }
            residual[i] = acc;
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

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != self.dims.len() {
            return Err(format!(
                "MarginalPriorFactor expects {} variables, got {}",
                self.dims.len(),
                variables.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::test_utils::assert_close;
    use apex_manifolds::LieGroup;
use apex_manifolds::Tangent;
    use apex_manifolds::se3::{SE3, SE3Tangent};

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    /// Local log over two SE3 blocks: right-minus against the linearization.
    fn se2_local_log(
        x0_a: &SE3,
        x0_b: &SE3,
    ) -> LocalLogFn {
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
        let factor = MarginalPriorFactor::new(vec![6, 6], sqrt_info, offset, se2_local_log(&a, &b))?;

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
        let factor = MarginalPriorFactor::new(vec![6, 6], sqrt_info, offset, se2_local_log(&a, &b))?;

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
        let factor = MarginalPriorFactor::new(vec![6, 6], sqrt_info.clone(), offset, se2_local_log(&a, &b))?;

        let a_v: Vec<f64> = a.as_param_slice().to_vec();
        let b_v: Vec<f64> = b.as_param_slice().to_vec();

        let mut residual = vec![0.0; 12];
        let mut jac_buf = vec![0.0; 144];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 12, 12);
        factor.linearize(&[&a_v, &b_v], &mut residual, Some(jac_mut));

        // Jacobian must be exactly S (column-major layout).
        for row in 0..12 {
            for col in 0..12 {
                assert_close(jac_buf[col * 12 + row], sqrt_info[(row, col)], 1e-12, "J vs S");
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

    #[test]
    fn rejects_dimension_mismatch() -> TestResult<()> {
        let a = sample_pose([0.0; 3], [0.0; 3]);
        let b = sample_pose([0.0; 3], [0.0; 3]);
        assert!(MarginalPriorFactor::new(
            vec![6, 6],
            DMatrix::identity(11, 12),
            DVector::zeros(12),
            se2_local_log(&a, &b),
        )
        .is_err());
        assert!(MarginalPriorFactor::new(
            vec![6, 6],
            DMatrix::identity(12, 12),
            DVector::zeros(11),
            se2_local_log(&a, &b),
        )
        .is_err());
        Ok(())
    }
}
