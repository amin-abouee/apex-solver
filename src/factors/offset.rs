//! A factor with one block evaluated at a fixed right offset: `X·Δ`.
//!
//! A measurement taken at a slightly different time than its state — a
//! camera image a few milliseconds after a LiDAR sweep, say — is often best
//! tied to that state through the motion `Δ` the IMU measured in between,
//! held constant. The wrapped factor sees `X' = X·Δ`; for a right
//! perturbation of `X`,
//!
//! ```text
//! X·Exp(ξ)·Δ = (X·Δ)·Exp(Ad(Δ⁻¹)·ξ)    ⇒    ∂r/∂ξ = ∂r/∂ξ' · Ad(Δ⁻¹)
//! ```
//!
//! so the block's Jacobian columns are the inner ones times `Ad(Δ⁻¹)`.

use faer::Mat;
use faer::prelude::ReborrowMut;

use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// `inner` with block `block` (of group `G`) evaluated at `X·Δ`.
pub struct RightOffset<F: Factor, G: LieGroup> {
    inner: F,
    block: usize,
    offset: G,
    /// `Ad(Δ⁻¹)`, `dof × dof`.
    adjoint: Vec<f64>,
    /// Column of the block in the inner Jacobian.
    column: usize,
}

impl<F: Factor, G: LieGroup> RightOffset<F, G> {
    /// Wrap `inner`, offsetting block `block` by `offset`. `column` is the
    /// block's first column in the inner Jacobian.
    ///
    /// # Errors
    ///
    /// A block column that does not fit the inner Jacobian.
    pub fn new(inner: F, block: usize, column: usize, offset: G) -> Result<Self, String> {
        let dof = G::TangentVector::DIM;
        if column + dof > inner.jacobian_shape().1 {
            return Err(format!(
                "RightOffset: block columns {column}..{} exceed the factor's {}",
                column + dof,
                inner.jacobian_shape().1
            ));
        }
        let adjoint = offset.inverse(None).adjoint();
        let adjoint = (0..dof * dof)
            .map(|k| adjoint[(k / dof, k % dof)])
            .collect();
        Ok(Self {
            inner,
            block,
            offset,
            adjoint,
            column,
        })
    }

    /// The wrapped factor.
    pub fn inner(&self) -> &F {
        &self.inner
    }
}

impl<F: Factor, G: LieGroup + Send + Sync + 'static> Factor for RightOffset<F, G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let moved = G::from_param_slice(params[self.block]).compose(&self.offset, None, None);
        let mut shifted: Vec<&[f64]> = params.to_vec();
        shifted[self.block] = moved.as_param_slice();
        let Some(mut jac) = jacobian else {
            self.inner.linearize(&shifted, residual, None);
            return;
        };
        let (rows, cols) = self.inner.jacobian_shape();
        let mut inner = Mat::<f64>::zeros(rows, cols);
        self.inner
            .linearize(&shifted, residual, Some(inner.as_mut()));
        let dof = G::TangentVector::DIM;
        for row in 0..rows {
            for col in 0..cols {
                let value = if (self.column..self.column + dof).contains(&col) {
                    let k = col - self.column;
                    (0..dof)
                        .map(|m| inner[(row, self.column + m)] * self.adjoint[m * dof + k])
                        .sum()
                } else {
                    inner[(row, col)]
                };
                *jac.rb_mut().get_mut(row, col) = value;
            }
        }
    }

    fn residual_dim(&self) -> usize {
        self.inner.residual_dim()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        self.inner.jacobian_shape()
    }

    fn whitens_internally(&self) -> bool {
        self.inner.whitens_internally()
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        self.inner.validate_variables(variables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::lidar::{LidarMapPlanesFactor, Plane, PlaneMatch};
    use apex_manifolds::se23::SE23;
    use nalgebra::{UnitQuaternion, Vector3};

    fn factor() -> Result<LidarMapPlanesFactor<SE23>, String> {
        let matches = (0..6)
            .map(|k| {
                let f = f64::from(k);
                Ok(PlaneMatch {
                    point: Vector3::new(1.0 + f, -0.5 * f, 0.3 + 0.2 * f),
                    plane: Plane::new(Vector3::new(0.1 * f, 1.0, -0.2), 0.5 - 0.1 * f)?,
                    sigma: 0.02,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        LidarMapPlanesFactor::new(matches)
    }

    /// The offset factor equals the inner one at `X·Δ`, and its Jacobian
    /// matches finite differences in `X`.
    #[test]
    fn offset_residual_and_jacobian_are_exact() -> Result<(), String> {
        let state = SE23::new(
            Vector3::new(0.4, -1.2, 0.3),
            Vector3::new(0.5, 0.1, -0.2),
            UnitQuaternion::from_euler_angles(0.2, -0.1, 0.7),
        );
        let offset = SE23::new(
            Vector3::new(0.012, -0.004, 0.002),
            Vector3::zeros(),
            UnitQuaternion::from_euler_angles(0.01, 0.005, -0.02),
        );
        let wrapped = RightOffset::new(factor()?, 0, 0, offset.clone())?;
        let (rows, cols) = wrapped.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut jac = Mat::<f64>::zeros(rows, cols);
        wrapped.linearize(&[state.as_param_slice()], &mut r, Some(jac.as_mut()));

        let moved = state.compose(&offset, None, None);
        let mut direct = vec![0.0; rows];
        factor()?.linearize(&[moved.as_param_slice()], &mut direct, None);
        assert!(r.iter().zip(&direct).all(|(a, b)| (a - b).abs() < 1e-12));

        let h = 1e-6;
        for col in 0..cols {
            let shifted = |sign: f64| {
                let mut t = vec![0.0; cols];
                t[col] = sign * h;
                let s = state.right_plus(
                    &<SE23 as LieGroup>::TangentVector::from_slice(&t),
                    None,
                    None,
                );
                let mut out = vec![0.0; rows];
                wrapped.linearize(&[s.as_param_slice()], &mut out, None);
                out
            };
            let (plus, minus) = (shifted(1.0), shifted(-1.0));
            for row in 0..rows {
                let fd = (plus[row] - minus[row]) / (2.0 * h);
                assert!(
                    (jac[(row, col)] - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                    "row {row} col {col}: {} vs {fd}",
                    jac[(row, col)]
                );
            }
        }
        assert!(RightOffset::new(factor()?, 0, 3, offset).is_err());
        Ok(())
    }
}
