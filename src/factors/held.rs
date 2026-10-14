//! A factor with some of its blocks held constant.
//!
//! Calibrated extrinsics, a frozen map anchor, a reference pose outside an
//! optimization window: a factor's block that the problem should not estimate
//! is often simplest to bake in rather than to add as a variable and pin. The
//! held blocks' parameter values are stored here; the wrapped factor sees them
//! at their original positions, and only the free blocks' Jacobian columns are
//! reported.

use faer::Mat;
use faer::prelude::ReborrowMut;

use crate::core::variable::ManifoldVariable;
use crate::core::{CoreError, CoreResult};
use crate::factors::{Factor, NormalBlocks};

/// `inner` with some parameter blocks fixed to given values.
pub struct HeldBlocks<F: Factor> {
    inner: F,
    /// Per inner block: `Some(values)` when held, `None` when free.
    blocks: Vec<Option<Vec<f64>>>,
    /// Per inner block, its tangent dimension (Jacobian column count).
    dofs: Vec<usize>,
}

impl<F: Factor> HeldBlocks<F> {
    /// Wrap `inner`, whose blocks have tangent dimensions `dofs`, holding
    /// block `i` at `held[i]` when that is `Some`.
    ///
    /// # Errors
    ///
    /// `dofs` and `held` of different lengths, column counts that do not add
    /// up to the inner Jacobian's ([`CoreError::DimensionMismatch`]), or no
    /// free block left ([`CoreError::InvalidInput`]).
    pub fn new(inner: F, dofs: Vec<usize>, held: Vec<Option<Vec<f64>>>) -> CoreResult<Self> {
        if dofs.len() != held.len() {
            return Err(CoreError::DimensionMismatch(format!(
                "HeldBlocks: {} block dimensions for {} blocks",
                dofs.len(),
                held.len()
            )));
        }
        if dofs.iter().sum::<usize>() != inner.jacobian_shape().1 {
            return Err(CoreError::DimensionMismatch(format!(
                "HeldBlocks: block dimensions sum to {}, the factor has {} columns",
                dofs.iter().sum::<usize>(),
                inner.jacobian_shape().1
            )));
        }
        if held.iter().all(Option::is_some) {
            return Err(CoreError::InvalidInput(
                "HeldBlocks: every block is held; nothing to estimate".to_string(),
            ));
        }
        Ok(Self {
            inner,
            blocks: held,
            dofs,
        })
    }

    /// The wrapped factor.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    fn free_columns(&self) -> usize {
        self.blocks
            .iter()
            .zip(&self.dofs)
            .filter(|(block, _)| block.is_none())
            .map(|(_, dof)| dof)
            .sum()
    }
}

impl<F: Factor> Factor for HeldBlocks<F> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let mut free = params.iter();
        let full: Vec<&[f64]> = self
            .blocks
            .iter()
            .filter_map(|block| match block {
                Some(values) => Some(values.as_slice()),
                None => free.next().copied(),
            })
            .collect();
        let Some(mut jac) = jacobian else {
            self.inner.linearize(&full, residual, None);
            return;
        };
        let (rows, cols) = self.inner.jacobian_shape();
        let mut inner_jacobian = Mat::<f64>::zeros(rows, cols);
        self.inner
            .linearize(&full, residual, Some(inner_jacobian.as_mut()));
        let (mut source, mut target) = (0, 0);
        for (block, &dof) in self.blocks.iter().zip(&self.dofs) {
            if block.is_none() {
                for col in 0..dof {
                    for row in 0..rows {
                        *jac.rb_mut().get_mut(row, target + col) =
                            inner_jacobian[(row, source + col)];
                    }
                }
                target += dof;
            }
            source += dof;
        }
    }

    /// The inner normal equations, held blocks dropped.
    fn accumulate_normal(
        &self,
        params: &[&[f64]],
        blocks: Option<&mut NormalBlocks<'_>>,
    ) -> Option<f64> {
        let mut free = params.iter();
        let full: Vec<&[f64]> = self
            .blocks
            .iter()
            .filter_map(|block| match block {
                Some(values) => Some(values.as_slice()),
                None => free.next().copied(),
            })
            .collect();
        let Some(blocks) = blocks else {
            return self.inner.accumulate_normal(&full, None);
        };
        let mut next = 0;
        let map: Vec<Option<usize>> = self
            .blocks
            .iter()
            .map(|block| {
                block.is_none().then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        self.inner
            .accumulate_normal(&full, Some(&mut blocks.remapped(&map)))
    }

    fn residual_dim(&self) -> usize {
        self.inner.residual_dim()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.inner.jacobian_shape().0, self.free_columns())
    }

    fn whitens_internally(&self) -> bool {
        self.inner.whitens_internally()
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        let free = self.blocks.iter().filter(|block| block.is_none()).count();
        if variables.len() != free {
            return Err(format!(
                "HeldBlocks: {} variables for {free} free blocks",
                variables.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::lidar::{LidarMapPlaneFactor, Plane};
    use apex_manifolds::LieGroup;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se23::SE23;
    use nalgebra::{UnitQuaternion, Vector3};

    fn state() -> SE23 {
        SE23::new(
            Vector3::new(1.0, -2.0, 0.5),
            Vector3::new(0.3, 0.1, -0.2),
            UnitQuaternion::from_euler_angles(0.1, -0.3, 0.7),
        )
    }

    fn extrinsic() -> SE3 {
        SE3::new(
            Vector3::new(-0.007, -0.007, 0.05),
            UnitQuaternion::from_euler_angles(3.1, 0.02, -1.56),
        )
    }

    fn factor() -> Result<LidarMapPlaneFactor<SE23>, CoreError> {
        LidarMapPlaneFactor::new(
            Vector3::new(4.0, -1.5, 0.8),
            Plane::new(Vector3::new(0.3, -0.5, 0.8), 1.7)?,
            0.05,
        )
    }

    /// Holding the extrinsic reproduces the full factor's residual and its
    /// state columns exactly.
    #[test]
    fn a_held_block_drops_its_columns_and_keeps_the_rest() -> Result<(), CoreError> {
        let full = factor()?;
        let s = state().as_param_slice().to_vec();
        let e = extrinsic().as_param_slice().to_vec();
        let mut full_residual = [0.0];
        let mut full_jacobian = Mat::<f64>::zeros(1, 15);
        full.linearize(&[&s, &e], &mut full_residual, Some(full_jacobian.as_mut()));

        let held = HeldBlocks::new(factor()?, vec![9, 6], vec![None, Some(e.clone())])?;
        assert_eq!(held.jacobian_shape(), (1, 9));
        assert_eq!(held.residual_dim(), 1);
        assert!(held.whitens_internally());
        let mut residual = [0.0];
        let mut jacobian = Mat::<f64>::zeros(1, 9);
        held.linearize(&[&s], &mut residual, Some(jacobian.as_mut()));
        assert_eq!(residual[0], full_residual[0]);
        for col in 0..9 {
            assert_eq!(jacobian[(0, col)], full_jacobian[(0, col)]);
        }

        // Holding the state instead leaves the extrinsic's columns.
        let other = HeldBlocks::new(factor()?, vec![9, 6], vec![Some(s.clone()), None])?;
        let mut jacobian = Mat::<f64>::zeros(1, 6);
        other.linearize(&[&e], &mut residual, Some(jacobian.as_mut()));
        for col in 0..6 {
            assert_eq!(jacobian[(0, col)], full_jacobian[(0, 9 + col)]);
        }
        let mut residual_only = [0.0];
        other.linearize(&[&e], &mut residual_only, None);
        assert_eq!(residual_only[0], full_residual[0]);
        assert_eq!(other.inner().sigma(), 0.05);
        Ok(())
    }

    /// Without inner normal equations there are none: use the Jacobian.
    #[test]
    fn no_inner_normal_equations_means_none() -> Result<(), CoreError> {
        let e = extrinsic().as_param_slice().to_vec();
        let held = HeldBlocks::new(factor()?, vec![9, 6], vec![None, Some(e)])?;
        assert_eq!(
            held.accumulate_normal(&[state().as_param_slice()], None),
            None
        );
        Ok(())
    }

    #[test]
    fn inconsistent_wrappings_are_rejected() -> Result<(), CoreError> {
        let e = extrinsic().as_param_slice().to_vec();
        assert!(HeldBlocks::new(factor()?, vec![9], vec![None, Some(e.clone())]).is_err());
        assert!(HeldBlocks::new(factor()?, vec![9, 5], vec![None, Some(e.clone())]).is_err());
        assert!(HeldBlocks::new(factor()?, vec![9, 6], vec![Some(e.clone()), Some(e)]).is_err());
        Ok(())
    }
}
