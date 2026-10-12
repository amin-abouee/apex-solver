//! A factor whose Jacobian is projected onto a subspace of its tangent —
//! solution remapping for degenerate measurements.
//!
//! In a long corridor a LiDAR scan cannot tell where along the corridor it
//! is: its information has a near-zero eigenvalue in that direction, and
//! whatever it says there is noise and mismatches. Projecting the factor's
//! Jacobian onto the well-constrained directions,
//!
//! ```text
//! J' = J·P,     P = I − Σ_k v_k v_kᵀ   (v_k the degenerate directions)
//! ```
//!
//! leaves its residual — and so the cost — unchanged, but removes its pull
//! and its information along `v_k`: a Gauss-Newton step there is decided by
//! the other factors (IMU, priors, the camera). Zhang, Kaess and Singh,
//! "On degeneracy of optimization-based state estimation problems" (ICRA
//! 2016) remap the solution the same way.

use faer::Mat;
use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, DVector};

use crate::core::variable::ManifoldVariable;
use crate::factors::{Factor, NormalBlocks};

/// `inner` with its Jacobian right-multiplied by a projection.
pub struct ProjectedFactor<F: Factor> {
    inner: F,
    projection: DMatrix<f64>,
}

impl<F: Factor> ProjectedFactor<F> {
    /// Remove `directions` (columns of the factor's full tangent, each made
    /// unit and orthogonalized) from `inner`'s Jacobian.
    ///
    /// # Errors
    ///
    /// A direction of the wrong length or not finite and non-zero.
    pub fn new(inner: F, directions: &[DVector<f64>]) -> Result<Self, String> {
        let n = inner.jacobian_shape().1;
        let mut basis: Vec<DVector<f64>> = Vec::new();
        for direction in directions {
            if direction.len() != n {
                return Err(format!(
                    "ProjectedFactor: direction of length {} for {n} columns",
                    direction.len()
                ));
            }
            let mut v = direction.clone();
            for b in &basis {
                v -= b * b.dot(&v);
            }
            let norm = v.norm();
            if !(norm.is_finite() && norm > 1e-12) {
                return Err("ProjectedFactor: degenerate or non-finite direction".to_string());
            }
            basis.push(v / norm);
        }
        let mut projection = DMatrix::identity(n, n);
        for b in &basis {
            projection -= b * b.transpose();
        }
        Ok(Self { inner, projection })
    }

    /// The wrapped factor.
    pub fn inner(&self) -> &F {
        &self.inner
    }
}

impl<F: Factor> Factor for ProjectedFactor<F> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let Some(mut jac) = jacobian else {
            self.inner.linearize(params, residual, None);
            return;
        };
        let (rows, cols) = self.inner.jacobian_shape();
        let mut inner = Mat::<f64>::zeros(rows, cols);
        self.inner.linearize(params, residual, Some(inner.as_mut()));
        for row in 0..rows {
            for col in 0..cols {
                let value: f64 = (0..cols)
                    .map(|k| inner[(row, k)] * self.projection[(k, col)])
                    .sum();
                *jac.rb_mut().get_mut(row, col) = value;
            }
        }
    }

    /// The inner normal equations projected, `PᵀHP`, `Pᵀg` — for a factor
    /// on one block (a projection across blocks is not block-wise).
    fn accumulate_normal(
        &self,
        params: &[&[f64]],
        blocks: Option<&mut NormalBlocks<'_>>,
    ) -> Option<f64> {
        if params.len() != 1 {
            return None;
        }
        match blocks {
            None => self.inner.accumulate_normal(params, None),
            Some(blocks) => self.inner.accumulate_normal(
                params,
                Some(&mut blocks.transformed(0, self.projection.clone())),
            ),
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
    use apex_manifolds::LieGroup;
    use apex_manifolds::se3::SE3;
    use nalgebra::{UnitQuaternion, Vector3};

    /// Points on two walls `x = ±2` only: a corridor along `y`.
    fn corridor() -> Result<LidarMapPlanesFactor<SE3>, String> {
        let mut matches = Vec::new();
        for k in 0..20 {
            let f = f64::from(k);
            for (normal, offset) in [(Vector3::x(), -2.0), (-Vector3::x(), -2.0)] {
                matches.push(PlaneMatch {
                    point: Vector3::new(-offset * normal.x, 0.3 * f - 3.0, 0.1 * f),
                    plane: Plane::new(normal, offset)?,
                    sigma: 0.02,
                });
            }
        }
        LidarMapPlanesFactor::new(matches)
    }

    fn jacobian<F: Factor>(f: &F, state: &SE3) -> (Vec<f64>, DMatrix<f64>) {
        let (rows, cols) = f.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut j = Mat::<f64>::zeros(rows, cols);
        f.linearize(&[state.as_param_slice()], &mut r, Some(j.as_mut()));
        (r, DMatrix::from_fn(rows, cols, |a, b| j[(a, b)]))
    }

    /// Projecting out the corridor's axis leaves the residual, zeroes the
    /// information along the axis, and keeps it across the corridor.
    #[test]
    fn projection_removes_only_the_degenerate_direction() -> Result<(), String> {
        let state = SE3::new(
            Vector3::new(0.1, 0.4, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.02),
        );
        let along = state.rotation_so3().rotation_matrix().transpose() * Vector3::y();
        let mut direction = DVector::zeros(6);
        direction.fixed_rows_mut::<3>(0).copy_from(&along);
        let projected = ProjectedFactor::new(corridor()?, std::slice::from_ref(&direction))?;
        let (r0, j0) = jacobian(&corridor()?, &state);
        let (r1, j1) = jacobian(&projected, &state);
        assert_eq!(r0, r1);
        let info0 = j0.transpose() * &j0;
        let info1 = j1.transpose() * &j1;
        assert!((info1.clone() * &direction).norm() < 1e-9 * info0.norm());
        let mut across = DVector::zeros(6);
        across
            .fixed_rows_mut::<3>(0)
            .copy_from(&(state.rotation_so3().rotation_matrix().transpose() * Vector3::x()));
        let (a0, a1) = (
            (across.transpose() * &info0 * &across)[0],
            (across.transpose() * &info1 * &across)[0],
        );
        assert!((a0 - a1).abs() < 1e-6 * a0, "{a0} vs {a1}");
        Ok(())
    }

    /// Its normal equations are the inner ones projected, `PᵀHP`.
    #[test]
    fn projected_normal_equations_match_its_jacobian() -> Result<(), String> {
        let state = SE3::new(
            Vector3::new(0.1, 0.4, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.02),
        );
        let mut direction = DVector::zeros(6);
        direction[1] = 1.0;
        direction[3] = 0.5;
        let projected = ProjectedFactor::new(corridor()?, &[direction])?;
        crate::factors::assert_normal_equations_match(
            &projected,
            &[state.as_param_slice()],
            &[6],
            1e-9,
        );
        Ok(())
    }

    #[test]
    fn construction_rejects_bad_directions() -> Result<(), String> {
        assert!(ProjectedFactor::new(corridor()?, &[DVector::zeros(5)]).is_err());
        assert!(ProjectedFactor::new(corridor()?, &[DVector::zeros(6)]).is_err());
        let v = DVector::from_element(6, 1.0);
        assert!(ProjectedFactor::new(corridor()?, &[v.clone(), v]).is_err());
        assert!(ProjectedFactor::new(corridor()?, &[]).is_ok());
        Ok(())
    }
}
