//! Many body-frame points against fixed map planes, in one factor — a whole
//! sweep's scan-to-map matches on one pose, without a residual block (and
//! its bookkeeping) per point.
//!
//! ```text
//! r_k = (n_kᵀ·(X ⊙ s_k) + d_k) / σ_k
//! ```
//!
//! The action is affine in the point for every group that acts on points, so
//! `∂(X⊙s)/∂X = J(0) + Σ_j s_j·(J(e_j) − J(0))` from four evaluations of
//! the action, shared by all points.

use std::marker::PhantomData;

use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::group_point::acts_on_points;
use crate::factors::common::validate::expect_manifolds;

use super::plane::Plane;

/// One point-to-plane match: a body-frame point, its map plane, its sigma.
#[derive(Clone, Debug)]
pub struct PlaneMatch {
    /// The point in the body frame (what the pose acts on).
    pub point: Vector3<f64>,
    /// The map plane, held constant.
    pub plane: Plane,
    /// Standard deviation of the point-to-plane distance \[m\].
    pub sigma: f64,
}

/// Point-to-plane matches on one pose; see the module docs.
///
/// Blocks `[X ∈ G]`. Whitens internally: register with `NoiseModel::null()`.
#[derive(Clone, Debug)]
pub struct LidarMapPlanesFactor<G: LieGroup> {
    matches: Vec<PlaneMatch>,
    group: PhantomData<fn() -> G>,
}

impl<G: LieGroup> LidarMapPlanesFactor<G> {
    /// A factor over `matches`.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, no match, a non-finite point or
    /// a sigma that is not finite and positive.
    pub fn new(matches: Vec<PlaneMatch>) -> Result<Self, String> {
        let name = "LidarMapPlanesFactor";
        if !acts_on_points::<G>() {
            return Err(format!("{name}: {} does not act on 3-D points", G::NAME));
        }
        if matches.is_empty() {
            return Err(format!("{name}: no match"));
        }
        for m in &matches {
            if !m.point.iter().all(|v| v.is_finite()) {
                return Err(format!("{name}: non-finite point"));
            }
            if !(m.sigma.is_finite() && m.sigma > 0.0) {
                return Err(format!(
                    "{name}: sigma must be finite and positive, got {}",
                    m.sigma
                ));
            }
        }
        Ok(Self {
            matches,
            group: PhantomData,
        })
    }

    /// Number of matches (residuals).
    pub fn len(&self) -> usize {
        self.matches.len()
    }

    /// Whether there are no matches (never, once constructed).
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }
}

impl<G: LieGroup + Send + Sync + 'static> Factor for LidarMapPlanesFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = G::from_param_slice(params[0]);
        let dof = state.tangent_dim();
        let at = |s: &Vector3<f64>| {
            let mut jacobian = G::zero_jacobian();
            let world = state.act(s, Some(&mut jacobian), None);
            let columns: Vec<Vector3<f64>> = (0..dof)
                .map(|c| Vector3::new(jacobian[(0, c)], jacobian[(1, c)], jacobian[(2, c)]))
                .collect();
            (world, columns)
        };
        let (b, j0) = at(&Vector3::zeros());
        let axes = [at(&Vector3::x()), at(&Vector3::y()), at(&Vector3::z())];
        let mut jac = jacobian;
        for (row, m) in self.matches.iter().enumerate() {
            let s = m.point;
            let world = b + (axes[0].0 - b) * s.x + (axes[1].0 - b) * s.y + (axes[2].0 - b) * s.z;
            let information = 1.0 / m.sigma;
            residual[row] = (m.plane.normal.dot(&world) + m.plane.offset) * information;
            if let Some(jac) = jac.as_mut() {
                for col in 0..dof {
                    let origin = j0[col];
                    let column = origin
                        + (axes[0].1[col] - origin) * s.x
                        + (axes[1].1[col] - origin) * s.y
                        + (axes[2].1[col] - origin) * s.z;
                    *jac.rb_mut().get_mut(row, col) = m.plane.normal.dot(&column) * information;
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        self.matches.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.matches.len(), G::TangentVector::DIM)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &[G::NAME],
            "LidarMapPlanesFactor expects one pose",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sim3::Sim3;
    use faer::Mat;
    use nalgebra::UnitQuaternion;

    fn matches() -> Result<Vec<PlaneMatch>, String> {
        (0..6)
            .map(|k| {
                let f = f64::from(k);
                Ok(PlaneMatch {
                    point: Vector3::new(1.0 + f, -0.5 * f, 0.3 + 0.2 * f),
                    plane: Plane::new(Vector3::new(0.1 * f, 1.0, -0.2), 0.5 - 0.1 * f)?,
                    sigma: 0.01 + 0.002 * f,
                })
            })
            .collect()
    }

    /// Residuals match the point-by-point formula, and the Jacobian finite
    /// differences, on every group.
    fn check<G: LieGroup + Send + Sync + 'static>(state: G) -> Result<(), String> {
        let ms = matches()?;
        let factor = LidarMapPlanesFactor::<G>::new(ms.clone())?;
        let (rows, cols) = factor.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut jac = Mat::<f64>::zeros(rows, cols);
        factor.linearize(&[state.as_param_slice()], &mut r, Some(jac.as_mut()));
        for (row, m) in ms.iter().enumerate() {
            let w = state.act(&m.point, None, None);
            let direct = (m.plane.normal.dot(&w) + m.plane.offset) / m.sigma;
            assert!(
                (r[row] - direct).abs() < 1e-9,
                "row {row}: {} vs {direct}",
                r[row]
            );
        }
        let h = 1e-6;
        for col in 0..cols {
            let shifted = |sign: f64| {
                let mut t = vec![0.0; cols];
                t[col] = sign * h;
                let moved = state.right_plus(&G::TangentVector::from_slice(&t), None, None);
                let mut out = vec![0.0; rows];
                factor.linearize(&[moved.as_param_slice()], &mut out, None);
                out
            };
            let (plus, minus) = (shifted(1.0), shifted(-1.0));
            for row in 0..rows {
                let fd = (plus[row] - minus[row]) / (2.0 * h);
                assert!(
                    (jac[(row, col)] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
                    "{} row {row} col {col}: {} vs {fd}",
                    G::NAME,
                    jac[(row, col)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn batched_matches_agree_with_single_points_on_every_group() -> Result<(), String> {
        let q = UnitQuaternion::from_euler_angles(0.2, -0.1, 0.7);
        let t = Vector3::new(0.4, -1.2, 0.3);
        check(SE3::new(t, q))?;
        check(SE23::new(t, Vector3::new(0.5, 0.1, -0.2), q))?;
        check(Sim3::new(t, q, 1.3))
    }

    #[test]
    fn construction_rejects_bad_input() -> Result<(), String> {
        assert!(LidarMapPlanesFactor::<SE3>::new(Vec::new()).is_err());
        let mut ms = matches()?;
        ms[0].sigma = 0.0;
        assert!(LidarMapPlanesFactor::<SE3>::new(ms).is_err());
        let factor = LidarMapPlanesFactor::<SE3>::new(matches()?)?;
        assert!(factor.len() == 6 && !factor.is_empty());
        Ok(())
    }
}
