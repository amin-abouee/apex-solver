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
use std::sync::Arc;

use faer::prelude::ReborrowMut;
use nalgebra::{SMatrix, Vector3};
use rayon::prelude::*;

use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::factors::common::group_point::acts_on_points;
use crate::factors::common::validate::expect_manifolds;
use crate::factors::{Factor, NormalBlocks};

use super::action::{Action, MAX_DOF, cached_action};
use super::plane::Plane;

/// Matches per parallel chunk of the normal equations.
const CHUNK: usize = 512;

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
    matches: Arc<[PlaneMatch]>,
    group: PhantomData<fn() -> G>,
}

impl<G: LieGroup> LidarMapPlanesFactor<G> {
    /// A factor over `matches` — a `Vec`, or an `Arc` shared with the
    /// caller without a copy.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, no match, a non-finite point or
    /// a sigma that is not finite and positive.
    pub fn new(matches: impl Into<Arc<[PlaneMatch]>>) -> Result<Self, String> {
        let matches = matches.into();
        let name = "LidarMapPlanesFactor";
        if !acts_on_points::<G>() || G::TangentVector::DIM > MAX_DOF {
            return Err(format!("{name}: {} does not act on 3-D points", G::NAME));
        }
        if matches.is_empty() {
            return Err(format!("{name}: no match"));
        }
        for m in matches.iter() {
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

/// A match's whitened residual `(nᵀ(A·s + b) + d)/σ` and its Jacobian row
/// `nᵀ(∂b + Σ s_k·∂(A·e_k))/σ`, zero past the group's dimension.
fn residual_and_row(action: &Action, m: &PlaneMatch) -> (f64, SMatrix<f64, 1, MAX_DOF>) {
    let (a, b, jacobian) = action;
    let s = m.point;
    let information = 1.0 / m.sigma;
    let n = m.plane.normal.transpose();
    let residual = (m.plane.normal.dot(&(a * s + b)) + m.plane.offset) * information;
    let row = (n * jacobian.origin
        + n * jacobian.axes[0] * s.x
        + n * jacobian.axes[1] * s.y
        + n * jacobian.axes[2] * s.z)
        * information;
    (residual, row)
}

impl<G: LieGroup + Send + Sync + 'static> Factor for LidarMapPlanesFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let action = cached_action::<G>(params[0]);
        let dof = G::TangentVector::DIM;
        let mut jac = jacobian;
        for (row, m) in self.matches.iter().enumerate() {
            let (r, jacobian_row) = residual_and_row(&action, m);
            residual[row] = r;
            if let Some(jac) = jac.as_mut() {
                for col in 0..dof {
                    *jac.rb_mut().get_mut(row, col) = jacobian_row[col];
                }
            }
        }
    }

    /// `Σ JᵀJ` and `Σ Jᵀr` row by row, in parallel chunks summed in order
    /// (deterministic), without the Jacobian.
    fn accumulate_normal(
        &self,
        params: &[&[f64]],
        blocks: Option<&mut NormalBlocks<'_>>,
    ) -> Option<f64> {
        let action = cached_action::<G>(params[0]);
        let action: &Action = &action;
        let want_normal = blocks.is_some();
        let chunks: Vec<(
            SMatrix<f64, MAX_DOF, MAX_DOF>,
            SMatrix<f64, MAX_DOF, 1>,
            f64,
        )> = self
            .matches
            .par_chunks(CHUNK)
            .map(|chunk| {
                let mut hessian = SMatrix::<f64, MAX_DOF, MAX_DOF>::zeros();
                let mut gradient = SMatrix::<f64, MAX_DOF, 1>::zeros();
                let mut cost = 0.0;
                for m in chunk {
                    let (r, row) = residual_and_row(action, m);
                    cost += r * r;
                    if want_normal {
                        hessian += row.transpose() * row;
                        gradient += row.transpose() * r;
                    }
                }
                (hessian, gradient, cost)
            })
            .collect();
        let mut hessian = SMatrix::<f64, MAX_DOF, MAX_DOF>::zeros();
        let mut gradient = SMatrix::<f64, MAX_DOF, 1>::zeros();
        let mut cost = 0.0;
        for (h, g, c) in &chunks {
            hessian += h;
            gradient += g;
            cost += c;
        }
        if let Some(blocks) = blocks {
            let dof = G::TangentVector::DIM;
            blocks.add_hessian(0, 0, &hessian.view((0, 0), (dof, dof)));
            blocks.add_gradient(0, &gradient.rows(0, dof));
        }
        Some(cost)
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
    use nalgebra::{DMatrix, DVector, UnitQuaternion};

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

    /// Over several parallel chunks, the normal equations equal the
    /// Jacobian's `JᵀJ`, `Jᵀr` and `‖r‖²`; and a shared `Arc` is not copied.
    #[test]
    fn normal_equations_match_the_jacobian() -> Result<(), String> {
        let base = matches()?;
        let many: Arc<[PlaneMatch]> = (0..3 * CHUNK + 7)
            .map(|k| {
                let mut m = base[k % base.len()].clone();
                m.point += Vector3::new(0.01 * k as f64, -0.003 * k as f64, 0.0);
                m
            })
            .collect();
        let factor = LidarMapPlanesFactor::<SE23>::new(Arc::clone(&many))?;
        assert!(std::ptr::eq(factor.matches.as_ptr(), many.as_ptr()));
        let state = SE23::new(
            Vector3::new(0.2, -0.4, 0.1),
            Vector3::new(1.0, 0.0, -0.5),
            UnitQuaternion::from_euler_angles(0.1, 0.2, -0.3),
        );
        let params = [state.as_param_slice()];
        let (mut hessian, mut gradient) = (DMatrix::zeros(9, 9), DVector::zeros(9));
        let cost = factor
            .accumulate_normal(
                &params,
                Some(&mut NormalBlocks::new(&mut hessian, &mut gradient, &[0])),
            )
            .ok_or("no normal equations")?;
        assert_eq!(factor.accumulate_normal(&params, None), Some(cost));
        let (rows, cols) = factor.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut j = Mat::<f64>::zeros(rows, cols);
        factor.linearize(&params, &mut r, Some(j.as_mut()));
        let j = DMatrix::from_fn(rows, cols, |a, b| j[(a, b)]);
        let r = DVector::from_column_slice(&r);
        assert!((j.transpose() * &j - &hessian).norm() < 1e-9 * hessian.norm());
        assert!((j.transpose() * &r - &gradient).norm() < 1e-9 * (1.0 + gradient.norm()));
        assert!((r.norm_squared() - cost).abs() < 1e-9 * cost.max(1.0));
        Ok(())
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
