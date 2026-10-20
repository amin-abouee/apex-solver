//! LiDAR plane bundle adjustment: one plane seen by several frames, the plane
//! itself eliminated — the window factor of BALM / BALM2 and Voxel-SLAM's
//! local mapping, generic over the pose group.
//!
//! A plane's points come from several frames `X_i` (each point `l_k` in its
//! LiDAR frame, carried to the world through `T_SL` and `X_i`), plus "fixed"
//! world points from frames already outside the window that anchor the plane.
//! The cost is the sum of squared distances of the free points to the plane
//! that best fits *all* of them:
//!
//! ```text
//! cost(X) = min_(n,q) Σ_k (nᵀ(p_k(X) − q))² / σ²
//! p_k = X_i ⊙ (T_SL ⊙ l_k) for a free point, the fixed world point otherwise
//! ```
//!
//! The fixed points count in the cost too (with no Jacobian), so the cost is
//! `N·λ_min/σ²` of the whole point set — BALM's eigenvalue objective. The
//! inner minimum is a PCA fit, solved in closed form on every evaluation. By the envelope theorem the gradient of `cost` is the gradient
//! with the plane held at its optimum, so the residual `r_k = nᵀ(w_k − q)/σ`
//! with the Jacobian `nᵀ·∂w_k/∂X_i / σ` gives the exact gradient `Jᵀr` — the
//! variable-projection form of BALM (BALM2 adds the plane's second-order
//! coupling, which Gauss-Newton forgoes here).

use std::marker::PhantomData;

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use apex_manifolds::capability::ActsOnPoints3;
use apex_manifolds::se3::SE3;

use crate::core::CoreError;
use crate::core::CoreResult;
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_manifolds;

use super::map_plane::lidar_point_chain;

/// Smallest gap between the two smallest scatter eigenvalues for a defined
/// normal.
const MIN_EIGEN_GAP: f64 = 1e-12;

/// The best-fit plane `(n, q)` of `points`: unit normal of the least scatter
/// and the centroid. `None` for fewer than three points or no unique normal.
pub fn fit_plane_pca(points: &[Vector3<f64>]) -> Option<(Vector3<f64>, Vector3<f64>)> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let centroid = points.iter().sum::<Vector3<f64>>() / n;
    let scatter = points
        .iter()
        .map(|p| (p - centroid) * (p - centroid).transpose())
        .sum::<Matrix3<f64>>()
        / n;
    let eigen = SymmetricEigen::new(scatter);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    if eigen.eigenvalues[order[1]] - eigen.eigenvalues[order[0]] < MIN_EIGEN_GAP {
        return None;
    }
    Some((eigen.eigenvectors.column(order[0]).into_owned(), centroid))
}

/// One plane over several frames, the plane eliminated; see the module docs.
///
/// Blocks `[X_1 … X_m ∈ G]`, the frames with points on the plane, in the
/// order of `points`. The LiDAR extrinsic is held at `extrinsic`.
///
/// Whitens internally: register with `NoiseModel::null()`.
#[derive(Clone)]
pub struct LidarPlaneBundleFactor<G: ActsOnPoints3> {
    extrinsic: SE3,
    /// Per block, its points in the LiDAR frame.
    points: Vec<Vec<Vector3<f64>>>,
    /// Points of the plane outside the window, in the world.
    fixed: Vec<Vector3<f64>>,
    /// Point-to-plane standard deviation [m].
    sigma: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> LidarPlaneBundleFactor<G> {
    /// A plane seen by `points.len()` frames.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, no frame, a frame without points,
    /// fewer than three points in all, non-finite input, or a `sigma` that is
    /// not finite and positive.
    pub fn new(
        extrinsic: SE3,
        points: Vec<Vec<Vector3<f64>>>,
        fixed: Vec<Vector3<f64>>,
        sigma: f64,
    ) -> CoreResult<Self> {
        let name = "LidarPlaneBundleFactor";
        if points.is_empty() || points.iter().any(Vec::is_empty) {
            return Err(CoreError::InvalidInput(format!(
                "{name}: every frame needs at least one point"
            )));
        }
        let free: usize = points.iter().map(Vec::len).sum();
        if free + fixed.len() < 3 {
            return Err(CoreError::InvalidInput(format!(
                "{name}: a plane needs three points, got {}",
                free + fixed.len()
            )));
        }
        let finite = points
            .iter()
            .flatten()
            .chain(&fixed)
            .all(|p| p.iter().all(|v| v.is_finite()));
        if !finite {
            return Err(CoreError::InvalidInput(format!("{name}: non-finite point")));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(CoreError::InvalidInput(format!(
                "{name}: sigma must be finite and positive, got {sigma}"
            )));
        }
        Ok(Self {
            extrinsic,
            points,
            fixed,
            sigma,
            group: PhantomData,
        })
    }

    /// Number of frames (blocks).
    pub fn frames(&self) -> usize {
        self.points.len()
    }

    fn free_points(&self) -> usize {
        self.points.iter().map(Vec::len).sum()
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for LidarPlaneBundleFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let dof = G::DOF;
        // Every free point carried to the world, with its frame's chain.
        let chains: Vec<(usize, super::map_plane::LidarPointChain)> = self
            .points
            .iter()
            .zip(params)
            .enumerate()
            .flat_map(|(block, (points, param))| {
                let state = G::from_param_slice(param);
                points
                    .iter()
                    .map(|point| (block, lidar_point_chain(&state, &self.extrinsic, point)))
                    .collect::<Vec<_>>()
            })
            .collect();
        let all: Vec<Vector3<f64>> = chains
            .iter()
            .map(|(_, chain)| chain.world)
            .chain(self.fixed.iter().copied())
            .collect();
        let mut jac = jacobian;
        if let Some(jac) = jac.as_mut() {
            for row in 0..jac.nrows() {
                for col in 0..jac.ncols() {
                    *jac.rb_mut().get_mut(row, col) = 0.0;
                }
            }
        }
        let Some((normal, centroid)) = fit_plane_pca(&all) else {
            residual.fill(0.0);
            return;
        };
        let information = 1.0 / self.sigma;
        for (row, (block, chain)) in chains.iter().enumerate() {
            residual[row] = information * normal.dot(&(chain.world - centroid));
            if let Some(jac) = jac.as_mut() {
                for (col, value) in chain.project_state(&normal).iter().enumerate() {
                    *jac.rb_mut().get_mut(row, block * dof + col) = information * value;
                }
            }
        }
        // The fixed points' residuals count too (their Jacobian rows stay
        // zero): the plane is fitted to every point, so only a cost over every
        // point has it as its minimizer — which is what makes `Jᵀr` exact.
        let free = chains.len();
        for (offset, point) in self.fixed.iter().enumerate() {
            residual[free + offset] = information * normal.dot(&(point - centroid));
        }
    }

    fn residual_dim(&self) -> usize {
        self.free_points() + self.fixed.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.residual_dim(), self.points.len() * G::DOF)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &vec![G::NAME; self.points.len()],
            "LidarPlaneBundleFactor expects one pose per frame",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sim3::Sim3;
    use apex_manifolds::{LieGroup, Tangent};
    use faer::Mat;
    use nalgebra::UnitQuaternion;

    fn extrinsic() -> SE3 {
        SE3::new(
            Vector3::new(-0.007, -0.007, 0.05),
            UnitQuaternion::from_euler_angles(3.1, 0.02, -1.56),
        )
    }

    /// Points on the plane `n·x = 2` (tilted), as seen from `state`, with a
    /// few millimetres of off-plane noise.
    fn plane_points<G: ActsOnPoints3>(state: &G, seed: usize) -> Vec<Vector3<f64>> {
        let normal = Vector3::new(0.2, -0.3, 0.93).normalize();
        let a = normal.cross(&Vector3::x()).normalize();
        let b = normal.cross(&a);
        let lidar_from_world = |w: &Vector3<f64>| {
            let body = state.inverse(None).act(w, None, None);
            extrinsic().inverse(None).act(&body, None, None)
        };
        (0..12)
            .map(|k| {
                let (i, j) = (f64::from((k % 4) as u32), f64::from((k / 4) as u32));
                let noise = 0.003 * f64::from(((k * 7 + seed) % 5) as u32) - 0.006;
                let w = normal * (2.0 + noise)
                    + a * (0.3 * i - 0.5)
                    + b * (0.4 * j - 0.4)
                    + a * (0.1 * seed as f64);
                lidar_from_world(&w)
            })
            .collect()
    }

    fn cost<G: ActsOnPoints3 + Send + Sync + 'static>(
        factor: &LidarPlaneBundleFactor<G>,
        states: &[G],
    ) -> f64 {
        let params: Vec<Vec<f64>> = states.iter().map(|s| s.as_param_slice().to_vec()).collect();
        let slices: Vec<&[f64]> = params.iter().map(Vec::as_slice).collect();
        let mut residual = vec![0.0; factor.residual_dim()];
        factor.linearize(&slices, &mut residual, None);
        0.5 * residual.iter().map(|r| r * r).sum::<f64>()
    }

    /// `Jᵀr` is the gradient of the plane-eliminated cost — checked against
    /// central differences of the cost itself, plane re-fitted each time.
    fn check_gradient<G: ActsOnPoints3 + Send + Sync + 'static>(
        states: Vec<G>,
    ) -> Result<(), CoreError> {
        const EPS: f64 = 1e-6;
        let dof = G::DOF;
        let points: Vec<Vec<Vector3<f64>>> = states
            .iter()
            .enumerate()
            .map(|(i, s)| plane_points(s, i))
            .collect();
        let fixed: Vec<Vector3<f64>> = (0..5)
            .map(|k| {
                Vector3::new(0.2, -0.3, 0.93).normalize() * 2.0
                    + Vector3::new(f64::from(k) * 0.2, 0.0, -f64::from(k) * 0.04)
            })
            .collect();
        let factor = LidarPlaneBundleFactor::<G>::new(extrinsic(), points, fixed, 0.02)?;
        // Perturb so the cost is not at its minimum.
        let perturbed: Vec<G> = states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let tangent: Vec<f64> = (0..dof)
                    .map(|k| 0.004 * ((i + k) % 3) as f64 - 0.004)
                    .collect();
                s.right_plus(&G::TangentVector::from_slice(&tangent), None, None)
            })
            .collect();
        let params: Vec<Vec<f64>> = perturbed
            .iter()
            .map(|s| s.as_param_slice().to_vec())
            .collect();
        let slices: Vec<&[f64]> = params.iter().map(Vec::as_slice).collect();
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0; rows];
        let mut jacobian = Mat::<f64>::zeros(rows, cols);
        factor.linearize(&slices, &mut residual, Some(jacobian.as_mut()));
        for col in 0..cols {
            let analytic: f64 = (0..rows).map(|r| jacobian[(r, col)] * residual[r]).sum();
            let (block, axis) = (col / dof, col % dof);
            let shifted = |sign: f64| {
                let mut moved = perturbed.clone();
                let mut tangent = vec![0.0; dof];
                tangent[axis] = sign * EPS;
                moved[block] = perturbed[block].right_plus(
                    &G::TangentVector::from_slice(&tangent),
                    None,
                    None,
                );
                cost(&factor, &moved)
            };
            let fd = (shifted(1.0) - shifted(-1.0)) / (2.0 * EPS);
            assert!(
                (fd - analytic).abs() < 1e-4 * (1.0 + analytic.abs()),
                "{} column {col}: fd {fd} vs {analytic}",
                G::NAME
            );
        }
        Ok(())
    }

    #[test]
    fn the_gradient_is_exact_on_se23() -> Result<(), CoreError> {
        check_gradient(vec![
            SE23::new(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1),
            ),
            SE23::new(
                Vector3::new(0.5, 0.1, 0.0),
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.02, 0.0, 0.2),
            ),
            SE23::new(
                Vector3::new(1.0, 0.3, 0.05),
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, -0.03, 0.35),
            ),
        ])
    }

    #[test]
    fn the_gradient_is_exact_on_se3_and_sim3() -> Result<(), CoreError> {
        check_gradient(vec![
            SE3::new(
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1),
            ),
            SE3::new(
                Vector3::new(0.5, 0.1, 0.0),
                UnitQuaternion::from_euler_angles(0.02, 0.0, 0.2),
            ),
        ])?;
        check_gradient(vec![
            Sim3::new(
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1),
                1.0,
            ),
            Sim3::new(
                Vector3::new(0.5, 0.1, 0.0),
                UnitQuaternion::from_euler_angles(0.02, 0.0, 0.2),
                1.1,
            ),
        ])
    }

    #[test]
    fn a_consistent_plane_costs_only_its_noise() -> Result<(), CoreError> {
        let states = vec![
            SE3::new(Vector3::zeros(), UnitQuaternion::identity()),
            SE3::new(
                Vector3::new(0.4, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.2),
            ),
        ];
        let points: Vec<Vec<Vector3<f64>>> = states
            .iter()
            .enumerate()
            .map(|(i, s)| plane_points(s, i))
            .collect();
        let factor = LidarPlaneBundleFactor::<SE3>::new(extrinsic(), points, Vec::new(), 0.02)?;
        // Millimetre noise against a 2 cm sigma: well under one per point.
        assert!(cost(&factor, &states) < 0.1 * factor.residual_dim() as f64);
        assert_eq!(factor.frames(), 2);
        assert!(factor.whitens_internally());
        Ok(())
    }

    #[test]
    fn bad_inputs_are_rejected() {
        let p = vec![Vector3::new(1.0, 0.0, 0.0)];
        assert!(LidarPlaneBundleFactor::<SE3>::new(extrinsic(), vec![], vec![], 0.02).is_err());
        assert!(
            LidarPlaneBundleFactor::<SE3>::new(extrinsic(), vec![vec![]], p.clone(), 0.02).is_err()
        );
        assert!(
            LidarPlaneBundleFactor::<SE3>::new(extrinsic(), vec![p.clone()], vec![], 0.02).is_err()
        );
        assert!(
            LidarPlaneBundleFactor::<SE3>::new(extrinsic(), vec![p.clone()], p.repeat(3), 0.0)
                .is_err()
        );
        assert!(fit_plane_pca(&[Vector3::x(), Vector3::y()]).is_none());
        let line: Vec<Vector3<f64>> = (0..5).map(|k| Vector3::x() * f64::from(k)).collect();
        assert!(fit_plane_pca(&line).is_none());
    }
}
