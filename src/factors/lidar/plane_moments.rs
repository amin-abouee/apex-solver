//! LiDAR plane bundle adjustment on point moments — Voxel-SLAM's local
//! mapping factor: one plane over several frames, the plane eliminated, each
//! frame's points summarized by their moments so the cost no longer grows with
//! the number of points.
//!
//! A frame's points `s_k` (body frame) reach the world affinely,
//! `w_k = A·s_k + b` (`A = R`, `b = t` for `SE3` and `SE₂(3)`, `A = sR` for
//! `Sim3`). With the plane `(n, d)`, `nᵀw + d = cᵀs̃` for `s̃ = (s, 1)` and
//! `c = (Aᵀn, nᵀb + d)`, so the frame's squared distances sum to
//!
//! ```text
//! Σ_k (nᵀw_k + d)² = cᵀ·M·c,     M = Σ_k s̃_k·s̃_kᵀ   (4 × 4, the moments)
//! ```
//!
//! Centred on the frame's mean `μ` (scatter `S = Σ(s_k−μ)(s_k−μ)ᵀ`), the cross
//! term vanishes and the sum splits into
//!
//! ```text
//! Σ_k (nᵀw_k + d)² = (Aᵀn)ᵀ·S·(Aᵀn) + N·(nᵀ(Aμ + b) + d)²
//! ```
//!
//! so with `S = L·Lᵀ` the frame's four residuals `[Lᵀ·Aᵀn, √N·(nᵀ(Aμ+b)+d)] / σ`
//! carry exactly the per-point cost. Centring is what keeps this exact: the
//! raw 4 × 4 moment matrix of points tens of metres away spans ten orders of
//! magnitude, and a square root of it is only accurate to ~1e-4 relative —
//! residuals in the thousands for a plane whose true cost is ~1e-2. World
//! points of frames already outside the window ("fixed") add the same four
//! residuals with `A = I`, `b = 0` and no Jacobian.
//!
//! The plane is the best fit to all the points, re-solved on every
//! evaluation: the cost is `N·λ_min/σ²` of the merged scatter, and by the
//! envelope theorem the Jacobian taken with the plane held at its optimum
//! gives the exact gradient — as for
//! [`LidarPlaneBundleFactor`](super::LidarPlaneBundleFactor), of which this is
//! the moment form.
//!
//! The action's Jacobian is affine in the point for every group that acts on
//! points, so `∂c/∂X` follows from the action at the origin and the unit axes:
//! `∂c_j/∂X = nᵀ(J(e_j) − J(0))`, `∂c_3/∂X = nᵀJ(0)`.

use std::marker::PhantomData;

use faer::prelude::ReborrowMut;
use std::rc::Rc;

use nalgebra::{Matrix3, SMatrix, SymmetricEigen, Vector3};

use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::factors::common::group_point::acts_on_points;
use crate::factors::common::validate::expect_manifolds;
use crate::factors::{Factor, NormalBlocks};

use super::action::{Action, ActionJacobian, MAX_DOF, cached_action};

/// Smallest gap between the two smallest scatter eigenvalues for a defined
/// normal.
const MIN_EIGEN_GAP: f64 = 1e-14;

/// First and second moments of a point set: `N`, `Σp`, `Σp·pᵀ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointMoments {
    /// Number of points.
    pub count: f64,
    /// `Σ p`.
    pub sum: Vector3<f64>,
    /// `Σ p·pᵀ`.
    pub outer: Matrix3<f64>,
}

impl Default for PointMoments {
    fn default() -> Self {
        Self {
            count: 0.0,
            sum: Vector3::zeros(),
            outer: Matrix3::zeros(),
        }
    }
}

impl PointMoments {
    /// The moments of `points`.
    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a Vector3<f64>>) -> Self {
        let mut moments = Self::default();
        for point in points {
            moments.push(point);
        }
        moments
    }

    /// Add one point.
    pub fn push(&mut self, point: &Vector3<f64>) {
        self.count += 1.0;
        self.sum += point;
        self.outer += point * point.transpose();
    }

    /// Merge another point set.
    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.outer += other.outer;
    }

    /// The moments of the points carried by `p ↦ A·p + b`.
    pub fn transformed(&self, a: &Matrix3<f64>, b: &Vector3<f64>) -> Self {
        let a_sum = a * self.sum;
        Self {
            count: self.count,
            sum: a_sum + b * self.count,
            outer: a * self.outer * a.transpose()
                + a_sum * b.transpose()
                + b * a_sum.transpose()
                + b * b.transpose() * self.count,
        }
    }

    /// Mean and scatter (`Σ(p − μ)(p − μ)ᵀ / N`); `None` without points.
    pub fn mean_and_scatter(&self) -> Option<(Vector3<f64>, Matrix3<f64>)> {
        if self.count <= 0.0 {
            return None;
        }
        let mean = self.sum / self.count;
        let scatter = self.outer / self.count - mean * mean.transpose();
        Some((mean, (scatter + scatter.transpose()) * 0.5))
    }

    /// The best-fit plane `(n, d)` with `nᵀp + d = 0`, and the scatter's
    /// eigenvalues in ascending order. `None` for fewer than three points or
    /// no unique normal.
    pub fn plane(&self) -> Option<(Vector3<f64>, f64, [f64; 3])> {
        if self.count < 3.0 {
            return None;
        }
        let (mean, scatter) = self.mean_and_scatter()?;
        let eigen = SymmetricEigen::new(scatter);
        let mut order = [0, 1, 2];
        order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
        let values = order.map(|k| eigen.eigenvalues[k]);
        if values[1] - values[0] < MIN_EIGEN_GAP {
            return None;
        }
        let normal = eigen.eigenvectors.column(order[0]).into_owned();
        Some((normal, -normal.dot(&mean), values))
    }

    /// The moments centred on their mean: `(N, μ, S)` with `S` the centred
    /// scatter `Σ(s−μ)(s−μ)ᵀ`. `None` without points.
    fn centred(&self) -> Option<Centred> {
        let (mean, scatter) = self.mean_and_scatter()?;
        Some(Centred {
            count: self.count,
            mean,
            scatter: scatter * self.count,
        })
    }
}

/// A point set centred on its mean, with its scatter.
#[derive(Clone, Copy, Debug)]
struct Centred {
    count: f64,
    mean: Vector3<f64>,
    /// `S = Σ(s−μ)(s−μ)ᵀ`.
    scatter: Matrix3<f64>,
}

impl Centred {
    /// `Lᵀ` with `L·Lᵀ = S`: the residual rows of [`Factor::linearize`].
    fn root(&self) -> Matrix3<f64> {
        let eigen = SymmetricEigen::new(self.scatter);
        let mut root = Matrix3::zeros();
        for k in 0..3 {
            let scale = eigen.eigenvalues[k].max(0.0).sqrt();
            root.row_mut(k)
                .copy_from(&(eigen.eigenvectors.column(k).transpose() * scale));
        }
        root
    }
}

/// One plane over several frames, from per-frame point moments; see the
/// module docs.
///
/// Blocks `[X_1 … X_m ∈ G]`, in the order of `frames`. Each frame's moments
/// are of its points in the body frame (what `X_i` acts on); `fixed` holds the
/// world points of frames outside the window.
///
/// Whitens internally: register with `NoiseModel::null()`.
#[derive(Clone)]
pub struct LidarPlaneMomentFactor<G: LieGroup> {
    frames: Vec<Centred>,
    fixed: Option<Centred>,
    sigma: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: LieGroup> LidarPlaneMomentFactor<G> {
    /// A plane seen by `frames.len()` frames, anchored by `fixed`, with
    /// point-to-plane standard deviation `sigma`.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, no frame, a frame without points,
    /// fewer than three points in all, non-finite moments, or a `sigma` that
    /// is not finite and positive.
    pub fn new(frames: Vec<PointMoments>, fixed: PointMoments, sigma: f64) -> Result<Self, String> {
        let name = "LidarPlaneMomentFactor";
        if !acts_on_points::<G>() || G::TangentVector::DIM > MAX_DOF {
            return Err(format!("{name}: {} does not act on 3-D points", G::NAME));
        }
        if frames.is_empty() || frames.iter().any(|m| m.count <= 0.0) {
            return Err(format!("{name}: every frame needs at least one point"));
        }
        let total = frames.iter().map(|m| m.count).sum::<f64>() + fixed.count;
        if total < 3.0 {
            return Err(format!("{name}: a plane needs three points, got {total}"));
        }
        let finite = |m: &PointMoments| {
            m.count.is_finite()
                && m.sum.iter().all(|v| v.is_finite())
                && m.outer.iter().all(|v| v.is_finite())
        };
        if !frames.iter().all(finite) || !finite(&fixed) {
            return Err(format!("{name}: non-finite moments"));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(format!(
                "{name}: sigma must be finite and positive, got {sigma}"
            ));
        }
        let frames = frames
            .iter()
            .map(PointMoments::centred)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("{name}: a frame without points"))?;
        Ok(Self {
            frames,
            fixed: fixed.centred(),
            sigma,
            group: PhantomData,
        })
    }

    /// Number of frames (blocks).
    pub fn frames(&self) -> usize {
        self.frames.len()
    }

    /// Total number of points, free and fixed.
    pub fn points(&self) -> f64 {
        self.frames.iter().map(|m| m.count).sum::<f64>() + self.fixed.map_or(0.0, |f| f.count)
    }
}

/// Unit eigenvector of the smallest eigenvalue of a symmetric `scatter`;
/// `None` when the two smallest are not separated.
fn smallest_axis(scatter: &Matrix3<f64>) -> Option<Vector3<f64>> {
    let eigen = SymmetricEigen::new(*scatter);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let gap = eigen.eigenvalues[order[1]] - eigen.eigenvalues[order[0]];
    (gap > MIN_EIGEN_GAP * eigen.eigenvalues[order[2]].abs().max(1.0))
        .then(|| eigen.eigenvectors.column(order[0]).into_owned())
}

/// A fitted plane `(n, d)` with `nᵀp + d = 0`.
type Fitted = (Vector3<f64>, f64);

impl<G: LieGroup + Send + Sync + 'static> LidarPlaneMomentFactor<G> {
    /// The states' actions and the best-fit plane `(n, d)` of every point:
    /// each frame's mean carried to the world, its scatter rotated. `None`
    /// when the plane has no unique normal.
    fn fit(&self, params: &[&[f64]]) -> (Vec<Rc<Action>>, Option<Fitted>) {
        let actions: Vec<Rc<Action>> = params
            .iter()
            .map(|param| cached_action::<G>(param))
            .collect();
        let centre = |action: &Action, frame: &Centred| action.0 * frame.mean + action.1;
        let mut total = 0.0;
        let mut weighted = Vector3::zeros();
        for (action, frame) in actions.iter().zip(&self.frames) {
            total += frame.count;
            weighted += centre(action, frame) * frame.count;
        }
        if let Some(fixed) = &self.fixed {
            total += fixed.count;
            weighted += fixed.mean * fixed.count;
        }
        let mean = weighted / total;
        let mut scatter = Matrix3::zeros();
        for (action, frame) in actions.iter().zip(&self.frames) {
            let a = &action.0;
            let offset = centre(action, frame) - mean;
            scatter +=
                a * frame.scatter * a.transpose() + offset * offset.transpose() * frame.count;
        }
        if let Some(fixed) = &self.fixed {
            let offset = fixed.mean - mean;
            scatter += fixed.scatter + offset * offset.transpose() * fixed.count;
        }
        let plane = smallest_axis(&scatter).map(|normal| (normal, -normal.dot(&mean)));
        (actions, plane)
    }

    /// `∂(Aᵀn)/∂X` (3 × dof) and `∂(nᵀ(Aμ + b))/∂X` (1 × dof) of a frame,
    /// zero past the group's dimension.
    fn derivatives(
        jacobian: &ActionJacobian,
        normal: &Vector3<f64>,
        mean: &Vector3<f64>,
    ) -> (SMatrix<f64, 3, MAX_DOF>, SMatrix<f64, 1, MAX_DOF>) {
        let n = normal.transpose();
        let mut d_an = SMatrix::<f64, 3, MAX_DOF>::zeros();
        let mut d_along = n * jacobian.origin;
        for (k, axis) in jacobian.axes.iter().enumerate() {
            let row = n * axis;
            d_an.set_row(k, &row);
            d_along += row * mean[k];
        }
        (d_an, d_along)
    }
}

impl<G: LieGroup + Send + Sync + 'static> Factor for LidarPlaneMomentFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let dof = G::TangentVector::DIM;
        let mut jac = jacobian;
        if let Some(jac) = jac.as_mut() {
            for row in 0..jac.nrows() {
                for col in 0..jac.ncols() {
                    *jac.rb_mut().get_mut(row, col) = 0.0;
                }
            }
        }
        let (actions, plane) = self.fit(params);
        let Some((normal, offset)) = plane else {
            residual.fill(0.0);
            return;
        };
        let information = 1.0 / self.sigma;
        for (block, (action, frame)) in actions.iter().zip(&self.frames).enumerate() {
            let (a, b, columns) = &**action;
            let root = frame.root();
            let a_n = a.transpose() * normal;
            let spread = root * a_n * information;
            let root_n = frame.count.sqrt();
            let along = normal.dot(&(a * frame.mean + b)) + offset;
            residual[4 * block..4 * block + 3].copy_from_slice(spread.as_slice());
            residual[4 * block + 3] = root_n * along * information;
            if let Some(jac) = jac.as_mut() {
                let (d_an, d_along) = Self::derivatives(columns, &normal, &frame.mean);
                let d_spread = root * d_an * information;
                for col in 0..dof {
                    for k in 0..3 {
                        *jac.rb_mut().get_mut(4 * block + k, block * dof + col) =
                            d_spread[(k, col)];
                    }
                    *jac.rb_mut().get_mut(4 * block + 3, block * dof + col) =
                        root_n * d_along[col] * information;
                }
            }
        }
        // The fixed points' residuals count too (no Jacobian): only a cost
        // over every point has the fitted plane as its minimizer, which is
        // what makes `Jᵀr` exact.
        let fixed_row = 4 * self.frames.len();
        match &self.fixed {
            Some(fixed) => {
                let spread = fixed.root() * normal * information;
                residual[fixed_row..fixed_row + 3].copy_from_slice(spread.as_slice());
                residual[fixed_row + 3] =
                    fixed.count.sqrt() * (normal.dot(&fixed.mean) + offset) * information;
            }
            None => residual[fixed_row..fixed_row + 4].fill(0.0),
        }
    }

    /// Per frame, from its scatter `S` and the derivatives `D = ∂(Aᵀn)/∂X`,
    /// `m = ∂(nᵀ(Aμ+b))/∂X`: `H += (DᵀSD + N·mᵀm)/σ²`,
    /// `g += (DᵀS·Aᵀn + N·mᵀ·e)/σ²` and cost `(nᵀASAᵀn + N·e²)/σ²` with `e`
    /// the mean's distance — what [`Factor::linearize`]'s rows sum to, with
    /// no square root of `S` and no Jacobian.
    fn accumulate_normal(
        &self,
        params: &[&[f64]],
        mut blocks: Option<&mut NormalBlocks<'_>>,
    ) -> Option<f64> {
        let (actions, plane) = self.fit(params);
        let Some((normal, offset)) = plane else {
            return Some(0.0);
        };
        let weight = 1.0 / (self.sigma * self.sigma);
        let mut cost = 0.0;
        let dof = G::TangentVector::DIM;
        for (block, (action, frame)) in actions.iter().zip(&self.frames).enumerate() {
            let (a, b, columns) = &**action;
            let a_n = a.transpose() * normal;
            let s_an = frame.scatter * a_n;
            let along = normal.dot(&(a * frame.mean + b)) + offset;
            cost += (a_n.dot(&s_an) + frame.count * along * along) * weight;
            if let Some(blocks) = blocks.as_deref_mut() {
                let (d_an, d_along) = Self::derivatives(columns, &normal, &frame.mean);
                let hessian = (d_an.transpose() * frame.scatter * d_an
                    + d_along.transpose() * d_along * frame.count)
                    * weight;
                let gradient = (d_an.transpose() * s_an
                    + d_along.transpose() * (frame.count * along))
                    * weight;
                blocks.add_hessian(block, block, &hessian.view((0, 0), (dof, dof)));
                blocks.add_gradient(block, &gradient.rows(0, dof));
            }
        }
        if let Some(fixed) = &self.fixed {
            let along = normal.dot(&fixed.mean) + offset;
            cost += (normal.dot(&(fixed.scatter * normal)) + fixed.count * along * along) * weight;
        }
        Some(cost)
    }

    fn residual_dim(&self) -> usize {
        4 * self.frames.len() + 4
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (
            self.residual_dim(),
            self.frames.len() * G::TangentVector::DIM,
        )
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &vec![G::NAME; self.frames.len()],
            "LidarPlaneMomentFactor expects one pose per frame",
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

    /// The normal equations straight from the scatter: the cost's gradient
    /// (by central differences) and the Jacobian path's `JᵀJ`, `Jᵀr`, cost.
    #[test]
    fn normal_equations_match_the_cost_and_the_jacobian() -> Result<(), String> {
        let poses = [pose(0.0), pose(1.0)];
        let factor = LidarPlaneMomentFactor::<SE3>::new(
            poses
                .iter()
                .enumerate()
                .map(|(seed, p)| PointMoments::from_points(&plane_points(p, seed)))
                .collect(),
            PointMoments::from_points(&plane_points(&SE3::identity(), 7)),
            0.01,
        )?;
        let params: Vec<&[f64]> = poses.iter().map(|p| p.as_param_slice()).collect();
        let (mut hessian, mut gradient) = (DMatrix::zeros(12, 12), DVector::zeros(12));
        let cost = factor
            .accumulate_normal(
                &params,
                Some(&mut NormalBlocks::new(&mut hessian, &mut gradient, &[0, 6])),
            )
            .ok_or("no normal equations")?;
        assert_eq!(factor.accumulate_normal(&params, None), Some(cost));

        let (rows, cols) = factor.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut j = Mat::<f64>::zeros(rows, cols);
        factor.linearize(&params, &mut r, Some(j.as_mut()));
        let j = DMatrix::from_fn(rows, cols, |a, b| j[(a, b)]);
        let r = DVector::from_column_slice(&r);
        assert!((j.transpose() * &j - &hessian).norm() < 1e-6 * hessian.norm());
        assert!((j.transpose() * &r - &gradient).norm() < 1e-6 * gradient.norm());
        assert!((r.norm_squared() - cost).abs() < 1e-6 * cost);

        // Holding the plane is exact at the optimum plane (envelope): the
        // cost's derivative is 2g.
        let step = 1e-6;
        for k in 0..12 {
            let cost_at = |s: f64| -> Option<f64> {
                let mut t = [0.0; 6];
                t[k % 6] = s;
                let moved: Vec<SE3> = poses
                    .iter()
                    .enumerate()
                    .map(|(i, p)| match i == k / 6 {
                        true => p.right_plus(
                            &<SE3 as LieGroup>::TangentVector::from_slice(&t),
                            None,
                            None,
                        ),
                        false => p.clone(),
                    })
                    .collect();
                let params: Vec<&[f64]> = moved.iter().map(|p| p.as_param_slice()).collect();
                factor.accumulate_normal(&params, None)
            };
            let numeric =
                (cost_at(step).ok_or("cost")? - cost_at(-step).ok_or("cost")?) / (4.0 * step);
            assert!(
                (numeric - gradient[k]).abs() < 1e-5 * (1.0 + gradient.norm()),
                "{k}: {numeric} vs {}",
                gradient[k]
            );
        }
        Ok(())
    }

    /// Body-frame points of a tilted plane `n·w = 2`, as seen from `state`,
    /// with millimetres of off-plane noise.
    fn plane_points<G: LieGroup>(state: &G, seed: usize) -> Vec<Vector3<f64>> {
        let normal = Vector3::new(0.2, -0.3, 0.93).normalize();
        let a = normal.cross(&Vector3::x()).normalize();
        let b = normal.cross(&a);
        (0..12)
            .map(|k| {
                let (i, j) = (f64::from((k % 4) as u32), f64::from((k / 4) as u32));
                let noise = 0.003 * f64::from(((k * 7 + seed) % 5) as u32) - 0.006;
                let w = normal * (2.0 + noise)
                    + a * (0.3 * i - 0.5 + 0.1 * seed as f64)
                    + b * (0.4 * j - 0.4);
                state.inverse(None).act(&w, None, None)
            })
            .collect()
    }

    fn cost(residual: &[f64]) -> f64 {
        residual.iter().map(|r| r * r).sum::<f64>()
    }

    /// The moment residuals carry the per-point cost exactly, and `Jᵀr`
    /// matches the finite-difference gradient of the cost.
    fn check_gradient<G: LieGroup + Send + Sync + 'static>(states: Vec<G>) -> Result<(), String> {
        let frames: Vec<Vec<Vector3<f64>>> = states
            .iter()
            .enumerate()
            .map(|(seed, state)| plane_points(state, seed))
            .collect();
        let fixed_points = plane_points(&states[0], 7)
            .iter()
            .map(|s| states[0].act(s, None, None))
            .collect::<Vec<_>>();
        let sigma = 0.02;
        let factor = LidarPlaneMomentFactor::<G>::new(
            frames.iter().map(PointMoments::from_points).collect(),
            PointMoments::from_points(&fixed_points),
            sigma,
        )?;
        // Perturb so the gradient is not zero.
        let perturbed: Vec<G> = states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let delta: Vec<f64> = (0..G::TangentVector::DIM)
                    .map(|k| 0.01 * ((i * 3 + k) % 5) as f64 - 0.02)
                    .collect();
                s.right_plus(&G::TangentVector::from_slice(&delta), None, None)
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

        // The cost equals the per-point point-to-plane sum at the fitted plane.
        let world: Vec<Vector3<f64>> = frames
            .iter()
            .zip(&perturbed)
            .flat_map(|(points, state)| points.iter().map(move |s| state.act(s, None, None)))
            .chain(fixed_points.iter().copied())
            .collect();
        let (normal, offset, _) = PointMoments::from_points(&world)
            .plane()
            .ok_or("no plane")?;
        let direct: f64 = world
            .iter()
            .map(|w| ((normal.dot(w) + offset) / sigma).powi(2))
            .sum();
        assert!(
            (cost(&residual) - direct).abs() < 1e-9 * direct.max(1.0),
            "{} vs {direct}",
            cost(&residual)
        );

        let dof = G::TangentVector::DIM;
        let h = 1e-6;
        for block in 0..perturbed.len() {
            for k in 0..dof {
                let mut gradient = 0.0;
                for row in 0..rows {
                    gradient += jacobian[(row, block * dof + k)] * residual[row];
                }
                let shifted = |sign: f64| {
                    let mut delta = vec![0.0; dof];
                    delta[k] = sign * h;
                    let mut p = params.clone();
                    p[block] = perturbed[block]
                        .right_plus(&G::TangentVector::from_slice(&delta), None, None)
                        .as_param_slice()
                        .to_vec();
                    let s: Vec<&[f64]> = p.iter().map(Vec::as_slice).collect();
                    let mut r = vec![0.0; rows];
                    factor.linearize(&s, &mut r, None);
                    cost(&r)
                };
                let numeric = (shifted(1.0) - shifted(-1.0)) / (4.0 * h);
                assert!(
                    (gradient - numeric).abs() < 1e-4 * numeric.abs().max(1.0),
                    "{} block {block} dof {k}: {gradient} vs {numeric}",
                    G::NAME
                );
            }
        }
        Ok(())
    }

    fn pose(i: f64) -> SE3 {
        SE3::new(
            Vector3::new(0.3 * i, -0.2 * i, 0.1),
            UnitQuaternion::from_euler_angles(0.1 * i, -0.05, 0.2 * i),
        )
    }

    #[test]
    fn moment_gradient_is_exact_on_se3() -> Result<(), String> {
        check_gradient((0..3).map(|i| pose(f64::from(i))).collect::<Vec<_>>())
    }

    #[test]
    fn moment_gradient_is_exact_on_se23() -> Result<(), String> {
        check_gradient(
            (0..3)
                .map(|i| {
                    let p = pose(f64::from(i));
                    SE23::new(
                        p.translation(),
                        Vector3::new(0.5, 0.0, -0.1),
                        p.rotation_quaternion(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn moment_gradient_is_exact_on_sim3() -> Result<(), String> {
        check_gradient(
            (0..2)
                .map(|i| {
                    let p = pose(f64::from(i));
                    Sim3::new(
                        p.translation(),
                        p.rotation_quaternion(),
                        1.0 + 0.1 * f64::from(i),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn transformed_moments_match_transformed_points() -> Result<(), String> {
        let points = plane_points(&pose(1.0), 2);
        let p = pose(2.0);
        let a = p.rotation_so3().rotation_matrix() * 1.3;
        let b = p.translation();
        let direct =
            PointMoments::from_points(&points.iter().map(|s| a * s + b).collect::<Vec<_>>());
        let carried = PointMoments::from_points(&points).transformed(&a, &b);
        assert!((direct.sum - carried.sum).norm() < 1e-12);
        assert!((direct.outer - carried.outer).norm() < 1e-10);
        let (normal, offset, values) = carried.plane().ok_or("no plane")?;
        assert!(values[0] < 1e-4 && values[1] > 1e-3, "{values:?}");
        assert!((normal.norm() - 1.0).abs() < 1e-12 && offset.is_finite());
        Ok(())
    }

    /// A plane of twelve points ~40 m from the body (Hilti site1_handheld_2,
    /// where the raw 4 × 4 moment form gave residuals in the thousands): the
    /// cost is `N·λ_min/σ²` to the digit.
    #[test]
    fn far_planes_keep_their_exact_cost() -> Result<(), String> {
        let moments = PointMoments {
            count: 12.0,
            sum: Vector3::new(-241.60453252371218, -418.42991819175455, -6.515154650408839),
            outer: Matrix3::new(
                4865.230606534443,
                8424.235208972244,
                131.19023951308878,
                8424.235208972244,
                14590.428639648177,
                227.2076983279844,
                131.19023951308878,
                227.2076983279844,
                4.641707333357514,
            ),
        };
        let rotation = Matrix3::from_column_slice(&[
            0.63635359169571,
            -0.7686413783581207,
            0.06515011751100691,
            -0.7695277019895543,
            -0.638419895487932,
            -0.015721097794173737,
            0.053677017490220025,
            -0.04013064316587261,
            -0.9977516270457535,
        ]);
        let translation = Vector3::new(0.5839251212596093, 0.7700903523873233, 0.9106204756308878);
        let state = SE3::new(translation, UnitQuaternion::from_matrix(&rotation));
        let sigma = 0.012247448713915891;
        let factor =
            LidarPlaneMomentFactor::<SE3>::new(vec![moments], PointMoments::default(), sigma)?;
        let mut residual = vec![0.0; factor.residual_dim()];
        factor.linearize(&[state.as_param_slice()], &mut residual, None);
        let world = moments.transformed(
            &state.rotation_so3().rotation_matrix(),
            &state.translation(),
        );
        let (_, _, values) = world.plane().ok_or("no plane")?;
        let expected = 12.0 * values[0] / (sigma * sigma);
        let cost = cost(&residual);
        assert!(
            (cost - expected).abs() < 1e-6 * expected,
            "{cost} vs {expected}"
        );
        Ok(())
    }

    #[test]
    fn construction_rejects_bad_input() {
        let one = PointMoments::from_points(&[Vector3::new(1.0, 0.0, 0.0)]);
        assert!(LidarPlaneMomentFactor::<SE3>::new(vec![], one, 0.1).is_err());
        assert!(
            LidarPlaneMomentFactor::<SE3>::new(vec![one], PointMoments::default(), 0.1).is_err()
        );
        assert!(LidarPlaneMomentFactor::<SE3>::new(vec![one, one, one], one, 0.0).is_err());
        assert!(
            LidarPlaneMomentFactor::<SE3>::new(vec![PointMoments::default()], one, 0.1).is_err()
        );
        let factor = LidarPlaneMomentFactor::<SE3>::new(vec![one, one], one, 0.1);
        assert!(factor.is_ok_and(|f| f.frames() == 2 && (f.points() - 3.0).abs() < 1e-12));
    }
}
