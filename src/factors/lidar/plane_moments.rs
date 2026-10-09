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
//! and with `M = L·Lᵀ` the frame's four residuals `r = Lᵀc / σ` carry exactly
//! the per-point cost. World points of frames already outside the window
//! ("fixed") add their own `M` with `c = (n, d)` and no Jacobian.
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
use nalgebra::{Matrix3, Matrix4, SymmetricEigen, Vector3, Vector4};

use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::group_point::acts_on_points;
use crate::factors::common::validate::expect_manifolds;

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

    /// The homogeneous moment matrix `Σ s̃·s̃ᵀ`, `s̃ = (s, 1)`.
    fn homogeneous(&self) -> Matrix4<f64> {
        let mut m = Matrix4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.outer);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.sum);
        m.fixed_view_mut::<1, 3>(3, 0)
            .copy_from(&self.sum.transpose());
        m[(3, 3)] = self.count;
        m
    }

    /// `Lᵀ` with `L·Lᵀ = Σ s̃·s̃ᵀ`, from the eigen-decomposition (the matrix
    /// is singular for coplanar points, which a Cholesky factor would refuse).
    fn root(&self) -> Matrix4<f64> {
        let eigen = SymmetricEigen::new(self.homogeneous());
        let mut root = Matrix4::zeros();
        for k in 0..4 {
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
    frames: Vec<PointMoments>,
    roots: Vec<Matrix4<f64>>,
    fixed: PointMoments,
    fixed_root: Matrix4<f64>,
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
        if !acts_on_points::<G>() {
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
        Ok(Self {
            roots: frames.iter().map(PointMoments::root).collect(),
            frames,
            fixed_root: fixed.root(),
            fixed,
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
        self.frames.iter().map(|m| m.count).sum::<f64>() + self.fixed.count
    }
}

/// The affine map `s ↦ A·s + b` of `state`'s action, and the action's
/// Jacobian `∂(X⊙s)/∂X` (3 × dof) at the origin and at each unit axis.
fn affine_action<G: LieGroup>(state: &G) -> (Matrix3<f64>, Vector3<f64>, [Vec<Vector3<f64>>; 4]) {
    let dof = state.tangent_dim();
    let at = |s: &Vector3<f64>| {
        let mut jacobian = G::zero_jacobian();
        let world = state.act(s, Some(&mut jacobian), None);
        let columns = (0..dof)
            .map(|col| Vector3::new(jacobian[(0, col)], jacobian[(1, col)], jacobian[(2, col)]))
            .collect::<Vec<_>>();
        (world, columns)
    };
    let (b, j0) = at(&Vector3::zeros());
    let (x, jx) = at(&Vector3::x());
    let (y, jy) = at(&Vector3::y());
    let (z, jz) = at(&Vector3::z());
    let a = Matrix3::from_columns(&[x - b, y - b, z - b]);
    (a, b, [j0, jx, jy, jz])
}

impl<G: LieGroup + Send + Sync + 'static> Factor for LidarPlaneMomentFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let dof = G::TangentVector::DIM;
        let actions: Vec<_> = params
            .iter()
            .map(|param| affine_action(&G::from_param_slice(param)))
            .collect();
        let mut total = self.fixed;
        for ((a, b, _), moments) in actions.iter().zip(&self.frames) {
            total.merge(&moments.transformed(a, b));
        }
        let mut jac = jacobian;
        if let Some(jac) = jac.as_mut() {
            for row in 0..jac.nrows() {
                for col in 0..jac.ncols() {
                    *jac.rb_mut().get_mut(row, col) = 0.0;
                }
            }
        }
        let Some((normal, offset, _)) = total.plane() else {
            residual.fill(0.0);
            return;
        };
        let information = 1.0 / self.sigma;
        for (block, ((a, b, columns), root)) in actions.iter().zip(&self.roots).enumerate() {
            let a_n = a.transpose() * normal;
            let c = Vector4::new(a_n.x, a_n.y, a_n.z, normal.dot(b) + offset);
            let r = root * c * information;
            residual[4 * block..4 * block + 4].copy_from_slice(r.as_slice());
            if let Some(jac) = jac.as_mut() {
                let [j0, jx, jy, jz] = columns;
                for col in 0..dof {
                    let origin = normal.dot(&j0[col]);
                    let dc = Vector4::new(
                        normal.dot(&jx[col]) - origin,
                        normal.dot(&jy[col]) - origin,
                        normal.dot(&jz[col]) - origin,
                        origin,
                    );
                    let dr = root * dc * information;
                    for k in 0..4 {
                        *jac.rb_mut().get_mut(4 * block + k, block * dof + col) = dr[k];
                    }
                }
            }
        }
        // The fixed points' residuals count too (no Jacobian): only a cost
        // over every point has the fitted plane as its minimizer, which is
        // what makes `Jᵀr` exact.
        let c = Vector4::new(normal.x, normal.y, normal.z, offset);
        let r = self.fixed_root * c * information;
        let fixed_row = 4 * self.frames.len();
        residual[fixed_row..fixed_row + 4].copy_from_slice(r.as_slice());
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
    use nalgebra::UnitQuaternion;

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
