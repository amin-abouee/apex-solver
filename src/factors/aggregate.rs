//! Many factors over a few variables, compressed into one exact factor.
//!
//! A sliding-window LiDAR adjustment has thousands of plane factors over ten
//! states: the stacked Jacobian has tens of thousands of rows, and forming
//! `JᵀJ` from it dominates the solve. Their sum, though, is fully described
//! at a linearization point by
//!
//! ```text
//! H = Σ Jᵢᵀ Jᵢ   (n × n, n the summed tangent dimension)
//! g = Σ Jᵢᵀ rᵢ          c = Σ ‖rᵢ‖²
//! ```
//!
//! With `H = U Λ Uᵀ`, the compressed factor reports
//!
//! ```text
//! J' = [ Λ^½ Uᵀ ; 0 ]      r' = [ Λ^-½ Uᵀ g ; √(c − gᵀH⁺g) ]
//! ```
//!
//! so `J'ᵀJ' = H`, `J'ᵀr' = g` (`g` lies in the range of `H`) and
//! `‖r'‖² = c`: the Gauss-Newton step and the cost — what Levenberg-Marquardt
//! accepts or rejects a step on — are exactly the parts' sum, from `n + 1`
//! rows. Each part's residual is taken as already whitened (as with
//! `NoiseModel::null()`), without a robust loss.

use faer::Mat;
use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Eigenvalues of `H` below this fraction of the largest are treated as null.
const RELATIVE_NULL: f64 = 1e-14;

/// A part: a factor and the indices of the aggregate's blocks it reads, in
/// its own block order.
pub struct AggregatePart {
    /// The factor.
    pub factor: Box<dyn Factor>,
    /// Indices into the aggregate's blocks.
    pub blocks: Vec<usize>,
}

/// The sum of `parts` as one factor; see the module docs.
pub struct AggregateFactor {
    parts: Vec<AggregatePart>,
    /// Tangent dimension of each aggregate block.
    dofs: Vec<usize>,
    /// Column offset of each aggregate block.
    offsets: Vec<usize>,
}

impl AggregateFactor {
    /// Aggregate `parts` over blocks of tangent dimensions `dofs`.
    ///
    /// # Errors
    ///
    /// No part, a part reading a block out of range, or a part whose
    /// Jacobian width does not match its blocks' dimensions.
    pub fn new(parts: Vec<AggregatePart>, dofs: Vec<usize>) -> Result<Self, String> {
        if parts.is_empty() {
            return Err("AggregateFactor: no part".to_string());
        }
        for (index, part) in parts.iter().enumerate() {
            let mut width = 0;
            for &block in &part.blocks {
                width += dofs.get(block).ok_or_else(|| {
                    format!(
                        "AggregateFactor: part {index} reads block {block} of {}",
                        dofs.len()
                    )
                })?;
            }
            if width != part.factor.jacobian_shape().1 {
                return Err(format!(
                    "AggregateFactor: part {index} has {} columns, its blocks {width}",
                    part.factor.jacobian_shape().1
                ));
            }
        }
        let offsets = dofs
            .iter()
            .scan(0, |offset, dof| {
                let start = *offset;
                *offset += dof;
                Some(start)
            })
            .collect();
        Ok(Self {
            parts,
            dofs,
            offsets,
        })
    }

    /// Number of aggregated parts.
    pub fn parts(&self) -> usize {
        self.parts.len()
    }

    fn width(&self) -> usize {
        self.dofs.iter().sum()
    }

    /// `H`, `g` and `c` of the parts at `params`.
    fn normal_equations(&self, params: &[&[f64]]) -> (DMatrix<f64>, DVector<f64>, f64) {
        let n = self.width();
        let zero = || (DMatrix::zeros(n, n), DVector::zeros(n), 0.0);
        self.parts
            .par_iter()
            .fold(zero, |(mut hessian, mut gradient, mut cost), part| {
                let (rows, cols) = part.factor.jacobian_shape();
                let part_params: Vec<&[f64]> = part.blocks.iter().map(|&b| params[b]).collect();
                let mut residual = vec![0.0; rows];
                let mut jacobian = Mat::<f64>::zeros(rows, cols);
                part.factor
                    .linearize(&part_params, &mut residual, Some(jacobian.as_mut()));
                let j = DMatrix::from_fn(rows, cols, |r, c| jacobian[(r, c)]);
                let r = DVector::from_column_slice(&residual);
                cost += r.norm_squared();
                let jt = j.transpose();
                let h = &jt * &j;
                let g = &jt * &r;
                // Scatter by block: the part's block `k` sits at its own
                // column offset and at the aggregate's.
                let mut local = 0;
                let starts: Vec<(usize, usize, usize)> = part
                    .blocks
                    .iter()
                    .map(|&block| {
                        let start = (local, self.offsets[block], self.dofs[block]);
                        local += self.dofs[block];
                        start
                    })
                    .collect();
                for &(la, ga, da) in &starts {
                    let mut target = gradient.rows_mut(ga, da);
                    target += g.rows(la, da);
                    for &(lb, gb, db) in &starts {
                        let mut target = hessian.view_mut((ga, gb), (da, db));
                        target += h.view((la, lb), (da, db));
                    }
                }
                (hessian, gradient, cost)
            })
            .reduce(zero, |(h1, g1, c1), (h2, g2, c2)| {
                (h1 + h2, g1 + g2, c1 + c2)
            })
    }

    /// The parts' summed cost alone.
    fn cost(&self, params: &[&[f64]]) -> f64 {
        self.parts
            .par_iter()
            .map(|part| {
                let part_params: Vec<&[f64]> = part.blocks.iter().map(|&b| params[b]).collect();
                let mut residual = vec![0.0; part.factor.residual_dim()];
                part.factor.linearize(&part_params, &mut residual, None);
                residual.iter().map(|r| r * r).sum::<f64>()
            })
            .sum()
    }
}

impl Factor for AggregateFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let n = self.width();
        let Some(mut jac) = jacobian else {
            // A cost evaluation (a trial step): only `‖r'‖² = c` matters.
            residual.fill(0.0);
            residual[n] = self.cost(params).sqrt();
            return;
        };
        let (hessian, gradient, cost) = self.normal_equations(params);
        let eigen = SymmetricEigen::new(hessian);
        let largest = eigen.eigenvalues.iter().fold(0.0_f64, |m, v| m.max(*v));
        let null = largest * RELATIVE_NULL;
        let mut explained = 0.0;
        for (k, (value, out)) in eigen
            .eigenvalues
            .iter()
            .zip(residual.iter_mut())
            .enumerate()
        {
            let axis = eigen.eigenvectors.column(k);
            let root = if *value > null { value.sqrt() } else { 0.0 };
            *out = if root > 0.0 {
                axis.dot(&gradient) / root
            } else {
                0.0
            };
            explained += *out * *out;
            for (col, value) in axis.iter().enumerate() {
                *jac.rb_mut().get_mut(k, col) = root * value;
            }
        }
        residual[n] = (cost - explained).max(0.0).sqrt();
        for col in 0..n {
            *jac.rb_mut().get_mut(n, col) = 0.0;
        }
    }

    fn residual_dim(&self) -> usize {
        self.width() + 1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.width() + 1, self.width())
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != self.dofs.len() {
            return Err(format!(
                "AggregateFactor expects {} blocks, got {}",
                self.dofs.len(),
                variables.len()
            ));
        }
        for part in &self.parts {
            let mine: Vec<&dyn ManifoldVariable> =
                part.blocks.iter().map(|&b| variables[b]).collect();
            part.factor.validate_variables(&mine)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::lidar::{LidarPlaneMomentFactor, PointMoments};
    use apex_manifolds::se3::SE3;
    use apex_manifolds::{LieGroup, Tangent};
    use nalgebra::{UnitQuaternion, Vector3};

    fn pose(i: f64) -> SE3 {
        SE3::new(
            Vector3::new(0.3 * i, -0.2 * i, 0.1),
            UnitQuaternion::from_euler_angles(0.1 * i, -0.05, 0.2 * i),
        )
    }

    /// Body-frame points of the plane `n·w = offset` seen from `state`.
    fn points(state: &SE3, normal: Vector3<f64>, offset: f64, seed: usize) -> PointMoments {
        let a = normal.cross(&Vector3::new(0.3, 0.5, 0.7)).normalize();
        let b = normal.cross(&a);
        let world: Vec<Vector3<f64>> = (0..9)
            .map(|k| {
                let noise = 0.002 * (((k * 7 + seed) % 5) as f64 - 2.0);
                normal * (offset + noise)
                    + a * (0.3 * (k % 3) as f64 - 0.3 + 0.05 * seed as f64)
                    + b * (0.3 * (k / 3) as f64 - 0.3)
            })
            .collect();
        let body: Vec<Vector3<f64>> = world
            .iter()
            .map(|w| state.inverse(None).act(w, None, None))
            .collect();
        PointMoments::from_points(&body)
    }

    /// Three planes over three poses, each plane on a different pair: the
    /// aggregate's `J'ᵀJ'`, `J'ᵀr'` and `‖r'‖²` equal the parts' sums.
    #[test]
    fn the_aggregate_preserves_normal_equations_and_cost() -> Result<(), String> {
        let poses: Vec<SE3> = (0..3).map(|i| pose(f64::from(i))).collect();
        let moved: Vec<SE3> = poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let t: Vec<f64> = (0..6)
                    .map(|k| 0.01 * ((i + k) % 4) as f64 - 0.015)
                    .collect();
                p.right_plus(
                    &<SE3 as LieGroup>::TangentVector::from_slice(&t),
                    None,
                    None,
                )
            })
            .collect();
        let planes = [
            (Vector3::new(0.1, 0.2, 1.0).normalize(), 2.0, [0, 1]),
            (Vector3::new(1.0, -0.1, 0.1).normalize(), 3.0, [1, 2]),
            (Vector3::new(0.2, 1.0, 0.0).normalize(), -1.5, [0, 2]),
        ];
        let make = |normal: Vector3<f64>,
                    offset: f64,
                    pair: [usize; 2]|
         -> Result<Box<dyn Factor>, String> {
            let frames = pair
                .iter()
                .enumerate()
                .map(|(seed, &i)| points(&poses[i], normal, offset, seed))
                .collect();
            Ok(Box::new(LidarPlaneMomentFactor::<SE3>::new(
                frames,
                points(&poses[0], normal, offset, 5).transformed(
                    &poses[0].rotation_so3().rotation_matrix(),
                    &poses[0].translation(),
                ),
                0.01,
            )?))
        };
        let parts = planes
            .iter()
            .map(|(n, d, pair)| {
                Ok(AggregatePart {
                    factor: make(*n, *d, *pair)?,
                    blocks: pair.to_vec(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let aggregate = AggregateFactor::new(parts, vec![6, 6, 6])?;
        let params: Vec<&[f64]> = moved.iter().map(|p| p.as_param_slice()).collect();
        let (hessian, gradient, cost) = aggregate.normal_equations(&params);

        let (rows, cols) = aggregate.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut j = Mat::<f64>::zeros(rows, cols);
        aggregate.linearize(&params, &mut r, Some(j.as_mut()));
        let jd = DMatrix::from_fn(rows, cols, |a, b| j[(a, b)]);
        let rd = DVector::from_column_slice(&r);
        let scale = hessian.norm();
        assert!((jd.transpose() * &jd - &hessian).norm() < 1e-9 * scale);
        assert!((jd.transpose() * &rd - &gradient).norm() < 1e-9 * (1.0 + gradient.norm()));
        assert!(
            (rd.norm_squared() - cost).abs() < 1e-9 * cost.max(1.0),
            "{} vs {cost}",
            rd.norm_squared()
        );
        let mut cost_only = vec![0.0; rows];
        aggregate.linearize(&params, &mut cost_only, None);
        let norm: f64 = cost_only.iter().map(|v| v * v).sum();
        assert!((norm - cost).abs() < 1e-9 * cost.max(1.0));
        assert_eq!(aggregate.parts(), 3);
        Ok(())
    }

    #[test]
    fn construction_rejects_bad_layouts() -> Result<(), String> {
        assert!(AggregateFactor::new(Vec::new(), vec![6]).is_err());
        let moments = PointMoments::from_points(&[
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ]);
        let part = |blocks: Vec<usize>| -> Result<AggregatePart, String> {
            Ok(AggregatePart {
                factor: Box::new(LidarPlaneMomentFactor::<SE3>::new(
                    vec![moments],
                    PointMoments::default(),
                    0.1,
                )?),
                blocks,
            })
        };
        assert!(AggregateFactor::new(vec![part(vec![3])?], vec![6]).is_err());
        assert!(AggregateFactor::new(vec![part(vec![0])?], vec![9]).is_err());
        assert!(AggregateFactor::new(vec![part(vec![0])?], vec![6]).is_ok());
        Ok(())
    }
}
