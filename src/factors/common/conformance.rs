//! Conformance harness: every factor's analytic Jacobian against the canonical
//! central difference, for any group it admits.
//!
//! A [`Block`] is one variable of a factor: its parameter vector, its tangent
//! dimension and how the group retracts a tangent step onto it. The check
//! perturbs each tangent axis of each block by `±STEP`, differences the
//! residuals and compares with the `linearize` Jacobian column by column.

use apex_manifolds::{LieGroup, Tangent};
use faer::mat::MatMut;

use crate::factors::Factor;

/// Central-difference step on the tangent space.
pub(crate) const STEP: f64 = 1e-6;

type Retract = Box<dyn Fn(&[f64], &[f64]) -> Vec<f64>>;

/// One variable of a factor under test.
pub(crate) struct Block {
    params: Vec<f64>,
    dof: usize,
    retract: Retract,
}

impl Block {
    /// A block on the Lie group `G`, retracted with `right_plus`.
    pub(crate) fn lie<G: LieGroup + 'static>(value: &G) -> Self {
        Self {
            params: value.as_param_slice().to_vec(),
            dof: value.tangent_dim(),
            retract: Box::new(|params, step| {
                let base = G::from_param_slice(params);
                let tangent = <G::TangentVector as Tangent<G>>::from_slice(step);
                base.right_plus(&tangent, None, None)
                    .as_param_slice()
                    .to_vec()
            }),
        }
    }

    /// A Euclidean block, retracted by addition.
    pub(crate) fn euclidean(values: &[f64]) -> Self {
        Self {
            params: values.to_vec(),
            dof: values.len(),
            retract: Box::new(|params, step| params.iter().zip(step).map(|(p, d)| p + d).collect()),
        }
    }

    fn moved(&self, axis: usize, delta: f64) -> Vec<f64> {
        let mut step = vec![0.0; self.dof];
        step[axis] = delta;
        (self.retract)(&self.params, &step)
    }
}

/// The residual and column-major Jacobian of `factor` at `params`.
pub(crate) fn evaluate<F: Factor + ?Sized>(factor: &F, params: &[&[f64]]) -> (Vec<f64>, Vec<f64>) {
    let (rows, cols) = factor.jacobian_shape();
    let mut residual = vec![0.0; factor.residual_dim()];
    let mut jacobian = vec![0.0; rows * cols];
    factor.linearize(
        params,
        &mut residual,
        Some(MatMut::from_column_major_slice_mut(
            &mut jacobian,
            rows,
            cols,
        )),
    );
    (residual, jacobian)
}

fn residual_at<F: Factor + ?Sized>(factor: &F, params: &[Vec<f64>]) -> Vec<f64> {
    let views: Vec<&[f64]> = params.iter().map(Vec::as_slice).collect();
    let mut residual = vec![0.0; factor.residual_dim()];
    factor.linearize(&views, &mut residual, None);
    residual
}

/// Largest absolute difference between the analytic Jacobian and the central
/// difference over all columns, or a message naming the first shape mismatch.
///
/// # Errors
///
/// The factor's Jacobian shape disagrees with the blocks' tangent dimensions.
pub(crate) fn max_jacobian_error<F: Factor + ?Sized>(
    factor: &F,
    blocks: &[Block],
) -> Result<f64, String> {
    let (rows, cols) = factor.jacobian_shape();
    let dofs: usize = blocks.iter().map(|b| b.dof).sum();
    if cols != dofs {
        return Err(format!("jacobian has {cols} columns, blocks span {dofs}"));
    }
    let views: Vec<&[f64]> = blocks.iter().map(|b| b.params.as_slice()).collect();
    let (_, analytic) = evaluate(factor, &views);
    let base: Vec<Vec<f64>> = blocks.iter().map(|b| b.params.clone()).collect();

    let mut worst = 0.0f64;
    let mut column = 0;
    for (index, block) in blocks.iter().enumerate() {
        for axis in 0..block.dof {
            let mut plus = base.clone();
            plus[index] = block.moved(axis, STEP);
            let mut minus = base.clone();
            minus[index] = block.moved(axis, -STEP);
            let up = residual_at(factor, &plus);
            let down = residual_at(factor, &minus);
            for row in 0..rows {
                let numeric = (up[row] - down[row]) / (2.0 * STEP);
                worst = worst.max((numeric - analytic[column * rows + row]).abs());
            }
            column += 1;
        }
    }
    Ok(worst)
}

/// Assert the analytic Jacobian matches the central difference within `tol`.
///
/// # Errors
///
/// A shape mismatch, or an error above `tol` (reported with its size).
pub(crate) fn finite_diff_jacobian_parity<F: Factor + ?Sized>(
    factor: &F,
    blocks: &[Block],
    tol: f64,
) -> Result<(), String> {
    let worst = max_jacobian_error(factor, blocks)?;
    if worst < tol {
        Ok(())
    } else {
        Err(format!("max Jacobian error {worst:.3e} exceeds {tol:.1e}"))
    }
}

/// Assert two factors give bit-equal residuals and Jacobians at `params`.
///
/// # Errors
///
/// The first differing entry, or a shape difference.
pub(crate) fn bit_parity<A: Factor + ?Sized, B: Factor + ?Sized>(
    old: &A,
    new: &B,
    params: &[&[f64]],
) -> Result<(), String> {
    if old.jacobian_shape() != new.jacobian_shape() {
        return Err("jacobian shapes differ".into());
    }
    let (r_old, j_old) = evaluate(old, params);
    let (r_new, j_new) = evaluate(new, params);
    let same = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    if !same(&r_old, &r_new) {
        return Err("residuals differ".into());
    }
    if !same(&j_old, &j_new) {
        return Err("jacobians differ".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::pose::{BetweenFactor, PriorFactor};
    use apex_manifolds::se2::SE2;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sgal3::SGal3;
    use apex_manifolds::sim3::Sim3;
    use apex_manifolds::so2::SO2;
    use apex_manifolds::so3::SO3;

    const TOL: f64 = 1e-5;

    fn between_parity<G: LieGroup + Send + Sync + 'static>() -> Result<(), String> {
        let factor = BetweenFactor::new(G::random());
        let a = G::random();
        let b = G::random();
        finite_diff_jacobian_parity(&factor, &[Block::lie(&a), Block::lie(&b)], TOL)
            .map_err(|e| format!("{}: {e}", G::NAME))
    }

    fn prior_parity<G: LieGroup + Send + Sync + 'static>() -> Result<(), String> {
        let factor = PriorFactor::new(G::random());
        let x = G::random();
        finite_diff_jacobian_parity(&factor, &[Block::lie(&x)], TOL)
            .map_err(|e| format!("{}: {e}", G::NAME))
    }

    #[test]
    fn between_factor_conforms_on_every_group() -> Result<(), String> {
        between_parity::<SO2>()?;
        between_parity::<SE2>()?;
        between_parity::<SO3>()?;
        between_parity::<SE3>()?;
        between_parity::<SE23>()?;
        between_parity::<SGal3>()?;
        between_parity::<Sim3>()
    }

    #[test]
    fn prior_factor_conforms_on_every_group() -> Result<(), String> {
        prior_parity::<SO2>()?;
        prior_parity::<SE2>()?;
        prior_parity::<SO3>()?;
        prior_parity::<SE3>()?;
        prior_parity::<SE23>()?;
        prior_parity::<SGal3>()?;
        prior_parity::<Sim3>()
    }

    #[test]
    fn a_wrong_jacobian_is_caught() {
        struct Wrong;
        impl Factor for Wrong {
            fn linearize(&self, p: &[&[f64]], r: &mut [f64], j: Option<MatMut<'_, f64>>) {
                r[0] = p[0][0] * p[0][0];
                if let Some(mut j) = j {
                    *j.rb_mut().get_mut(0, 0) = 1.0;
                }
            }
            fn residual_dim(&self) -> usize {
                1
            }
            fn jacobian_shape(&self) -> (usize, usize) {
                (1, 1)
            }
        }
        use faer::prelude::ReborrowMut;
        let result = finite_diff_jacobian_parity(&Wrong, &[Block::euclidean(&[3.0])], TOL);
        assert!(result.is_err());
    }

    #[test]
    fn bit_parity_accepts_identical_and_rejects_different_factors() -> Result<(), String> {
        let relative = SE3::random();
        let a = SE3::random();
        let b = SE3::random();
        let params: [&[f64]; 2] = [a.as_param_slice(), b.as_param_slice()];
        let first = BetweenFactor::new(relative.clone());
        bit_parity(&first, &BetweenFactor::new(relative), &params)?;
        match bit_parity(&first, &BetweenFactor::new(SE3::random()), &params) {
            Ok(()) => Err("different measurements compared equal".into()),
            Err(_) => Ok(()),
        }
    }

    mod lidar {
        use super::*;
        use crate::factors::lidar::{
            GicpFactor, IcpFactor, LidarEdgeFactor, Plane, PointToPlaneFactor, PoseToPointFactor,
            PrecomputedPlane,
        };
        use apex_manifolds::capability::ActsOnPoints3;
        use nalgebra::{Matrix3, Vector3};

        const POINT: [f64; 3] = [0.4, -0.7, 2.0];

        fn pair<G: ActsOnPoints3 + 'static>() -> [Block; 2] {
            [Block::lie(&G::random()), Block::euclidean(&POINT)]
        }

        fn two<G: ActsOnPoints3 + 'static>() -> [Block; 2] {
            [Block::lie(&G::random()), Block::lie(&G::random())]
        }

        fn conforms<G: ActsOnPoints3 + Send + Sync + 'static>() -> Result<(), String> {
            let name = |e: String| format!("{}: {e}", G::NAME);
            let plane = Plane::new(Vector3::new(0.2, 0.3, 0.9), -0.5).map_err(|e| e.to_string())?;
            let cov = Matrix3::identity() * 0.05;
            finite_diff_jacobian_parity(
                &PoseToPointFactor::<G>::new(Vector3::new(1.0, 2.0, 3.0)),
                &pair::<G>(),
                TOL,
            )
            .map_err(name)?;
            finite_diff_jacobian_parity(&PointToPlaneFactor::<G>::new(plane), &pair::<G>(), TOL)
                .map_err(name)?;
            let gicp =
                GicpFactor::<G>::new(Vector3::new(1.0, 2.0, 3.0), cov, cov, Matrix3::identity())
                    .map_err(|e| e.to_string())?;
            finite_diff_jacobian_parity(&gicp, &pair::<G>(), TOL).map_err(name)?;
            let icp = IcpFactor::<G, _>::new(
                PrecomputedPlane::new(Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.2, 1.0)),
                Vector3::new(0.3, 0.1, -0.4),
                0.05,
            );
            finite_diff_jacobian_parity(&icp, &two::<G>(), TOL).map_err(name)?;
            let edge = LidarEdgeFactor::<G>::new_isotropic(
                Vector3::new(0.3, 0.1, -0.4),
                Vector3::new(0.0, 0.5, 0.0),
                Vector3::new(1.0, 0.2, 0.1),
                0.05,
            );
            finite_diff_jacobian_parity(&edge, &two::<G>(), TOL).map_err(name)
        }

        #[test]
        fn the_lidar_correspondence_factors_conform_on_every_admitted_group() -> Result<(), String>
        {
            conforms::<SO3>()?;
            conforms::<SE3>()?;
            conforms::<SE23>()?;
            conforms::<Sim3>()
        }
    }
}
