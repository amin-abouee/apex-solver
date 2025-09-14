pub use nalgebra as na;

use crate::manifold::{LieGroup, se2::SE2, se3::SE3};

pub trait Factor: Send + Sync {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>);
    fn get_dimension(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE2 {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
}

impl BetweenFactorSE2 {
    pub fn new(dx: f64, dy: f64, dtheta: f64) -> Self {
        Self { dx, dy, dtheta }
    }

    pub fn from_se2(relative_pose: SE2) -> Self {
        Self {
            dx: relative_pose.translation().x,
            dy: relative_pose.translation().y,
            dtheta: relative_pose.angle(),
        }
    }
}

impl Factor for BetweenFactorSE2 {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>) {
        // Use standard SE2 formulation with [x, y, theta] parameter ordering
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];

        // Create nalgebra Isometry2 objects from pose parameters
        let se2_origin_k0 = na::Isometry2::new(
            na::Vector2::new(t_origin_k0[0], t_origin_k0[1]), // x, y from indices 0, 1
            t_origin_k0[2],                                   // theta from index 2
        );
        let se2_origin_k1 = na::Isometry2::new(
            na::Vector2::new(t_origin_k1[0], t_origin_k1[1]), // x, y from indices 0, 1
            t_origin_k1[2],                                   // theta from index 2
        );

        // Create measurement from dx, dy, dtheta relative transformation
        let se2_k0_k1 = na::Isometry2::new(na::Vector2::new(self.dx, self.dy), self.dtheta);

        // Compute residual: se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1
        let se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;

        // Extract residual as [dx, dy, dtheta]
        let residual = na::dvector![
            se2_diff.translation.x,
            se2_diff.translation.y,
            se2_diff.rotation.angle()
        ];

        // Use numerical differentiation for Jacobian computation
        let epsilon = 1e-8;
        let mut jacobian = na::DMatrix::<f64>::zeros(3, 6);

        // Numerical Jacobian for k0 (parameters 0, 1, 2)
        for i in 0..3 {
            let mut params_plus = params[0].clone();
            params_plus[i] += epsilon;
            let se2_k0_plus = na::Isometry2::new(
                na::Vector2::new(params_plus[1], params_plus[2]),
                params_plus[0],
            );
            let se2_diff_plus = se2_origin_k1.inverse() * se2_k0_plus * se2_k0_k1;
            let residual_plus = na::dvector![
                se2_diff_plus.translation.x,
                se2_diff_plus.translation.y,
                se2_diff_plus.rotation.angle()
            ];

            let grad = (residual_plus - &residual) / epsilon;
            jacobian.column_mut(i).copy_from(&grad);
        }

        // Numerical Jacobian for k1 (parameters 3, 4, 5)
        for i in 0..3 {
            let mut params_plus = params[1].clone();
            params_plus[i] += epsilon;
            let se2_k1_plus = na::Isometry2::new(
                na::Vector2::new(params_plus[1], params_plus[2]),
                params_plus[0],
            );
            let se2_diff_plus = se2_k1_plus.inverse() * se2_origin_k0 * se2_k0_k1;
            let residual_plus = na::dvector![
                se2_diff_plus.translation.x,
                se2_diff_plus.translation.y,
                se2_diff_plus.rotation.angle()
            ];

            let grad = (residual_plus - &residual) / epsilon;
            jacobian.column_mut(3 + i).copy_from(&grad);
        }

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE3 {
    pub relative_pose: SE3,
}

impl BetweenFactorSE3 {
    pub fn new(relative_pose: SE3) -> Self {
        Self { relative_pose }
    }
}

impl Factor for BetweenFactorSE3 {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>) {
        // Use standard SE3 formulation with [tx, ty, tz, qw, qx, qy, qz] parameter ordering
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];

        // Convert parameters to SE3 objects using apex-solver conversion
        let se3_origin_k0 = SE3::from(t_origin_k0.clone());
        let se3_origin_k1 = SE3::from(t_origin_k1.clone());

        // Compute residual: se3_diff = se3_origin_k1.inverse() * se3_origin_k0 * se3_k0_k1
        // For SE3, we use the direct multiplication approach
        let se3_k1_inv = se3_origin_k1.inverse(None);
        let temp = se3_origin_k0.compose(&self.relative_pose, None, None);
        let se3_diff = se3_k1_inv.compose(&temp, None, None);
        let residual = se3_diff.log(None).to_vector();

        // Use numerical differentiation for Jacobian computation
        let epsilon = 1e-8;
        let mut jacobian = na::DMatrix::<f64>::zeros(6, 14);

        // Helper function to compute residual
        let relative_pose_copy = self.relative_pose.clone();
        let compute_residual =
            |params0: &na::DVector<f64>, params1: &na::DVector<f64>| -> na::DVector<f64> {
                let se3_k0 = SE3::from(params0.clone());
                let se3_k1 = SE3::from(params1.clone());
                let se3_k1_inv = se3_k1.inverse(None);
                let temp = se3_k0.compose(&relative_pose_copy, None, None);
                let diff = se3_k1_inv.compose(&temp, None, None);
                diff.log(None).to_vector()
            };

        let base_residual = compute_residual(t_origin_k0, t_origin_k1);

        // Numerical Jacobian for k0 (7 parameters)
        for i in 0..7 {
            let mut params_plus = t_origin_k0.clone();
            params_plus[i] += epsilon;
            let residual_plus = compute_residual(&params_plus, t_origin_k1);
            let grad = (residual_plus - &base_residual) / epsilon;
            jacobian.column_mut(i).copy_from(&grad);
        }

        // Numerical Jacobian for k1 (7 parameters)
        for i in 0..7 {
            let mut params_plus = t_origin_k1.clone();
            params_plus[i] += epsilon;
            let residual_plus = compute_residual(t_origin_k0, &params_plus);
            let grad = (residual_plus - &base_residual) / epsilon;
            jacobian.column_mut(7 + i).copy_from(&grad);
        }

        // Return full 6x14 Jacobian (6 residual dims × 14 total parameters)
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        6
    }
}

#[derive(Debug, Clone)]
pub struct PriorFactor {
    pub data: na::DVector<f64>,
}
impl Factor for PriorFactor {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let residual = &params[0] - &self.data;
        let jacobian = na::DMatrix::<f64>::identity(residual.nrows(), residual.nrows());
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.data.len()
    }
}
