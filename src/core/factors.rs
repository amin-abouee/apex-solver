pub use nalgebra as na;

use crate::manifold::{LieGroup, se2::SE2, se3::SE3};

pub trait Factor: Send + Sync {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>);
    fn get_dimension(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE2 {
    pub relative_pose: SE2,
}

impl BetweenFactorSE2 {
    pub fn new(dx: f64, dy: f64, dtheta: f64) -> Self {
        let relative_pose = SE2::from_xy_angle(dx, dy, dtheta);
        Self { relative_pose }
    }

    pub fn from_se2(relative_pose: SE2) -> Self {
        Self { relative_pose }
    }
}

impl Factor for BetweenFactorSE2 {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>) {
        // TEMPORARY: Use numerical jacobians that match tiny-solver exactly
        // Input: params = [theta, x, y] for each pose (TINY-SOLVER FORMAT)
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];

        // Create Isometry2 exactly like tiny-solver: params[0] = theta, params[1] = x, params[2] = y
        let se2_origin_k0 = na::Isometry2::new(
            na::Vector2::new(t_origin_k0[1], t_origin_k0[2]), // x, y
            t_origin_k0[0],                                   // theta
        );
        let se2_origin_k1 = na::Isometry2::new(
            na::Vector2::new(t_origin_k1[1], t_origin_k1[2]), // x, y
            t_origin_k1[0],                                   // theta
        );
        let se2_k0_k1 = na::Isometry2::new(
            na::Vector2::new(self.relative_pose.x(), self.relative_pose.y()),
            self.relative_pose.angle(),
        );

        // Exact tiny-solver residual computation
        let se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;
        let residual = na::dvector![
            se2_diff.translation.x,
            se2_diff.translation.y,
            se2_diff.rotation.angle()
        ];

        // Compute numerical jacobian (matching tiny-solver's automatic differentiation)
        let eps = 1e-8;
        let mut jacobian = na::DMatrix::zeros(3, 6);

        // Tiny-solver compatible residual function
        let compute_residual = |params: &[na::DVector<f64>]| -> na::DVector<f64> {
            let k0 = &params[0];
            let k1 = &params[1];
            let iso_k0 = na::Isometry2::new(na::Vector2::new(k0[1], k0[2]), k0[0]);
            let iso_k1 = na::Isometry2::new(na::Vector2::new(k1[1], k1[2]), k1[0]);
            let iso_measured = na::Isometry2::new(
                na::Vector2::new(self.relative_pose.x(), self.relative_pose.y()),
                self.relative_pose.angle(),
            );
            let diff = iso_k1.inverse() * iso_k0 * iso_measured;
            na::dvector![
                diff.translation.x,
                diff.translation.y,
                diff.rotation.angle()
            ]
        };

        for param_idx in 0..2 {
            for component in 0..3 {
                let mut params_plus = [t_origin_k0.clone(), t_origin_k1.clone()];
                let mut params_minus = [t_origin_k0.clone(), t_origin_k1.clone()];

                params_plus[param_idx][component] += eps;
                params_minus[param_idx][component] -= eps;

                let residual_plus = compute_residual(&params_plus);
                let residual_minus = compute_residual(&params_minus);

                let numerical_derivative = (&residual_plus - &residual_minus) / (2.0 * eps);

                let col_idx = param_idx * 3 + component;
                for row in 0..3 {
                    jacobian[(row, col_idx)] = numerical_derivative[row];
                }
            }
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
        let se3_origin_k0 = SE3::from(params[0].clone());
        let se3_origin_k1 = SE3::from(params[1].clone());
        let se3_k0_k1_measured = &self.relative_pose;

        // Step 1: se3_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = na::Matrix6::zeros();
        let se3_k1_inv = se3_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se3_k1_inv * se3_origin_k0
        let mut j_compose1_wrt_k1_inv = na::Matrix6::zeros();
        let mut j_compose1_wrt_k0 = na::Matrix6::zeros();
        let se3_temp = se3_k1_inv.compose(
            &se3_origin_k0,
            Some(&mut j_compose1_wrt_k1_inv),
            Some(&mut j_compose1_wrt_k0),
        );

        // Step 3: se3_temp * se3_k0_k1_measured
        let mut j_compose2_wrt_temp = na::Matrix6::zeros();
        let se3_diff = se3_temp.compose(se3_k0_k1_measured, Some(&mut j_compose2_wrt_temp), None);

        // Step 4: se3_diff.log()
        let mut j_log_wrt_diff = na::Matrix6::zeros();
        let residual = se3_diff.log(Some(&mut j_log_wrt_diff));

        // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
        let j_temp_wrt_k1 = j_compose1_wrt_k1_inv * j_k1_inv_wrt_k1;
        let j_diff_wrt_k0 = j_compose2_wrt_temp * j_compose1_wrt_k0;
        let j_diff_wrt_k1 = j_compose2_wrt_temp * j_temp_wrt_k1;

        let jacobian_wrt_k0 = j_log_wrt_diff * j_diff_wrt_k0;
        let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

        let mut jacobian = na::DMatrix::<f64>::zeros(6, 12);
        jacobian
            .fixed_view_mut::<6, 6>(0, 0)
            .copy_from(&jacobian_wrt_k0);
        jacobian
            .fixed_view_mut::<6, 6>(0, 6)
            .copy_from(&jacobian_wrt_k1);

        (residual.into(), jacobian)
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
