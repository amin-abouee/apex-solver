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
    pub fn new(relative_pose: SE2) -> Self {
        Self { relative_pose }
    }
}

impl Factor for BetweenFactorSE2 {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let se2_origin_k0 = SE2::from(params[0].clone());
        let se2_origin_k1 = SE2::from(params[1].clone());
        // let se2_origin_k0 = SE2::from_xy_angle(t_origin_k0[1], t_origin_k0[2], t_origin_k0[0]);
        // let se2_origin_k1 = SE2::from_xy_angle(t_origin_k1[1], t_origin_k1[2], t_origin_k1[0]);
        // let se2_k0_k1_measured = SE2::from_xy_angle(self.dx, self.dy, self.dtheta);
        let se2_k0_k1_measured = &self.relative_pose;

        // Predicted measurement: T_k0_k1 = (T_w_k0)^-1 * T_w_k1
        let mut j_predicted_wrt_k0 = na::Matrix3::zeros();
        let mut j_predicted_wrt_k1 = na::Matrix3::zeros();
        let se2_k0_k1_predicted = se2_origin_k0.between(
            &se2_origin_k1,
            Some(&mut j_predicted_wrt_k0),
            Some(&mut j_predicted_wrt_k1),
        );

        // Compute residual and Jacobians
        // residual = log( (T_k0_k1_measured)^-1 * T_k0_k1_predicted )
        let mut j_residual_wrt_predicted = na::Matrix3::zeros();
        let residual = se2_k0_k1_predicted.right_minus(
            se2_k0_k1_measured,
            Some(&mut j_residual_wrt_predicted),
            None,
        );

        // Chain rule for Jacobians
        let jacobian_wrt_k0 = j_residual_wrt_predicted * j_predicted_wrt_k0;
        let jacobian_wrt_k1 = j_residual_wrt_predicted * j_predicted_wrt_k1;

        let mut jacobian = na::DMatrix::<f64>::zeros(3, 6);
        jacobian
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&jacobian_wrt_k0);
        jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&jacobian_wrt_k1);

        (residual.into(), jacobian)
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
