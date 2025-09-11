pub use nalgebra as na;

use crate::manifold::{LieGroup, se2::SE2, se3::SE3};

pub trait Factor: Send + Sync {
    fn linearize(&self, params: &[na::DVector<f64>]) -> (na::DVector<f64>, na::DMatrix<f64>);
    fn get_dimension(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE2 {
    // pub dx: f64,
    // pub dy: f64,
    // pub dtheta: f64,
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

        (residual.to_vector(), jacobian)
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
        // let p0 = &params[0];
        // let p1 = &params[1];
        let se3_origin_k0 = SE3::from(params[0].clone());
        let se3_origin_k1 = SE3::from(params[1].clone());

        // let se3_k0_k1_measured = SE3::from_translation_quaternion(
        //     na::Vector3::new(self.dtx, self.dty, self.dtz),
        //     na::Quaternion::new(self.dqw, self.dqx, self.dqy, self.dqz),
        // );
        let se3_k0_k1_measured = &self.relative_pose;

        // Predicted measurement: T_k0_k1 = (T_w_k0)^-1 * T_w_k1
        let mut j_predicted_wrt_k0 = na::Matrix6::zeros();
        let mut j_predicted_wrt_k1 = na::Matrix6::zeros();
        let se3_k0_k1_predicted = se3_origin_k0.between(
            &se3_origin_k1,
            Some(&mut j_predicted_wrt_k0),
            Some(&mut j_predicted_wrt_k1),
        );

        // Compute residual and Jacobians
        // residual = log( (T_k0_k1_measured)^-1 * T_k0_k1_predicted )
        let mut j_residual_wrt_predicted = na::Matrix6::zeros();
        let residual = se3_k0_k1_predicted.right_minus(
            se3_k0_k1_measured,
            Some(&mut j_residual_wrt_predicted),
            None,
        );

        // Chain rule for Jacobians
        let jacobian_wrt_k0 = j_residual_wrt_predicted * j_predicted_wrt_k0;
        let jacobian_wrt_k1 = j_residual_wrt_predicted * j_predicted_wrt_k1;

        let mut jacobian = na::DMatrix::<f64>::zeros(6, 12);
        jacobian
            .fixed_view_mut::<6, 6>(0, 0)
            .copy_from(&jacobian_wrt_k0);
        jacobian
            .fixed_view_mut::<6, 6>(0, 6)
            .copy_from(&jacobian_wrt_k1);

        (residual.to_vector(), jacobian)
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
