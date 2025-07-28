pub use nalgebra as na;

use crate::manifold::se3::SE3;

pub trait Factor: Send + Sync {
    fn residual_with_jacobian(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, na::DMatrix<f64>);
}

pub trait FactorImpl: Factor {}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE2 {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
}
impl Factor for BetweenFactorSE2 {
    fn residual_with_jacobian(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];
        let se2_origin_k0 = na::Isometry2::new(
            na::Vector2::new(t_origin_k0[1], t_origin_k0[2]),
            t_origin_k0[0],
        );
        let se2_origin_k1 = na::Isometry2::new(
            na::Vector2::new(t_origin_k1[1], t_origin_k1[2]),
            t_origin_k1[0],
        );
        let se2_k0_k1 = na::Isometry2::new(
            na::Vector2::new(self.dx, self.dy),
            self.dtheta,
        );

        let se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;
        let residual = na::dvector![
            se2_diff.translation.x,
            se2_diff.translation.y,
            se2_diff.rotation.angle()
        ];

        let mut jacobian = na::DMatrix::<f64>::zeros(3, 6);
        let theta0 = t_origin_k0[0];
        let (s, c) = (theta0.sin(), theta0.cos());
        jacobian.fixed_slice_mut::<3, 3>(0, 0).copy_from(&na::Matrix3::new(
            -c, -s, -s * self.dx + c * self.dy,
            s,  -c, -c * self.dx - s * self.dy,
            0.0, 0.0, -1.0
        ));
        jacobian.fixed_slice_mut::<3, 3>(0, 3).copy_from(&na::Matrix3::identity());

        (residual, jacobian)
    }
}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE3 {
    pub dtx: f64,
    pub dty: f64,
    pub dtz: f64,
    pub dqx: f64,
    pub dqy: f64,
    pub dqz: f64,
    pub dqw: f64,
}
impl Factor for BetweenFactorSE3 {
    fn residual_with_jacobian(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];
        let se3_origin_k0 = SE3::from_vec(t_origin_k0.as_view());
        let se3_origin_k1 = SE3::from_vec(t_origin_k1.as_view());

        let se3_k0_k1 = SE3::from_vec(
            na::dvector![self.dqx, self.dqy, self.dqz, self.dqw, self.dtx, self.dty, self.dtz,]
                .as_view(),
        );

        let se3_diff = se3_origin_k1.inverse() * se3_origin_k0 * se3_k0_k1;
        let residual = se3_diff.log();

        let mut jacobian = na::DMatrix::<f64>::zeros(6, 12);
        let ad_se3_diff_inv = se3_diff.inverse().ad();
        jacobian.fixed_slice_mut::<6, 6>(0, 0).copy_from(&ad_se3_diff_inv);
        jacobian.fixed_slice_mut::<6, 6>(0, 6).copy_from(&-na::Matrix6::<f64>::identity());

        (residual, jacobian)
    }
}

#[derive(Debug, Clone)]
pub struct PriorFactor {
    pub v: na::DVector<f64>,
}
impl Factor for PriorFactor {
    fn residual_with_jacobian(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let residual = params[0].clone() - self.v.clone();
        let jacobian = na::DMatrix::<f64>::identity(residual.nrows(), residual.nrows());
        (residual, jacobian)
    }
}
