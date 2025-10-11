use crate::manifold::{LieGroup, se2::SE2, se3::SE3};
use faer::{Col, Mat};

pub trait Factor: Send + Sync {
    fn linearize(&self, params: &[Col<f64>]) -> (Col<f64>, Mat<f64>);
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
    fn linearize(&self, params: &[Col<f64>]) -> (Col<f64>, Mat<f64>) {
        // Use numerical jacobians with SE2 manifold operations
        // Input: params = [theta, x, y] for each pose (TINY-SOLVER FORMAT)
        let se2_origin_k0 = SE2::from(params[0].clone());
        let se2_origin_k1 = SE2::from(params[1].clone());

        // Compute residual: se2_k1_inv * se2_k0 * se2_k0_k1_measured
        let se2_k1_inv = se2_origin_k1.inverse(None);
        let se2_temp = se2_k1_inv.compose(&se2_origin_k0, None, None);
        let se2_diff = se2_temp.compose(&self.relative_pose, None, None);

        // Convert to tangent space (residual)
        let tangent = se2_diff.log(None);
        let residual: Col<f64> = tangent.into();

        // Compute numerical jacobian
        let eps = 1e-8;
        let mut jacobian = Mat::zeros(3, 6);

        // Residual function for numerical differentiation
        let compute_residual = |params: &[Col<f64>]| -> Col<f64> {
            let k0 = SE2::from(params[0].clone());
            let k1 = SE2::from(params[1].clone());
            let k1_inv = k1.inverse(None);
            let temp = k1_inv.compose(&k0, None, None);
            let diff = temp.compose(&self.relative_pose, None, None);
            let tang = diff.log(None);
            tang.into()
        };

        for param_idx in 0..2 {
            for component in 0..3 {
                let mut params_plus = [params[0].clone(), params[1].clone()];
                let mut params_minus = [params[0].clone(), params[1].clone()];

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
    fn linearize(&self, params: &[Col<f64>]) -> (Col<f64>, Mat<f64>) {
        let se3_origin_k0 = SE3::from(params[0].clone());
        let se3_origin_k1 = SE3::from(params[1].clone());
        let se3_k0_k1_measured = &self.relative_pose;

        // Step 1: se3_origin_k1.inverse()
        let mut j_k1_inv_wrt_k1 = Mat::<f64>::zeros(6, 6);
        let se3_k1_inv = se3_origin_k1.inverse(Some(&mut j_k1_inv_wrt_k1));

        // Step 2: se3_k1_inv * se3_origin_k0
        let mut j_compose1_wrt_k1_inv = Mat::<f64>::zeros(6, 6);
        let mut j_compose1_wrt_k0 = Mat::<f64>::zeros(6, 6);
        let se3_temp = se3_k1_inv.compose(
            &se3_origin_k0,
            Some(&mut j_compose1_wrt_k1_inv),
            Some(&mut j_compose1_wrt_k0),
        );

        // Step 3: se3_temp * se3_k0_k1_measured
        let mut j_compose2_wrt_temp = Mat::<f64>::zeros(6, 6);
        let se3_diff = se3_temp.compose(se3_k0_k1_measured, Some(&mut j_compose2_wrt_temp), None);

        // Step 4: se3_diff.log()
        let mut j_log_wrt_diff = Mat::<f64>::zeros(6, 6);
        let residual = se3_diff.log(Some(&mut j_log_wrt_diff));

        // Chain rule: d(residual)/d(k0) and d(residual)/d(k1)
        let j_temp_wrt_k1 = &j_compose1_wrt_k1_inv * &j_k1_inv_wrt_k1;
        let j_diff_wrt_k0 = &j_compose2_wrt_temp * &j_compose1_wrt_k0;
        let j_diff_wrt_k1 = &j_compose2_wrt_temp * &j_temp_wrt_k1;

        let jacobian_wrt_k0 = &j_log_wrt_diff * &j_diff_wrt_k0;
        let jacobian_wrt_k1 = &j_log_wrt_diff * &j_diff_wrt_k1;

        let mut jacobian = Mat::<f64>::zeros(6, 12);
        jacobian
            .as_mut()
            .submatrix_mut(0, 0, 6, 6)
            .copy_from(&jacobian_wrt_k0);
        jacobian
            .as_mut()
            .submatrix_mut(0, 6, 6, 6)
            .copy_from(&jacobian_wrt_k1);

        (residual.into(), jacobian)
    }

    fn get_dimension(&self) -> usize {
        6
    }
}

#[derive(Debug, Clone)]
pub struct PriorFactor {
    pub data: Col<f64>,
}
impl Factor for PriorFactor {
    fn linearize(&self, params: &[Col<f64>]) -> (Col<f64>, Mat<f64>) {
        let residual = &params[0] - &self.data;
        let jacobian = Mat::<f64>::identity(residual.nrows(), residual.nrows());
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.data.nrows()
    }
}
