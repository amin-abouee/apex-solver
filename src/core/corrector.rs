use nalgebra::{DMatrix, DVector};
/// https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.cc
use std::ops::Mul;

pub(crate) struct Corrector {
    sqrt_rho1: f64,
    residual_scaling: f64,
    alpha_sq_norm: f64,
}

impl Corrector {
    pub fn new(sq_norm: f64, rho: &[f64; 3]) -> Corrector {
        let sqrt_rho1 = rho[1].sqrt();
        // is always well formed.
        if (sq_norm == 0.0) || (rho[2] <= 0.0) {
            let residual_scaling = sqrt_rho1;
            let alpha_sq_norm = 0.0;
            Corrector {
                sqrt_rho1,
                residual_scaling,
                alpha_sq_norm,
            }
        } else {
            let d = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

            let alpha = 1.0 - d.sqrt();

            let residual_scaling = sqrt_rho1 / (1.0 - alpha);
            let alpha_sq_norm = alpha / sq_norm;
            Corrector {
                sqrt_rho1,
                residual_scaling,
                alpha_sq_norm,
            }
        }
    }
    pub fn correct_jacobian(&self, residuals: &DVector<f64>, jacobian: &mut DMatrix<f64>) {
        // The common case (rho[2] <= 0).
        if self.alpha_sq_norm == 0.0 {
            *jacobian = jacobian.clone().mul(self.sqrt_rho1);
            return;
        }

        //  J = sqrt(rho) * (J - alpha^2 r * r' J)
        let r_rtj = residuals.clone() * residuals.transpose() * jacobian.clone();
        *jacobian = (jacobian.clone() - r_rtj.mul(self.alpha_sq_norm)).mul(self.sqrt_rho1);
    }
    pub fn correct_residuals(&self, residuals: &mut DVector<f64>) {
        *residuals = residuals.clone().mul(self.residual_scaling);
    }
}
