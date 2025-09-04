use nalgebra as na;
// use rayon::prelude::*;

use crate::core::corrector::Corrector;
use crate::core::factors::FactorImpl;
use crate::core::loss_functions::Loss;
use crate::core::variable::Variable;
use crate::manifold::LieGroup;

pub struct ResidualBlock {
    pub residual_block_id: usize,
    pub dim_residual: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub factor: Box<dyn FactorImpl + Send>,
    pub loss_func: Option<Box<dyn Loss + Send>>,
}

impl ResidualBlock {
    pub fn new(
        residual_block_id: usize,
        dim_residual: usize,
        residual_row_start_idx: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn FactorImpl + Send>,
        loss_func: Option<Box<dyn Loss + Send>>,
    ) -> Self {
        ResidualBlock {
            residual_block_id,
            dim_residual,
            residual_row_start_idx,
            variable_key_list: variable_key_size_list
                .iter()
                .map(|s| s.to_string())
                .collect(),
            factor,
            loss_func,
        }
    }

    pub fn residual_and_jacobian<M>(
        &self,
        variables: &[&Variable<M>],
    ) -> (na::DVector<f64>, na::DMatrix<f64>)
    where
        M: LieGroup + Clone + Into<na::DVector<f64>>,
        M::TangentVector: crate::manifold::Tangent<M>,
    {
        let param_vec: Vec<_> = variables.iter().map(|v| v.value.clone().into()).collect();
        let (mut residual, mut jacobian) = self.factor.residual_with_jacobian(&param_vec);
        let squared_norm = residual.norm_squared();
        if let Some(loss_func) = self.loss_func.as_ref() {
            let rho = loss_func.evaluate(squared_norm);
            let corrector = Corrector::new(squared_norm, &rho);
            corrector.correct_jacobian(&residual, &mut jacobian);
            corrector.correct_residuals(&mut residual);
        }
        (residual, jacobian)
    }
}


