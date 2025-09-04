use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

// use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer_ext::IntoFaer;
use nalgebra as na;
use rayon::prelude::*;

use crate::core::variable::Variable;
use crate::core::{factors, residual_block, loss_functions};
use crate::manifold::LieGroup;

type ResidualBlockId = usize;

/// Generic Problem struct that works with any manifold type M
///
/// This struct represents an optimization problem where all variables
/// live on the same manifold type M. For problems with mixed manifold types,
/// multiple Problem instances can be used or a higher-level coordinator.
pub struct Problem<M: LieGroup> {
    pub total_residual_dimension: usize,
    residual_id_count: usize,
    residual_blocks: HashMap<ResidualBlockId, residual_block::ResidualBlock>,
    pub fixed_variable_indexes: HashMap<String, HashSet<usize>>,
    pub variable_bounds: HashMap<String, HashMap<usize, (f64, f64)>>,
    _phantom: std::marker::PhantomData<M>,
}
impl<M> Default for Problem<M>
where
    M: LieGroup + Clone + Send + Sync + 'static + Into<na::DVector<f64>>,
    M::TangentVector: crate::manifold::Tangent<M>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M> Problem<M>
where
    M: LieGroup + Clone + Send + Sync + 'static + Into<na::DVector<f64>>,
    M::TangentVector: crate::manifold::Tangent<M>,
{
    pub fn new() -> Self {
        Self {
            total_residual_dimension: 0,
            residual_id_count: 0,
            residual_blocks: HashMap::new(),
            fixed_variable_indexes: HashMap::new(),
            variable_bounds: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_variable_name_to_col_idx_dict(
        &self,
        variables: &HashMap<String, Variable<M>>,
    ) -> HashMap<String, usize> {
        let mut count_col_idx = 0;
        let mut variable_name_to_col_idx_dict = HashMap::new();
        variables
            .iter()
            .for_each(|(var_name, variable)| {
                variable_name_to_col_idx_dict.insert(var_name.to_owned(), count_col_idx);
                count_col_idx += variable.get_size();
            });
        variable_name_to_col_idx_dict
    }

    pub fn add_residual_block(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn factors::FactorImpl + Send>,
        loss_func: Option<Box<dyn loss_functions::Loss + Send>>,
    ) -> ResidualBlockId {
        self.residual_blocks.insert(
            self.residual_id_count,
            residual_block::ResidualBlock::new(
                self.residual_id_count,
                dim_residual,
                self.total_residual_dimension,
                variable_key_size_list,
                factor,
                loss_func,
            ),
        );
        let block_id = self.residual_id_count;
        self.residual_id_count += 1;

        self.total_residual_dimension += dim_residual;

        block_id
    }

    pub fn remove_residual_block(
        &mut self,
        block_id: ResidualBlockId,
    ) -> Option<residual_block::ResidualBlock> {
        if let Some(residual_block) = self.residual_blocks.remove(&block_id) {
            self.total_residual_dimension -= residual_block.dim_residual;
            Some(residual_block)
        } else {
            None
        }
    }

    pub fn fix_variable(&mut self, var_to_fix: &str, idx: usize) {
        if let Some(var_mut) = self.fixed_variable_indexes.get_mut(var_to_fix) {
            var_mut.insert(idx);
        } else {
            self.fixed_variable_indexes
                .insert(var_to_fix.to_owned(), HashSet::from([idx]));
        }
    }

    pub fn unfix_variable(&mut self, var_to_unfix: &str) {
        self.fixed_variable_indexes.remove(var_to_unfix);
    }

    pub fn set_variable_bounds(
        &mut self,
        var_to_bound: &str,
        idx: usize,
        lower_bound: f64,
        upper_bound: f64,
    ) {
        if lower_bound > upper_bound {
            log::error!("lower bound is larger than upper bound");
        } else if let Some(var_mut) = self.variable_bounds.get_mut(var_to_bound) {
            var_mut.insert(idx, (lower_bound, upper_bound));
        } else {
            self.variable_bounds.insert(
                var_to_bound.to_owned(),
                HashMap::from([(idx, (lower_bound, upper_bound))]),
            );
        }
    }

    pub fn remove_variable_bounds(&mut self, var_to_unbound: &str) {
        self.variable_bounds.remove(var_to_unbound);
    }

    /// Initialize variables from initial values
    ///
    /// This method requires that the manifold type M can be constructed from
    /// a vector representation. For most manifolds, this means the vector
    /// should contain the appropriate number of parameters.
    pub fn initialize_variables(
        &self,
        initial_values: &HashMap<String, na::DVector<f64>>,
    ) -> HashMap<String, Variable<M>>
    where
        M: From<na::DVector<f64>>,
    {
        let variables: HashMap<String, Variable<M>> = initial_values
            .iter()
            .map(|(k, v)| {
                let manifold_value = M::from(v.clone());
                let mut variable = Variable::new(manifold_value);

                if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                    variable.fixed_indices = indexes.clone();
                }
                if let Some(bounds) = self.variable_bounds.get(k) {
                    variable.bounds = bounds.clone();
                }

                (k.to_owned(), variable)
            })
            .collect();
        variables
    }

    pub fn compute_residual_and_jacobian(
        &self,
        variables: &HashMap<String, Variable<M>>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
    ) -> (faer::Mat<f64>, na::DMatrix<f64>) {
        let total_residual = Arc::new(Mutex::new(na::DVector::<f64>::zeros(
            self.total_residual_dimension,
        )));
        let total_jacobian = Arc::new(Mutex::new(na::DMatrix::<f64>::zeros(
            self.total_residual_dimension,
            variable_name_to_col_idx_dict.len(),
        )));

        self.residual_blocks
            .par_iter()
            .for_each(|(_, residual_block)| {
                let mut vars: Vec<&Variable<M>> = Vec::new();
                for var_key in &residual_block.variable_key_list {
                    if let Some(variable) = variables.get(var_key) {
                        vars.push(variable);
                    };
                }
                let (res, jac) = residual_block.residual_and_jacobian(&vars);
                {
                    let mut total_residual = total_residual.lock().unwrap();
                    total_residual
                        .rows_mut(
                            residual_block.residual_row_start_idx,
                            residual_block.dim_residual,
                        )
                        .copy_from(&res);
                }
                {
                    let mut total_jacobian = total_jacobian.lock().unwrap();
                    let mut current_col = 0;
                    for (_i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                        if let Some(col_idx) = variable_name_to_col_idx_dict.get(var_key) {
                            let variable = variables.get(var_key).unwrap();
                            let tangent_size = variable.get_size();
                            total_jacobian
                                .view_mut(
                                    (residual_block.residual_row_start_idx, *col_idx),
                                    (residual_block.dim_residual, tangent_size),
                                )
                                .copy_from(&jac.view(
                                    (0, current_col),
                                    (residual_block.dim_residual, tangent_size),
                                ));
                            current_col += tangent_size;
                        }
                    }
                }
            });

        let total_residual = Arc::try_unwrap(total_residual)
            .unwrap()
            .into_inner()
            .unwrap();
        let total_jacobian = Arc::try_unwrap(total_jacobian)
            .unwrap()
            .into_inner()
            .unwrap();

        let residual_faer = total_residual.view_range(.., ..).into_faer().to_owned();
        (residual_faer, total_jacobian)
    }
}
