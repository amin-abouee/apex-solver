use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

// use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer_ext::IntoFaer;
use nalgebra as na;
use rayon::prelude::*;

use crate::core::variable::Variable;
use crate::core::{factors, loss_functions, residual_block};
use crate::manifold::{ManifoldType, rn::Rn, se2::SE2, se3::SE3, so2::SO2, so3::SO3};

/// Enum to handle mixed manifold variable types
#[derive(Clone, Debug)]
pub enum VariableEnum {
    Rn(Variable<Rn>),
    SE2(Variable<SE2>),
    SE3(Variable<SE3>),
    SO2(Variable<SO2>),
    SO3(Variable<SO3>),
}

impl VariableEnum {
    /// Get the tangent space size for this variable
    pub fn get_size(&self) -> usize {
        match self {
            VariableEnum::Rn(var) => var.get_size(),
            VariableEnum::SE2(var) => var.get_size(),
            VariableEnum::SE3(var) => var.get_size(),
            VariableEnum::SO2(var) => var.get_size(),
            VariableEnum::SO3(var) => var.get_size(),
        }
    }

    /// Convert to DVector for use with Factor trait
    pub fn to_vector(&self) -> na::DVector<f64> {
        match self {
            VariableEnum::Rn(var) => var.value.clone().into(),
            VariableEnum::SE2(var) => var.value.clone().into(),
            VariableEnum::SE3(var) => var.value.clone().into(),
            VariableEnum::SO2(var) => var.value.clone().into(),
            VariableEnum::SO3(var) => var.value.clone().into(),
        }
    }
}

pub struct Problem {
    pub total_residual_dimension: usize,
    residual_id_count: usize,
    residual_blocks: HashMap<usize, residual_block::ResidualBlock>,
    pub fixed_variable_indexes: HashMap<String, HashSet<usize>>,
    pub variable_bounds: HashMap<String, HashMap<usize, (f64, f64)>>,
}
impl Default for Problem {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem {
    pub fn new() -> Self {
        Self {
            total_residual_dimension: 0,
            residual_id_count: 0,
            residual_blocks: HashMap::new(),
            fixed_variable_indexes: HashMap::new(),
            variable_bounds: HashMap::new(),
        }
    }

    // pub fn get_variable_name_to_col_idx_dict(
    //     &self,
    //     variables: &HashMap<String, Variable<M>>,
    // ) -> HashMap<String, usize> {
    //     let mut count_col_idx = 0;
    //     let mut variable_name_to_col_idx_dict = HashMap::new();
    //     variables.iter().for_each(|(var_name, variable)| {
    //         variable_name_to_col_idx_dict.insert(var_name.to_owned(), count_col_idx);
    //         count_col_idx += variable.get_size();
    //     });
    //     variable_name_to_col_idx_dict
    // }

    pub fn add_residual_block(
        &mut self,
        variable_key_size_list: &[&str],
        factor: Box<dyn factors::Factor + Send>,
        loss_func: Option<Box<dyn loss_functions::Loss + Send>>,
    ) -> usize {
        let new_residual_dimension = factor.get_dimension();
        self.residual_blocks.insert(
            self.residual_id_count,
            residual_block::ResidualBlock::new(
                self.residual_id_count,
                self.total_residual_dimension,
                variable_key_size_list,
                factor,
                loss_func,
            ),
        );
        let block_id = self.residual_id_count;
        self.residual_id_count += 1;

        self.total_residual_dimension += new_residual_dimension;

        block_id
    }

    pub fn remove_residual_block(
        &mut self,
        block_id: usize,
    ) -> Option<residual_block::ResidualBlock> {
        if let Some(residual_block) = self.residual_blocks.remove(&block_id) {
            self.total_residual_dimension -= residual_block.factor.get_dimension();
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

    /// Initialize variables from initial values with mixed manifold types
    ///
    /// This method handles a collection of different manifold types and creates
    /// appropriate Variable instances for each based on the ManifoldType.
    pub fn initialize_variables(
        &self,
        initial_values: &HashMap<String, (ManifoldType, na::DVector<f64>)>,
    ) -> HashMap<String, VariableEnum> {
        let variables: HashMap<String, VariableEnum> = initial_values
            .iter()
            .map(|(k, v)| {
                let variable_enum = match v.0 {
                    ManifoldType::SO2 => {
                        let mut var = Variable::new(SO2::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SO2(var)
                    }
                    ManifoldType::SO3 => {
                        let mut var = Variable::new(SO3::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SO3(var)
                    }
                    ManifoldType::SE2 => {
                        let mut var = Variable::new(SE2::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SE2(var)
                    }
                    ManifoldType::SE3 => {
                        let mut var = Variable::new(SE3::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SE3(var)
                    }
                    ManifoldType::RN => {
                        let mut var = Variable::new(Rn::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::Rn(var)
                    }
                };

                (k.to_owned(), variable_enum)
            })
            .collect();
        variables
    }

    /// Compute residual and jacobian for mixed manifold types
    pub fn compute_residual_and_jacobian_mixed(
        &self,
        variables: &HashMap<String, VariableEnum>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
    ) -> (faer::Mat<f64>, na::DMatrix<f64>) {
        // Calculate total degrees of freedom
        let total_dof: usize = variables.values().map(|var| var.get_size()).sum();

        let total_residual = Arc::new(Mutex::new(na::DVector::<f64>::zeros(
            self.total_residual_dimension,
        )));
        let total_jacobian = Arc::new(Mutex::new(na::DMatrix::<f64>::zeros(
            self.total_residual_dimension,
            total_dof,
        )));

        self.residual_blocks
            .par_iter()
            .for_each(|(_, residual_block)| {
                let mut param_vectors: Vec<na::DVector<f64>> = Vec::new();
                let mut var_sizes: Vec<usize> = Vec::new();

                for var_key in &residual_block.variable_key_list {
                    if let Some(variable) = variables.get(var_key) {
                        param_vectors.push(variable.to_vector());
                        var_sizes.push(variable.get_size());
                    }
                }

                let (res, jac) = residual_block.factor.linearize(&param_vectors);

                {
                    let mut total_residual = total_residual.lock().unwrap();
                    total_residual
                        .rows_mut(
                            residual_block.residual_row_start_idx,
                            residual_block.factor.get_dimension(),
                        )
                        .copy_from(&res);
                }
                {
                    let mut total_jacobian = total_jacobian.lock().unwrap();
                    let mut current_col = 0;
                    for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                        if let Some(col_idx) = variable_name_to_col_idx_dict.get(var_key) {
                            let tangent_size = var_sizes[i];
                            total_jacobian
                                .view_mut(
                                    (residual_block.residual_row_start_idx, *col_idx),
                                    (residual_block.factor.get_dimension(), tangent_size),
                                )
                                .copy_from(&jac.view(
                                    (0, current_col),
                                    (residual_block.factor.get_dimension(), tangent_size),
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
