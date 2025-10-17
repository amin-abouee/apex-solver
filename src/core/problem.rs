use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};

use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer_ext::IntoFaer;
use nalgebra::DVector;
use rayon::prelude::*;

use crate::core::variable::Variable;
use crate::core::{factors, loss_functions, residual_block};
use crate::manifold::{ManifoldType, rn::Rn, se2::SE2, se3::SE3, so2::SO2, so3::SO3};

/// Symbolic structure for sparse matrix operations
pub struct SymbolicStructure {
    pub pattern: SymbolicSparseColMat<usize>,
    pub order: Argsort<usize>,
}

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
    pub fn to_vector(&self) -> DVector<f64> {
        match self {
            VariableEnum::Rn(var) => var.value.clone().into(),
            VariableEnum::SE2(var) => var.value.clone().into(),
            VariableEnum::SE3(var) => var.value.clone().into(),
            VariableEnum::SO2(var) => var.value.clone().into(),
            VariableEnum::SO3(var) => var.value.clone().into(),
        }
    }

    /// Apply a tangent space step to update this variable.
    ///
    /// This method applies a manifold plus operation: x_new = x ⊞ δx
    /// where δx is a tangent vector. It supports all manifold types.
    ///
    /// # Arguments
    /// * `step_slice` - View into the full step vector for this variable's DOF
    ///
    /// # Implementation Notes
    /// Uses explicit clone instead of unsafe memory copy (`IntoNalgebra`) for small vectors.
    /// This is safe and performant for typical manifold dimensions (1-6 DOF).
    ///
    pub fn apply_tangent_step(&mut self, step_slice: faer::MatRef<f64>) {
        match self {
            VariableEnum::SE3(var) => {
                // SE3 has 6 DOF in tangent space
                let step_data: Vec<f64> = (0..6).map(|i| step_slice[(i, 0)]).collect();
                let step_dvector = DVector::from_vec(step_data);
                let tangent = crate::manifold::se3::SE3Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SE2(var) => {
                // SE2 has 3 DOF in tangent space
                let step_data: Vec<f64> = (0..3).map(|i| step_slice[(i, 0)]).collect();
                let step_dvector = DVector::from_vec(step_data);
                let tangent = crate::manifold::se2::SE2Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SO3(var) => {
                // SO3 has 3 DOF in tangent space
                let step_data: Vec<f64> = (0..3).map(|i| step_slice[(i, 0)]).collect();
                let step_dvector = DVector::from_vec(step_data);
                let tangent = crate::manifold::so3::SO3Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SO2(var) => {
                // SO2 has 1 DOF in tangent space
                let step_data = step_slice[(0, 0)];
                let step_dvector = DVector::from_vec(vec![step_data]);
                let tangent = crate::manifold::so2::SO2Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::Rn(var) => {
                // Rn has dynamic size
                let size = var.get_size();
                let step_data: Vec<f64> = (0..size).map(|i| step_slice[(i, 0)]).collect();
                let step_dvector = DVector::from_vec(step_data);
                let tangent = crate::manifold::rn::RnTangent::new(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
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

    pub fn initialize_variables(
        &self,
        initial_values: &HashMap<String, (ManifoldType, DVector<f64>)>,
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

    /// Get the number of residual blocks
    pub fn num_residual_blocks(&self) -> usize {
        self.residual_blocks.len()
    }

    /// Build symbolic structure for sparse Jacobian computation
    ///
    /// This method constructs the sparsity pattern of the Jacobian matrix before numerical
    /// computation. It determines which entries in the Jacobian will be non-zero based on
    /// the structure of the optimization problem (which residual blocks connect which variables).
    ///
    /// # Purpose
    /// - Pre-allocates memory for sparse matrix operations
    /// - Enables efficient sparse linear algebra (avoiding dense operations)
    /// - Computed once at the beginning, used throughout optimization
    ///
    /// # Arguments
    /// * `variables` - Map of variable names to their values and properties (SE2, SE3, etc.)
    /// * `variable_index_sparce_matrix` - Map from variable name to starting column index in Jacobian
    /// * `total_dof` - Total degrees of freedom (number of columns in Jacobian)
    ///
    /// # Returns
    /// A `SymbolicStructure` containing:
    /// - `pattern`: The symbolic sparse column matrix structure (row/col indices of non-zeros)
    /// - `order`: An ordering/permutation for efficient numerical computation
    ///
    /// # Algorithm
    /// For each residual block:
    /// 1. Identify which variables it depends on
    /// 2. For each (residual_dimension × variable_dof) block, mark entries as non-zero
    /// 3. Convert to optimized sparse matrix representation
    ///
    /// # Example Structure
    /// For a simple problem with 3 SE2 poses (9 DOF total):
    /// - Between(x0, x1): Creates 3×6 block at rows 0-2, cols 0-5
    /// - Between(x1, x2): Creates 3×6 block at rows 3-5, cols 3-8
    /// - Prior(x0): Creates 3×3 block at rows 6-8, cols 0-2
    ///
    /// Result: 9×9 sparse Jacobian with 45 non-zero entries
    pub fn build_symbolic_structure(
        &self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_sparce_matrix: &HashMap<String, usize>,
        total_dof: usize,
    ) -> crate::core::ApexResult<SymbolicStructure> {
        // Vector to accumulate all (row, col) pairs that will be non-zero in the Jacobian
        // Each Pair represents one entry in the sparse matrix
        let mut indices = Vec::<Pair<usize, usize>>::new();

        // Iterate through all residual blocks (factors/constraints) in the problem
        // Each residual block contributes a block of entries to the Jacobian
        self.residual_blocks.iter().for_each(|(_, residual_block)| {
            // Create local indexing for this residual block's variables
            // Maps each variable to its local starting index and size within this factor
            // Example: For Between(x0, x1) with SE2: [(0, 3), (3, 3)]
            //   - x0 starts at local index 0, has 3 DOF
            //   - x1 starts at local index 3, has 3 DOF
            let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
            let mut count_variable_local_idx: usize = 0;

            // Build the local index mapping for this residual block
            for var_key in &residual_block.variable_key_list {
                if let Some(variable) = variables.get(var_key) {
                    // Store (local_start_index, dof_size) for this variable
                    variable_local_idx_size_list
                        .push((count_variable_local_idx, variable.get_size()));
                    count_variable_local_idx += variable.get_size();
                }
            }

            // For each variable in this residual block, generate Jacobian entries
            for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                if let Some(variable_global_idx) = variable_index_sparce_matrix.get(var_key) {
                    // Get the DOF size for this variable
                    let (_, var_size) = variable_local_idx_size_list[i];

                    // Generate all (row, col) pairs for the Jacobian block:
                    // ∂(residual) / ∂(variable)
                    //
                    // For a residual block with dimension R and variable with DOF V:
                    // Creates R × V entries in the Jacobian

                    // Iterate over each residual dimension (rows)
                    for row_idx in 0..residual_block.factor.get_dimension() {
                        // Iterate over each variable DOF (columns)
                        for col_idx in 0..var_size {
                            // Compute global row index:
                            // Start from this residual block's first row, add offset
                            let global_row_idx = residual_block.residual_row_start_idx + row_idx;

                            // Compute global column index:
                            // Start from this variable's first column, add offset
                            let global_col_idx = variable_global_idx + col_idx;

                            // Record this (row, col) pair as a non-zero entry
                            indices.push(Pair::new(global_row_idx, global_col_idx));
                        }
                    }
                }
            }
        });

        // Convert the list of (row, col) pairs into an optimized symbolic sparse matrix
        // This performs:
        // 1. Duplicate elimination (same entry might be referenced multiple times)
        // 2. Sorting for column-wise storage format
        // 3. Computing a fill-reducing ordering for numerical stability
        // 4. Allocating the symbolic structure (no values yet, just pattern)
        let (pattern, order) = SymbolicSparseColMat::try_new_from_indices(
            self.total_residual_dimension, // Number of rows (total residual dimension)
            total_dof,                     // Number of columns (total DOF)
            &indices,                      // List of non-zero entry locations
        )
        .map_err(|_| {
            crate::core::ApexError::MatrixOperation(
                "Failed to build symbolic sparse matrix structure".to_string(),
            )
        })?;

        // Return the symbolic structure that will be filled with numerical values later
        Ok(SymbolicStructure { pattern, order })
    }

    /// Compute residual and sparse Jacobian for mixed manifold types
    pub fn compute_residual_and_jacobian_sparse(
        &self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_sparce_matrix: &HashMap<String, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> crate::core::ApexResult<(faer::Mat<f64>, SparseColMat<usize, f64>)> {
        let total_residual = Arc::new(Mutex::new(DVector::<f64>::zeros(
            self.total_residual_dimension,
        )));

        let jacobian_values: Result<Vec<Vec<f64>>, crate::core::ApexError> = self
            .residual_blocks
            .par_iter()
            .map(|(_, residual_block)| {
                self.compute_residual_and_jacobian_block(
                    residual_block,
                    variables,
                    variable_index_sparce_matrix,
                    &total_residual,
                )
            })
            .collect();

        let jacobian_values: Vec<f64> = jacobian_values?.into_iter().flatten().collect();

        let total_residual = Arc::try_unwrap(total_residual)
            .map_err(|_| {
                crate::core::ApexError::ThreadError(
                    "Failed to unwrap Arc for total residual".to_string(),
                )
            })?
            .into_inner()
            .map_err(|_| {
                crate::core::ApexError::ThreadError(
                    "Failed to extract mutex inner value for total residual".to_string(),
                )
            })?;

        let residual_faer = total_residual.view_range(.., ..).into_faer().to_owned();
        let jacobian_sparse = SparseColMat::new_from_argsort(
            symbolic_structure.pattern.clone(),
            &symbolic_structure.order,
            jacobian_values.as_slice(),
        )
        .map_err(|_| {
            crate::core::ApexError::MatrixOperation(
                "Failed to create sparse Jacobian from argsort".to_string(),
            )
        })?;

        Ok((residual_faer, jacobian_sparse))
    }

    fn compute_residual_and_jacobian_block(
        &self,
        residual_block: &residual_block::ResidualBlock,
        variables: &HashMap<String, VariableEnum>,
        variable_index_sparce_matrix: &HashMap<String, usize>,
        total_residual: &Arc<Mutex<DVector<f64>>>,
    ) -> crate::core::ApexResult<Vec<f64>> {
        let mut param_vectors: Vec<DVector<f64>> = Vec::new();
        let mut var_sizes: Vec<usize> = Vec::new();
        let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
        let mut count_variable_local_idx: usize = 0;

        for var_key in &residual_block.variable_key_list {
            if let Some(variable) = variables.get(var_key) {
                param_vectors.push(variable.to_vector());
                let var_size = variable.get_size();
                var_sizes.push(var_size);
                variable_local_idx_size_list.push((count_variable_local_idx, var_size));
                count_variable_local_idx += var_size;
            }
        }

        let (res, jac) = residual_block.factor.linearize(&param_vectors);

        // Update total residual
        {
            let mut total_residual = total_residual.lock().map_err(|_| {
                crate::core::ApexError::ThreadError(
                    "Failed to acquire lock on total residual".to_string(),
                )
            })?;
            total_residual
                .rows_mut(
                    residual_block.residual_row_start_idx,
                    residual_block.factor.get_dimension(),
                )
                .copy_from(&res);
        }

        // Extract Jacobian values in the correct order
        let mut local_jacobian_values = Vec::new();
        for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
            if variable_index_sparce_matrix.contains_key(var_key) {
                let (variable_local_idx, var_size) = variable_local_idx_size_list[i];
                let variable_jac = jac.view((0, variable_local_idx), (jac.shape().0, var_size));

                for row_idx in 0..jac.shape().0 {
                    for col_idx in 0..var_size {
                        local_jacobian_values.push(variable_jac[(row_idx, col_idx)]);
                    }
                }
            } else {
                return Err(crate::core::ApexError::InvalidInput(format!(
                    "Missing key {} in variable-to-column-index mapping",
                    var_key
                )));
            }
        }

        Ok(local_jacobian_values)
    }

    /// Log residual vector to a text file
    pub fn log_residual_to_file(
        &self,
        residual: &DVector<f64>,
        filename: &str,
    ) -> Result<(), std::io::Error> {
        let mut file = File::create(filename)?;
        writeln!(file, "# Residual vector - {} elements", residual.len())?;
        for (i, &value) in residual.iter().enumerate() {
            writeln!(file, "{}: {:.12}", i, value)?;
        }
        Ok(())
    }

    /// Log sparse Jacobian matrix to a text file
    pub fn log_sparse_jacobian_to_file(
        &self,
        jacobian: &SparseColMat<usize, f64>,
        filename: &str,
    ) -> Result<(), std::io::Error> {
        let mut file = File::create(filename)?;
        writeln!(
            file,
            "# Sparse Jacobian matrix - {} x {} ({} non-zeros)",
            jacobian.nrows(),
            jacobian.ncols(),
            jacobian.compute_nnz()
        )?;
        writeln!(file, "# Matrix saved as dimensions and non-zero count only")?;
        writeln!(file, "# For detailed access, convert to dense matrix first")?;
        Ok(())
    }

    /// Log variables to a text file
    pub fn log_variables_to_file(
        &self,
        variables: &HashMap<String, VariableEnum>,
        filename: &str,
    ) -> Result<(), std::io::Error> {
        let mut file = File::create(filename)?;
        writeln!(file, "# Variables - {} total", variables.len())?;
        writeln!(file, "# Format: variable_name: [values...]")?;

        let mut sorted_vars: Vec<_> = variables.keys().collect();
        sorted_vars.sort();

        for var_name in sorted_vars {
            let var_vector = variables[var_name].to_vector();
            write!(file, "{}: [", var_name)?;
            for (i, &value) in var_vector.iter().enumerate() {
                write!(file, "{:.12}", value)?;
                if i < var_vector.len() - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factors::{BetweenFactorSE2, BetweenFactorSE3, PriorFactor};
    use crate::core::loss_functions::HuberLoss;
    use crate::manifold::se3::SE3;
    use nalgebra::{Quaternion, Vector3, dvector};
    use std::collections::HashMap;

    /// Create a test SE2 dataset with 10 vertices in a loop
    fn create_se2_test_problem() -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();

        // Create 10 SE2 poses in a rough circle pattern
        let poses = vec![
            (0.0, 0.0, 0.0),    // x0: origin
            (1.0, 0.0, 0.1),    // x1: move right
            (1.5, 1.0, 0.5),    // x2: move up-right
            (1.0, 2.0, 1.0),    // x3: move up
            (0.0, 2.5, 1.5),    // x4: move up-left
            (-1.0, 2.0, 2.0),   // x5: move left
            (-1.5, 1.0, 2.5),   // x6: move down-left
            (-1.0, 0.0, 3.0),   // x7: move down
            (-0.5, -0.5, -2.8), // x8: move down-right
            (0.5, -0.5, -2.3),  // x9: back towards origin
        ];

        // Add vertices using [x, y, theta] ordering
        for (i, (x, y, theta)) in poses.iter().enumerate() {
            let var_name = format!("x{}", i);
            let se2_data = dvector![*x, *y, *theta];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }

        // Add chain of between factors
        for i in 0..9 {
            let from_pose = poses[i];
            let to_pose = poses[i + 1];

            // Compute relative transformation
            let dx = to_pose.0 - from_pose.0;
            let dy = to_pose.1 - from_pose.1;
            let dtheta = to_pose.2 - from_pose.2;

            let between_factor = BetweenFactorSE2::new(dx, dy, dtheta);
            problem.add_residual_block(
                &[&format!("x{}", i), &format!("x{}", i + 1)],
                Box::new(between_factor),
                Some(Box::new(HuberLoss::new(1.0).unwrap())),
            );
        }

        // Add loop closure from x9 back to x0
        let dx = poses[0].0 - poses[9].0;
        let dy = poses[0].1 - poses[9].1;
        let dtheta = poses[0].2 - poses[9].2;

        let loop_closure = BetweenFactorSE2::new(dx, dy, dtheta);
        problem.add_residual_block(
            &["x9", "x0"],
            Box::new(loop_closure),
            Some(Box::new(HuberLoss::new(1.0).unwrap())),
        );

        // Add prior factor for x0
        let prior_factor = PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        };
        problem.add_residual_block(&["x0"], Box::new(prior_factor), None);

        (problem, initial_values)
    }

    /// Create a test SE3 dataset with 8 vertices in a 3D pattern
    fn create_se3_test_problem() -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();

        // Create 8 SE3 poses in a rough 3D cube pattern
        let poses = vec![
            // Bottom face of cube
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),   // x0: origin
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.995), // x1: +X
            (1.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.98),  // x2: +X+Y
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.955), // x3: +Y
            // Top face of cube
            (0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.995), // x4: +Z
            (1.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.99),  // x5: +X+Z
            (1.0, 1.0, 1.0, 0.1, 0.0, 0.2, 0.975), // x6: +X+Y+Z
            (0.0, 1.0, 1.0, 0.1, 0.0, 0.3, 0.95),  // x7: +Y+Z
        ];

        // Add vertices using [tx, ty, tz, qw, qx, qy, qz] ordering
        for (i, (tx, ty, tz, qx, qy, qz, qw)) in poses.iter().enumerate() {
            let var_name = format!("x{}", i);
            let se3_data = dvector![*tx, *ty, *tz, *qw, *qx, *qy, *qz];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }

        // Add between factors connecting the cube edges
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // Vertical edges
        ];

        for (from_idx, to_idx) in edges {
            let from_pose = poses[from_idx];
            let to_pose = poses[to_idx];

            // Create a simple relative transformation (simplified for testing)
            let relative_se3 = SE3::from_translation_quaternion(
                Vector3::new(
                    to_pose.0 - from_pose.0, // dx
                    to_pose.1 - from_pose.1, // dy
                    to_pose.2 - from_pose.2, // dz
                ),
                Quaternion::new(1.0, 0.0, 0.0, 0.0), // identity quaternion
            );

            let between_factor = BetweenFactorSE3::new(relative_se3);
            problem.add_residual_block(
                &[&format!("x{}", from_idx), &format!("x{}", to_idx)],
                Box::new(between_factor),
                Some(Box::new(HuberLoss::new(1.0).unwrap())),
            );
        }

        // Add prior factor for x0
        let prior_factor = PriorFactor {
            data: dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        };
        problem.add_residual_block(&["x0"], Box::new(prior_factor), None);

        (problem, initial_values)
    }

    #[test]
    fn test_problem_construction_se2() {
        let (problem, initial_values) = create_se2_test_problem();

        // Test basic problem properties
        assert_eq!(problem.num_residual_blocks(), 11); // 9 between + 1 loop closure + 1 prior
        assert_eq!(problem.total_residual_dimension, 33); // 11 * 3
        assert_eq!(initial_values.len(), 10);

        println!("SE2 Problem construction test passed");
        println!("Residual blocks: {}", problem.num_residual_blocks());
        println!("Total residual dim: {}", problem.total_residual_dimension);
        println!("Variables: {}", initial_values.len());
    }

    #[test]
    fn test_problem_construction_se3() {
        let (problem, initial_values) = create_se3_test_problem();

        // Test basic problem properties
        assert_eq!(problem.num_residual_blocks(), 13); // 12 between + 1 prior
        assert_eq!(problem.total_residual_dimension, 79); // 12 * 6 + 1 * 7 (SE3 between factors are 6-dim, prior factor is 7-dim)
        assert_eq!(initial_values.len(), 8);

        println!("SE3 Problem construction test passed");
        println!("Residual blocks: {}", problem.num_residual_blocks());
        println!("Total residual dim: {}", problem.total_residual_dimension);
        println!("Variables: {}", initial_values.len());
    }

    #[test]
    fn test_variable_initialization_se2() {
        let (problem, initial_values) = create_se2_test_problem();

        // Test variable initialization
        let variables = problem.initialize_variables(&initial_values);
        assert_eq!(variables.len(), 10);

        // Test variable sizes
        for (name, var) in &variables {
            assert_eq!(
                var.get_size(),
                3,
                "SE2 variable {} should have size 3",
                name
            );
        }

        // Test conversion to DVector
        for (name, var) in &variables {
            let vec = var.to_vector();
            assert_eq!(
                vec.len(),
                3,
                "SE2 variable {} vector should have length 3",
                name
            );
        }

        println!("SE2 Variable initialization test passed");
        println!("Variables created: {}", variables.len());
    }

    #[test]
    fn test_variable_initialization_se3() {
        let (problem, initial_values) = create_se3_test_problem();

        // Test variable initialization
        let variables = problem.initialize_variables(&initial_values);
        assert_eq!(variables.len(), 8);

        // Test variable sizes
        for (name, var) in &variables {
            assert_eq!(
                var.get_size(),
                6,
                "SE3 variable {} should have size 6 (DOF)",
                name
            );
        }

        // Test conversion to DVector
        for (name, var) in &variables {
            let vec = var.to_vector();
            assert_eq!(
                vec.len(),
                7,
                "SE3 variable {} vector should have length 7",
                name
            );
        }

        println!("SE3 Variable initialization test passed");
        println!("Variables created: {}", variables.len());
    }

    #[test]
    fn test_column_mapping_se2() {
        let (problem, initial_values) = create_se2_test_problem();
        let variables = problem.initialize_variables(&initial_values);

        // Create column mapping for variables
        let mut variable_index_sparce_matrix = HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<_> = variables.keys().collect();
        sorted_vars.sort(); // Ensure consistent ordering

        for var_name in sorted_vars {
            variable_index_sparce_matrix.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Test total degrees of freedom
        let total_dof: usize = variables.values().map(|v| v.get_size()).sum();
        assert_eq!(total_dof, 30); // 10 variables * 3 DOF each
        assert_eq!(col_offset, 30);

        // Test each variable has correct column mapping
        for (var_name, &col_idx) in &variable_index_sparce_matrix {
            assert!(
                col_idx < total_dof,
                "Column index {} for {} should be < {}",
                col_idx,
                var_name,
                total_dof
            );
        }

        println!("SE2 Column mapping test passed");
        println!("Total DOF: {}", total_dof);
        println!("Variable mappings: {}", variable_index_sparce_matrix.len());
    }

    #[test]
    fn test_symbolic_structure_se2() {
        let (problem, initial_values) = create_se2_test_problem();
        let variables = problem.initialize_variables(&initial_values);

        // Create column mapping
        let mut variable_index_sparce_matrix = HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<_> = variables.keys().collect();
        sorted_vars.sort();

        for var_name in sorted_vars {
            variable_index_sparce_matrix.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Build symbolic structure
        let symbolic_structure = problem
            .build_symbolic_structure(&variables, &variable_index_sparce_matrix, col_offset)
            .unwrap();

        // Test symbolic structure dimensions
        assert_eq!(
            symbolic_structure.pattern.nrows(),
            problem.total_residual_dimension
        );
        assert_eq!(symbolic_structure.pattern.ncols(), 30); // total DOF

        println!("SE2 Symbolic structure test passed");
        println!(
            "Symbolic matrix: {} x {}",
            symbolic_structure.pattern.nrows(),
            symbolic_structure.pattern.ncols()
        );
    }

    #[test]
    fn test_residual_jacobian_computation_se2() {
        let (problem, initial_values) = create_se2_test_problem();
        let variables = problem.initialize_variables(&initial_values);

        // Create column mapping
        let mut variable_index_sparce_matrix = HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<_> = variables.keys().collect();
        sorted_vars.sort();

        for var_name in sorted_vars {
            variable_index_sparce_matrix.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Test sparse computation
        let symbolic_structure = problem
            .build_symbolic_structure(&variables, &variable_index_sparce_matrix, col_offset)
            .unwrap();
        let (residual_sparse, jacobian_sparse) = problem
            .compute_residual_and_jacobian_sparse(
                &variables,
                &variable_index_sparce_matrix,
                &symbolic_structure,
            )
            .unwrap();

        // Test sparse dimensions
        assert_eq!(residual_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.ncols(), 30);

        println!("SE2 Residual/Jacobian computation test passed");
        println!("Residual dimensions: {}", residual_sparse.nrows());
        println!(
            "Jacobian dimensions: {} x {}",
            jacobian_sparse.nrows(),
            jacobian_sparse.ncols()
        );
    }

    #[test]
    fn test_residual_jacobian_computation_se3() {
        let (problem, initial_values) = create_se3_test_problem();
        let variables = problem.initialize_variables(&initial_values);

        // Create column mapping
        let mut variable_index_sparce_matrix = HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<_> = variables.keys().collect();
        sorted_vars.sort();

        for var_name in sorted_vars {
            variable_index_sparce_matrix.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Test sparse computation
        let symbolic_structure = problem
            .build_symbolic_structure(&variables, &variable_index_sparce_matrix, col_offset)
            .unwrap();
        let (residual_sparse, jacobian_sparse) = problem
            .compute_residual_and_jacobian_sparse(
                &variables,
                &variable_index_sparce_matrix,
                &symbolic_structure,
            )
            .unwrap();

        // Test sparse dimensions match
        assert_eq!(residual_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.ncols(), 48); // 8 variables * 6 DOF each

        println!("SE3 Residual/Jacobian computation test passed");
        println!("Residual dimensions: {}", residual_sparse.nrows());
        println!(
            "Jacobian dimensions: {} x {}",
            jacobian_sparse.nrows(),
            jacobian_sparse.ncols()
        );
    }

    #[test]
    fn test_residual_block_operations() {
        let mut problem = Problem::new();

        // Test adding residual blocks
        let block_id1 = problem.add_residual_block(
            &["x0", "x1"],
            Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1)),
            Some(Box::new(HuberLoss::new(1.0).unwrap())),
        );

        let block_id2 = problem.add_residual_block(
            &["x0"],
            Box::new(PriorFactor {
                data: dvector![0.0, 0.0, 0.0],
            }),
            None,
        );

        assert_eq!(problem.num_residual_blocks(), 2);
        assert_eq!(problem.total_residual_dimension, 6); // 3 + 3
        assert_eq!(block_id1, 0);
        assert_eq!(block_id2, 1);

        // Test removing residual blocks
        let removed_block = problem.remove_residual_block(block_id1);
        assert!(removed_block.is_some());
        assert_eq!(problem.num_residual_blocks(), 1);
        assert_eq!(problem.total_residual_dimension, 3); // Only prior factor remains

        // Test removing non-existent block
        let non_existent = problem.remove_residual_block(999);
        assert!(non_existent.is_none());

        println!("Residual block operations test passed");
        println!("Block operations working correctly");
    }

    #[test]
    fn test_variable_constraints() {
        let mut problem = Problem::new();

        // Test fixing variables
        problem.fix_variable("x0", 0);
        problem.fix_variable("x0", 1);
        problem.fix_variable("x1", 2);

        assert!(problem.fixed_variable_indexes.contains_key("x0"));
        assert!(problem.fixed_variable_indexes.contains_key("x1"));
        assert_eq!(problem.fixed_variable_indexes["x0"].len(), 2);
        assert_eq!(problem.fixed_variable_indexes["x1"].len(), 1);

        // Test unfixing variables
        problem.unfix_variable("x0");
        assert!(!problem.fixed_variable_indexes.contains_key("x0"));
        assert!(problem.fixed_variable_indexes.contains_key("x1"));

        // Test variable bounds
        problem.set_variable_bounds("x2", 0, -1.0, 1.0);
        problem.set_variable_bounds("x2", 1, -2.0, 2.0);
        problem.set_variable_bounds("x3", 0, 0.0, 5.0);

        assert!(problem.variable_bounds.contains_key("x2"));
        assert!(problem.variable_bounds.contains_key("x3"));
        assert_eq!(problem.variable_bounds["x2"].len(), 2);
        assert_eq!(problem.variable_bounds["x3"].len(), 1);

        // Test removing bounds
        problem.remove_variable_bounds("x2");
        assert!(!problem.variable_bounds.contains_key("x2"));
        assert!(problem.variable_bounds.contains_key("x3"));

        println!("Variable constraints test passed");
        println!("Fix/unfix and bounds operations working correctly");
    }

    // Helper function for the known 5-pose test case
    #[allow(dead_code)]
    fn create_simple_5pose_se2_test() -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();

        // Use the exact same data from our successful simple debug test
        let poses = [
            ("x0", [0.000000, 0.000000, 0.000000]),
            ("x1", [-0.012958, 1.030390, 0.011350]),
            ("x2", [-0.026183, 2.043445, -0.060422]),
            ("x3", [-0.021350, 3.070548, -0.094779]),
            ("x4", [1.545440, 3.079976, 0.909609]),
        ];

        for (name, data) in &poses {
            initial_values.insert(
                name.to_string(),
                (ManifoldType::SE2, dvector![data[1], data[2], data[0]]),
            );
        }

        // Add the exact same between factors
        let between_factors = [
            ("x0", "x1", 1.030390, 0.011350, -0.012958),
            ("x1", "x2", 1.013900, -0.058639, -0.013225),
            ("x2", "x3", 1.027650, -0.007456, 0.004833),
            ("x3", "x4", -0.012016, 1.004360, 1.566790),
        ];

        for (from, to, dx, dy, dtheta) in &between_factors {
            problem.add_residual_block(
                &[from, to],
                Box::new(BetweenFactorSE2::new(*dx, *dy, *dtheta)),
                Some(Box::new(HuberLoss::new(1.0).unwrap())),
            );
        }

        // Add prior factor
        problem.add_residual_block(
            &["x0"],
            Box::new(PriorFactor {
                data: dvector![0.000000, 0.000000, 0.000000],
            }),
            None,
        );

        (problem, initial_values)
    }
}
