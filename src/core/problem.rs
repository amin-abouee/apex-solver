//! Optimization problem definition and sparse Jacobian computation.
//!
//! The `Problem` struct is the central component that defines a factor graph optimization problem.
//! It manages residual blocks (constraints), variables, and the construction of sparse Jacobian
//! matrices for efficient nonlinear least squares optimization.
//!
//! # Factor Graph Representation
//!
//! The optimization problem is represented as a bipartite factor graph:
//!
//! ```text
//! Variables:  x0 --- x1 --- x2 --- x3
//!              |      |      |      |
//! Factors:    f0     f1     f2     f3 (constraints/measurements)
//! ```
//!
//! Each factor connects one or more variables and contributes a residual (error) term to the
//! overall cost function:
//!
//! ```text
//! minimize Σ_i ρ(||r_i(x)||²)
//! ```
//!
//! where `r_i(x)` is the residual for factor i, and ρ is an optional robust loss function.
//!
//! # Key Responsibilities
//!
//! 1. **Residual Block Management**: Add/remove factors and track their structure
//! 2. **Variable Management**: Initialize variables with manifold types and constraints
//! 3. **Sparsity Pattern**: Build symbolic structure for efficient sparse linear algebra
//! 4. **Linearization**: Compute residuals and Jacobians in parallel
//! 5. **Covariance**: Extract per-variable uncertainty estimates after optimization
//!
//! # Sparse Jacobian Structure
//!
//! The Jacobian matrix `J = ∂r/∂x` is sparse because each factor only depends on a small
//! subset of variables. For example, a between factor connecting x0 and x1 contributes
//! a 3×6 block (SE2) or 6×12 block (SE3) to the Jacobian, leaving the rest as zeros.
//!
//! The Problem pre-computes the sparsity pattern once, then efficiently fills in the
//! numerical values during each iteration.
//!
//! # Mixed Manifold Support
//!
//! The Problem supports mixed manifold types in a single optimization problem via
//! [`VariableEnum`]. This allows:
//! - SE2 and SE3 poses in the same graph
//! - SO3 rotations with R³ landmarks
//! - Any combination of supported manifolds
//!
//! # Example: Building a Problem
//!
//! ```
//! use apex_solver::core::problem::Problem;
//! use apex_solver::factors::{BetweenFactorSE2, PriorFactor};
//! use apex_solver::core::loss_functions::HuberLoss;
//! use apex_solver::manifold::ManifoldType;
//! use nalgebra::{DVector, dvector};
//! use std::collections::HashMap;
//!
//! let mut problem = Problem::new();
//!
//! // Add prior factor to anchor the first pose
//! let prior = Box::new(PriorFactor {
//!     data: dvector![0.0, 0.0, 0.0],
//! });
//! problem.add_residual_block(&["x0"], prior, None);
//!
//! // Add between factor with robust loss
//! let between = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));
//! let loss: Option<Box<dyn apex_solver::core::loss_functions::LossFunction + Send>> =
//!     Some(Box::new(HuberLoss::new(1.0).unwrap()));
//! problem.add_residual_block(&["x0", "x1"], between, loss);
//!
//! // Initialize variables
//! let mut initial_values = HashMap::new();
//! initial_values.insert("x0".to_string(), (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]));
//! initial_values.insert("x1".to_string(), (ManifoldType::SE2, dvector![0.9, 0.1, 0.12]));
//!
//! let variables = problem.initialize_variables(&initial_values);
//!
//! println!("Problem has {} residual blocks", problem.num_residual_blocks());
//! println!("Total residual dimension: {}", problem.total_residual_dimension);
//! ```

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Error, Write},
    sync::{Arc, Mutex},
};

use faer::{
    Col, Mat, MatRef,
    sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat},
};
use nalgebra::DVector;
use rayon::prelude::*;
use tracing::warn;

use crate::{
    core::{
        CoreError, corrector::Corrector, loss_functions::LossFunction,
        residual_block::ResidualBlock, variable::Variable,
    },
    error::{ApexError, ApexResult},
    factors::Factor,
    linalg::{SparseLinearSolver, extract_variable_covariances},
    manifold::{ManifoldType, rn, se2, se3, so2, so3},
};

/// Symbolic structure for sparse matrix operations.
///
/// Contains the sparsity pattern (which entries are non-zero) and an ordering
/// for efficient numerical computation. This is computed once at the beginning
/// and reused throughout optimization.
///
/// # Fields
///
/// - `pattern`: The symbolic sparse column matrix structure (row/col indices of non-zeros)
/// - `order`: A fill-reducing ordering/permutation for numerical stability
pub struct SymbolicStructure {
    pub pattern: SymbolicSparseColMat<usize>,
    pub order: Argsort<usize>,
}

/// Enum to handle mixed manifold variable types
#[derive(Clone)]
pub enum VariableEnum {
    Rn(Variable<rn::Rn>),
    SE2(Variable<se2::SE2>),
    SE3(Variable<se3::SE3>),
    SO2(Variable<so2::SO2>),
    SO3(Variable<so3::SO3>),
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
    pub fn apply_tangent_step(&mut self, step_slice: MatRef<f64>) {
        match self {
            VariableEnum::SE3(var) => {
                // SE3 has 6 DOF in tangent space
                let mut step_data: Vec<f64> = (0..6).map(|i| step_slice[(i, 0)]).collect();

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 6 {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let step_dvector = DVector::from_vec(step_data);
                let tangent = se3::SE3Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SE2(var) => {
                // SE2 has 3 DOF in tangent space
                let mut step_data: Vec<f64> = (0..3).map(|i| step_slice[(i, 0)]).collect();

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 3 {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let step_dvector = DVector::from_vec(step_data);
                let tangent = se2::SE2Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SO3(var) => {
                // SO3 has 3 DOF in tangent space
                let mut step_data: Vec<f64> = (0..3).map(|i| step_slice[(i, 0)]).collect();

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 3 {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let step_dvector = DVector::from_vec(step_data);
                let tangent = so3::SO3Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SO2(var) => {
                // SO2 has 1 DOF in tangent space
                let mut step_data = step_slice[(0, 0)];

                // Enforce fixed indices: zero out step if index 0 is fixed
                if var.fixed_indices.contains(&0) {
                    step_data = 0.0;
                }

                let step_dvector = DVector::from_vec(vec![step_data]);
                let tangent = so2::SO2Tangent::from(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::Rn(var) => {
                // Rn has dynamic size
                let size = var.get_size();
                let mut step_data: Vec<f64> = (0..size).map(|i| step_slice[(i, 0)]).collect();

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < size {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let step_dvector = DVector::from_vec(step_data);
                let tangent = rn::RnTangent::new(step_dvector);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
        }
    }

    /// Get the covariance matrix for this variable (if computed).
    ///
    /// Returns `None` if covariance has not been computed.
    ///
    /// # Returns
    /// Reference to the covariance matrix in tangent space
    pub fn get_covariance(&self) -> Option<&Mat<f64>> {
        match self {
            VariableEnum::Rn(var) => var.get_covariance(),
            VariableEnum::SE2(var) => var.get_covariance(),
            VariableEnum::SE3(var) => var.get_covariance(),
            VariableEnum::SO2(var) => var.get_covariance(),
            VariableEnum::SO3(var) => var.get_covariance(),
        }
    }

    /// Set the covariance matrix for this variable.
    ///
    /// The covariance matrix should be square with dimension equal to
    /// the tangent space dimension of this variable.
    ///
    /// # Arguments
    /// * `cov` - Covariance matrix in tangent space
    pub fn set_covariance(&mut self, cov: Mat<f64>) {
        match self {
            VariableEnum::Rn(var) => var.set_covariance(cov),
            VariableEnum::SE2(var) => var.set_covariance(cov),
            VariableEnum::SE3(var) => var.set_covariance(cov),
            VariableEnum::SO2(var) => var.set_covariance(cov),
            VariableEnum::SO3(var) => var.set_covariance(cov),
        }
    }

    /// Clear the covariance matrix for this variable.
    pub fn clear_covariance(&mut self) {
        match self {
            VariableEnum::Rn(var) => var.clear_covariance(),
            VariableEnum::SE2(var) => var.clear_covariance(),
            VariableEnum::SE3(var) => var.clear_covariance(),
            VariableEnum::SO2(var) => var.clear_covariance(),
            VariableEnum::SO3(var) => var.clear_covariance(),
        }
    }

    /// Get the bounds for this variable.
    ///
    /// Returns a reference to the bounds map where keys are indices and values are (lower, upper) pairs.
    pub fn get_bounds(&self) -> &HashMap<usize, (f64, f64)> {
        match self {
            VariableEnum::Rn(var) => &var.bounds,
            VariableEnum::SE2(var) => &var.bounds,
            VariableEnum::SE3(var) => &var.bounds,
            VariableEnum::SO2(var) => &var.bounds,
            VariableEnum::SO3(var) => &var.bounds,
        }
    }

    /// Get the fixed indices for this variable.
    ///
    /// Returns a reference to the set of indices that should remain fixed during optimization.
    pub fn get_fixed_indices(&self) -> &HashSet<usize> {
        match self {
            VariableEnum::Rn(var) => &var.fixed_indices,
            VariableEnum::SE2(var) => &var.fixed_indices,
            VariableEnum::SE3(var) => &var.fixed_indices,
            VariableEnum::SO2(var) => &var.fixed_indices,
            VariableEnum::SO3(var) => &var.fixed_indices,
        }
    }

    /// Set the value of this variable from a vector representation.
    ///
    /// This is used to update the variable after applying constraints (bounds and fixed indices).
    pub fn set_from_vector(&mut self, vec: &DVector<f64>) {
        match self {
            VariableEnum::Rn(var) => {
                var.set_value(rn::Rn::new(vec.clone()));
            }
            VariableEnum::SE2(var) => {
                let new_se2: se2::SE2 = vec.clone().into();
                var.set_value(new_se2);
            }
            VariableEnum::SE3(var) => {
                let new_se3: se3::SE3 = vec.clone().into();
                var.set_value(new_se3);
            }
            VariableEnum::SO2(var) => {
                let new_so2: so2::SO2 = vec.clone().into();
                var.set_value(new_so2);
            }
            VariableEnum::SO3(var) => {
                let new_so3: so3::SO3 = vec.clone().into();
                var.set_value(new_so3);
            }
        }
    }
}

/// The optimization problem definition for factor graph optimization.
///
/// Manages residual blocks (factors/constraints), variables, and the sparse Jacobian structure.
/// Supports mixed manifold types (SE2, SE3, SO2, SO3, Rn) in a single problem and provides
/// efficient parallel residual/Jacobian computation.
///
/// # Architecture
///
/// The Problem acts as a container and coordinator:
/// - Stores all residual blocks (factors with optional loss functions)
/// - Tracks the global structure (which variables connect to which factors)
/// - Builds and maintains the sparse Jacobian pattern
/// - Provides parallel residual/Jacobian evaluation using rayon
/// - Manages variable constraints (fixed indices, bounds)
///
/// # Workflow
///
/// 1. **Construction**: Create a new Problem with `Problem::new()`
/// 2. **Add Factors**: Use `add_residual_block()` to add constraints
/// 3. **Initialize Variables**: Use `initialize_variables()` with initial values
/// 4. **Build Sparsity**: Use `build_symbolic_structure()` once before optimization
/// 5. **Linearize**: Call `compute_residual_and_jacobian_sparse()` each iteration
/// 6. **Extract Covariance**: Use `compute_and_set_covariances()` after convergence
///
/// # Example
///
/// ```
/// use apex_solver::core::problem::Problem;
/// use apex_solver::factors::BetweenFactorSE2;
/// use apex_solver::manifold::ManifoldType;
/// use nalgebra::dvector;
/// use std::collections::HashMap;
///
/// let mut problem = Problem::new();
///
/// // Add a between factor
/// let factor = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));
/// problem.add_residual_block(&["x0", "x1"], factor, None);
///
/// // Initialize variables
/// let mut initial = HashMap::new();
/// initial.insert("x0".to_string(), (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]));
/// initial.insert("x1".to_string(), (ManifoldType::SE2, dvector![1.0, 0.0, 0.1]));
///
/// let variables = problem.initialize_variables(&initial);
/// assert_eq!(variables.len(), 2);
/// ```
pub struct Problem {
    /// Total dimension of the stacked residual vector (sum of all residual block dimensions)
    pub total_residual_dimension: usize,

    /// Counter for assigning unique IDs to residual blocks
    residual_id_count: usize,

    /// Map from residual block ID to ResidualBlock instance
    residual_blocks: HashMap<usize, ResidualBlock>,

    /// Variables with fixed indices (e.g., fix first pose's x,y coordinates)
    /// Maps variable name -> set of indices to fix
    pub fixed_variable_indexes: HashMap<String, HashSet<usize>>,

    /// Variable bounds (box constraints on individual DOF)
    /// Maps variable name -> (index -> (lower_bound, upper_bound))
    pub variable_bounds: HashMap<String, HashMap<usize, (f64, f64)>>,
}
impl Default for Problem {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem {
    /// Create a new empty optimization problem.
    ///
    /// # Returns
    ///
    /// A new `Problem` with no residual blocks or variables
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::problem::Problem;
    ///
    /// let problem = Problem::new();
    /// assert_eq!(problem.num_residual_blocks(), 0);
    /// assert_eq!(problem.total_residual_dimension, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            total_residual_dimension: 0,
            residual_id_count: 0,
            residual_blocks: HashMap::new(),
            fixed_variable_indexes: HashMap::new(),
            variable_bounds: HashMap::new(),
        }
    }

    /// Add a residual block (factor with optional loss function) to the problem.
    ///
    /// This is the primary method for building the factor graph. Each call adds one constraint
    /// connecting one or more variables.
    ///
    /// # Arguments
    ///
    /// * `variable_key_size_list` - Names of the variables this factor connects (order matters)
    /// * `factor` - The factor implementation that computes residuals and Jacobians
    /// * `loss_func` - Optional robust loss function for outlier rejection
    ///
    /// # Returns
    ///
    /// The unique ID assigned to this residual block
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::problem::Problem;
    /// use apex_solver::factors::{BetweenFactorSE2, PriorFactor};
    /// use apex_solver::core::loss_functions::HuberLoss;
    /// use nalgebra::dvector;
    ///
    /// let mut problem = Problem::new();
    ///
    /// // Add prior factor (unary constraint)
    /// let prior = Box::new(PriorFactor { data: dvector![0.0, 0.0, 0.0] });
    /// let id1 = problem.add_residual_block(&["x0"], prior, None);
    ///
    /// // Add between factor with robust loss (binary constraint)
    /// let between = Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1));
    /// let loss: Option<Box<dyn apex_solver::core::loss_functions::LossFunction + Send>> =
    ///     Some(Box::new(HuberLoss::new(1.0).unwrap()));
    /// let id2 = problem.add_residual_block(&["x0", "x1"], between, loss);
    ///
    /// assert_eq!(id1, 0);
    /// assert_eq!(id2, 1);
    /// assert_eq!(problem.num_residual_blocks(), 2);
    /// ```
    pub fn add_residual_block(
        &mut self,
        variable_key_size_list: &[&str],
        factor: Box<dyn Factor + Send>,
        loss_func: Option<Box<dyn LossFunction + Send>>,
    ) -> usize {
        let new_residual_dimension = factor.get_dimension();
        self.residual_blocks.insert(
            self.residual_id_count,
            ResidualBlock::new(
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

    pub fn remove_residual_block(&mut self, block_id: usize) -> Option<ResidualBlock> {
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
            warn!("lower bound is larger than upper bound");
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

    /// Initialize variables from initial values with manifold types.
    ///
    /// Converts raw initial values into typed `Variable<M>` instances wrapped in `VariableEnum`.
    /// This method also applies any fixed indices or bounds that were set via `fix_variable()`
    /// or `set_variable_bounds()`.
    ///
    /// # Arguments
    ///
    /// * `initial_values` - Map from variable name to (manifold type, initial value vector)
    ///
    /// # Returns
    ///
    /// Map from variable name to `VariableEnum` (typed variables ready for optimization)
    ///
    /// # Manifold Formats
    ///
    /// - **SE2**: `[x, y, theta]` (3 elements)
    /// - **SE3**: `[tx, ty, tz, qw, qx, qy, qz]` (7 elements)
    /// - **SO2**: `[theta]` (1 element)
    /// - **SO3**: `[qw, qx, qy, qz]` (4 elements)
    /// - **Rn**: `[x1, x2, ..., xn]` (n elements)
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::problem::Problem;
    /// use apex_solver::manifold::ManifoldType;
    /// use nalgebra::dvector;
    /// use std::collections::HashMap;
    ///
    /// let problem = Problem::new();
    ///
    /// let mut initial = HashMap::new();
    /// initial.insert("pose0".to_string(), (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]));
    /// initial.insert("pose1".to_string(), (ManifoldType::SE2, dvector![1.0, 0.0, 0.1]));
    /// initial.insert("landmark".to_string(), (ManifoldType::RN, dvector![5.0, 3.0]));
    ///
    /// let variables = problem.initialize_variables(&initial);
    /// assert_eq!(variables.len(), 3);
    /// ```
    pub fn initialize_variables(
        &self,
        initial_values: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> HashMap<String, VariableEnum> {
        let variables: HashMap<String, VariableEnum> = initial_values
            .iter()
            .map(|(k, v)| {
                let variable_enum = match v.0 {
                    ManifoldType::SO2 => {
                        let mut var = Variable::new(so2::SO2::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SO2(var)
                    }
                    ManifoldType::SO3 => {
                        let mut var = Variable::new(so3::SO3::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SO3(var)
                    }
                    ManifoldType::SE2 => {
                        let mut var = Variable::new(se2::SE2::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SE2(var)
                    }
                    ManifoldType::SE3 => {
                        let mut var = Variable::new(se3::SE3::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SE3(var)
                    }
                    ManifoldType::RN => {
                        let mut var = Variable::new(rn::Rn::from(v.1.clone()));
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
    ) -> ApexResult<SymbolicStructure> {
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
        .map_err(|e| {
            CoreError::SymbolicStructure(
                "Failed to build symbolic sparse matrix structure".to_string(),
            )
            .log_with_source(e)
        })?;

        // Return the symbolic structure that will be filled with numerical values later
        Ok(SymbolicStructure { pattern, order })
    }

    /// Compute only the residual vector for the current variable values.
    ///
    /// This is an optimized version that skips Jacobian computation when only the cost
    /// function value is needed (e.g., during initialization or step evaluation).
    ///
    /// # Arguments
    ///
    /// * `variables` - Current variable values (from `initialize_variables()` or updated)
    ///
    /// # Returns
    ///
    /// Residual vector as N×1 column matrix (N = total residual dimension)
    ///
    /// # Performance
    ///
    /// Approximately **2x faster** than `compute_residual_and_jacobian_sparse()` since it:
    /// - Skips Jacobian computation for each residual block
    /// - Avoids Jacobian matrix assembly and storage
    /// - Only parallelizes residual evaluation
    ///
    /// # When to Use
    ///
    /// - **Initial cost computation**: When setting up optimization state
    /// - **Step evaluation**: When computing new cost after applying parameter updates
    /// - **Cost-only queries**: When you don't need gradients
    ///
    /// Use `compute_residual_and_jacobian_sparse()` when you need both residual and Jacobian
    /// (e.g., in the main optimization iteration loop for linearization).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Initial cost evaluation (no Jacobian needed)
    /// let residual = problem.compute_residual_sparse(&variables)?;
    /// let initial_cost = residual.norm_l2() * residual.norm_l2();
    /// ```
    pub fn compute_residual_sparse(
        &self,
        variables: &HashMap<String, VariableEnum>,
    ) -> ApexResult<Mat<f64>> {
        let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(self.total_residual_dimension)));

        // Compute residuals in parallel (no Jacobian needed)
        let result: Result<Vec<()>, ApexError> = self
            .residual_blocks
            .par_iter()
            .map(|(_, residual_block)| {
                self.compute_residual_block(residual_block, variables, &total_residual)
            })
            .collect();

        result?;

        let total_residual = Arc::try_unwrap(total_residual)
            .map_err(|_| {
                CoreError::ParallelComputation(
                    "Failed to unwrap Arc for total residual".to_string(),
                )
                .log()
            })?
            .into_inner()
            .map_err(|e| {
                CoreError::ParallelComputation(
                    "Failed to extract mutex inner value for total residual".to_string(),
                )
                .log_with_source(e)
            })?;

        // Convert faer Col to Mat (column vector as n×1 matrix)
        let residual_faer = total_residual.as_ref().as_mat().to_owned();
        Ok(residual_faer)
    }

    /// Compute residual vector and sparse Jacobian matrix for the current variable values.
    ///
    /// This is the core linearization method called during each optimization iteration. It:
    /// 1. Evaluates all residual blocks in parallel using rayon
    /// 2. Assembles the full residual vector
    /// 3. Constructs the sparse Jacobian matrix using the precomputed symbolic structure
    ///
    /// # Arguments
    ///
    /// * `variables` - Current variable values (from `initialize_variables()` or updated)
    /// * `variable_index_sparce_matrix` - Map from variable name to starting column in Jacobian
    /// * `symbolic_structure` - Precomputed sparsity pattern (from `build_symbolic_structure()`)
    ///
    /// # Returns
    ///
    /// Tuple `(residual, jacobian)` where:
    /// - `residual`: N×1 column matrix (total residual dimension)
    /// - `jacobian`: N×M sparse matrix (N = residual dim, M = total DOF)
    ///
    /// # Performance
    ///
    /// This method is highly optimized:
    /// - **Parallel evaluation**: Each residual block is evaluated independently using rayon
    /// - **Sparse storage**: Only non-zero Jacobian entries are stored and computed
    /// - **Memory efficient**: Preallocated sparse structure avoids dynamic allocations
    ///
    /// Typically accounts for 40-60% of total optimization time (including sparse matrix ops).
    ///
    /// # When to Use
    ///
    /// Use this method in the main optimization loop when you need both residual and Jacobian
    /// for linearization. For cost-only evaluation, use `compute_residual_sparse()` instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Inside optimizer loop:
    /// let (residual, jacobian) = problem.compute_residual_and_jacobian_sparse(
    ///     &variables,
    ///     &variable_index_map,
    ///     &symbolic_structure,
    /// )?;
    ///
    /// // Use for linear system: J^T J dx = -J^T r
    /// ```
    pub fn compute_residual_and_jacobian_sparse(
        &self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_sparce_matrix: &HashMap<String, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> ApexResult<(Mat<f64>, SparseColMat<usize, f64>)> {
        let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(self.total_residual_dimension)));

        // OPTIMIZATION: Pre-allocate exact size to avoid double allocation and flattening
        // This eliminates the Vec<Vec<f64>> → Vec<f64> conversion overhead
        let total_nnz = symbolic_structure.pattern.compute_nnz();

        // Collect block results with pre-computed sizes
        let jacobian_blocks: Result<Vec<(usize, Vec<f64>)>, ApexError> = self
            .residual_blocks
            .par_iter()
            .map(|(_, residual_block)| {
                let values = self.compute_residual_and_jacobian_block(
                    residual_block,
                    variables,
                    variable_index_sparce_matrix,
                    &total_residual,
                )?;
                let size = values.len();
                Ok((size, values))
            })
            .collect();

        let jacobian_blocks = jacobian_blocks?;

        // Pre-allocate final vec with exact size
        let mut jacobian_values = Vec::with_capacity(total_nnz);
        for (_size, mut block_values) in jacobian_blocks {
            jacobian_values.append(&mut block_values);
        }

        let total_residual = Arc::try_unwrap(total_residual)
            .map_err(|_| {
                CoreError::ParallelComputation(
                    "Failed to unwrap Arc for total residual".to_string(),
                )
                .log()
            })?
            .into_inner()
            .map_err(|e| {
                CoreError::ParallelComputation(
                    "Failed to extract mutex inner value for total residual".to_string(),
                )
                .log_with_source(e)
            })?;

        // Convert faer Col to Mat (column vector as n×1 matrix)
        let residual_faer = total_residual.as_ref().as_mat().to_owned();
        let jacobian_sparse = SparseColMat::new_from_argsort(
            symbolic_structure.pattern.clone(),
            &symbolic_structure.order,
            jacobian_values.as_slice(),
        )
        .map_err(|e| {
            CoreError::SymbolicStructure(
                "Failed to create sparse Jacobian from argsort".to_string(),
            )
            .log_with_source(e)
        })?;

        Ok((residual_faer, jacobian_sparse))
    }

    /// Compute only the residual for a single residual block (no Jacobian).
    ///
    /// Helper method for `compute_residual_sparse()`.
    fn compute_residual_block(
        &self,
        residual_block: &ResidualBlock,
        variables: &HashMap<String, VariableEnum>,
        total_residual: &Arc<Mutex<Col<f64>>>,
    ) -> ApexResult<()> {
        let mut param_vectors: Vec<DVector<f64>> = Vec::new();

        for var_key in &residual_block.variable_key_list {
            if let Some(variable) = variables.get(var_key) {
                param_vectors.push(variable.to_vector());
            }
        }

        // Compute only residual (linearize still computes Jacobian internally,
        // but we don't extract/store it)
        let (mut res, _) = residual_block.factor.linearize(&param_vectors, false);

        // Apply loss function if present (critical for robust optimization)
        if let Some(loss_func) = &residual_block.loss_func {
            let squared_norm = res.dot(&res);
            let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
            corrector.correct_residuals(&mut res);
        }

        let mut total_residual = total_residual.lock().map_err(|e| {
            CoreError::ParallelComputation("Failed to acquire lock on total residual".to_string())
                .log_with_source(e)
        })?;

        // Copy residual values from nalgebra DVector to faer Col
        let start_idx = residual_block.residual_row_start_idx;
        let dim = residual_block.factor.get_dimension();
        let mut total_residual_mut = total_residual.as_mut();
        for i in 0..dim {
            total_residual_mut[start_idx + i] = res[i];
        }

        Ok(())
    }

    /// Compute residual and Jacobian for a single residual block.
    ///
    /// Helper method for `compute_residual_and_jacobian_sparse()`.
    fn compute_residual_and_jacobian_block(
        &self,
        residual_block: &ResidualBlock,
        variables: &HashMap<String, VariableEnum>,
        variable_index_sparce_matrix: &HashMap<String, usize>,
        total_residual: &Arc<Mutex<Col<f64>>>,
    ) -> ApexResult<Vec<f64>> {
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

        let (mut res, jac_opt) = residual_block.factor.linearize(&param_vectors, true);
        let mut jac = jac_opt.ok_or_else(|| {
            CoreError::FactorLinearization(
                "Factor returned None for Jacobian when compute_jacobian=true".to_string(),
            )
            .log()
        })?;

        // Apply loss function if present (critical for robust optimization)
        if let Some(loss_func) = &residual_block.loss_func {
            let squared_norm = res.dot(&res);
            let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
            corrector.correct_jacobian(&res, &mut jac);
            corrector.correct_residuals(&mut res);
        }

        // Update total residual
        {
            let mut total_residual = total_residual.lock().map_err(|e| {
                CoreError::ParallelComputation(
                    "Failed to acquire lock on total residual".to_string(),
                )
                .log_with_source(e)
            })?;

            // Copy residual values from nalgebra DVector to faer Col
            let start_idx = residual_block.residual_row_start_idx;
            let dim = residual_block.factor.get_dimension();
            let mut total_residual_mut = total_residual.as_mut();
            for i in 0..dim {
                total_residual_mut[start_idx + i] = res[i];
            }
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
                return Err(CoreError::Variable(format!(
                    "Missing key {} in variable-to-column-index mapping",
                    var_key
                ))
                .log()
                .into());
            }
        }

        Ok(local_jacobian_values)
    }

    /// Log residual vector to a text file
    pub fn log_residual_to_file(
        &self,
        residual: &nalgebra::DVector<f64>,
        filename: &str,
    ) -> Result<(), Error> {
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
    ) -> Result<(), Error> {
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
    ) -> Result<(), Error> {
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

    /// Compute per-variable covariances and set them in Variable objects
    ///
    /// This method computes the full covariance matrix by inverting the Hessian
    /// from the linear solver, then extracts per-variable covariance blocks and
    /// stores them in the corresponding Variable objects.
    ///
    /// # Arguments
    /// * `linear_solver` - Mutable reference to the linear solver containing the cached Hessian
    /// * `variables` - Mutable map of variables where covariances will be stored
    /// * `variable_index_map` - Map from variable names to their starting column indices
    ///
    /// # Returns
    /// `Some(HashMap)` containing per-variable covariance matrices if successful, `None` otherwise
    ///
    pub fn compute_and_set_covariances(
        &self,
        linear_solver: &mut Box<dyn SparseLinearSolver>,
        variables: &mut HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> Option<HashMap<String, Mat<f64>>> {
        // Compute the full covariance matrix (H^{-1}) using the linear solver
        linear_solver.compute_covariance_matrix()?;
        let full_cov = linear_solver.get_covariance_matrix()?.clone();

        // Extract per-variable covariance blocks from the full matrix
        let per_var_covariances =
            extract_variable_covariances(&full_cov, variables, variable_index_map);

        // Set covariances in Variable objects for easy access
        for (var_name, cov) in &per_var_covariances {
            if let Some(var) = variables.get_mut(var_name) {
                var.set_covariance(cov.clone());
            }
        }

        Some(per_var_covariances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::loss_functions::HuberLoss;
    use crate::factors::{BetweenFactorSE2, BetweenFactorSE3, PriorFactor};
    use crate::manifold::{ManifoldType, se3::SE3};
    use nalgebra::{Quaternion, Vector3, dvector};
    use std::collections::HashMap;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Create a test SE2 dataset with 10 vertices in a loop
    fn create_se2_test_problem() -> Result<
        (
            Problem,
            HashMap<String, (ManifoldType, nalgebra::DVector<f64>)>,
        ),
        Box<dyn std::error::Error>,
    > {
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
                Some(Box::new(HuberLoss::new(1.0)?)),
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
            Some(Box::new(HuberLoss::new(1.0)?)),
        );

        // Add prior factor for x0
        let prior_factor = PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        };
        problem.add_residual_block(&["x0"], Box::new(prior_factor), None);

        Ok((problem, initial_values))
    }

    /// Create a test SE3 dataset with 8 vertices in a 3D pattern
    fn create_se3_test_problem()
    -> Result<(Problem, HashMap<String, (ManifoldType, DVector<f64>)>), Box<dyn std::error::Error>>
    {
        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();

        // Create 8 SE3 poses in a rough 3D cube pattern
        let poses = [
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
                Some(Box::new(HuberLoss::new(1.0)?)),
            );
        }

        // Add prior factor for x0
        let prior_factor = PriorFactor {
            data: dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        };
        problem.add_residual_block(&["x0"], Box::new(prior_factor), None);

        Ok((problem, initial_values))
    }

    #[test]
    fn test_problem_construction_se2() -> TestResult {
        let (problem, initial_values) = create_se2_test_problem()?;

        // Test basic problem properties
        assert_eq!(problem.num_residual_blocks(), 11); // 9 between + 1 loop closure + 1 prior
        assert_eq!(problem.total_residual_dimension, 33); // 11 * 3
        assert_eq!(initial_values.len(), 10);

        Ok(())
    }

    #[test]
    fn test_problem_construction_se3() -> TestResult {
        let (problem, initial_values) = create_se3_test_problem()?;

        // Test basic problem properties
        assert_eq!(problem.num_residual_blocks(), 13); // 12 between + 1 prior
        assert_eq!(problem.total_residual_dimension, 79); // 12 * 6 + 1 * 7 (SE3 between factors are 6-dim, prior factor is 7-dim)
        assert_eq!(initial_values.len(), 8);

        Ok(())
    }

    #[test]
    fn test_variable_initialization_se2() -> TestResult {
        let (problem, initial_values) = create_se2_test_problem()?;

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

        Ok(())
    }

    #[test]
    fn test_variable_initialization_se3() -> TestResult {
        let (problem, initial_values) = create_se3_test_problem()?;

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

        Ok(())
    }

    #[test]
    fn test_column_mapping_se2() -> TestResult {
        let (problem, initial_values) = create_se2_test_problem()?;
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

        Ok(())
    }

    #[test]
    fn test_symbolic_structure_se2() -> TestResult {
        let (problem, initial_values) = create_se2_test_problem()?;
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
        let symbolic_structure = problem.build_symbolic_structure(
            &variables,
            &variable_index_sparce_matrix,
            col_offset,
        )?;

        // Test symbolic structure dimensions
        assert_eq!(
            symbolic_structure.pattern.nrows(),
            problem.total_residual_dimension
        );
        assert_eq!(symbolic_structure.pattern.ncols(), 30); // total DOF

        Ok(())
    }

    #[test]
    fn test_residual_jacobian_computation_se2() -> TestResult {
        let (problem, initial_values) = create_se2_test_problem()?;
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
        let symbolic_structure = problem.build_symbolic_structure(
            &variables,
            &variable_index_sparce_matrix,
            col_offset,
        )?;
        let (residual_sparse, jacobian_sparse) = problem.compute_residual_and_jacobian_sparse(
            &variables,
            &variable_index_sparce_matrix,
            &symbolic_structure,
        )?;

        // Test sparse dimensions
        assert_eq!(residual_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.ncols(), 30);

        Ok(())
    }

    #[test]
    fn test_residual_jacobian_computation_se3() -> TestResult {
        let (problem, initial_values) = create_se3_test_problem()?;
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
        let symbolic_structure = problem.build_symbolic_structure(
            &variables,
            &variable_index_sparce_matrix,
            col_offset,
        )?;
        let (residual_sparse, jacobian_sparse) = problem.compute_residual_and_jacobian_sparse(
            &variables,
            &variable_index_sparce_matrix,
            &symbolic_structure,
        )?;

        // Test sparse dimensions match
        assert_eq!(residual_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.nrows(), problem.total_residual_dimension);
        assert_eq!(jacobian_sparse.ncols(), 48); // 8 variables * 6 DOF each

        Ok(())
    }

    #[test]
    fn test_residual_block_operations() -> TestResult {
        let mut problem = Problem::new();

        // Test adding residual blocks
        let block_id1 = problem.add_residual_block(
            &["x0", "x1"],
            Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1)),
            Some(Box::new(HuberLoss::new(1.0)?)),
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

        Ok(())
    }

    #[test]
    fn test_variable_constraints() -> TestResult {
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

        Ok(())
    }
}
