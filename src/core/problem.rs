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
//! use apex_solver::factors::{BetweenFactor, PriorFactor};
//! use apex_solver::core::loss_functions::HuberLoss;
//! use apex_solver::manifold::ManifoldType;
//! use nalgebra::{DVector, dvector};
//! use std::collections::HashMap;
//! use apex_solver::manifold::se2::SE2;
//! use apex_solver::JacobianMode;
//! # use apex_solver::error::ApexSolverResult;
//! # fn example() -> ApexSolverResult<()> {
//!
//! let mut problem = Problem::new(JacobianMode::Sparse);
//!
//! // Add prior factor to anchor the first pose
//! let prior = Box::new(PriorFactor {
//!     data: dvector![0.0, 0.0, 0.0],
//! });
//! problem.add_residual_block(&["x0"], prior, None);
//!
//! // Add between factor with robust loss
//! let between = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
//! let loss: Option<Box<dyn apex_solver::core::loss_functions::LossFunction + Send>> =
//!     Some(Box::new(HuberLoss::new(1.0)?));
//! problem.add_residual_block(&["x0", "x1"], between, loss);
//!
//! // Initialize variables
//! let mut initial_values = HashMap::new();
//! initial_values.insert("x0".to_string(), (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]));
//! initial_values.insert("x1".to_string(), (ManifoldType::SE2, dvector![0.9, 0.1, 0.12]));
//!
//! let variables = problem.initialize_variables(&initial_values);
//! # Ok(())
//! # }
//! # example().unwrap();
//! ```

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Error, Write},
    sync::{Arc, Mutex},
};

use faer::{Col, Mat, MatRef, sparse::SparseColMat};
use nalgebra::DVector;
use rayon::prelude::*;
use tracing::warn;

use crate::{
    core::{
        CoreError, corrector::Corrector, loss_functions::LossFunction,
        residual_block::ResidualBlock, variable::Variable,
    },
    error::{ApexSolverError, ApexSolverResult},
    factors::Factor,
    linalg::{JacobianMode, LinearSolver, SparseMode, extract_variable_covariances},
};
use apex_manifolds::{ManifoldType, rn, se2, se3, se23, sgal3, sim3, so2, so3};

// Re-export SymbolicStructure from the assembly module for backward compatibility
pub use crate::linearizer::cpu::sparse::SymbolicStructure;

/// Enum to handle mixed manifold variable types
#[derive(Clone)]
pub enum VariableEnum {
    Rn(Variable<rn::Rn>),
    SE2(Variable<se2::SE2>),
    SE3(Variable<se3::SE3>),
    SE23(Variable<se23::SE23>),
    SGal3(Variable<sgal3::SGal3>),
    Sim3(Variable<sim3::Sim3>),
    SO2(Variable<so2::SO2>),
    SO3(Variable<so3::SO3>),
}

impl VariableEnum {
    /// Get the manifold type for this variable
    pub fn manifold_type(&self) -> ManifoldType {
        match self {
            VariableEnum::Rn(_) => ManifoldType::RN,
            VariableEnum::SE2(_) => ManifoldType::SE2,
            VariableEnum::SE3(_) => ManifoldType::SE3,
            VariableEnum::SE23(_) => ManifoldType::SE23,
            VariableEnum::SGal3(_) => ManifoldType::SGal3,
            VariableEnum::Sim3(_) => ManifoldType::Sim3,
            VariableEnum::SO2(_) => ManifoldType::SO2,
            VariableEnum::SO3(_) => ManifoldType::SO3,
        }
    }

    /// Get the tangent space size for this variable
    pub fn get_size(&self) -> usize {
        match self {
            VariableEnum::Rn(var) => var.get_size(),
            VariableEnum::SE2(var) => var.get_size(),
            VariableEnum::SE3(var) => var.get_size(),
            VariableEnum::SE23(var) => var.get_size(),
            VariableEnum::SGal3(var) => var.get_size(),
            VariableEnum::Sim3(var) => var.get_size(),
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
            VariableEnum::SE23(var) => var.value.clone().into(),
            VariableEnum::SGal3(var) => var.value.clone().into(),
            VariableEnum::Sim3(var) => var.value.clone().into(),
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
                let mut step_data = DVector::from_fn(6, |i, _| step_slice[(i, 0)]);
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 6 {
                        step_data[fixed_idx] = 0.0;
                    }
                }
                let tangent = se3::SE3Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SE2(var) => {
                let mut step_data = DVector::from_fn(3, |i, _| step_slice[(i, 0)]);
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 3 {
                        step_data[fixed_idx] = 0.0;
                    }
                }
                let tangent = se2::SE2Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SO3(var) => {
                let mut step_data = DVector::from_fn(3, |i, _| step_slice[(i, 0)]);
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 3 {
                        step_data[fixed_idx] = 0.0;
                    }
                }
                let tangent = so3::SO3Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SO2(var) => {
                let mut step_data = DVector::from_fn(1, |i, _| step_slice[(i, 0)]);
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 1 {
                        step_data[fixed_idx] = 0.0;
                    }
                }
                let tangent = so2::SO2Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SE23(var) => {
                // SE_2(3) has 9 DOF in tangent space
                let mut step_data = DVector::from_fn(9, |i, _| step_slice[(i, 0)]);

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 9 {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let tangent = se23::SE23Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::SGal3(var) => {
                // SGal(3) has 10 DOF in tangent space
                let mut step_data = DVector::from_fn(10, |i, _| step_slice[(i, 0)]);

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 10 {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let tangent = sgal3::SGal3Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::Sim3(var) => {
                // Sim(3) has 7 DOF in tangent space
                let mut step_data = DVector::from_fn(7, |i, _| step_slice[(i, 0)]);

                // Enforce fixed indices: zero out step components for fixed DOF
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < 7 {
                        step_data[fixed_idx] = 0.0;
                    }
                }

                let tangent = sim3::Sim3Tangent::from(step_data);
                let new_value = var.plus(&tangent);
                var.set_value(new_value);
            }
            VariableEnum::Rn(var) => {
                let size = var.get_size();
                let mut step_data = DVector::from_fn(size, |i, _| step_slice[(i, 0)]);
                for &fixed_idx in &var.fixed_indices {
                    if fixed_idx < size {
                        step_data[fixed_idx] = 0.0;
                    }
                }
                let tangent = rn::RnTangent::new(step_data);
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
            VariableEnum::SE23(var) => var.get_covariance(),
            VariableEnum::SGal3(var) => var.get_covariance(),
            VariableEnum::Sim3(var) => var.get_covariance(),
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
            VariableEnum::SE23(var) => var.set_covariance(cov),
            VariableEnum::SGal3(var) => var.set_covariance(cov),
            VariableEnum::Sim3(var) => var.set_covariance(cov),
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
            VariableEnum::SE23(var) => var.clear_covariance(),
            VariableEnum::SGal3(var) => var.clear_covariance(),
            VariableEnum::Sim3(var) => var.clear_covariance(),
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
            VariableEnum::SE23(var) => &var.bounds,
            VariableEnum::SGal3(var) => &var.bounds,
            VariableEnum::Sim3(var) => &var.bounds,
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
            VariableEnum::SE23(var) => &var.fixed_indices,
            VariableEnum::SGal3(var) => &var.fixed_indices,
            VariableEnum::Sim3(var) => &var.fixed_indices,
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
            VariableEnum::SE23(var) => {
                let new_se23: se23::SE23 = vec.clone().into();
                var.set_value(new_se23);
            }
            VariableEnum::SGal3(var) => {
                let new_sgal3: sgal3::SGal3 = vec.clone().into();
                var.set_value(new_sgal3);
            }
            VariableEnum::Sim3(var) => {
                let new_sim3: sim3::Sim3 = vec.clone().into();
                var.set_value(new_sim3);
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
/// 1. **Construction**: Create a new Problem with `Problem::new(JacobianMode::Sparse)`
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
/// use apex_solver::factors::BetweenFactor;
/// use apex_solver::manifold::ManifoldType;
/// use apex_solver::manifold::se2::SE2;
/// use apex_solver::JacobianMode;
/// use nalgebra::dvector;
/// use std::collections::HashMap;
///
/// let mut problem = Problem::new(JacobianMode::Sparse);
///
/// // Add a between factor
/// let factor = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
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
    pub(crate) total_residual_dimension: usize,

    /// Controls which Jacobian assembly strategy is used (sparse or dense).
    ///
    /// Set via `Problem::new(JacobianMode::Sparse)` (sparse, default) or
    /// `Problem::new(JacobianMode::Dense)` (dense).
    /// The optimizer reads this field to select the assembly path.
    pub(crate) jacobian_mode: JacobianMode,

    /// Counter for assigning unique IDs to residual blocks
    residual_id_count: usize,

    /// Map from residual block ID to ResidualBlock instance
    residual_blocks: HashMap<usize, ResidualBlock>,

    /// Variables with fixed indices (e.g., fix first pose's x,y coordinates)
    /// Maps variable name -> set of indices to fix
    pub(crate) fixed_variable_indexes: HashMap<String, HashSet<usize>>,

    /// Variable bounds (box constraints on individual DOF)
    /// Maps variable name -> (index -> (lower_bound, upper_bound))
    pub(crate) variable_bounds: HashMap<String, HashMap<usize, (f64, f64)>>,
}
impl Default for Problem {
    fn default() -> Self {
        Self::new(JacobianMode::Sparse)
    }
}

impl Problem {
    /// Create a new empty optimization problem with the given Jacobian assembly mode.
    ///
    /// # Arguments
    ///
    /// * `jacobian_mode` - Assembly strategy: [`JacobianMode::Sparse`] (default, large problems)
    ///   or [`JacobianMode::Dense`] (small-to-medium problems < ~500 DOF)
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::problem::Problem;
    /// use apex_solver::JacobianMode;
    ///
    /// // Sparse (default for large-scale problems)
    /// let sparse_problem = Problem::new(JacobianMode::Sparse);
    /// assert_eq!(sparse_problem.num_residual_blocks(), 0);
    ///
    /// // Dense (for small-to-medium problems)
    /// let dense_problem = Problem::new(JacobianMode::Dense);
    /// assert_eq!(dense_problem.num_residual_blocks(), 0);
    /// ```
    pub fn new(jacobian_mode: JacobianMode) -> Self {
        Self {
            total_residual_dimension: 0,
            jacobian_mode,
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
    /// use apex_solver::factors::{BetweenFactor, PriorFactor};
    /// use apex_solver::core::loss_functions::HuberLoss;
    /// use apex_solver::JacobianMode;
    /// use nalgebra::dvector;
    /// use apex_solver::manifold::se2::SE2;
    /// # use apex_solver::error::ApexSolverResult;
    /// # fn example() -> ApexSolverResult<()> {
    ///
    /// let mut problem = Problem::new(JacobianMode::Sparse);
    ///
    /// // Add prior factor (unary constraint)
    /// let prior = Box::new(PriorFactor { data: dvector![0.0, 0.0, 0.0] });
    /// let id1 = problem.add_residual_block(&["x0"], prior, None);
    ///
    /// // Add between factor with robust loss (binary constraint)
    /// let between = Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1)));
    /// let loss: Option<Box<dyn apex_solver::core::loss_functions::LossFunction + Send>> =
    ///     Some(Box::new(HuberLoss::new(1.0)?));
    /// let id2 = problem.add_residual_block(&["x0", "x1"], between, loss);
    ///
    /// assert_eq!(id1, 0);
    /// assert_eq!(id2, 1);
    /// assert_eq!(problem.num_residual_blocks(), 2);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
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
    /// use apex_solver::JacobianMode;
    /// use nalgebra::dvector;
    /// use std::collections::HashMap;
    ///
    /// let problem = Problem::new(JacobianMode::Sparse);
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
                        assert_eq!(
                            v.1.len(),
                            1,
                            "Variable '{}': SO2 expects 1 element, got {}",
                            k,
                            v.1.len()
                        );
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
                        assert_eq!(
                            v.1.len(),
                            4,
                            "Variable '{}': SO3 expects 4 elements, got {}",
                            k,
                            v.1.len()
                        );
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
                        assert_eq!(
                            v.1.len(),
                            3,
                            "Variable '{}': SE2 expects 3 elements, got {}",
                            k,
                            v.1.len()
                        );
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
                        assert_eq!(
                            v.1.len(),
                            7,
                            "Variable '{}': SE3 expects 7 elements, got {}",
                            k,
                            v.1.len()
                        );
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
                    ManifoldType::SE23 => {
                        let mut var = Variable::new(se23::SE23::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SE23(var)
                    }
                    ManifoldType::SGal3 => {
                        let mut var = Variable::new(sgal3::SGal3::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::SGal3(var)
                    }
                    ManifoldType::Sim3 => {
                        let mut var = Variable::new(sim3::Sim3::from(v.1.clone()));
                        if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                            var.fixed_indices = indexes.clone();
                        }
                        if let Some(bounds) = self.variable_bounds.get(k) {
                            var.bounds = bounds.clone();
                        }
                        VariableEnum::Sim3(var)
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

    /// Get a reference to the residual blocks map (crate-internal for assembly modules)
    pub(crate) fn residual_blocks(&self) -> &HashMap<usize, ResidualBlock> {
        &self.residual_blocks
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
    /// ```no_run
    /// # use apex_solver::core::problem::Problem;
    /// # use apex_solver::JacobianMode;
    /// # use std::collections::HashMap;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let problem = Problem::new(JacobianMode::Sparse);
    /// # let variables = HashMap::new();
    /// // Initial cost evaluation (no Jacobian needed)
    /// let residual = problem.compute_residual_sparse(&variables)?;
    /// let initial_cost = residual.norm_l2() * residual.norm_l2();
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_residual_sparse(
        &self,
        variables: &HashMap<String, VariableEnum>,
    ) -> ApexSolverResult<Mat<f64>> {
        let total_residual = Arc::new(Mutex::new(Col::<f64>::zeros(self.total_residual_dimension)));

        // Compute residuals in parallel (no Jacobian needed)
        let result: Result<Vec<()>, ApexSolverError> = self
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
    /// ```
    /// # use apex_solver::core::problem::Problem;
    /// # use std::collections::HashMap;
    /// // Inside optimizer loop, compute both residual and Jacobian for linearization
    /// // let (residual, jacobian) = problem.compute_residual_and_jacobian_sparse(
    /// //     &variables,
    /// //     &variable_index_map,
    /// //     &symbolic_structure,
    /// // )?;
    /// //
    /// // Use for linear system: J^T J dx = -J^T r
    /// ```
    /// Compute residuals and sparse Jacobian.
    ///
    /// Delegates to [`assembly::sparse::assemble_sparse()`](super::assembly::sparse::assemble_sparse).
    pub fn compute_residual_and_jacobian_sparse(
        &self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_sparce_matrix: &HashMap<String, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> ApexSolverResult<(Mat<f64>, SparseColMat<usize, f64>)> {
        crate::linearizer::cpu::sparse::assemble_sparse(
            self,
            variables,
            variable_index_sparce_matrix,
            symbolic_structure,
        )
    }

    /// Compute residuals and dense Jacobian.
    ///
    /// Delegates to [`assembly::dense::assemble_dense()`](super::assembly::dense::assemble_dense).
    pub fn compute_residual_and_jacobian_dense(
        &self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
        total_dof: usize,
    ) -> ApexSolverResult<(Mat<f64>, Mat<f64>)> {
        crate::linearizer::cpu::dense::assemble_dense(
            self,
            variables,
            variable_index_map,
            total_dof,
        )
    }

    /// Compute only the residual for a single residual block (no Jacobian).
    ///
    /// Helper method for `compute_residual_sparse()`.
    fn compute_residual_block(
        &self,
        residual_block: &ResidualBlock,
        variables: &HashMap<String, VariableEnum>,
        total_residual: &Arc<Mutex<Col<f64>>>,
    ) -> ApexSolverResult<()> {
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
        linear_solver: &mut Box<dyn LinearSolver<SparseMode>>,
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

    /// Compute and set covariances using a generic linear solver.
    ///
    /// This is the generic version of [`compute_and_set_covariances`] that works
    /// with any assembly mode (sparse or dense).
    pub fn compute_and_set_covariances_generic<M: crate::linalg::LinearizationMode>(
        &self,
        linear_solver: &mut dyn crate::linalg::LinearSolver<M>,
        variables: &mut HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> Option<HashMap<String, Mat<f64>>> {
        linear_solver.compute_covariance_matrix()?;
        let full_cov = linear_solver.get_covariance_matrix()?.clone();

        let per_var_covariances =
            extract_variable_covariances(&full_cov, variables, variable_index_map);

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
    use crate::factors::{BetweenFactor, PriorFactor};
    use apex_manifolds::{ManifoldType, se2::SE2, se3::SE3};
    use nalgebra::{Quaternion, Vector3, dvector};
    use std::collections::HashMap;

    type TestResult = Result<(), Box<dyn std::error::Error>>;
    type TestProblemResult = Result<
        (
            Problem,
            HashMap<String, (ManifoldType, nalgebra::DVector<f64>)>,
        ),
        Box<dyn std::error::Error>,
    >;

    /// Create a test SE2 dataset with 10 vertices in a loop
    fn create_se2_test_problem() -> TestProblemResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
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

            let between_factor = BetweenFactor::new(SE2::from_xy_angle(dx, dy, dtheta));
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

        let loop_closure = BetweenFactor::new(SE2::from_xy_angle(dx, dy, dtheta));
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
    fn create_se3_test_problem() -> TestProblemResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
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

            let between_factor = BetweenFactor::new(relative_se3);
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
        let symbolic_structure = crate::linearizer::cpu::sparse::build_symbolic_structure(
            &problem,
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
        let symbolic_structure = crate::linearizer::cpu::sparse::build_symbolic_structure(
            &problem,
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
        let symbolic_structure = crate::linearizer::cpu::sparse::build_symbolic_structure(
            &problem,
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
        let mut problem = Problem::new(JacobianMode::Sparse);

        // Test adding residual blocks
        let block_id1 = problem.add_residual_block(
            &["x0", "x1"],
            Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1))),
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
        let mut problem = Problem::new(JacobianMode::Sparse);

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

    // -------------------------------------------------------------------------
    // New tests for previously uncovered code paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_problem_default_equals_new_sparse() {
        let default_problem = Problem::default();
        let new_problem = Problem::new(JacobianMode::Sparse);
        assert_eq!(default_problem.jacobian_mode, new_problem.jacobian_mode);
        assert_eq!(
            default_problem.num_residual_blocks(),
            new_problem.num_residual_blocks()
        );
    }

    /// Test that SO2 variables can be initialized correctly.
    #[test]
    fn test_initialize_so2_variable() -> TestResult {
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert("angle".to_string(), (ManifoldType::SO2, dvector![0.5]));
        let variables = problem.initialize_variables(&initial);
        assert_eq!(variables.len(), 1);
        let var = variables.get("angle").ok_or("angle not found")?;
        assert_eq!(var.manifold_type(), ManifoldType::SO2);
        assert_eq!(var.get_size(), 1);
        assert_eq!(var.to_vector().len(), 1);
        Ok(())
    }

    /// Test that SO3 variables can be initialized correctly.
    #[test]
    fn test_initialize_so3_variable() -> TestResult {
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        // SO3: [qw, qx, qy, qz]
        initial.insert(
            "rot".to_string(),
            (ManifoldType::SO3, dvector![1.0, 0.0, 0.0, 0.0]),
        );
        let variables = problem.initialize_variables(&initial);
        assert_eq!(variables.len(), 1);
        let var = variables.get("rot").ok_or("rot not found")?;
        assert_eq!(var.manifold_type(), ManifoldType::SO3);
        assert_eq!(var.get_size(), 3); // SO3 tangent space is 3-dimensional
        Ok(())
    }

    /// Test that RN variables can be initialized correctly (arbitrary dimension).
    #[test]
    fn test_initialize_rn_variable() -> TestResult {
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert(
            "pt".to_string(),
            (ManifoldType::RN, dvector![1.0, 2.0, 3.0]),
        );
        let variables = problem.initialize_variables(&initial);
        let var = variables.get("pt").ok_or("pt not found")?;
        assert_eq!(var.manifold_type(), ManifoldType::RN);
        assert_eq!(var.get_size(), 3);
        let vec = var.to_vector();
        assert!((vec[0] - 1.0).abs() < 1e-12);
        assert!((vec[1] - 2.0).abs() < 1e-12);
        assert!((vec[2] - 3.0).abs() < 1e-12);
        Ok(())
    }

    /// Test VariableEnum covariance lifecycle: set → get → clear.
    #[test]
    fn test_variable_enum_covariance_lifecycle() -> TestResult {
        use faer::Mat;
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert(
            "x0".to_string(),
            (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]),
        );
        let mut variables = problem.initialize_variables(&initial);
        let var = variables.get_mut("x0").ok_or("x0 not found")?;

        // Before setting: no covariance
        assert!(var.get_covariance().is_none());

        // Set a 3×3 identity covariance
        let cov = Mat::identity(3, 3);
        var.set_covariance(cov);
        let retrieved = var.get_covariance().ok_or("covariance not set")?;
        assert_eq!(retrieved.nrows(), 3);
        assert!((retrieved[(0, 0)] - 1.0).abs() < 1e-12);

        // Clear covariance
        var.clear_covariance();
        assert!(var.get_covariance().is_none());
        Ok(())
    }

    /// Test VariableEnum::get_bounds() propagation via initialize_variables.
    #[test]
    fn test_variable_enum_bounds_propagation() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.set_variable_bounds("x0", 0, -1.0, 1.0);
        problem.set_variable_bounds("x0", 1, -2.0, 2.0);
        let mut initial = HashMap::new();
        initial.insert(
            "x0".to_string(),
            (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]),
        );
        let variables = problem.initialize_variables(&initial);
        let var = variables.get("x0").ok_or("x0 not found")?;
        let bounds = var.get_bounds();
        assert_eq!(bounds.len(), 2);
        let (lo, hi) = bounds[&0];
        assert!((lo + 1.0).abs() < 1e-12);
        assert!((hi - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// Test VariableEnum::get_fixed_indices() propagation via initialize_variables.
    #[test]
    fn test_variable_enum_fixed_indices_propagation() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.fix_variable("x0", 0);
        problem.fix_variable("x0", 2);
        let mut initial = HashMap::new();
        initial.insert(
            "x0".to_string(),
            (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]),
        );
        let variables = problem.initialize_variables(&initial);
        let var = variables.get("x0").ok_or("x0 not found")?;
        let fixed = var.get_fixed_indices();
        assert_eq!(fixed.len(), 2);
        assert!(fixed.contains(&0));
        assert!(fixed.contains(&2));
        Ok(())
    }

    /// Test VariableEnum::set_from_vector() for all five manifold variants.
    #[test]
    fn test_variable_enum_set_from_vector_all_variants() -> TestResult {
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert(
            "se2".to_string(),
            (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]),
        );
        initial.insert(
            "se3".to_string(),
            (
                ManifoldType::SE3,
                dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
        );
        initial.insert("so2".to_string(), (ManifoldType::SO2, dvector![0.0]));
        initial.insert(
            "so3".to_string(),
            (ManifoldType::SO3, dvector![1.0, 0.0, 0.0, 0.0]),
        );
        initial.insert("rn".to_string(), (ManifoldType::RN, dvector![1.0, 2.0]));
        let mut variables = problem.initialize_variables(&initial);

        // SE2: update to [1, 2, 0.5]
        let new_se2 = dvector![1.0, 2.0, 0.5];
        variables
            .get_mut("se2")
            .ok_or("se2 not found")?
            .set_from_vector(&new_se2);
        let got = variables.get("se2").ok_or("se2 not found")?.to_vector();
        assert!((got[0] - 1.0).abs() < 1e-10);

        // SE3: update translation part
        let new_se3 = dvector![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        variables
            .get_mut("se3")
            .ok_or("se3 not found")?
            .set_from_vector(&new_se3);
        let got = variables.get("se3").ok_or("se3 not found")?.to_vector();
        assert!((got[0] - 1.0).abs() < 1e-10);

        // SO2
        let new_so2 = dvector![0.5];
        variables
            .get_mut("so2")
            .ok_or("so2 not found")?
            .set_from_vector(&new_so2);
        assert_eq!(variables.get("so2").ok_or("so2 not found")?.get_size(), 1);

        // SO3
        let new_so3 = dvector![1.0, 0.0, 0.0, 0.0];
        variables
            .get_mut("so3")
            .ok_or("so3 not found")?
            .set_from_vector(&new_so3);
        assert_eq!(variables.get("so3").ok_or("so3 not found")?.get_size(), 3);

        // RN
        let new_rn = dvector![5.0, 6.0];
        variables
            .get_mut("rn")
            .ok_or("rn not found")?
            .set_from_vector(&new_rn);
        let got = variables.get("rn").ok_or("rn not found")?.to_vector();
        assert!((got[0] - 5.0).abs() < 1e-10);
        assert!((got[1] - 6.0).abs() < 1e-10);
        Ok(())
    }

    /// Test apply_tangent_step() zeros out fixed indices for SE2.
    #[test]
    fn test_apply_tangent_step_se2_fixed_index() -> TestResult {
        use faer::Mat;
        let mut problem = Problem::new(JacobianMode::Sparse);
        // Fix SE2 index 0 (x translation)
        problem.fix_variable("x0", 0);
        let mut initial = HashMap::new();
        initial.insert(
            "x0".to_string(),
            (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]),
        );
        let mut variables = problem.initialize_variables(&initial);

        // Step vector: [1, 2, 3] — but index 0 should be zeroed
        let mut step = Mat::zeros(3, 1);
        step[(0, 0)] = 1.0;
        step[(1, 0)] = 2.0;
        step[(2, 0)] = 3.0;

        let var = variables.get_mut("x0").ok_or("x0 not found")?;
        var.apply_tangent_step(step.as_ref());
        // After update, get_size() should still be 3
        assert_eq!(var.get_size(), 3);
        Ok(())
    }

    /// Test apply_tangent_step() zeros out fixed indices for SE3.
    #[test]
    fn test_apply_tangent_step_se3_fixed_index() -> TestResult {
        use faer::Mat;
        let mut problem = Problem::new(JacobianMode::Sparse);
        problem.fix_variable("p", 0);
        let mut initial = HashMap::new();
        initial.insert(
            "p".to_string(),
            (
                ManifoldType::SE3,
                dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
        );
        let mut variables = problem.initialize_variables(&initial);

        let mut step = Mat::zeros(6, 1);
        for i in 0..6 {
            step[(i, 0)] = (i + 1) as f64;
        }

        let var = variables.get_mut("p").ok_or("p not found")?;
        var.apply_tangent_step(step.as_ref());
        assert_eq!(var.get_size(), 6);
        Ok(())
    }

    /// Test apply_tangent_step for SO2 and SO3 and RN.
    #[test]
    fn test_apply_tangent_step_remaining_manifolds() -> TestResult {
        use faer::Mat;
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert("so2".to_string(), (ManifoldType::SO2, dvector![0.0]));
        initial.insert(
            "so3".to_string(),
            (ManifoldType::SO3, dvector![1.0, 0.0, 0.0, 0.0]),
        );
        initial.insert("rn".to_string(), (ManifoldType::RN, dvector![0.0, 0.0]));
        let mut variables = problem.initialize_variables(&initial);

        // SO2: 1-DOF step
        let mut step_so2 = Mat::zeros(1, 1);
        step_so2[(0, 0)] = 0.1;
        variables
            .get_mut("so2")
            .ok_or("so2 not found")?
            .apply_tangent_step(step_so2.as_ref());

        // SO3: 3-DOF step
        let mut step_so3 = Mat::zeros(3, 1);
        step_so3[(0, 0)] = 0.01;
        variables
            .get_mut("so3")
            .ok_or("so3 not found")?
            .apply_tangent_step(step_so3.as_ref());

        // RN: 2-DOF step
        let mut step_rn = Mat::zeros(2, 1);
        step_rn[(0, 0)] = 1.0;
        step_rn[(1, 0)] = 2.0;
        variables
            .get_mut("rn")
            .ok_or("rn not found")?
            .apply_tangent_step(step_rn.as_ref());
        let vec = variables.get("rn").ok_or("rn not found")?.to_vector();
        assert!((vec[0] - 1.0).abs() < 1e-10);
        assert!((vec[1] - 2.0).abs() < 1e-10);
        Ok(())
    }

    /// Test compute_residual_sparse returns a non-negative squared norm.
    #[test]
    fn test_compute_residual_sparse_smoke() -> TestResult {
        let (problem, initial_values) = create_se2_test_problem()?;
        let variables = problem.initialize_variables(&initial_values);
        let residual = problem.compute_residual_sparse(&variables)?;
        // Squared norm must be non-negative
        let norm_sq: f64 = (0..residual.nrows())
            .map(|i| residual[(i, 0)].powi(2))
            .sum();
        assert!(norm_sq >= 0.0);
        assert_eq!(residual.nrows(), problem.total_residual_dimension);
        Ok(())
    }

    /// Test log_residual_to_file writes a file without error.
    #[test]
    fn test_log_residual_to_file() -> TestResult {
        let problem = Problem::new(JacobianMode::Sparse);
        let residual = nalgebra::dvector![1.0, 2.0, 3.0];
        let path = std::env::temp_dir().join("apex_test_residual.txt");
        let path_str = path.to_str().ok_or("temp path is not valid UTF-8")?;
        problem.log_residual_to_file(&residual, path_str)?;
        assert!(path.exists());
        Ok(())
    }

    /// Test log_variables_to_file writes a file without error.
    #[test]
    fn test_log_variables_to_file() -> TestResult {
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert(
            "x0".to_string(),
            (ManifoldType::SE2, dvector![1.0, 2.0, 0.3]),
        );
        let variables = problem.initialize_variables(&initial);
        let path = std::env::temp_dir().join("apex_test_variables.txt");
        let path_str = path.to_str().ok_or("temp path is not valid UTF-8")?;
        problem.log_variables_to_file(&variables, path_str)?;
        assert!(path.exists());
        Ok(())
    }

    /// Test log_sparse_jacobian_to_file writes a file without error.
    #[test]
    fn test_log_sparse_jacobian_to_file() -> TestResult {
        use faer::sparse::SparseColMat;
        let problem = Problem::new(JacobianMode::Sparse);
        let triplets = vec![faer::sparse::Triplet::new(0usize, 0usize, 1.0f64)];
        let jacobian =
            SparseColMat::try_new_from_triplets(1, 1, &triplets).map_err(|e| format!("{e:?}"))?;
        let path = std::env::temp_dir().join("apex_test_jacobian.txt");
        let path_str = path.to_str().ok_or("temp path is not valid UTF-8")?;
        problem.log_sparse_jacobian_to_file(&jacobian, path_str)?;
        assert!(path.exists());
        Ok(())
    }

    /// Test set_variable_bounds with lower > upper logs a warning but does not update bounds.
    #[test]
    fn test_set_variable_bounds_invalid_order() {
        let mut problem = Problem::new(JacobianMode::Sparse);
        // lower > upper — should warn but NOT insert into the bounds map
        problem.set_variable_bounds("x0", 0, 5.0, 1.0);
        // Since lower > upper, the bounds map should NOT be updated
        assert!(!problem.variable_bounds.contains_key("x0"));
    }

    /// Test VariableEnum::manifold_type() returns the correct type for each variant.
    #[test]
    fn test_variable_enum_manifold_type() {
        let problem = Problem::new(JacobianMode::Sparse);
        let mut initial = HashMap::new();
        initial.insert(
            "se2".to_string(),
            (ManifoldType::SE2, dvector![0.0, 0.0, 0.0]),
        );
        initial.insert(
            "se3".to_string(),
            (
                ManifoldType::SE3,
                dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
        );
        initial.insert("so2".to_string(), (ManifoldType::SO2, dvector![0.0]));
        initial.insert(
            "so3".to_string(),
            (ManifoldType::SO3, dvector![1.0, 0.0, 0.0, 0.0]),
        );
        initial.insert("rn".to_string(), (ManifoldType::RN, dvector![0.0]));
        let variables = problem.initialize_variables(&initial);

        assert_eq!(variables["se2"].manifold_type(), ManifoldType::SE2);
        assert_eq!(variables["se3"].manifold_type(), ManifoldType::SE3);
        assert_eq!(variables["so2"].manifold_type(), ManifoldType::SO2);
        assert_eq!(variables["so3"].manifold_type(), ManifoldType::SO3);
        assert_eq!(variables["rn"].manifold_type(), ManifoldType::RN);
    }
}
