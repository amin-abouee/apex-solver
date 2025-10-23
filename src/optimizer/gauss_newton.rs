//! Gauss-Newton optimization algorithm implementation
//!
//! The Gauss-Newton algorithm is an iterative method for solving non-linear least squares problems.
//! It approximates the Hessian using only first-order derivatives (J^T·J) and takes pure Newton steps.
//!
//! This implementation includes:
//! - Direct Newton step computation without damping
//! - Robust numerical factorization (Cholesky or QR)
//! - Comprehensive optimization summaries
//! - Support for both sparse Cholesky and QR factorizations
//! - Optional Jacobi scaling for improved convergence

use crate::{core::problem, error, linalg, manifold, optimizer};
use faer::sparse;
use std::{collections, fmt, time};

/// Summary statistics for the Gauss-Newton optimization process.
#[derive(Debug, Clone)]
pub struct GaussNewtonSummary {
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Average cost reduction per iteration
    pub average_cost_reduction: f64,
    /// Maximum gradient norm encountered
    pub max_gradient_norm: f64,
    /// Final gradient norm
    pub final_gradient_norm: f64,
    /// Maximum parameter update norm
    pub max_parameter_update_norm: f64,
    /// Final parameter update norm
    pub final_parameter_update_norm: f64,
    /// Total time elapsed
    pub total_time: time::Duration,
    /// Average time per iteration
    pub average_time_per_iteration: time::Duration,
}

impl fmt::Display for GaussNewtonSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Gauss-Newton Optimization Summary ===")?;
        writeln!(f, "Initial cost:              {:.6e}", self.initial_cost)?;
        writeln!(f, "Final cost:                {:.6e}", self.final_cost)?;
        writeln!(
            f,
            "Cost reduction:            {:.6e} ({:.2}%)",
            self.initial_cost - self.final_cost,
            100.0 * (self.initial_cost - self.final_cost) / self.initial_cost.max(1e-12)
        )?;
        writeln!(f, "Total iterations:          {}", self.iterations)?;
        writeln!(
            f,
            "Average cost reduction:    {:.6e}",
            self.average_cost_reduction
        )?;
        writeln!(
            f,
            "Max gradient norm:         {:.6e}",
            self.max_gradient_norm
        )?;
        writeln!(
            f,
            "Final gradient norm:       {:.6e}",
            self.final_gradient_norm
        )?;
        writeln!(
            f,
            "Max parameter update norm: {:.6e}",
            self.max_parameter_update_norm
        )?;
        writeln!(
            f,
            "Final param update norm:   {:.6e}",
            self.final_parameter_update_norm
        )?;
        writeln!(f, "Total time:                {:?}", self.total_time)?;
        writeln!(
            f,
            "Average time per iteration: {:?}",
            self.average_time_per_iteration
        )?;
        Ok(())
    }
}

/// Configuration for Gauss-Newton solver.
#[derive(Clone)]
pub struct GaussNewtonConfig {
    /// Type of linear solver for the linear systems
    pub linear_solver_type: linalg::LinearSolverType,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for cost function
    pub cost_tolerance: f64,
    /// Convergence tolerance for parameter updates
    pub parameter_tolerance: f64,
    /// Convergence tolerance for gradient norm
    pub gradient_tolerance: f64,
    /// Timeout duration
    pub timeout: Option<time::Duration>,
    /// Enable detailed logging
    pub verbose: bool,
    /// Use Jacobi column scaling (preconditioning)
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// This can improve convergence for problems with mixed parameter scales
    /// (e.g., positions in meters + angles in radians) but adds ~5-10% overhead.
    ///
    /// Default: false (Gauss-Newton is typically used on well-conditioned problems)
    pub use_jacobi_scaling: bool,
    /// Small regularization to ensure J^T·J is positive definite
    ///
    /// Pure Gauss-Newton (λ=0) can fail when J^T·J is singular or near-singular.
    /// Adding a tiny diagonal regularization (e.g., 1e-10) ensures numerical stability
    /// while maintaining the fast convergence of Gauss-Newton.
    ///
    /// Default: 1e-10 (very small, practically identical to pure Gauss-Newton)
    pub min_diagonal: f64,
    /// Compute per-variable covariance matrices (uncertainty estimation)
    ///
    /// When enabled, computes covariance by inverting the Hessian matrix after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and optimier::SolverResult.
    ///
    /// Default: false (to avoid performance overhead)
    pub compute_covariances: bool,

    /// Enable real-time visualization (graphical debugging).
    ///
    /// When enabled, optimization progress is logged to a Rerun viewer.
    /// Note: Has zero overhead when disabled.
    ///
    /// Default: false
    pub enable_visualization: bool,
}

impl Default for GaussNewtonConfig {
    fn default() -> Self {
        Self {
            linear_solver_type: linalg::LinearSolverType::default(),
            max_iterations: 100,
            cost_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            timeout: None,
            verbose: false,
            use_jacobi_scaling: false,
            min_diagonal: 1e-10,
            compute_covariances: false,
            enable_visualization: false,
        }
    }
}

impl GaussNewtonConfig {
    /// Create a new Gauss-Newton configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the linear solver type
    pub fn with_linear_solver_type(mut self, linear_solver_type: linalg::LinearSolverType) -> Self {
        self.linear_solver_type = linear_solver_type;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the cost tolerance
    pub fn with_cost_tolerance(mut self, cost_tolerance: f64) -> Self {
        self.cost_tolerance = cost_tolerance;
        self
    }

    /// Set the parameter tolerance
    pub fn with_parameter_tolerance(mut self, parameter_tolerance: f64) -> Self {
        self.parameter_tolerance = parameter_tolerance;
        self
    }

    /// Set the gradient tolerance
    pub fn with_gradient_tolerance(mut self, gradient_tolerance: f64) -> Self {
        self.gradient_tolerance = gradient_tolerance;
        self
    }

    /// Set the timeout duration
    pub fn with_timeout(mut self, timeout: time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enable or disable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable or disable Jacobi column scaling (preconditioning).
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// Can improve convergence for mixed-scale problems but adds ~5-10% overhead.
    pub fn with_jacobi_scaling(mut self, use_jacobi_scaling: bool) -> Self {
        self.use_jacobi_scaling = use_jacobi_scaling;
        self
    }

    /// Set the minimum diagonal regularization for numerical stability.
    ///
    /// A small value (e.g., 1e-10) ensures J^T·J is positive definite while
    /// maintaining the fast convergence of pure Gauss-Newton.
    pub fn with_min_diagonal(mut self, min_diagonal: f64) -> Self {
        self.min_diagonal = min_diagonal;
        self
    }

    /// Enable or disable covariance computation (uncertainty estimation).
    ///
    /// When enabled, computes the full covariance matrix by inverting the Hessian
    /// after convergence, then extracts per-variable covariance blocks.
    pub fn with_compute_covariances(mut self, compute_covariances: bool) -> Self {
        self.compute_covariances = compute_covariances;
        self
    }

    /// Enable real-time visualization.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable visualization
    pub fn with_visualization(mut self, enable: bool) -> Self {
        self.enable_visualization = enable;
        self
    }
}

/// State for optimization iteration
struct OptimizationState {
    variables: collections::HashMap<String, problem::VariableEnum>,
    variable_index_map: collections::HashMap<String, usize>,
    sorted_vars: Vec<String>,
    symbolic_structure: problem::SymbolicStructure,
    current_cost: f64,
    initial_cost: f64,
}

/// Result from step computation
struct StepResult {
    step: faer::Mat<f64>,
    gradient_norm: f64,
}

/// Result from cost evaluation
struct CostEvaluation {
    new_cost: f64,
    cost_reduction: f64,
}

/// Gauss-Newton solver for nonlinear least squares optimization.
pub struct GaussNewton {
    config: GaussNewtonConfig,
    jacobi_scaling: Option<sparse::SparseColMat<usize, f64>>,
    visualizer: Option<optimizer::visualization::OptimizationVisualizer>,
}

impl Default for GaussNewton {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussNewton {
    /// Create a new Gauss-Newton solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(GaussNewtonConfig::default())
    }

    /// Create a new Gauss-Newton solver with the given configuration.
    pub fn with_config(config: GaussNewtonConfig) -> Self {
        // Create visualizer if enabled (zero overhead when disabled)
        let visualizer = if config.enable_visualization {
            optimizer::visualization::OptimizationVisualizer::new(true).ok()
        } else {
            None
        };

        Self {
            config,
            jacobi_scaling: None,
            visualizer,
        }
    }

    /// Create the appropriate linear solver based on configuration
    fn create_linear_solver(&self) -> Box<dyn linalg::SparseLinearSolver> {
        match self.config.linear_solver_type {
            linalg::LinearSolverType::SparseCholesky => {
                Box::new(linalg::SparseCholeskySolver::new())
            }
            linalg::LinearSolverType::SparseQR => Box::new(linalg::SparseQRSolver::new()),
        }
    }

    /// Check convergence criteria
    fn check_convergence(
        &self,
        iteration: usize,
        cost_change: f64,
        parameter_update_norm: f64,
        gradient_norm: f64,
        elapsed: time::Duration,
    ) -> Option<optimizer::OptimizationStatus> {
        // Check timeout
        if let Some(timeout) = self.config.timeout
            && elapsed >= timeout
        {
            return Some(optimizer::OptimizationStatus::Timeout);
        }

        // Check maximum iterations
        if iteration >= self.config.max_iterations {
            return Some(optimizer::OptimizationStatus::MaxIterationsReached);
        }

        // Check cost tolerance
        if iteration > 0 && cost_change.abs() < self.config.cost_tolerance {
            return Some(optimizer::OptimizationStatus::CostToleranceReached);
        }

        // Check parameter tolerance
        if iteration > 0 && parameter_update_norm < self.config.parameter_tolerance {
            return Some(optimizer::OptimizationStatus::ParameterToleranceReached);
        }

        // Check gradient tolerance
        if gradient_norm < self.config.gradient_tolerance {
            return Some(optimizer::OptimizationStatus::GradientToleranceReached);
        }

        None
    }

    /// Create Jacobi scaling matrix from Jacobian
    fn create_jacobi_scaling(
        &self,
        jacobian: &sparse::SparseColMat<usize, f64>,
    ) -> sparse::SparseColMat<usize, f64> {
        use faer::sparse::Triplet;

        let cols = jacobian.ncols();
        let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
            .map(|c| {
                // Compute column norm: sqrt(sum(J_col^2))
                let col_norm_squared: f64 = jacobian
                    .triplet_iter()
                    .filter(|t| t.col == c)
                    .map(|t| t.val * t.val)
                    .sum();
                let col_norm = col_norm_squared.sqrt();
                // Scaling factor: 1.0 / (1.0 + col_norm)
                let scaling = 1.0 / (1.0 + col_norm);
                Triplet::new(c, c, scaling)
            })
            .collect();

        sparse::SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
            .expect("Failed to create Jacobi scaling matrix")
    }

    /// Apply Jacobi scaling to Jacobian: J_scaled = J * S
    fn apply_jacobi_scaling(
        &self,
        jacobian: &sparse::SparseColMat<usize, f64>,
        scaling: &sparse::SparseColMat<usize, f64>,
    ) -> sparse::SparseColMat<usize, f64> {
        jacobian * scaling
    }

    /// Apply inverse Jacobi scaling to step: dx_final = S * dx_scaled
    fn apply_inverse_jacobi_scaling(
        &self,
        step: &faer::Mat<f64>,
        scaling: &sparse::SparseColMat<usize, f64>,
    ) -> faer::Mat<f64> {
        scaling * step
    }

    /// Initialize optimization state from problem and initial parameters
    fn initialize_optimization_state(
        &self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<OptimizationState, error::ApexError> {
        // Initialize variables from initial values
        let variables = problem.initialize_variables(initial_params);

        // Create column mapping for variables
        let mut variable_index_map = collections::HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<String> = variables.keys().cloned().collect();
        sorted_vars.sort();

        for var_name in &sorted_vars {
            variable_index_map.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Build symbolic structure for sparse operations
        let symbolic_structure =
            problem.build_symbolic_structure(&variables, &variable_index_map, col_offset)?;

        // Initial cost evaluation (residual only, no Jacobian needed)
        let residual = problem.compute_residual_sparse(&variables)?;

        let residual_norm = residual.norm_l2();
        let current_cost = residual_norm * residual_norm;
        let initial_cost = current_cost;

        if self.config.verbose {
            println!(
                "Starting Gauss-Newton optimization with {} max iterations",
                self.config.max_iterations
            );
            println!("Initial cost: {:.6e}", current_cost);
        }

        Ok(OptimizationState {
            variables,
            variable_index_map,
            sorted_vars,
            symbolic_structure,
            current_cost,
            initial_cost,
        })
    }

    /// Process Jacobian by creating and applying Jacobi scaling if enabled
    fn process_jacobian(
        &mut self,
        jacobian: &sparse::SparseColMat<usize, f64>,
        iteration: usize,
    ) -> sparse::SparseColMat<usize, f64> {
        // Create Jacobi scaling on first iteration if enabled
        if iteration == 0 && self.config.use_jacobi_scaling {
            let scaling = self.create_jacobi_scaling(jacobian);

            if self.config.verbose {
                println!("Jacobi scaling computed for {} columns", scaling.ncols());
            }

            self.jacobi_scaling = Some(scaling);
        }

        // Apply Jacobi scaling to Jacobian if enabled
        if self.config.use_jacobi_scaling {
            self.apply_jacobi_scaling(jacobian, self.jacobi_scaling.as_ref().unwrap())
        } else {
            jacobian.clone()
        }
    }

    /// Compute Gauss-Newton step by solving the normal equations
    fn compute_gauss_newton_step(
        &self,
        residuals: &faer::Mat<f64>,
        scaled_jacobian: &sparse::SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn linalg::SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Solve the Gauss-Newton equation: J^T·J·Δx = -J^T·r
        // Use min_diagonal for numerical stability (tiny regularization)
        let residuals_owned = residuals.as_ref().to_owned();
        let scaled_step = linear_solver
            .solve_augmented_equation(&residuals_owned, scaled_jacobian, self.config.min_diagonal)
            .ok()?;

        // Get gradient from the solver (J^T * r)
        let gradient = linear_solver.get_gradient()?;
        // Compute gradient norm for convergence check
        let gradient_norm = gradient.norm_l2();

        if self.config.verbose {
            println!("Gradient (J^T*r) norm: {:.12e}", gradient_norm);
        }

        // Apply inverse Jacobi scaling to get final step (if enabled)
        let step = if self.config.use_jacobi_scaling {
            self.apply_inverse_jacobi_scaling(&scaled_step, self.jacobi_scaling.as_ref().unwrap())
        } else {
            scaled_step.clone()
        };

        if self.config.verbose {
            println!("Step norm: {:.12e}", step.norm_l2());
        }

        Some(StepResult {
            step,
            gradient_norm,
        })
    }

    /// Apply step to parameters and evaluate new cost
    fn apply_step_and_evaluate_cost(
        &self,
        step_result: &StepResult,
        state: &mut OptimizationState,
        problem: &problem::Problem,
    ) -> error::ApexResult<CostEvaluation> {
        // Apply parameter updates using manifold operations
        let _step_norm = optimizer::apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost (residual only, no Jacobian needed for step evaluation)
        let new_residual = problem.compute_residual_sparse(&state.variables)?;
        let new_residual_norm = new_residual.norm_l2();
        let new_cost = new_residual_norm * new_residual_norm;

        // Compute cost reduction
        let cost_reduction = state.current_cost - new_cost;

        // Update current cost
        state.current_cost = new_cost;

        Ok(CostEvaluation {
            new_cost,
            cost_reduction,
        })
    }

    /// Log iteration details if verbose mode is enabled
    fn log_iteration(&self, iteration: usize, cost_eval: &CostEvaluation, step_norm: f64) {
        if !self.config.verbose {
            return;
        }

        println!(
            "Iteration {}: cost = {:.6e}, reduction = {:.6e}, step_norm = {:.6e}",
            iteration + 1,
            cost_eval.new_cost,
            cost_eval.cost_reduction,
            step_norm
        );
    }

    /// Create optimization summary
    #[allow(clippy::too_many_arguments)]
    fn create_summary(
        &self,
        initial_cost: f64,
        final_cost: f64,
        iterations: usize,
        max_gradient_norm: f64,
        final_gradient_norm: f64,
        max_parameter_update_norm: f64,
        final_parameter_update_norm: f64,
        total_cost_reduction: f64,
        total_time: time::Duration,
    ) -> GaussNewtonSummary {
        GaussNewtonSummary {
            initial_cost,
            final_cost,
            iterations,
            average_cost_reduction: if iterations > 0 {
                total_cost_reduction / iterations as f64
            } else {
                0.0
            },
            max_gradient_norm,
            final_gradient_norm,
            max_parameter_update_norm,
            final_parameter_update_norm,
            total_time,
            average_time_per_iteration: if iterations > 0 {
                total_time / iterations as u32
            } else {
                time::Duration::from_secs(0)
            },
        }
    }

    pub fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<collections::HashMap<String, problem::VariableEnum>>,
        error::ApexError,
    > {
        let start_time = time::Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1; // Initial cost evaluation
        let mut jacobian_evaluations = 0;

        // Initialize optimization state
        let mut state = self.initialize_optimization_state(problem, initial_params)?;

        // Create linear solver
        let mut linear_solver = self.create_linear_solver();

        // Initialize summary tracking variables
        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;
        let mut final_gradient_norm;
        let mut final_parameter_update_norm;

        // Main optimization loop
        loop {
            // Evaluate residuals and Jacobian
            let (residuals, jacobian) = problem.compute_residual_and_jacobian_sparse(
                &state.variables,
                &state.variable_index_map,
                &state.symbolic_structure,
            )?;
            jacobian_evaluations += 1;

            if self.config.verbose {
                println!("\n=== GAUSS-NEWTON ITERATION {} ===", iteration);
                println!("Current cost: {:.12e}", state.current_cost);
                println!(
                    "Residuals shape: ({}, {})",
                    residuals.nrows(),
                    residuals.ncols()
                );
                println!("Residuals norm: {:.12e}", residuals.norm_l2());
                println!(
                    "Jacobian shape: ({}, {})",
                    jacobian.nrows(),
                    jacobian.ncols()
                );
            }

            // Process Jacobian (apply scaling if enabled)
            let scaled_jacobian = self.process_jacobian(&jacobian, iteration);

            // Compute Gauss-Newton step
            let step_result = match self.compute_gauss_newton_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(error::ApexError::Solver(
                        "Linear solver failed to solve Gauss-Newton system".to_string(),
                    ));
                }
            };

            // Update tracking variables
            max_gradient_norm = max_gradient_norm.max(step_result.gradient_norm);
            final_gradient_norm = step_result.gradient_norm;
            let step_norm = step_result.step.norm_l2();
            max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
            final_parameter_update_norm = step_norm;

            // Apply step and evaluate new cost
            let cost_eval = self.apply_step_and_evaluate_cost(&step_result, &mut state, problem)?;
            cost_evaluations += 1;
            total_cost_reduction += cost_eval.cost_reduction;

            // Log iteration
            self.log_iteration(iteration, &cost_eval, step_norm);

            // Rerun visualization
            if let Some(ref vis) = self.visualizer {
                if let Err(e) = vis.log_scalars(
                    iteration,
                    state.current_cost,
                    step_result.gradient_norm,
                    0.0, // Gauss-Newton doesn't use damping/trust region
                    step_norm,
                    None, // No step quality rho in Gauss-Newton
                ) {
                    eprintln!("[WARNING] Failed to log scalars: {}", e);
                }

                // Log expensive visualizations (Hessian, gradient, manifolds)
                if let Err(e) = vis.log_hessian(linear_solver.get_hessian(), iteration) {
                    eprintln!("[WARNING] Failed to log Hessian: {}", e);
                }
                if let Err(e) = vis.log_gradient(linear_solver.get_gradient(), iteration) {
                    eprintln!("[WARNING] Failed to log gradient: {}", e);
                }
                if let Err(e) = vis.log_manifolds(&state.variables, iteration) {
                    eprintln!("[WARNING] Failed to log manifolds: {}", e);
                }
            }

            // Check convergence
            let elapsed = start_time.elapsed();
            if let Some(status) = self.check_convergence(
                iteration,
                cost_eval.cost_reduction,
                step_norm,
                step_result.gradient_norm,
                elapsed,
            ) {
                let summary = self.create_summary(
                    state.initial_cost,
                    state.current_cost,
                    iteration + 1,
                    max_gradient_norm,
                    final_gradient_norm,
                    max_parameter_update_norm,
                    final_parameter_update_norm,
                    total_cost_reduction,
                    elapsed,
                );

                if self.config.verbose {
                    println!("{}", summary);
                }

                // Compute covariances if enabled
                let covariances = if self.config.compute_covariances {
                    problem.compute_and_set_covariances(
                        &mut linear_solver,
                        &mut state.variables,
                        &state.variable_index_map,
                    )
                } else {
                    None
                };

                return Ok(optimizer::SolverResult {
                    status,
                    iterations: iteration + 1,
                    initial_cost: state.initial_cost,
                    final_cost: state.current_cost,
                    parameters: state.variables.into_iter().collect(),
                    elapsed_time: elapsed,
                    convergence_info: Some(optimizer::ConvergenceInfo {
                        final_gradient_norm,
                        final_parameter_update_norm,
                        cost_evaluations,
                        jacobian_evaluations,
                    }),
                    covariances,
                });
            }

            iteration += 1;
        }
    }
}

impl optimizer::Solver for GaussNewton {
    type Config = GaussNewtonConfig;
    type Error = error::ApexError;

    fn new() -> Self {
        Self::default()
    }

    fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &std::collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<std::collections::HashMap<String, problem::VariableEnum>>,
        Self::Error,
    > {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{factors, problem},
        manifold, optimizer,
    };
    use nalgebra::dvector;
    use std::collections;

    /// Custom Rosenbrock Factor 1: r1 = 10(x2 - x1²)
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor1;

    impl factors::Factor for RosenbrockFactor1 {
        fn linearize(
            &self,
            params: &[nalgebra::DVector<f64>],
            compute_jacobian: bool,
        ) -> (nalgebra::DVector<f64>, Option<nalgebra::DMatrix<f64>>) {
            let x1 = params[0][0];
            let x2 = params[1][0];

            // Residual: r1 = 10(x2 - x1²)
            let residual = nalgebra::dvector![10.0 * (x2 - x1 * x1)];

            // Jacobian: ∂r1/∂x1 = -20*x1, ∂r1/∂x2 = 10
            let jacobian = if compute_jacobian {
                let mut jac = nalgebra::DMatrix::zeros(1, 2);
                jac[(0, 0)] = -20.0 * x1;
                jac[(0, 1)] = 10.0;
                Some(jac)
            } else {
                None
            };

            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
    }

    /// Custom Rosenbrock Factor 2: r2 = 1 - x1
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor2;

    impl factors::Factor for RosenbrockFactor2 {
        fn linearize(
            &self,
            params: &[nalgebra::DVector<f64>],
            compute_jacobian: bool,
        ) -> (nalgebra::DVector<f64>, Option<nalgebra::DMatrix<f64>>) {
            let x1 = params[0][0];

            // Residual: r2 = 1 - x1
            let residual = nalgebra::dvector![1.0 - x1];

            // Jacobian: ∂r2/∂x1 = -1
            let jacobian = if compute_jacobian {
                Some(nalgebra::DMatrix::from_element(1, 1, -1.0))
            } else {
                None
            };

            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_rosenbrock_optimization() {
        // Rosenbrock function test:
        // Minimize: r1² + r2² where
        //   r1 = 10(x2 - x1²)
        //   r2 = 1 - x1
        // Starting point: [-1.2, 1.0]
        // Expected minimum: [1.0, 1.0]

        let mut problem = problem::Problem::new();
        let mut initial_values = collections::HashMap::new();

        // Add variables using Rn manifold (Euclidean space)
        initial_values.insert(
            "x1".to_string(),
            (manifold::ManifoldType::RN, dvector![-1.2]),
        );
        initial_values.insert(
            "x2".to_string(),
            (manifold::ManifoldType::RN, dvector![1.0]),
        );

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Configure Gauss-Newton optimizer
        let config = optimizer::gauss_newton::GaussNewtonConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10);

        let mut solver = optimizer::GaussNewton::with_config(config);
        let result = solver.optimize(&problem, &initial_values).unwrap();

        // Extract final values
        let x1_final = result.parameters.get("x1").unwrap().to_vector()[0];
        let x2_final = result.parameters.get("x2").unwrap().to_vector()[0];

        println!("Rosenbrock optimization result (Gauss-Newton):");
        println!("  Status: {:?}", result.status);
        println!("  Initial cost: {:.6e}", result.initial_cost);
        println!("  Final cost: {:.6e}", result.final_cost);
        println!("  Iterations: {}", result.iterations);
        println!("  x1: {:.6} (expected 1.0)", x1_final);
        println!("  x2: {:.6} (expected 1.0)", x2_final);

        // Verify convergence to [1.0, 1.0]
        assert!(
            matches!(
                result.status,
                optimizer::OptimizationStatus::Converged
                    | optimizer::OptimizationStatus::CostToleranceReached
                    | optimizer::OptimizationStatus::ParameterToleranceReached
                    | optimizer::OptimizationStatus::GradientToleranceReached
            ),
            "Optimization should converge"
        );
        assert!(
            (x1_final - 1.0).abs() < 1e-4,
            "x1 should converge to 1.0, got {}",
            x1_final
        );
        assert!(
            (x2_final - 1.0).abs() < 1e-4,
            "x2 should converge to 1.0, got {}",
            x2_final
        );
        assert!(
            result.final_cost < 1e-6,
            "Final cost should be near zero, got {}",
            result.final_cost
        );
    }
}
