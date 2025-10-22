//! Levenberg-Marquardt algorithm implementation.
//!
//! The Levenberg-Marquardt algorithm is a popular optimization method for
//! nonlinear least squares problems. It interpolates between the Gauss-Newton
//! algorithm and gradient descent by adding a damping parameter.
//!
//! This implementation includes:
//! - Adaptive damping parameter adjustment
//! - Trust region strategy
//! - Robust numerical factorization
//! - Comprehensive optimization summaries
//! - Support for both sparse Cholesky and QR factorizations

use crate::core::problem::{Problem, SymbolicStructure, VariableEnum};
use crate::linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use crate::optimizer::{
    ConvergenceInfo, OptimizationStatus, SolverResult, visualization::OptimizationVisualizer,
};
use faer::{Mat, sparse::SparseColMat};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Summary statistics for the Levenberg-Marquardt optimization process.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardtSummary {
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Number of successful steps (cost decreased)
    pub successful_steps: usize,
    /// Number of unsuccessful steps (cost increased, damping increased)
    pub unsuccessful_steps: usize,
    /// Final damping parameter value
    pub final_damping: f64,
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
    pub total_time: std::time::Duration,
    /// Average time per iteration
    pub average_time_per_iteration: std::time::Duration,
}

impl fmt::Display for LevenbergMarquardtSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Levenberg-Marquardt Optimization Summary")?;
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
            "Successful steps:          {} ({:.1}%)",
            self.successful_steps,
            100.0 * self.successful_steps as f64 / self.iterations.max(1) as f64
        )?;
        writeln!(
            f,
            "Unsuccessful steps:        {} ({:.1}%)",
            self.unsuccessful_steps,
            100.0 * self.unsuccessful_steps as f64 / self.iterations.max(1) as f64
        )?;
        writeln!(f, "Final damping parameter:   {:.6e}", self.final_damping)?;
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

/// Configuration for Levenberg-Marquardt solver.
#[derive(Clone)]
pub struct LevenbergMarquardtConfig {
    /// Type of linear solver for the linear systems
    pub linear_solver_type: LinearSolverType,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for cost function
    pub cost_tolerance: f64,
    /// Convergence tolerance for parameter updates
    pub parameter_tolerance: f64,
    /// Convergence tolerance for gradient norm
    pub gradient_tolerance: f64,
    /// Timeout duration
    pub timeout: Option<Duration>,
    /// Enable detailed logging
    pub verbose: bool,
    /// Initial damping parameter
    pub damping: f64,
    /// Minimum damping parameter
    pub damping_min: f64,
    /// Maximum damping parameter
    pub damping_max: f64,
    /// Damping increase factor (when step rejected)
    pub damping_increase_factor: f64,
    /// Damping decrease factor (when step accepted)
    pub damping_decrease_factor: f64,
    /// Damping nu parameter
    pub damping_nu: f64,
    /// Trust region radius
    pub trust_region_radius: f64,
    /// Minimum step quality for acceptance
    pub min_step_quality: f64,
    /// Good step quality threshold
    pub good_step_quality: f64,
    /// Minimum diagonal value for regularization
    pub min_diagonal: f64,
    /// Maximum diagonal value for regularization
    pub max_diagonal: f64,
    /// Use Jacobi column scaling (preconditioning)
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// This can improve convergence for problems with mixed parameter scales
    /// (e.g., positions in meters + angles in radians) but adds ~5-10% overhead.
    ///
    /// Default: false (to avoid performance overhead and faster convergence)
    pub use_jacobi_scaling: bool,
    /// Compute per-variable covariance matrices (uncertainty estimation)
    ///
    /// When enabled, computes covariance by inverting the Hessian matrix after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and SolverResult.
    ///
    /// Default: false (to avoid performance overhead)
    pub compute_covariances: bool,
    /// Enable real-time Rerun visualization (graphical debugging)
    ///
    /// When enabled, logs optimization progress to Rerun viewer including:
    /// - Time series plots (cost, gradient norm, damping, step quality)
    /// - Sparse Hessian heat map visualization
    /// - Gradient vector visualization
    /// - Manifold state updates (for SE2/SE3 problems)
    ///
    /// This is automatically disabled in release builds (zero overhead).
    /// Use `verbose` for terminal output; this is for graphical visualization.
    ///
    /// Default: false
    pub enable_visualization: bool,
}

impl Default for LevenbergMarquardtConfig {
    fn default() -> Self {
        Self {
            linear_solver_type: LinearSolverType::default(),
            max_iterations: 100,
            cost_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            timeout: None,
            verbose: false,
            damping: 1e-4,
            damping_min: 1e-12,
            damping_max: 1e12,
            damping_increase_factor: 10.0,
            damping_decrease_factor: 0.3,
            damping_nu: 2.0,
            trust_region_radius: 1e4,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            min_diagonal: 1e-6,
            max_diagonal: 1e32,
            use_jacobi_scaling: false,
            compute_covariances: false,
            enable_visualization: false,
        }
    }
}

impl LevenbergMarquardtConfig {
    /// Create a new Levenberg-Marquardt configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the linear solver type
    pub fn with_linear_solver_type(mut self, linear_solver_type: LinearSolverType) -> Self {
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
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enable or disable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the initial damping parameter.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Set the damping parameter bounds.
    pub fn with_damping_bounds(mut self, min: f64, max: f64) -> Self {
        self.damping_min = min;
        self.damping_max = max;
        self
    }

    /// Set the damping adjustment factors.
    pub fn with_damping_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.damping_increase_factor = increase;
        self.damping_decrease_factor = decrease;
        self
    }

    /// Set the trust region parameters.
    pub fn with_trust_region(mut self, radius: f64, min_quality: f64, good_quality: f64) -> Self {
        self.trust_region_radius = radius;
        self.min_step_quality = min_quality;
        self.good_step_quality = good_quality;
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

    /// Enable or disable covariance computation (uncertainty estimation).
    ///
    /// When enabled, computes the full covariance matrix by inverting the Hessian
    /// after convergence, then extracts per-variable covariance blocks.
    pub fn with_compute_covariances(mut self, compute_covariances: bool) -> Self {
        self.compute_covariances = compute_covariances;
        self
    }

    /// Enable real-time visualization (graphical debugging).
    ///
    /// When enabled, optimization progress is logged to a Rerun viewer with:
    /// - Time series plots of cost, gradient norm, damping, step quality
    /// - Sparse Hessian matrix visualization as heat map
    /// - Gradient vector visualization
    /// - Real-time manifold state updates (for SE2/SE3 problems)
    ///
    /// Note: Has zero overhead when disabled. Use `verbose` for terminal logging.
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
    variables: HashMap<String, VariableEnum>,
    variable_index_map: HashMap<String, usize>,
    sorted_vars: Vec<String>,
    symbolic_structure: SymbolicStructure,
    current_cost: f64,
    initial_cost: f64,
}

/// Result from step computation
struct StepResult {
    step: Mat<f64>,
    gradient_norm: f64,
    predicted_reduction: f64,
}

/// Result from step evaluation
struct StepEvaluation {
    accepted: bool,
    cost_reduction: f64,
    rho: f64,
}

/// Levenberg-Marquardt solver for nonlinear least squares optimization.
pub struct LevenbergMarquardt {
    config: LevenbergMarquardtConfig,
    jacobi_scaling: Option<SparseColMat<usize, f64>>,
    visualizer: Option<OptimizationVisualizer>,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenbergMarquardt {
    /// Create a new Levenberg-Marquardt solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(LevenbergMarquardtConfig::default())
    }

    /// Create a new Levenberg-Marquardt solver with the given configuration.
    pub fn with_config(config: LevenbergMarquardtConfig) -> Self {
        // Create visualizer if enabled (zero overhead when disabled)
        let visualizer = if config.enable_visualization {
            match OptimizationVisualizer::new(true) {
                Ok(vis) => Some(vis),
                Err(e) => {
                    eprintln!("[WARNING] Failed to create visualizer: {}", e);
                    None
                }
            }
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
    fn create_linear_solver(&self) -> Box<dyn SparseLinearSolver> {
        match self.config.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        }
    }

    /// Update damping parameter based on step quality using trust region approach
    /// Reference: Introduction to Optimization and Data Fitting
    /// Algorithm 6.18
    fn update_damping(&mut self, rho: f64) -> bool {
        if rho > 0.0 {
            // Step accepted - decrease damping
            let coff = 2.0 * rho - 1.0;
            self.config.damping *= (1.0_f64 / 3.0).max(1.0 - coff * coff * coff);
            self.config.damping = self.config.damping.max(self.config.damping_min);
            self.config.damping_nu = 2.0;
            true
        } else {
            // Step rejected - increase damping
            self.config.damping *= self.config.damping_nu;
            self.config.damping_nu *= 2.0;
            self.config.damping = self.config.damping.min(self.config.damping_max);
            false
        }
    }

    /// Compute step quality ratio (actual vs predicted reduction)
    fn compute_step_quality(
        &self,
        current_cost: f64,
        new_cost: f64,
        predicted_reduction: f64,
    ) -> f64 {
        let actual_reduction = current_cost - new_cost;
        if predicted_reduction.abs() < 1e-15 {
            if actual_reduction > 0.0 { 1.0 } else { 0.0 }
        } else {
            actual_reduction / predicted_reduction
        }
    }

    /// Compute predicted cost reduction from linear model
    /// Standard LM formula: 0.5 * step^T * (damping * step - gradient)
    fn compute_predicted_reduction(&self, step: &Mat<f64>, gradient: &Mat<f64>) -> f64 {
        // Standard Levenberg-Marquardt predicted reduction formula
        // predicted_reduction = -step^T * gradient - 0.5 * step^T * H * step
        //                     = 0.5 * step^T * (damping * step - gradient)
        let diff = self.config.damping * step - gradient;
        (0.5 * step.transpose() * &diff)[(0, 0)]
    }

    /// Check convergence criteria
    fn check_convergence(
        &self,
        iteration: usize,
        cost_change: f64,
        parameter_update_norm: f64,
        gradient_norm: f64,
        elapsed: std::time::Duration,
    ) -> Option<OptimizationStatus> {
        // Check timeout
        if let Some(timeout) = self.config.timeout
            && elapsed >= timeout
        {
            return Some(OptimizationStatus::Timeout);
        }

        // Check maximum iterations
        if iteration >= self.config.max_iterations {
            return Some(OptimizationStatus::MaxIterationsReached);
        }

        // Check cost tolerance
        // TODO: Implement cost tolerance check
        if iteration > 0 && cost_change.abs() < self.config.cost_tolerance {
            return Some(OptimizationStatus::CostToleranceReached);
        }

        // Check parameter tolerance
        // TODO: Implement parameter tolerance check
        if iteration > 0 && parameter_update_norm < self.config.parameter_tolerance {
            return Some(OptimizationStatus::ParameterToleranceReached);
        }

        // Check gradient tolerance
        if gradient_norm < self.config.gradient_tolerance {
            return Some(OptimizationStatus::GradientToleranceReached);
        }

        None
    }

    /// Create Jacobi scaling matrix from Jacobian
    fn create_jacobi_scaling(
        &self,
        jacobian: &SparseColMat<usize, f64>,
    ) -> SparseColMat<usize, f64> {
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

        SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
            .expect("Failed to create Jacobi scaling matrix")
    }

    /// Apply Jacobi scaling to Jacobian: J_scaled = J * S
    // fn apply_jacobi_scaling(
    //     &self,
    //     jacobian: &SparseColMat<usize, f64>,
    //     scaling: &SparseColMat<usize, f64>,
    // ) -> SparseColMat<usize, f64> {
    //     jacobian * scaling
    // }

    /// Apply inverse Jacobi scaling to step: dx_final = S * dx_scaled
    fn apply_inverse_jacobi_scaling(
        &self,
        step: &Mat<f64>,
        scaling: &SparseColMat<usize, f64>,
    ) -> Mat<f64> {
        scaling * step
    }

    /// Initialize optimization state from problem and initial parameters
    fn initialize_optimization_state(
        &self,
        problem: &Problem,
        initial_params: &HashMap<String, (crate::manifold::ManifoldType, nalgebra::DVector<f64>)>,
    ) -> Result<OptimizationState, crate::core::ApexError> {
        // Initialize variables from initial values
        let variables = problem.initialize_variables(initial_params);

        // Create column mapping for variables
        let mut variable_index_map = HashMap::new();
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

        // Initial cost evaluation using sparse interface
        let (residual, _) = problem.compute_residual_and_jacobian_sparse(
            &variables,
            &variable_index_map,
            &symbolic_structure,
        )?;

        let residual_norm = residual.norm_l2();
        let current_cost = residual_norm * residual_norm;
        let initial_cost = current_cost;

        if self.config.verbose {
            println!(
                "Starting Levenberg-Marquardt optimization with {} max iterations",
                self.config.max_iterations
            );
            println!(
                "Initial cost: {:.6e}, initial damping: {:.6e}",
                current_cost, self.config.damping
            );
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
        jacobian: &SparseColMat<usize, f64>,
        iteration: usize,
    ) -> SparseColMat<usize, f64> {
        // Create Jacobi scaling on first iteration if enabled
        if iteration == 0 {
            let scaling = self.create_jacobi_scaling(jacobian);

            if self.config.verbose {
                println!("Jacobi scaling computed for {} columns", scaling.ncols());
            }

            self.jacobi_scaling = Some(scaling);
        }

        // self.apply_jacobi_scaling(jacobian, self.jacobi_scaling.as_ref().unwrap())
        jacobian * self.jacobi_scaling.as_ref().unwrap()
    }

    /// Compute optimization step by solving the augmented system
    fn compute_optimization_step(
        &self,
        residuals: &Mat<f64>,
        scaled_jacobian: &SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Solve augmented equation: (J_scaled^T * J_scaled + λI) * dx_scaled = -J_scaled^T * r
        let residuals_owned = residuals.as_ref().to_owned();
        let scaled_step = linear_solver
            .solve_augmented_equation(&residuals_owned, scaled_jacobian, self.config.damping)
            .ok()?;

        // Get cached gradient and Hessian from the solver
        let gradient = linear_solver.get_gradient()?;
        let hessian = linear_solver.get_hessian()?;
        let gradient_norm = gradient.norm_l2();

        if self.config.verbose {
            println!("Gradient (J^T*r) norm: {:.12e}", gradient_norm);
            println!("Hessian shape: ({}, {})", hessian.nrows(), hessian.ncols());
            println!("Damping parameter: {:.12e}", self.config.damping);
        }

        // Apply inverse Jacobi scaling to get final step (if enabled)
        let step = if self.config.use_jacobi_scaling {
            self.apply_inverse_jacobi_scaling(&scaled_step, self.jacobi_scaling.as_ref().unwrap())
        } else {
            scaled_step.clone()
        };

        if self.config.verbose {
            println!(
                "Linear step (scaled_step) norm: {:.12e}",
                scaled_step.norm_l2()
            );
            println!("Final step norm: {:.12e}", step.norm_l2());
        }

        // Compute predicted reduction using scaled values
        let predicted_reduction = self.compute_predicted_reduction(&scaled_step, &gradient);

        if self.config.verbose {
            println!("Predicted reduction: {:.12e}", predicted_reduction);
        }

        Some(StepResult {
            step,
            gradient_norm,
            predicted_reduction,
        })
    }

    /// Evaluate and apply step, handling acceptance/rejection based on step quality
    fn evaluate_and_apply_step(
        &mut self,
        step_result: &StepResult,
        state: &mut OptimizationState,
        problem: &Problem,
    ) -> crate::core::ApexResult<StepEvaluation> {
        // Apply parameter updates using manifold operations
        let _step_norm = crate::optimizer::apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost using sparse interface
        let (new_residual, _) = problem.compute_residual_and_jacobian_sparse(
            &state.variables,
            &state.variable_index_map,
            &state.symbolic_structure,
        )?;
        let new_residual_norm = new_residual.norm_l2();
        let new_cost = new_residual_norm * new_residual_norm;

        // Compute step quality
        let rho = self.compute_step_quality(
            state.current_cost,
            new_cost,
            step_result.predicted_reduction,
        );

        if self.config.verbose {
            println!("=== RHO CALCULATION DETAILS ===");
            println!("Old cost: {:.12e}", state.current_cost);
            println!("New cost: {:.12e}", new_cost);
            let actual_reduction = state.current_cost - new_cost;
            println!("Actual cost reduction: {:.12e}", actual_reduction);
            println!(
                "Predicted cost reduction: {:.12e}",
                step_result.predicted_reduction
            );
            println!("Rho (actual/predicted): {:.12e}", rho);
        }

        // Update damping and decide whether to accept step
        let accepted = self.update_damping(rho);

        let cost_reduction = if accepted {
            // Accept the step - parameters already updated
            let reduction = state.current_cost - new_cost;
            state.current_cost = new_cost;
            reduction
        } else {
            // Reject the step - revert parameter changes
            crate::optimizer::apply_negative_parameter_step(
                &mut state.variables,
                step_result.step.as_ref(),
                &state.sorted_vars,
            );
            0.0
        };

        Ok(StepEvaluation {
            accepted,
            cost_reduction,
            rho,
        })
    }

    /// Create optimization summary
    #[allow(clippy::too_many_arguments)]
    fn create_summary(
        &self,
        initial_cost: f64,
        final_cost: f64,
        iterations: usize,
        successful_steps: usize,
        unsuccessful_steps: usize,
        max_gradient_norm: f64,
        final_gradient_norm: f64,
        max_parameter_update_norm: f64,
        final_parameter_update_norm: f64,
        total_cost_reduction: f64,
        total_time: std::time::Duration,
    ) -> LevenbergMarquardtSummary {
        LevenbergMarquardtSummary {
            initial_cost,
            final_cost,
            iterations,
            successful_steps,
            unsuccessful_steps,
            final_damping: self.config.damping,
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
                std::time::Duration::from_secs(0)
            },
        }
    }

    pub fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &std::collections::HashMap<
            String,
            (crate::manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<SolverResult<std::collections::HashMap<String, VariableEnum>>, crate::core::ApexError>
    {
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1; // Initial cost evaluation
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;

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
                println!("\n=== APEX-SOLVER DEBUG ITERATION {} ===", iteration);
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
            let scaled_jacobian = if self.config.use_jacobi_scaling {
                self.process_jacobian(&jacobian, iteration)
            } else {
                jacobian
            };

            if self.config.verbose {
                println!(
                    "Scaled Jacobian shape: ({}, {})",
                    scaled_jacobian.nrows(),
                    scaled_jacobian.ncols()
                );
            }

            // Compute optimization step
            let step_result = match self.compute_optimization_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(crate::core::ApexError::Solver(
                        "Linear solver failed to solve augmented system".to_string(),
                    ));
                }
            };

            // Update tracking variables
            max_gradient_norm = max_gradient_norm.max(step_result.gradient_norm);
            final_gradient_norm = step_result.gradient_norm;
            let step_norm = step_result.step.norm_l2();
            max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
            final_parameter_update_norm = step_norm;

            // Evaluate and apply step (handles accept/reject)
            let step_eval = self.evaluate_and_apply_step(&step_result, &mut state, problem)?;
            cost_evaluations += 1;

            // Update counters based on acceptance
            if step_eval.accepted {
                successful_steps += 1;
                total_cost_reduction += step_eval.cost_reduction;
            } else {
                unsuccessful_steps += 1;
            }

            // Rerun visualization
            if let Some(ref vis) = self.visualizer {
                if let Err(e) = vis.log_scalars(
                    iteration,
                    state.current_cost,
                    step_result.gradient_norm,
                    self.config.damping,
                    step_norm,
                    Some(step_eval.rho),
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

            // Check convergence (only after accepted steps)
            if step_eval.accepted {
                let elapsed = start_time.elapsed();
                if let Some(status) = self.check_convergence(
                    iteration,
                    step_eval.cost_reduction,
                    step_norm,
                    step_result.gradient_norm,
                    elapsed,
                ) {
                    let summary = self.create_summary(
                        state.initial_cost,
                        state.current_cost,
                        iteration + 1,
                        successful_steps,
                        unsuccessful_steps,
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

                    // Log convergence to Rerun
                    if let Some(ref vis) = self.visualizer {
                        let _ = vis.log_convergence(&format!("Converged: {}", status));
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

                    return Ok(SolverResult {
                        status,
                        iterations: iteration + 1,
                        initial_cost: state.initial_cost,
                        final_cost: state.current_cost,
                        parameters: state.variables.into_iter().collect(),
                        elapsed_time: elapsed,
                        convergence_info: Some(ConvergenceInfo {
                            final_gradient_norm,
                            final_parameter_update_norm,
                            cost_evaluations,
                            jacobian_evaluations,
                        }),
                        covariances,
                    });
                }
            }

            // Check max iterations
            let elapsed = start_time.elapsed();
            if iteration >= self.config.max_iterations {
                let summary = self.create_summary(
                    state.initial_cost,
                    state.current_cost,
                    iteration,
                    successful_steps,
                    unsuccessful_steps,
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

                return Ok(SolverResult {
                    parameters: state.variables,
                    status: OptimizationStatus::MaxIterationsReached,
                    initial_cost: state.initial_cost,
                    final_cost: state.current_cost,
                    iterations: iteration,
                    elapsed_time: elapsed,
                    convergence_info: Some(ConvergenceInfo {
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
// Implement Solver trait
impl crate::optimizer::Solver for LevenbergMarquardt {
    type Config = LevenbergMarquardtConfig;
    type Error = crate::core::ApexError;

    fn new() -> Self {
        Self::default()
    }

    fn optimize(
        &mut self,
        problem: &crate::core::problem::Problem,
        initial_params: &std::collections::HashMap<
            String,
            (crate::manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        crate::optimizer::SolverResult<
            std::collections::HashMap<String, crate::core::problem::VariableEnum>,
        >,
        Self::Error,
    > {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factors::Factor;
    use crate::core::problem::Problem;
    use crate::manifold::ManifoldType;
    use nalgebra::{DMatrix, DVector, dvector};
    use std::collections::HashMap;

    /// Custom Rosenbrock Factor 1: r1 = 10(x2 - x1²)
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor1;

    impl Factor for RosenbrockFactor1 {
        fn linearize(&self, params: &[DVector<f64>]) -> (DVector<f64>, DMatrix<f64>) {
            let x1 = params[0][0];
            let x2 = params[1][0];

            // Residual: r1 = 10(x2 - x1²)
            let residual = dvector![10.0 * (x2 - x1 * x1)];

            // Jacobian: ∂r1/∂x1 = -20*x1, ∂r1/∂x2 = 10
            let mut jacobian = DMatrix::zeros(1, 2);
            jacobian[(0, 0)] = -20.0 * x1;
            jacobian[(0, 1)] = 10.0;

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

    impl Factor for RosenbrockFactor2 {
        fn linearize(&self, params: &[DVector<f64>]) -> (DVector<f64>, DMatrix<f64>) {
            let x1 = params[0][0];

            // Residual: r2 = 1 - x1
            let residual = dvector![1.0 - x1];

            // Jacobian: ∂r2/∂x1 = -1
            let jacobian = DMatrix::from_element(1, 1, -1.0);

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

        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();

        // Add variables using Rn manifold (Euclidean space)
        initial_values.insert("x1".to_string(), (ManifoldType::RN, dvector![-1.2]));
        initial_values.insert("x2".to_string(), (ManifoldType::RN, dvector![1.0]));

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Configure Levenberg-Marquardt optimizer
        let config = LevenbergMarquardtConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10);

        let mut solver = LevenbergMarquardt::with_config(config);
        let result = solver.optimize(&problem, &initial_values).unwrap();

        // Extract final values
        let x1_final = result.parameters.get("x1").unwrap().to_vector()[0];
        let x2_final = result.parameters.get("x2").unwrap().to_vector()[0];

        println!("Rosenbrock optimization result:");
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
                OptimizationStatus::Converged
                    | OptimizationStatus::CostToleranceReached
                    | OptimizationStatus::ParameterToleranceReached
                    | OptimizationStatus::GradientToleranceReached
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
