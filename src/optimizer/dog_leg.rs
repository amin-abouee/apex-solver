//! Dog Leg optimization algorithm implementation.
//!
//! The Dog Leg algorithm is a trust region method that combines the Gauss-Newton
//! direction with the steepest descent direction to find an optimal step within
//! a trust region.
//!
//! This implementation includes:
//! - Trust region management with adaptive radius adjustment
//! - Dog leg step computation (interpolation between steepest descent and Gauss-Newton)
//! - Robust numerical factorization
//! - Comprehensive optimization summaries
//! - Support for both sparse Cholesky and QR factorizations

use crate::core::problem::{Problem, VariableEnum};
use crate::linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use crate::optimizer::{ConvergenceInfo, OptimizationStatus, SolverResult};
use faer::{Mat, sparse::SparseColMat};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Summary statistics for the Dog Leg optimization process.
#[derive(Debug, Clone)]
pub struct DogLegSummary {
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Number of successful steps (cost decreased)
    pub successful_steps: usize,
    /// Number of unsuccessful steps (cost increased, step rejected)
    pub unsuccessful_steps: usize,
    /// Final trust region radius
    pub final_trust_region_radius: f64,
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
    pub total_time: Duration,
    /// Average time per iteration
    pub average_time_per_iteration: Duration,
}

impl fmt::Display for DogLegSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Dog Leg Optimization Summary ===")?;
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
        writeln!(
            f,
            "Final trust region radius: {:.6e}",
            self.final_trust_region_radius
        )?;
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

/// Configuration for Dog Leg solver.
#[derive(Clone)]
pub struct DogLegConfig {
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
    /// Initial trust region radius
    pub trust_region_radius: f64,
    /// Minimum trust region radius
    pub trust_region_min: f64,
    /// Maximum trust region radius
    pub trust_region_max: f64,
    /// Trust region increase factor (for good steps, rho > 0.75)
    pub trust_region_increase_factor: f64,
    /// Trust region decrease factor (for poor steps, rho < 0.25)
    pub trust_region_decrease_factor: f64,
    /// Minimum step quality for acceptance (typically 0.0)
    pub min_step_quality: f64,
    /// Good step quality threshold (typically 0.75)
    pub good_step_quality: f64,
    /// Poor step quality threshold (typically 0.25)
    pub poor_step_quality: f64,
    /// Use Jacobi column scaling (preconditioning)
    pub use_jacobi_scaling: bool,
}

impl Default for DogLegConfig {
    fn default() -> Self {
        Self {
            linear_solver_type: LinearSolverType::default(),
            max_iterations: 100,
            cost_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            timeout: None,
            verbose: false,
            trust_region_radius: 1.0,
            trust_region_min: 1e-12,
            trust_region_max: 1e12,
            trust_region_increase_factor: 2.0,
            trust_region_decrease_factor: 0.5,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            poor_step_quality: 0.25,
            use_jacobi_scaling: false,
        }
    }
}

impl DogLegConfig {
    /// Create a new Dog Leg configuration with default values.
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

    /// Set the initial trust region radius
    pub fn with_trust_region_radius(mut self, radius: f64) -> Self {
        self.trust_region_radius = radius;
        self
    }

    /// Set the trust region radius bounds
    pub fn with_trust_region_bounds(mut self, min: f64, max: f64) -> Self {
        self.trust_region_min = min;
        self.trust_region_max = max;
        self
    }

    /// Set the trust region adjustment factors
    pub fn with_trust_region_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.trust_region_increase_factor = increase;
        self.trust_region_decrease_factor = decrease;
        self
    }

    /// Set the trust region quality thresholds
    pub fn with_step_quality_thresholds(
        mut self,
        min_quality: f64,
        poor_quality: f64,
        good_quality: f64,
    ) -> Self {
        self.min_step_quality = min_quality;
        self.poor_step_quality = poor_quality;
        self.good_step_quality = good_quality;
        self
    }

    /// Enable or disable Jacobi column scaling (preconditioning)
    pub fn with_jacobi_scaling(mut self, use_jacobi_scaling: bool) -> Self {
        self.use_jacobi_scaling = use_jacobi_scaling;
        self
    }
}

/// State for optimization iteration
#[allow(dead_code)]
struct OptimizationState {
    variables: HashMap<String, VariableEnum>,
    variable_index_map: HashMap<String, usize>,
    sorted_vars: Vec<String>,
    symbolic_structure: crate::core::problem::SymbolicStructure,
    current_cost: f64,
    initial_cost: f64,
}

/// Result from step computation
#[allow(dead_code)]
struct StepResult {
    step: Mat<f64>,
    scaled_step: Mat<f64>,
    gradient: Mat<f64>,
    gradient_norm: f64,
    predicted_reduction: f64,
    step_type: StepType,
}

/// Type of step taken
#[derive(Debug, Clone, Copy)]
enum StepType {
    /// Full Gauss-Newton step
    GaussNewton,
    /// Scaled steepest descent (Cauchy point)
    SteepestDescent,
    /// Dog leg interpolation
    DogLeg,
}

impl fmt::Display for StepType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepType::GaussNewton => write!(f, "GN"),
            StepType::SteepestDescent => write!(f, "SD"),
            StepType::DogLeg => write!(f, "DL"),
        }
    }
}

/// Result from step evaluation
#[allow(dead_code)]
struct StepEvaluation {
    accepted: bool,
    new_cost: f64,
    cost_reduction: f64,
    rho: f64,
}

/// Dog Leg solver for nonlinear least squares optimization.
pub struct DogLeg {
    config: DogLegConfig,
    jacobi_scaling: Option<SparseColMat<usize, f64>>,
}

impl Default for DogLeg {
    fn default() -> Self {
        Self::new()
    }
}

impl DogLeg {
    /// Create a new Dog Leg solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(DogLegConfig::default())
    }

    /// Create a new Dog Leg solver with the given configuration.
    pub fn with_config(config: DogLegConfig) -> Self {
        Self {
            config,
            jacobi_scaling: None,
        }
    }

    /// Set the initial trust region radius
    pub fn with_trust_region_radius(mut self, radius: f64) -> Self {
        self.config.trust_region_radius = radius;
        self
    }

    /// Set the trust region radius bounds
    pub fn with_trust_region_bounds(mut self, min: f64, max: f64) -> Self {
        self.config.trust_region_min = min;
        self.config.trust_region_max = max;
        self
    }

    /// Set the trust region adjustment factors
    pub fn with_trust_region_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.config.trust_region_increase_factor = increase;
        self.config.trust_region_decrease_factor = decrease;
        self
    }

    /// Create the appropriate linear solver based on configuration
    fn create_linear_solver(&self) -> Box<dyn SparseLinearSolver> {
        match self.config.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        }
    }

    /// Compute the Cauchy point (steepest descent step)
    /// Returns the optimal step along the negative gradient direction
    fn compute_cauchy_point(
        &self,
        gradient: &Mat<f64>,
        hessian: &SparseColMat<usize, f64>,
    ) -> Mat<f64> {
        // Optimal step size along steepest descent: α = (g^T*g) / (g^T*H*g)
        let g_norm_sq_mat = gradient.transpose() * gradient;
        let g_norm_sq = g_norm_sq_mat[(0, 0)];

        let h_g = hessian * gradient;
        let g_h_g_mat = gradient.transpose() * &h_g;
        let g_h_g = g_h_g_mat[(0, 0)];

        // Avoid division by zero
        let alpha = if g_h_g.abs() > 1e-15 {
            g_norm_sq / g_h_g
        } else {
            1.0
        };

        // Return steepest descent step: -α * gradient
        let mut cauchy = Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            cauchy[(i, 0)] = -alpha * gradient[(i, 0)];
        }
        cauchy
    }

    /// Compute the dog leg step by interpolating between Cauchy point and Gauss-Newton step
    fn compute_dog_leg_step(
        &self,
        h_sd: &Mat<f64>,
        h_gn: &Mat<f64>,
        delta: f64,
    ) -> (Mat<f64>, StepType) {
        let gn_norm = h_gn.norm_l2();
        let sd_norm = h_sd.norm_l2();

        if gn_norm <= delta {
            // Full Gauss-Newton step fits in trust region
            (h_gn.clone(), StepType::GaussNewton)
        } else if sd_norm >= delta {
            // Scale steepest descent to trust region boundary
            let mut scaled_sd = Mat::zeros(h_sd.nrows(), 1);
            let scale = delta / sd_norm;
            for i in 0..h_sd.nrows() {
                scaled_sd[(i, 0)] = h_sd[(i, 0)] * scale;
            }
            (scaled_sd, StepType::SteepestDescent)
        } else {
            // Compute dog leg: h_sd + β*(h_gn - h_sd)
            // where β is chosen so ||dog_leg|| = delta
            // We need to solve: ||h_sd + β*(h_gn - h_sd)||^2 = delta^2

            // Let v = h_gn - h_sd
            let mut v = Mat::zeros(h_sd.nrows(), 1);
            for i in 0..h_sd.nrows() {
                v[(i, 0)] = h_gn[(i, 0)] - h_sd[(i, 0)];
            }

            // Quadratic equation: ||h_sd + β*v||^2 = delta^2
            // Expanding: h_sd^T*h_sd + 2*β*h_sd^T*v + β^2*v^T*v = delta^2
            // β^2*(v^T*v) + β*(2*h_sd^T*v) + (h_sd^T*h_sd - delta^2) = 0

            let a = v.transpose() * &v;
            let b_mat = h_sd.transpose() * &v;
            let c = sd_norm * sd_norm - delta * delta;

            let a_val = a[(0, 0)];
            let b_val = 2.0 * b_mat[(0, 0)];
            let c_val = c;

            // Solve quadratic: a*β^2 + b*β + c = 0
            let discriminant = b_val * b_val - 4.0 * a_val * c_val;
            let beta = if discriminant >= 0.0 && a_val.abs() > 1e-15 {
                // Take the positive root (we want β ∈ [0, 1])
                (-b_val + discriminant.sqrt()) / (2.0 * a_val)
            } else {
                1.0
            };

            // Clamp beta to [0, 1]
            let beta = beta.max(0.0).min(1.0);

            // Compute dog leg step
            let mut dog_leg = Mat::zeros(h_sd.nrows(), 1);
            for i in 0..h_sd.nrows() {
                dog_leg[(i, 0)] = h_sd[(i, 0)] + beta * v[(i, 0)];
            }

            (dog_leg, StepType::DogLeg)
        }
    }

    /// Update trust region radius based on step quality
    fn update_trust_region(&mut self, rho: f64, step_norm: f64) -> bool {
        if rho > self.config.good_step_quality {
            // Good step, increase trust region
            self.config.trust_region_radius = (self.config.trust_region_radius
                * self.config.trust_region_increase_factor)
                .min(self.config.trust_region_max);
            true
        } else if rho < self.config.poor_step_quality {
            // Poor step, decrease trust region
            self.config.trust_region_radius = (step_norm
                * self.config.trust_region_decrease_factor)
                .max(self.config.trust_region_min);
            false
        } else {
            // Moderate step, keep trust region unchanged
            true
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
    fn compute_predicted_reduction(
        &self,
        step: &Mat<f64>,
        gradient: &Mat<f64>,
        hessian: &SparseColMat<usize, f64>,
    ) -> f64 {
        // Dog Leg predicted reduction: -step^T * gradient - 0.5 * step^T * H * step
        let linear_term = step.transpose() * gradient;
        let hessian_step = hessian * step;
        let quadratic_term = step.transpose() * &hessian_step;

        -linear_term[(0, 0)] - 0.5 * quadratic_term[(0, 0)]
    }

    /// Check convergence criteria
    fn check_convergence(
        &self,
        iteration: usize,
        cost_change: f64,
        parameter_update_norm: f64,
        gradient_norm: f64,
        elapsed: Duration,
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
        if iteration > 0 && cost_change.abs() < self.config.cost_tolerance {
            return Some(OptimizationStatus::CostToleranceReached);
        }

        // Check parameter tolerance
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
                let col_norm_squared: f64 = jacobian
                    .triplet_iter()
                    .filter(|t| t.col == c)
                    .map(|t| t.val * t.val)
                    .sum();
                let col_norm = col_norm_squared.sqrt();
                let scaling = 1.0 / (1.0 + col_norm);
                Triplet::new(c, c, scaling)
            })
            .collect();

        SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
            .expect("Failed to create Jacobi scaling matrix")
    }

    /// Apply Jacobi scaling to Jacobian
    fn apply_jacobi_scaling(
        &self,
        jacobian: &SparseColMat<usize, f64>,
        scaling: &SparseColMat<usize, f64>,
    ) -> SparseColMat<usize, f64> {
        jacobian * scaling
    }

    /// Apply inverse Jacobi scaling to step
    fn apply_inverse_jacobi_scaling(
        &self,
        step: &Mat<f64>,
        scaling: &SparseColMat<usize, f64>,
    ) -> Mat<f64> {
        scaling * step
    }

    /// Initialize optimization state
    fn initialize_optimization_state(
        &self,
        problem: &Problem,
        initial_params: &HashMap<String, (crate::manifold::ManifoldType, nalgebra::DVector<f64>)>,
    ) -> Result<OptimizationState, crate::core::ApexError> {
        let variables = problem.initialize_variables(initial_params);

        let mut variable_index_map = HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<String> = variables.keys().cloned().collect();
        sorted_vars.sort();

        for var_name in &sorted_vars {
            variable_index_map.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        let symbolic_structure =
            problem.build_symbolic_structure(&variables, &variable_index_map, col_offset)?;

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
                "Starting Dog Leg optimization with {} max iterations",
                self.config.max_iterations
            );
            println!(
                "Initial cost: {:.6e}, initial trust region radius: {:.6e}",
                current_cost, self.config.trust_region_radius
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

    /// Process Jacobian (apply scaling if enabled)
    fn process_jacobian(
        &mut self,
        jacobian: &SparseColMat<usize, f64>,
        iteration: usize,
    ) -> SparseColMat<usize, f64> {
        if iteration == 0 && self.config.use_jacobi_scaling {
            let scaling = self.create_jacobi_scaling(jacobian);
            if self.config.verbose {
                println!("Jacobi scaling computed for {} columns", scaling.ncols());
            }
            self.jacobi_scaling = Some(scaling);
        }

        if self.config.use_jacobi_scaling {
            self.apply_jacobi_scaling(jacobian, self.jacobi_scaling.as_ref().unwrap())
        } else {
            jacobian.clone()
        }
    }

    /// Compute dog leg optimization step
    fn compute_optimization_step(
        &self,
        residuals: &Mat<f64>,
        scaled_jacobian: &SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Solve for Gauss-Newton step: (J^T*J + λI) * h_gn = -J^T*r
        // Use tiny damping for numerical stability (effectively Gauss-Newton)
        let residuals_owned = residuals.as_ref().to_owned();
        let scaled_gn_step = linear_solver
            .solve_augmented_equation(&residuals_owned, scaled_jacobian, 1e-12)
            .ok()?;

        // Get gradient and Hessian (cached by solve_augmented_equation)
        let solver_gradient = linear_solver.get_gradient()?;
        let hessian = linear_solver.get_hessian()?;
        let gradient = -solver_gradient;
        let gradient_norm = gradient.norm_l2();

        // Compute Cauchy point (steepest descent step)
        let cauchy_step = self.compute_cauchy_point(&gradient, hessian);

        // Compute dog leg step based on trust region radius
        let (scaled_step, step_type) = self.compute_dog_leg_step(
            &cauchy_step,
            &scaled_gn_step,
            self.config.trust_region_radius,
        );

        // Apply inverse Jacobi scaling if enabled
        let step = if self.config.use_jacobi_scaling {
            self.apply_inverse_jacobi_scaling(&scaled_step, self.jacobi_scaling.as_ref().unwrap())
        } else {
            scaled_step.clone()
        };

        // Compute predicted reduction
        let predicted_reduction =
            self.compute_predicted_reduction(&scaled_step, &gradient, hessian);

        if self.config.verbose {
            println!("Gradient norm: {:.12e}", gradient_norm);
            println!("Step type: {}", step_type);
            println!("Step norm: {:.12e}", step.norm_l2());
            println!("Predicted reduction: {:.12e}", predicted_reduction);
        }

        Some(StepResult {
            step,
            scaled_step,
            gradient,
            gradient_norm,
            predicted_reduction,
            step_type,
        })
    }

    /// Evaluate and apply step
    fn evaluate_and_apply_step(
        &mut self,
        step_result: &StepResult,
        state: &mut OptimizationState,
        problem: &Problem,
    ) -> crate::core::ApexResult<StepEvaluation> {
        // Apply parameter updates
        let step_norm = crate::optimizer::apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost
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
            println!("=== STEP QUALITY EVALUATION ===");
            println!("Old cost: {:.12e}", state.current_cost);
            println!("New cost: {:.12e}", new_cost);
            println!("Actual reduction: {:.12e}", state.current_cost - new_cost);
            println!(
                "Predicted reduction: {:.12e}",
                step_result.predicted_reduction
            );
            println!("Rho: {:.12e}", rho);
        }

        // Update trust region and decide acceptance
        let accepted = rho > self.config.min_step_quality;
        let trust_region_updated = self.update_trust_region(rho, step_norm);

        let cost_reduction = if accepted {
            let reduction = state.current_cost - new_cost;
            state.current_cost = new_cost;
            reduction
        } else {
            // Reject step - revert changes
            crate::optimizer::apply_negative_parameter_step(
                &mut state.variables,
                step_result.step.as_ref(),
                &state.sorted_vars,
            );
            0.0
        };

        if self.config.verbose {
            println!(
                "Step {}, Trust region updated: {}, New radius: {:.6e}",
                if accepted { "ACCEPTED" } else { "REJECTED" },
                trust_region_updated,
                self.config.trust_region_radius
            );
        }

        Ok(StepEvaluation {
            accepted,
            new_cost,
            cost_reduction,
            rho,
        })
    }

    /// Log iteration details
    fn log_iteration(
        &self,
        iteration: usize,
        step_eval: &StepEvaluation,
        step_norm: f64,
        step_type: StepType,
        current_cost: f64,
    ) {
        if !self.config.verbose {
            return;
        }

        let status = if step_eval.accepted {
            "[ACCEPTED]"
        } else {
            "[REJECTED]"
        };

        println!(
            "Iteration {}: cost = {:.6e}, reduction = {:.6e}, radius = {:.6e}, step_norm = {:.6e}, rho = {:.3}, type = {} {}",
            iteration + 1,
            current_cost,
            step_eval.cost_reduction,
            self.config.trust_region_radius,
            step_norm,
            step_eval.rho,
            step_type,
            status
        );
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
        total_time: Duration,
    ) -> DogLegSummary {
        DogLegSummary {
            initial_cost,
            final_cost,
            iterations,
            successful_steps,
            unsuccessful_steps,
            final_trust_region_radius: self.config.trust_region_radius,
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
                Duration::from_secs(0)
            },
        }
    }

    /// Minimize the optimization problem using Dog Leg algorithm
    pub fn minimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (crate::manifold::ManifoldType, nalgebra::DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, crate::core::ApexError> {
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1;
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;

        let mut state = self.initialize_optimization_state(problem, initial_params)?;
        let mut linear_solver = self.create_linear_solver();

        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;
        let mut final_gradient_norm;
        let mut final_parameter_update_norm;

        loop {
            // Evaluate residuals and Jacobian
            let (residuals, jacobian) = problem.compute_residual_and_jacobian_sparse(
                &state.variables,
                &state.variable_index_map,
                &state.symbolic_structure,
            )?;
            jacobian_evaluations += 1;

            if self.config.verbose {
                println!("\n=== DOG LEG ITERATION {} ===", iteration);
                println!("Current cost: {:.12e}", state.current_cost);
                println!(
                    "Trust region radius: {:.12e}",
                    self.config.trust_region_radius
                );
            }

            // Process Jacobian
            let scaled_jacobian = self.process_jacobian(&jacobian, iteration);

            // Compute dog leg step
            let step_result = match self.compute_optimization_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(crate::core::ApexError::Solver(
                        "Linear solver failed to solve system".to_string(),
                    ));
                }
            };

            // Update tracking
            max_gradient_norm = max_gradient_norm.max(step_result.gradient_norm);
            final_gradient_norm = step_result.gradient_norm;
            let step_norm = step_result.step.norm_l2();
            max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
            final_parameter_update_norm = step_norm;

            // Evaluate and apply step
            let step_eval = self.evaluate_and_apply_step(&step_result, &mut state, problem)?;
            cost_evaluations += 1;

            if step_eval.accepted {
                successful_steps += 1;
                total_cost_reduction += step_eval.cost_reduction;
            } else {
                unsuccessful_steps += 1;
            }

            // Log iteration
            self.log_iteration(
                iteration,
                &step_eval,
                step_norm,
                step_result.step_type,
                state.current_cost,
            );

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

                    return Ok(SolverResult {
                        status,
                        iterations: iteration + 1,
                        init_cost: state.initial_cost,
                        final_cost: state.current_cost,
                        parameters: state.variables.into_iter().collect(),
                        elapsed_time: elapsed,
                        convergence_info: Some(ConvergenceInfo {
                            final_gradient_norm,
                            final_parameter_update_norm,
                            cost_evaluations,
                            jacobian_evaluations,
                        }),
                    });
                }
            }

            // Check max iterations
            if iteration >= self.config.max_iterations {
                let elapsed = start_time.elapsed();
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

                return Ok(SolverResult {
                    parameters: state.variables,
                    status: OptimizationStatus::MaxIterationsReached,
                    init_cost: state.initial_cost,
                    final_cost: state.current_cost,
                    iterations: iteration,
                    elapsed_time: elapsed,
                    convergence_info: Some(ConvergenceInfo {
                        final_gradient_norm,
                        final_parameter_update_norm,
                        cost_evaluations,
                        jacobian_evaluations,
                    }),
                });
            }

            iteration += 1;
        }
    }
}

impl crate::optimizer::Solver for DogLeg {
    type Config = DogLegConfig;
    type Error = crate::core::ApexError;

    fn new() -> Self {
        Self::default()
    }

    fn minimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (crate::manifold::ManifoldType, nalgebra::DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, Self::Error> {
        self.minimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dog_leg_creation() {
        let solver = DogLeg::new();
        assert!(solver.config.trust_region_radius > 0.0);
    }

    #[test]
    fn test_trust_region_configuration() {
        let solver = DogLeg::new()
            .with_trust_region_radius(2.0)
            .with_trust_region_bounds(1e-15, 1e15);

        assert_eq!(solver.config.trust_region_radius, 2.0);
        assert_eq!(solver.config.trust_region_min, 1e-15);
        assert_eq!(solver.config.trust_region_max, 1e15);
    }

    #[test]
    fn test_trust_region_factors() {
        let solver = DogLeg::new().with_trust_region_factors(3.0, 0.25);

        assert_eq!(solver.config.trust_region_increase_factor, 3.0);
        assert_eq!(solver.config.trust_region_decrease_factor, 0.25);
    }

    #[test]
    fn test_config_builder() {
        let config = DogLegConfig::new()
            .with_max_iterations(50)
            .with_cost_tolerance(1e-6)
            .with_verbose(true)
            .with_trust_region_radius(5.0);

        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.cost_tolerance, 1e-6);
        assert!(config.verbose);
        assert_eq!(config.trust_region_radius, 5.0);
    }
}
