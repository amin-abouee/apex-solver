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

use crate::{core::problem, error, linalg, manifold, optimizer};
use faer::sparse;
use std::{collections, fmt, time};

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
    pub total_time: time::Duration,
    /// Average time per iteration
    pub average_time_per_iteration: time::Duration,
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

    // Ceres-style adaptive mu regularization parameters
    /// Initial mu regularization parameter for Gauss-Newton step
    pub initial_mu: f64,
    /// Minimum mu regularization parameter
    pub min_mu: f64,
    /// Maximum mu regularization parameter
    pub max_mu: f64,
    /// Factor to increase mu when linear solver fails
    pub mu_increase_factor: f64,

    // Ceres-style step reuse optimization
    /// Enable step reuse after rejection (Ceres-style efficiency optimization)
    pub enable_step_reuse: bool,

    /// Compute per-variable covariance matrices (uncertainty estimation)
    ///
    /// When enabled, computes covariance by inverting the Hessian matrix after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and optimizer::SolverResult.
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

impl Default for DogLegConfig {
    fn default() -> Self {
        Self {
            linear_solver_type: linalg::LinearSolverType::default(),
            max_iterations: 100,
            cost_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            timeout: None,
            verbose: false,
            // Ceres-style: larger initial radius for better global convergence
            trust_region_radius: 1e4,
            trust_region_min: 1e-12,
            trust_region_max: 1e12,
            // Ceres uses adaptive increase (max(radius, 3*step_norm)),
            // but we keep factor for simpler config
            trust_region_increase_factor: 3.0,
            trust_region_decrease_factor: 0.5,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            poor_step_quality: 0.25,
            // Ceres-style: Enable diagonal scaling by default for elliptical trust region
            use_jacobi_scaling: true,

            // Ceres-style adaptive mu regularization defaults
            // Start with more conservative regularization to avoid singular Hessian
            initial_mu: 1e-4,
            min_mu: 1e-8,
            max_mu: 1.0,
            mu_increase_factor: 10.0,

            // Ceres-style step reuse optimization
            enable_step_reuse: true,

            compute_covariances: false,
            enable_visualization: false,
        }
    }
}

impl DogLegConfig {
    /// Create a new Dog Leg configuration with default values.
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

    /// Set adaptive mu regularization parameters (Ceres-style)
    pub fn with_mu_params(
        mut self,
        initial_mu: f64,
        min_mu: f64,
        max_mu: f64,
        increase_factor: f64,
    ) -> Self {
        self.initial_mu = initial_mu;
        self.min_mu = min_mu;
        self.max_mu = max_mu;
        self.mu_increase_factor = increase_factor;
        self
    }

    /// Enable or disable step reuse optimization (Ceres-style)
    pub fn with_step_reuse(mut self, enable_step_reuse: bool) -> Self {
        self.enable_step_reuse = enable_step_reuse;
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
struct StepEvaluation {
    accepted: bool,
    cost_reduction: f64,
    rho: f64,
}

/// Dog Leg solver for nonlinear least squares optimization.
pub struct DogLeg {
    config: DogLegConfig,
    jacobi_scaling: Option<sparse::SparseColMat<usize, f64>>,
    visualizer: Option<optimizer::visualization::OptimizationVisualizer>,

    // Adaptive mu regularization (Ceres-style)
    mu: f64,
    min_mu: f64,
    max_mu: f64,
    mu_increase_factor: f64,

    // Step reuse mechanism (Ceres-style efficiency optimization)
    reuse_step_on_rejection: bool,
    cached_gn_step: Option<faer::Mat<f64>>,
    cached_cauchy_point: Option<faer::Mat<f64>>,
    cached_gradient: Option<faer::Mat<f64>>,
    cached_alpha: Option<f64>,
    cache_reuse_count: usize, // Track consecutive reuses to prevent staleness
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
        // Create visualizer if enabled (zero overhead when disabled)
        let visualizer = if config.enable_visualization {
            optimizer::visualization::OptimizationVisualizer::new(true).ok()
        } else {
            None
        };

        Self {
            // Initialize adaptive mu from config
            mu: config.initial_mu,
            min_mu: config.min_mu,
            max_mu: config.max_mu,
            mu_increase_factor: config.mu_increase_factor,

            // Initialize step reuse mechanism (disabled initially, enabled after first rejection)
            reuse_step_on_rejection: false,
            cached_gn_step: None,
            cached_cauchy_point: None,
            cached_gradient: None,
            cached_alpha: None,
            cache_reuse_count: 0,

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

    /// Compute the Cauchy point (steepest descent step)
    /// Returns the optimal step along the negative gradient direction
    /// Compute Cauchy point and optimal step length for steepest descent
    ///
    /// Returns (alpha, cauchy_point) where:
    /// - alpha: optimal step length α = ||g||² / (g^T H g)
    /// - cauchy_point: p_c = -α * g (the Cauchy point)
    ///
    /// This is the optimal point along the steepest descent direction within
    /// the quadratic approximation of the objective function.
    fn compute_cauchy_point_and_alpha(
        &self,
        gradient: &faer::Mat<f64>,
        hessian: &sparse::SparseColMat<usize, f64>,
    ) -> (f64, faer::Mat<f64>) {
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

        // Compute Cauchy point: p_c = -α * gradient
        let mut cauchy_point = faer::Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            cauchy_point[(i, 0)] = -alpha * gradient[(i, 0)];
        }

        (alpha, cauchy_point)
    }

    /// Compute the dog leg step using Powell's Dog Leg method
    ///
    /// The dog leg path consists of two segments:
    /// 1. From origin to Cauchy point (optimal along steepest descent)
    /// 2. From Cauchy point to Gauss-Newton step
    ///
    /// Arguments:
    /// - steepest_descent_dir: -gradient (steepest descent direction, not scaled)
    /// - cauchy_point: p_c = α * (-gradient), the optimal steepest descent step
    /// - h_gn: Gauss-Newton step
    /// - delta: Trust region radius
    ///
    /// Returns: (step, step_type)
    fn compute_dog_leg_step(
        &self,
        steepest_descent_dir: &faer::Mat<f64>,
        cauchy_point: &faer::Mat<f64>,
        h_gn: &faer::Mat<f64>,
        delta: f64,
    ) -> (faer::Mat<f64>, StepType) {
        let gn_norm = h_gn.norm_l2();
        let cauchy_norm = cauchy_point.norm_l2();
        let sd_norm = steepest_descent_dir.norm_l2();

        // Case 1: Full Gauss-Newton step fits in trust region
        if gn_norm <= delta {
            return (h_gn.clone(), StepType::GaussNewton);
        }

        // Case 2: Even Cauchy point is outside trust region
        // Scale steepest descent direction to boundary: (delta / ||δ_sd||) * δ_sd
        if cauchy_norm >= delta {
            let mut scaled_sd = faer::Mat::zeros(steepest_descent_dir.nrows(), 1);
            let scale = delta / sd_norm;
            for i in 0..steepest_descent_dir.nrows() {
                scaled_sd[(i, 0)] = steepest_descent_dir[(i, 0)] * scale;
            }
            return (scaled_sd, StepType::SteepestDescent);
        }

        // Case 3: Dog leg interpolation between Cauchy point and GN step
        // Use Ceres-style numerically robust formula
        //
        // Following Ceres solver implementation for numerical stability:
        // Compute intersection of trust region boundary with line from Cauchy point to GN step
        //
        // Let v = δ_gn - p_c
        // Solve: ||p_c + β*v||² = delta²
        // This gives: a*β² + 2*b*β + c = 0
        // where:
        //   a = v^T·v = ||v||²
        //   b = p_c^T·v
        //   c = p_c^T·p_c - delta² = ||p_c||² - delta²

        let mut v = faer::Mat::zeros(cauchy_point.nrows(), 1);
        for i in 0..cauchy_point.nrows() {
            v[(i, 0)] = h_gn[(i, 0)] - cauchy_point[(i, 0)];
        }

        // Compute coefficients
        let v_squared_norm = v.transpose() * &v;
        let a = v_squared_norm[(0, 0)];

        let pc_dot_v = cauchy_point.transpose() * &v;
        let b = pc_dot_v[(0, 0)];

        let c = cauchy_norm * cauchy_norm - delta * delta;

        // Ceres-style numerically robust beta computation
        // Uses two different formulas based on sign of b to avoid catastrophic cancellation
        let d_squared = b * b - a * c;

        let beta = if d_squared < 0.0 {
            // Should not happen geometrically, but handle gracefully
            1.0
        } else if a.abs() < 1e-15 {
            // Degenerate case: v is nearly zero
            1.0
        } else {
            let d = d_squared.sqrt();

            // Ceres formula: choose formula based on sign of b to avoid cancellation
            // If b <= 0: beta = (-b + d) / a  (standard formula, no cancellation)
            // If b > 0:  beta = -c / (b + d)  (alternative formula, avoids cancellation)
            if b <= 0.0 { (-b + d) / a } else { -c / (b + d) }
        };

        // Clamp beta to [0, 1] for safety
        let beta = beta.clamp(0.0, 1.0);

        // Compute dog leg step: p_dl = p_c + β*(δ_gn - p_c)
        let mut dog_leg = faer::Mat::zeros(cauchy_point.nrows(), 1);
        for i in 0..cauchy_point.nrows() {
            dog_leg[(i, 0)] = cauchy_point[(i, 0)] + beta * v[(i, 0)];
        }

        (dog_leg, StepType::DogLeg)
    }

    /// Update trust region radius based on step quality (Ceres-style)
    fn update_trust_region(&mut self, rho: f64, step_norm: f64) -> bool {
        if rho > self.config.good_step_quality {
            // Good step, increase trust region (Ceres-style: max(radius, 3*step_norm))
            let new_radius = self.config.trust_region_radius.max(3.0 * step_norm);
            self.config.trust_region_radius = new_radius.min(self.config.trust_region_max);

            // Decrease mu on successful step (Ceres-style adaptive regularization)
            self.mu = (self.mu / (0.5 * self.mu_increase_factor)).max(self.min_mu);

            // Clear reuse flag and invalidate cache on acceptance (parameters have moved)
            self.reuse_step_on_rejection = false;
            self.cached_gn_step = None;
            self.cached_cauchy_point = None;
            self.cached_gradient = None;
            self.cached_alpha = None;
            self.cache_reuse_count = 0;

            true
        } else if rho < self.config.poor_step_quality {
            // Poor step, decrease trust region
            self.config.trust_region_radius = (self.config.trust_region_radius
                * self.config.trust_region_decrease_factor)
                .max(self.config.trust_region_min);

            // Enable step reuse flag for next iteration (Ceres-style)
            self.reuse_step_on_rejection = self.config.enable_step_reuse;

            false
        } else {
            // Moderate step, keep trust region unchanged
            // Clear reuse flag and invalidate cache on acceptance (parameters have moved)
            self.reuse_step_on_rejection = false;
            self.cached_gn_step = None;
            self.cached_cauchy_point = None;
            self.cached_gradient = None;
            self.cached_alpha = None;
            self.cache_reuse_count = 0;

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
        step: &faer::Mat<f64>,
        gradient: &faer::Mat<f64>,
        hessian: &sparse::SparseColMat<usize, f64>,
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

        sparse::SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
            .expect("Failed to create Jacobi scaling matrix")
    }

    /// Initialize optimization state
    fn initialize_optimization_state(
        &self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<OptimizationState, error::ApexError> {
        let variables = problem.initialize_variables(initial_params);

        let mut variable_index_map = collections::HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<String> = variables.keys().cloned().collect();
        sorted_vars.sort();

        for var_name in &sorted_vars {
            variable_index_map.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        let symbolic_structure =
            problem.build_symbolic_structure(&variables, &variable_index_map, col_offset)?;

        // Initial cost evaluation (residual only, no Jacobian needed)
        let residual = problem.compute_residual_sparse(&variables)?;

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

    /// Process Jacobian by creating and applying Jacobi scaling if enabled
    fn process_jacobian(
        &mut self,
        jacobian: &sparse::SparseColMat<usize, f64>,
        iteration: usize,
    ) -> sparse::SparseColMat<usize, f64> {
        // Create Jacobi scaling on first iteration if enabled
        if iteration == 0 {
            let scaling = self.create_jacobi_scaling(jacobian);

            if self.config.verbose {
                println!("Jacobi scaling computed for {} columns", scaling.ncols());
            }

            self.jacobi_scaling = Some(scaling);
        }
        jacobian * self.jacobi_scaling.as_ref().unwrap()
    }

    /// Compute dog leg optimization step
    fn compute_optimization_step(
        &mut self,
        residuals: &faer::Mat<f64>,
        scaled_jacobian: &sparse::SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn linalg::SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Check if we can reuse cached step (Ceres-style optimization)
        // Safety limit: prevent excessive reuse that could lead to stale gradient/Hessian
        const MAX_CACHE_REUSE: usize = 5;

        if self.reuse_step_on_rejection
            && self.config.enable_step_reuse
            && self.cache_reuse_count < MAX_CACHE_REUSE
        {
            if let (Some(cached_gn), Some(cached_cauchy), Some(cached_grad), Some(_cached_a)) = (
                &self.cached_gn_step,
                &self.cached_cauchy_point,
                &self.cached_gradient,
                &self.cached_alpha,
            ) {
                // Increment reuse counter
                self.cache_reuse_count += 1;

                if self.config.verbose {
                    println!(
                        "  Reusing cached GN step and Cauchy point (step was rejected, reuse #{}/{})",
                        self.cache_reuse_count, MAX_CACHE_REUSE
                    );
                }

                let gradient_norm = cached_grad.norm_l2();
                let mut steepest_descent = faer::Mat::zeros(cached_grad.nrows(), 1);
                for i in 0..cached_grad.nrows() {
                    steepest_descent[(i, 0)] = -cached_grad[(i, 0)];
                }

                let (scaled_step, step_type) = self.compute_dog_leg_step(
                    &steepest_descent,
                    cached_cauchy,
                    cached_gn,
                    self.config.trust_region_radius,
                );

                let step = if self.config.use_jacobi_scaling {
                    self.jacobi_scaling.as_ref().unwrap() * &scaled_step
                } else {
                    scaled_step.clone()
                };

                let hessian = linear_solver.get_hessian()?;
                let predicted_reduction =
                    self.compute_predicted_reduction(&scaled_step, cached_grad, hessian);

                return Some(StepResult {
                    step,
                    gradient_norm,
                    predicted_reduction,
                    step_type,
                });
            }
        }

        // Not reusing, compute fresh step
        if self.reuse_step_on_rejection
            && self.cache_reuse_count >= MAX_CACHE_REUSE
            && self.config.verbose
        {
            println!(
                "  Cache reuse limit reached ({}), forcing fresh computation to avoid stale gradient",
                MAX_CACHE_REUSE
            );
        }
        // 1. Solve for Gauss-Newton step with adaptive mu regularization (Ceres-style)
        let residuals_owned = residuals.as_ref().to_owned();
        let mut scaled_gn_step = None;
        let mut mu_attempts = 0;

        // Try to solve with current mu, increasing if necessary
        while mu_attempts < 10 && self.mu <= self.max_mu {
            let damping = self.mu;

            if let Ok(step) =
                linear_solver.solve_augmented_equation(&residuals_owned, scaled_jacobian, damping)
            {
                scaled_gn_step = Some(step);
                break;
            }

            // Increase mu (Ceres-style)
            self.mu = (self.mu * self.mu_increase_factor).min(self.max_mu);
            mu_attempts += 1;

            if self.config.verbose && mu_attempts < 10 {
                println!("  Linear solve failed, increasing mu to {:.6e}", self.mu);
            }
        }

        let scaled_gn_step = scaled_gn_step?;

        // 2. Get gradient and Hessian (cached by solve_augmented_equation)
        let gradient = linear_solver.get_gradient()?;
        let hessian = linear_solver.get_hessian()?;
        let gradient_norm = gradient.norm_l2();

        // 3. Compute steepest descent direction: δ_sd = -gradient
        let mut steepest_descent = faer::Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            steepest_descent[(i, 0)] = -gradient[(i, 0)];
        }

        // 4. Compute Cauchy point and optimal step length
        let (alpha, cauchy_point) = self.compute_cauchy_point_and_alpha(gradient, hessian);

        // 5. Compute dog leg step based on trust region radius
        let (scaled_step, step_type) = self.compute_dog_leg_step(
            &steepest_descent,
            &cauchy_point,
            &scaled_gn_step,
            self.config.trust_region_radius,
        );

        // 6. Apply inverse Jacobi scaling if enabled
        let step = if self.config.use_jacobi_scaling {
            self.jacobi_scaling.as_ref().unwrap() * &scaled_step
        } else {
            scaled_step.clone()
        };

        // 7. Compute predicted reduction
        let predicted_reduction = self.compute_predicted_reduction(&scaled_step, gradient, hessian);

        // 8. Cache step components for potential reuse (Ceres-style)
        self.cached_gn_step = Some(scaled_gn_step.clone());
        self.cached_cauchy_point = Some(cauchy_point.clone());
        self.cached_gradient = Some(gradient.clone());
        self.cached_alpha = Some(alpha);

        if self.config.verbose {
            println!("Gradient norm: {:.12e}", gradient_norm);
            println!("Mu (regularization): {:.12e}", self.mu);
            println!("Alpha (Cauchy step length): {:.12e}", alpha);
            println!("Cauchy point norm: {:.12e}", cauchy_point.norm_l2());
            println!("GN step norm: {:.12e}", scaled_gn_step.norm_l2());
            println!("Step type: {}", step_type);
            println!("Step norm: {:.12e}", step.norm_l2());
            println!("Predicted reduction: {:.12e}", predicted_reduction);
        }

        Some(StepResult {
            step,
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
        problem: &problem::Problem,
    ) -> error::ApexResult<StepEvaluation> {
        // Apply parameter updates
        let step_norm = optimizer::apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost (residual only, no Jacobian needed for step evaluation)
        let new_residual = problem.compute_residual_sparse(&state.variables)?;
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
        // Filter out numerical noise with small threshold
        let accepted = rho > 1e-4;
        let _trust_region_updated = self.update_trust_region(rho, step_norm);

        let cost_reduction = if accepted {
            let reduction = state.current_cost - new_cost;
            state.current_cost = new_cost;
            reduction
        } else {
            // Reject step - revert changes
            optimizer::apply_negative_parameter_step(
                &mut state.variables,
                step_result.step.as_ref(),
                &state.sorted_vars,
            );
            0.0
        };

        if self.config.verbose {
            println!(
                "Step {}, New radius: {:.6e}",
                if accepted { "ACCEPTED" } else { "REJECTED" },
                self.config.trust_region_radius
            );
        }

        Ok(StepEvaluation {
            accepted,
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
        total_time: time::Duration,
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
                time::Duration::from_secs(0)
            },
        }
    }

    /// Minimize the optimization problem using Dog Leg algorithm
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

            // Process Jacobian (apply scaling if enabled)
            let scaled_jacobian = if self.config.use_jacobi_scaling {
                self.process_jacobian(&jacobian, iteration)
            } else {
                jacobian
            };

            // Compute dog leg step
            let step_result = match self.compute_optimization_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(error::ApexError::Solver(
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

            // Rerun visualization
            if let Some(ref vis) = self.visualizer {
                if let Err(e) = vis.log_scalars(
                    iteration,
                    state.current_cost,
                    step_result.gradient_norm,
                    self.config.trust_region_radius,
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
                    parameters: state.variables,
                    status: optimizer::OptimizationStatus::MaxIterationsReached,
                    initial_cost: state.initial_cost,
                    final_cost: state.current_cost,
                    iterations: iteration,
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

impl optimizer::Solver for DogLeg {
    type Config = DogLegConfig;
    type Error = error::ApexError;

    fn new() -> Self {
        Self::default()
    }

    fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<collections::HashMap<String, problem::VariableEnum>>,
        Self::Error,
    > {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::factors, manifold};
    use nalgebra;

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
            (manifold::ManifoldType::RN, nalgebra::dvector![-1.2]),
        );
        initial_values.insert(
            "x2".to_string(),
            (manifold::ManifoldType::RN, nalgebra::dvector![1.0]),
        );

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Configure Dog Leg optimizer with appropriate trust region
        let config = DogLegConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10)
            .with_trust_region_radius(10.0); // Start with larger trust region

        let mut solver = DogLeg::with_config(config);
        let result = solver.optimize(&problem, &initial_values).unwrap();

        // Extract final values
        let x1_final = result.parameters.get("x1").unwrap().to_vector()[0];
        let x2_final = result.parameters.get("x2").unwrap().to_vector()[0];

        println!("Rosenbrock optimization result (Dog Leg):");
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
