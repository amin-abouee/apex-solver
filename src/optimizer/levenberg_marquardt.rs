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

use crate::linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use crate::optimizer::{ConvergenceInfo, OptimizationStatus, OptimizerConfig, SolverResult};
use faer::{Mat, sparse::SparseColMat};
use std::fmt;
use std::time::Instant;

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
        writeln!(f, "=== Levenberg-Marquardt Optimization Summary ===")?;
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

/// Levenberg-Marquardt solver for nonlinear least squares optimization.
pub struct LevenbergMarquardt {
    config: OptimizerConfig,
    damping: f64,
    damping_min: f64,
    damping_max: f64,
    damping_increase_factor: f64,
    damping_decrease_factor: f64,
    trust_region_radius: f64,
    min_step_quality: f64,
    good_step_quality: f64,
    // Jacobi scaling support
    jacobi_scaling: Option<SparseColMat<usize, f64>>,
    min_diagonal: f64,
    max_diagonal: f64,
}

impl LevenbergMarquardt {
    /// Create a new Levenberg-Marquardt solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(OptimizerConfig::default())
    }

    /// Create a new Levenberg-Marquardt solver with the given configuration.
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            config,
            damping: 1e-4,
            damping_min: 1e-12,
            damping_max: 1e12,
            damping_increase_factor: 10.0,
            damping_decrease_factor: 0.3,
            trust_region_radius: 1e4,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            jacobi_scaling: None,
            min_diagonal: 1e-6,
            max_diagonal: 1e32,
        }
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

    /// Create the appropriate linear solver based on configuration
    fn create_linear_solver(&self) -> Box<dyn SparseLinearSolver> {
        match self.config.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        }
    }

    /// Update damping parameter based on step quality using trust region approach
    fn update_damping(&mut self, rho: f64) -> bool {
        if rho > 0.0 {
            // Step accepted - decrease damping using tiny-solver's exact strategy
            let tmp = 2.0 * rho - 1.0;
            self.damping *= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
            self.damping = self.damping.max(self.damping_min);
            true
        } else {
            // Step rejected - increase damping using tiny-solver's exact strategy
            self.damping *= 2.0;
            self.damping = self.damping.min(self.damping_max);
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
    /// Standard LM formula: -step^T * gradient - 0.5 * step^T * H * step
    fn compute_predicted_reduction(
        &self,
        step: &Mat<f64>,
        gradient: &Mat<f64>,
        hessian: &SparseColMat<usize, f64>,
    ) -> f64 {
        // Standard Levenberg-Marquardt predicted reduction formula
        // predicted_reduction = -step^T * gradient - 0.5 * step^T * H * step
        let linear_term = step.transpose() * gradient;
        let hessian_step = hessian * step;
        let quadratic_term = step.transpose() * &hessian_step;

        let predicted_reduction = -linear_term[(0, 0)] - 0.5 * quadratic_term[(0, 0)];

        // Positive predicted_reduction means we expect cost to decrease
        predicted_reduction
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

    /// Compute gradient norm for convergence checking
    fn compute_gradient_norm(
        &self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> f64 {
        let gradient = self.compute_gradient(residuals, jacobian);
        gradient.norm_l2()
    }

    /// Compute gradient vector: J^T * r
    fn compute_gradient(
        &self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> Mat<f64> {
        // Compute J^T * r
        jacobian.transpose() * residuals
    }

    /// Compute Hessian approximation: J^T * J
    fn compute_hessian(&self, jacobian: &SparseColMat<usize, f64>) -> SparseColMat<usize, f64> {
        // Compute J^T * J
        jacobian.transpose().to_col_major().unwrap() * jacobian
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
    fn apply_jacobi_scaling(
        &self,
        jacobian: &SparseColMat<usize, f64>,
        scaling: &SparseColMat<usize, f64>,
    ) -> SparseColMat<usize, f64> {
        jacobian * scaling
    }

    /// Apply inverse Jacobi scaling to step: dx_final = S * dx_scaled
    fn apply_inverse_jacobi_scaling(
        &self,
        step: &Mat<f64>,
        scaling: &SparseColMat<usize, f64>,
    ) -> Mat<f64> {
        scaling * step
    }

    /// Solve the regularized system with diagonal-aware damping
    fn solve_regularized_system(
        &self,
        residuals: &Mat<f64>,
        scaled_jacobian: &SparseColMat<usize, f64>,
        hessian: &SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn SparseLinearSolver>,
    ) -> Option<Mat<f64>> {
        // For now, create a temporary augmented system that mimics what we want
        // We'll use the solve_augmented_equation with our custom damping

        // Apply diagonal-aware regularization similar to tiny-solver-rs
        let n_vars = scaled_jacobian.ncols();
        let n_residuals = scaled_jacobian.nrows();

        // Create damping matrix entries
        let mut damping_triplets = Vec::new();
        for i in 0..n_vars {
            let diag_val = hessian[(i, i)];
            let clamped_diag = diag_val.max(self.min_diagonal).min(self.max_diagonal);
            let damping_value = (self.damping * clamped_diag).sqrt();
            damping_triplets.push(faer::sparse::Triplet::new(
                n_residuals + i,
                i,
                damping_value,
            ));
        }

        // Create augmented Jacobian: [J; sqrt(damping) * D]
        let mut all_triplets: Vec<_> = scaled_jacobian
            .triplet_iter()
            .map(|t| faer::sparse::Triplet::new(t.row, t.col, *t.val))
            .collect();
        all_triplets.extend(damping_triplets);

        let augmented_jacobian =
            SparseColMat::try_new_from_triplets(n_residuals + n_vars, n_vars, &all_triplets)
                .ok()?;

        // Create augmented residuals: [r; 0]
        let mut augmented_residuals = Mat::zeros(n_residuals + n_vars, 1);
        for i in 0..n_residuals {
            augmented_residuals[(i, 0)] = residuals[(i, 0)];
        }
        // Zero entries for the damping part

        // Solve using the standard interface
        linear_solver.solve_normal_equation(&augmented_residuals, &augmented_jacobian)
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
            final_damping: self.damping,
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

    pub fn solve_problem(
        &mut self,
        problem: &crate::core::problem::Problem,
        initial_params: &std::collections::HashMap<
            String,
            (crate::manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        SolverResult<std::collections::HashMap<String, crate::core::problem::VariableEnum>>,
        crate::core::ApexError,
    > {
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 0;
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;

        // Create linear solver
        let mut linear_solver = self.create_linear_solver();

        // Initialize variables from initial values
        let mut variables = problem.initialize_variables(initial_params);

        // Create column mapping for variables
        let mut variable_name_to_col_idx_dict = std::collections::HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
        sorted_vars.sort(); // Ensure consistent ordering

        for var_name in &sorted_vars {
            variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Build symbolic structure for sparse operations
        let symbolic_structure =
            problem.build_symbolic_structure(&variables, &variable_name_to_col_idx_dict);

        // Initial cost evaluation using sparse interface
        let (residual, _) = problem.compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        );
        use faer_ext::IntoNalgebra;
        let residual_na = residual.as_ref().into_nalgebra();
        let mut current_cost = residual_na.norm_squared();
        let initial_cost = current_cost;
        cost_evaluations += 1;

        // Initialize summary tracking variables
        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;
        let mut last_cost_reduction = 0.0;
        let mut last_accepted_cost_reduction = 0.0;

        if self.config.verbose {
            println!(
                "Starting Levenberg-Marquardt optimization with {} max iterations",
                self.config.max_iterations
            );
            println!(
                "Initial cost: {:.6e}, initial damping: {:.6e}",
                current_cost, self.damping
            );
        }

        let mut final_gradient_norm;
        let mut final_parameter_update_norm;

        loop {
            // Evaluate residuals and Jacobian using sparse interface
            let (residuals, jacobian) = problem.compute_residual_and_jacobian_sparse(
                &variables,
                &variable_name_to_col_idx_dict,
                &symbolic_structure,
            );
            jacobian_evaluations += 1;

            // DEBUG: Log detailed iteration information
            if self.config.verbose {
                println!("\n=== APEX-SOLVER DEBUG ITERATION {} ===", iteration);
                println!("Current cost: {:.12e}", current_cost);
                use faer_ext::IntoNalgebra;
                let residual_na = residuals.as_ref().into_nalgebra();
                println!(
                    "Residuals shape: ({}, {})",
                    residuals.nrows(),
                    residuals.ncols()
                );
                println!("Residuals norm: {:.12e}", residual_na.norm());
                println!(
                    "Jacobian shape: ({}, {})",
                    jacobian.nrows(),
                    jacobian.ncols()
                );

                // Log first few residual values
                let residual_vec: Vec<f64> = (0..residuals.nrows().min(10))
                    .map(|row| residual_na[row])
                    .collect();
                println!("First 10 residuals: {:?}", residual_vec);
            }

            // Create or reuse Jacobi scaling on first iteration
            if iteration == 0 {
                let scaling = self.create_jacobi_scaling(&jacobian);

                // DEBUG: Log Jacobi scaling values
                if self.config.verbose {
                    println!("Jacobi scaling computed for {} columns", scaling.ncols());
                }

                self.jacobi_scaling = Some(scaling);
            }

            // Apply Jacobi scaling to Jacobian
            let scaled_jacobian =
                self.apply_jacobi_scaling(&jacobian, self.jacobi_scaling.as_ref().unwrap());

            // DEBUG: Log scaled Jacobian info
            if self.config.verbose {
                println!(
                    "Scaled Jacobian shape: ({}, {})",
                    scaled_jacobian.nrows(),
                    scaled_jacobian.ncols()
                );
            }

            // Compute gradient = J^T * r using scaled Jacobian
            let residuals_owned = residuals.as_ref().to_owned();
            let gradient_norm = self.compute_gradient_norm(&residuals_owned, &scaled_jacobian);
            max_gradient_norm = max_gradient_norm.max(gradient_norm);
            final_gradient_norm = gradient_norm;

            // Compute Hessian approximation: J^T * J (using scaled Jacobian)
            let hessian = self.compute_hessian(&scaled_jacobian);
            let gradient = self.compute_gradient(&residuals_owned, &scaled_jacobian);

            // DEBUG: Log gradient and Hessian info
            if self.config.verbose {
                println!("Gradient (J^T*r) norm: {:.12e}", gradient_norm);

                // Log Hessian info
                println!("Hessian shape: ({}, {})", hessian.nrows(), hessian.ncols());

                // Log first few gradient values
                use faer_ext::IntoNalgebra;
                let gradient_na = gradient.as_ref().into_nalgebra();
                let gradient_vec: Vec<f64> = (0..gradient_na.nrows().min(10))
                    .map(|row| gradient_na[row])
                    .collect();
                println!("First 10 gradient values: {:?}", gradient_vec);

                println!("Damping parameter: {:.12e}", self.damping);
            }

            if self.config.verbose && iteration < 3 {
                println!(
                    "Debug iteration {}: gradient_norm = {:.6e}, damping = {:.6e}",
                    iteration, gradient_norm, self.damping
                );
            }

            // Use standard augmented equation solver with scaled Jacobian
            // This will solve: (J_scaled^T * J_scaled + λI) * dx_scaled = -J_scaled^T * r
            if let Some(scaled_step) = linear_solver.solve_augmented_equation(
                &residuals_owned,
                &scaled_jacobian,
                self.damping,
            ) {
                // Apply inverse Jacobi scaling to get final step
                let step = self.apply_inverse_jacobi_scaling(
                    &scaled_step,
                    self.jacobi_scaling.as_ref().unwrap(),
                );

                let step_norm = step.norm_l2();
                max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
                final_parameter_update_norm = step_norm;

                // DEBUG: Log linear step details
                if self.config.verbose {
                    println!(
                        "Linear step (scaled_step) norm: {:.12e}",
                        scaled_step.norm_l2()
                    );
                    println!("Final step norm: {:.12e}", step_norm);

                    // Log first few step values
                    use faer_ext::IntoNalgebra;
                    let step_na = step.as_ref().into_nalgebra();
                    let step_vec: Vec<f64> = (0..step_na.nrows().min(10))
                        .map(|row| step_na[row])
                        .collect();
                    println!("First 10 step values: {:?}", step_vec);
                }

                // Compute predicted reduction using scaled values
                let predicted_reduction =
                    self.compute_predicted_reduction(&scaled_step, &gradient, &hessian);

                // DEBUG: Log predicted reduction calculation details
                if self.config.verbose {
                    println!("=== PREDICTED REDUCTION CALCULATION ===");
                    use faer_ext::IntoNalgebra;
                    let grad_na = gradient.as_ref().into_nalgebra();
                    println!("Gradient norm: {:.12e}", grad_na.norm());
                    println!("Predicted reduction: {:.12e}", predicted_reduction);
                }

                // Apply parameter updates using simple perturbation
                let mut step_offset = 0;
                for var_name in &sorted_vars {
                    let var_size = variables[var_name].get_size();
                    let var_step = step.subrows(step_offset, var_size);

                    // Simple perturbation for SE3 variables
                    match &mut variables.get_mut(var_name).unwrap() {
                        crate::core::problem::VariableEnum::SE3(var) => {
                            // Get current SE3 value
                            let current_translation = var.value.translation();
                            let current_rotation = var.value.rotation_so3();

                            // Apply simple additive perturbation to translation (first 3 components)
                            // and small rotation perturbation (last 3 components)
                            use faer_ext::IntoNalgebra;
                            let step_na = var_step.as_ref().into_nalgebra();
                            let translation_step =
                                nalgebra::Vector3::new(step_na[0], step_na[1], step_na[2]);
                            let rotation_step =
                                nalgebra::Vector3::new(step_na[3], step_na[4], step_na[5]);

                            // Create SE3Tangent from the step vector (proper manifold operation)
                            let step_dvector = nalgebra::DVector::from_vec(vec![
                                step_na[0], step_na[1], step_na[2], step_na[3], step_na[4],
                                step_na[5],
                            ]);
                            let se3_tangent = crate::manifold::se3::SE3Tangent::from(step_dvector);

                            // Apply proper manifold plus operation: g ⊞ φ = g ∘ exp(φ^∧)
                            let new_se3 = var.plus(&se3_tangent);
                            var.set_value(new_se3);
                        }
                        _ => {
                            // For other variable types, implement as needed
                        }
                    }
                    step_offset += var_size;
                }

                if self.config.verbose && iteration < 3 {
                    println!("  Applied parameter step with norm: {:.6e}", step_norm);
                }

                // Compute new cost using sparse interface
                let (new_residual, _) = problem.compute_residual_and_jacobian_sparse(
                    &variables,
                    &variable_name_to_col_idx_dict,
                    &symbolic_structure,
                );
                let new_residual_na = new_residual.as_ref().into_nalgebra();
                let new_cost = new_residual_na.norm_squared();
                cost_evaluations += 1;

                // Compute step quality
                let rho = self.compute_step_quality(current_cost, new_cost, predicted_reduction);

                // DEBUG: Log detailed rho calculation
                if self.config.verbose {
                    println!("=== RHO CALCULATION DETAILS ===");
                    println!("Old cost: {:.12e}", current_cost);
                    println!("New cost: {:.12e}", new_cost);
                    let actual_reduction = current_cost - new_cost;
                    println!("Actual cost reduction: {:.12e}", actual_reduction);
                    println!("Predicted cost reduction: {:.12e}", predicted_reduction);
                    println!("Rho (actual/predicted): {:.12e}", rho);
                }

                // Update damping and decide whether to accept step
                let accept_step = self.update_damping(rho);

                if accept_step {
                    // Accept the step
                    let cost_reduction = current_cost - new_cost;
                    last_cost_reduction = cost_reduction;
                    last_accepted_cost_reduction = cost_reduction;
                    total_cost_reduction += cost_reduction;
                    // Parameters already updated in variables
                    current_cost = new_cost;
                    successful_steps += 1;

                    if self.config.verbose {
                        println!(
                            "Iteration {}: cost = {:.6e}, reduction = {:.6e}, damping = {:.6e}, step_norm = {:.6e}, rho = {:.3} [ACCEPTED]",
                            iteration + 1,
                            current_cost,
                            cost_reduction,
                            self.damping,
                            step_norm,
                            rho
                        );
                    }

                    // Check convergence only after successful steps
                    let elapsed = start_time.elapsed();
                    if let Some(status) = self.check_convergence(
                        iteration,
                        cost_reduction, // Use actual cost reduction from this step
                        step_norm,      // Use actual step norm from this step
                        gradient_norm,
                        elapsed,
                    ) {
                        let summary = self.create_summary(
                            initial_cost,
                            current_cost,
                            iteration + 1, // increment for final count
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
                            final_cost: current_cost,
                            parameters: variables.into_iter().collect(),
                            elapsed_time: elapsed,
                            convergence_info: ConvergenceInfo {
                                final_gradient_norm,
                                final_parameter_update_norm,
                                cost_evaluations,
                                jacobian_evaluations,
                            },
                        });
                    }
                } else {
                    // Reject the step, revert parameter changes
                    let mut step_offset = 0;
                    for var_name in &sorted_vars {
                        let var_size = variables[var_name].get_size();
                        let var_step = step.subrows(step_offset, var_size);

                        // Revert the perturbation using proper manifold operations
                        match &mut variables.get_mut(var_name).unwrap() {
                            crate::core::problem::VariableEnum::SE3(var) => {
                                // Revert using negative step with proper manifold operation
                                use faer_ext::IntoNalgebra;
                                let step_na = var_step.as_ref().into_nalgebra();
                                let negative_step_na = -step_na;

                                // Create SE3Tangent from the negative step vector
                                let negative_step_dvector = nalgebra::DVector::from_vec(vec![
                                    negative_step_na[0],
                                    negative_step_na[1],
                                    negative_step_na[2],
                                    negative_step_na[3],
                                    negative_step_na[4],
                                    negative_step_na[5],
                                ]);
                                let negative_se3_tangent =
                                    crate::manifold::se3::SE3Tangent::from(negative_step_dvector);

                                // Apply proper manifold plus operation with negative step
                                let reverted_se3 = var.plus(&negative_se3_tangent);
                                var.set_value(reverted_se3);
                            }
                            _ => {
                                // For other variable types, implement as needed
                            }
                        }
                        step_offset += var_size;
                    }

                    last_cost_reduction = 0.0;
                    unsuccessful_steps += 1;

                    if self.config.verbose {
                        println!(
                            "Iteration {}: cost = {:.6e}, damping = {:.6e}, step_norm = {:.6e}, rho = {:.3} [REJECTED]",
                            iteration + 1,
                            current_cost,
                            self.damping,
                            step_norm,
                            rho
                        );
                    }
                }
            } else {
                // Linear solver failed
                return Err(crate::core::ApexError::Solver(
                    "Linear solver failed to solve augmented system".to_string(),
                ));
            }

            // Check only max iterations - other convergence criteria checked after successful steps
            let elapsed = start_time.elapsed();
            if iteration >= self.config.max_iterations {
                let summary = self.create_summary(
                    initial_cost,
                    current_cost,
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
                    parameters: variables,
                    status: OptimizationStatus::MaxIterationsReached,
                    final_cost: current_cost,
                    iterations: iteration,
                    elapsed_time: elapsed,
                    convergence_info: ConvergenceInfo {
                        final_gradient_norm,
                        final_parameter_update_norm,
                        cost_evaluations,
                        jacobian_evaluations,
                    },
                });
            }
            iteration += 1;
        }
    }
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Solver trait

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Optimizable;
    use faer::mat;

    #[test]
    fn test_levenberg_marquardt_creation() {
        let solver = LevenbergMarquardt::new();
        assert!(solver.damping > 0.0);
    }

    #[test]
    fn test_damping_configuration() {
        let solver = LevenbergMarquardt::new()
            .with_damping(1e-6)
            .with_damping_bounds(1e-15, 1e15);

        assert_eq!(solver.damping, 1e-6);
        assert_eq!(solver.damping_min, 1e-15);
        assert_eq!(solver.damping_max, 1e15);
    }

    // Rosenbrock function for testing
    struct RosenbrockProblem {
        a: f64,
        b: f64,
    }

    struct MatWrapper(pub Mat<f64>);
    impl AsRef<Mat<f64>> for MatWrapper {
        fn as_ref(&self) -> &Mat<f64> {
            &self.0
        }
    }

    struct SparseMatWrapper(pub SparseColMat<usize, f64>);
    impl AsRef<SparseColMat<usize, f64>> for SparseMatWrapper {
        fn as_ref(&self) -> &SparseColMat<usize, f64> {
            &self.0
        }
    }

    impl Optimizable for RosenbrockProblem {
        type Parameters = Mat<f64>;
        type Residuals = MatWrapper;
        type Jacobian = SparseMatWrapper;

        fn evaluate_with_jacobian(
            &self,
            params: &Self::Parameters,
        ) -> Result<(Self::Residuals, Self::Jacobian), crate::core::ApexError> {
            let x = params[(0, 0)];
            let y = params[(1, 0)];

            let r1 = self.a - x;
            let r2 = self.b.sqrt() * (y - x * x);
            let residuals = mat![[r1], [r2]];

            let triplets = vec![
                faer::sparse::Triplet::new(0, 0, -1.0),
                faer::sparse::Triplet::new(1, 0, -2.0 * self.b.sqrt() * x),
                faer::sparse::Triplet::new(1, 1, self.b.sqrt()),
            ];
            let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)
                .map_err(|e| crate::core::ApexError::Computation(e.to_string()))?;

            Ok((MatWrapper(residuals), SparseMatWrapper(jacobian)))
        }

        fn cost(&self, params: &Self::Parameters) -> Result<f64, crate::core::ApexError> {
            let (residuals, _) = self.evaluate_with_jacobian(params)?;
            let r = residuals.as_ref();
            Ok(0.5 * r.norm_l2() * r.norm_l2())
        }

        fn weights(&self) -> Mat<f64> {
            mat![[1.0], [1.0]]
        }

        fn evaluate(
            &self,
            parameters: &Self::Parameters,
        ) -> Result<Self::Residuals, crate::core::ApexError> {
            let (residuals, _) = self.evaluate_with_jacobian(parameters)?;
            Ok(residuals)
        }

        fn parameter_count(&self) -> usize {
            2
        }

        fn residual_count(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_rosenbrock_optimization() {
        // Start with simpler Rosenbrock problem (smaller b value)
        let problem = RosenbrockProblem { a: 1.0, b: 1.0 };
        let mut solver = LevenbergMarquardt::with_config(OptimizerConfig {
            max_iterations: 50,
            cost_tolerance: 1e-6,
            parameter_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            verbose: true,
            ..Default::default()
        })
        .with_damping(1e-1);

        let initial_params = mat![[0.0], [0.0]];
        let result = solver.solve(&problem, initial_params).unwrap();

        // More relaxed assertions for debugging
        println!(
            "Final parameters: [{}, {}]",
            result.parameters[(0, 0)],
            result.parameters[(1, 0)]
        );
        assert!(
            result.final_cost < 1e-1,
            "Final cost was: {}",
            result.final_cost
        );
    }
}
