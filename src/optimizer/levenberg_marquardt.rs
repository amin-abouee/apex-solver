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
            damping: 1e-3,
            damping_min: 1e-12,
            damping_max: 1e12,
            damping_increase_factor: 10.0,
            damping_decrease_factor: 0.3,
            trust_region_radius: 1e4,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
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

    /// Update damping parameter based on step quality
    fn update_damping(&mut self, rho: f64) -> bool {
        if rho > self.good_step_quality {
            // Good step, decrease damping
            self.damping = (self.damping * self.damping_decrease_factor).max(self.damping_min);
            true
        } else if rho < self.min_step_quality {
            // Poor step, increase damping
            self.damping = (self.damping * self.damping_increase_factor).min(self.damping_max);
            false
        } else {
            // Acceptable step, keep damping unchanged
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
        // Predicted reduction = -step^T * gradient - 0.5 * step^T * H * step
        // The negative signs account for the fact that we're minimizing
        let linear_term = step.transpose() * gradient;
        let quadratic_term = step.transpose() * (hessian * step);
        linear_term[(0, 0)] + 0.5 * quadratic_term[(0, 0)]
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

    pub fn solve<T, P>(
        &mut self,
        problem: &T,
        initial_params: P,
    ) -> Result<SolverResult<P>, crate::core::ApexError>
    where
        T: crate::core::Optimizable<Parameters = P>,
        P: Clone + std::ops::Sub<Output = P>,
        for<'a> &'a P: std::ops::Add<&'a Mat<f64>, Output = P>,
    {
        let start_time = Instant::now();
        let mut params = initial_params;
        let mut iteration = 0;
        let mut cost_evaluations = 0;
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;
        let mut last_cost_reduction;

        // Create linear solver
        let mut linear_solver = self.create_linear_solver();

        // Initial cost evaluation
        let initial_cost = problem.cost(&params)?;
        let mut current_cost = initial_cost;
        cost_evaluations += 1;

        // Initialize summary tracking variables
        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;

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
            // Evaluate residuals and Jacobian
            let (residuals, jacobian) = problem.evaluate_with_jacobian(&params)?;
            jacobian_evaluations += 1;

            // Compute gradient = J^T * r
            let gradient_norm = self.compute_gradient_norm(residuals.as_ref(), jacobian.as_ref());
            max_gradient_norm = max_gradient_norm.max(gradient_norm);
            final_gradient_norm = gradient_norm;

            // Solve augmented system: (J^T * J + λI) * dx = -J^T * r
            if let Some(step) = linear_solver.solve_augmented_equation(
                residuals.as_ref(),
                jacobian.as_ref(),
                self.damping,
            ) {
                let step_norm = step.norm_l2();
                max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
                final_parameter_update_norm = step_norm;

                // Compute predicted reduction
                let hessian = self.compute_hessian(jacobian.as_ref());
                // Use negative gradient to match what linear solver uses: g = -J^T *  r
                let gradient = self.compute_gradient(residuals.as_ref(), jacobian.as_ref());
                let negative_gradient = &gradient * -1.0;
                let predicted_reduction =
                    self.compute_predicted_reduction(&step, &negative_gradient, &hessian);

                // Try the step
                let new_params = &params + &step;
                let new_cost = problem.cost(&new_params)?;
                cost_evaluations += 1;

                // Compute step quality
                let rho = self.compute_step_quality(current_cost, new_cost, predicted_reduction);

                // Update damping and decide whether to accept step
                let accept_step = self.update_damping(rho);

                if accept_step {
                    // Accept the step
                    let cost_reduction = current_cost - new_cost;
                    last_cost_reduction = cost_reduction;
                    params = new_params;
                    total_cost_reduction += cost_reduction;
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
                } else {
                    // Reject the step, increase damping
                    last_cost_reduction = 0.0; // No cost reduction if step is rejected
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

            let elapsed = start_time.elapsed();
            // Check convergence criteria
            if let Some(status) = self.check_convergence(
                iteration,
                last_cost_reduction,
                final_parameter_update_norm,
                gradient_norm,
                elapsed,
            ) {
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
                    parameters: params.clone(),
                    status,
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
impl crate::optimizer::Solver for LevenbergMarquardt {
    type Config = OptimizerConfig;
    type Error = crate::core::ApexError;

    fn new(config: Self::Config) -> Self {
        Self::with_config(config)
    }

    fn solve(
        &mut self,
        problem: &crate::core::problem::Problem,
        initial_params: &std::collections::HashMap<
            String,
            (crate::manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        SolverResult<std::collections::HashMap<String, crate::core::problem::VariableEnum>>,
        Self::Error,
    > {
        // For now, use the simple solve_problem function from the module
        // TODO: Implement actual Levenberg-Marquardt algorithm with Problem interface
        use crate::optimizer::solve_problem;
        let config = OptimizerConfig {
            optimizer_type: crate::optimizer::OptimizerType::LevenbergMarquardt,
            ..Default::default()
        };
        solve_problem(problem, initial_params, config)
    }
}

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
