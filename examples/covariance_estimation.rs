use apex_manifolds::se2::SE2;
use apex_solver::{
    JacobianMode, ManifoldType,
    core::problem::Problem,
    factors::{BetweenFactor, PriorFactor},
    init_logger,
    linalg::LinearSolverType,
    linalg::covariance::{Covariance, CovarianceAlgorithm, CovarianceOptions},
    optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig},
};
use nalgebra::dvector;
use tracing::{info, warn};

fn main() {
    init_logger();

    info!("Covariance Estimation Example\n");

    let mut problem = Problem::new(JacobianMode::Sparse);

    info!("Creating pose graph with 4 SE2 poses...");

    let x0 = problem.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
    let x1 = problem.add_variable(ManifoldType::SE2, dvector![0.95, 0.05, 0.02]);
    let x2 = problem.add_variable(ManifoldType::SE2, dvector![1.9, 0.1, 0.12]);
    let x3 = problem.add_variable(ManifoldType::SE2, dvector![2.85, 0.15, 0.05]);

    problem.add_residual_block(
        &[x0, x1],
        Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0))),
        None,
    );
    problem.add_residual_block(
        &[x1, x2],
        Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1))),
        None,
    );
    problem.add_residual_block(
        &[x2, x3],
        Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, -0.1))),
        None,
    );
    problem.add_residual_block(
        &[x0, x3],
        Box::new(BetweenFactor::new(SE2::from_xy_angle(3.0, 0.0, 0.0))),
        None,
    );

    // Anchor the gauge. `BetweenFactor` constraints are all relative, so without a
    // prior the whole graph can be rigidly translated and rotated at zero cost:
    // H = JᵀJ is singular with a 3-dimensional null space, and the covariance is
    // undefined. Previously this was masked by Levenberg-Marquardt damping leaking
    // into the covariance (H + λI is invertible even when H is not) — the bug
    // behind issue #38. Now it would be reported as a rank deficiency, so the
    // example fixes the gauge properly.
    problem.add_residual_block(
        &[x0],
        Box::new(PriorFactor {
            data: dvector![0.0, 0.0, 0.0],
        }),
        None,
    );

    info!("Added 3 odometry constraints, 1 loop closure, and 1 prior on x0\n");

    info!("Configuring Levenberg-Marquardt optimizer with covariance estimation...");
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(50)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-6)
        .with_compute_covariances(true);

    let mut solver = LevenbergMarquardt::with_config(config);

    info!("Running optimization...\n");
    match solver.optimize(&mut problem) {
        Ok(result) => {
            info!("Optimization succeeded!");
            info!("  Status: {:?}", result.status);
            info!("  Iterations: {}", result.iterations);
            info!("  Initial cost: {:.6e}", result.initial_cost);
            info!("  Final cost: {:.6e}", result.final_cost);
            info!(
                "  Cost reduction: {:.2}%\n",
                100.0 * (result.initial_cost - result.final_cost) / result.initial_cost
            );

            info!("Optimized poses:");
            for (key, variable) in result.parameters.iter() {
                let vec = variable.to_dvector();
                info!(
                    "  {:?}: x={:.4}, y={:.4}, theta={:.4}",
                    key, vec[0], vec[1], vec[2]
                );
            }

            if let Some(covariances) = &result.covariances {
                info!("Covariance Analysis");
                info!("Covariances computed for {} variables", covariances.len());

                for (label, key) in [("x0", x0), ("x1", x1), ("x2", x2), ("x3", x3)] {
                    if let Some(cov) = covariances.get(key) {
                        info!("Variable '{}':", label);
                        info!("  Full 3x3 covariance matrix (tangent space):");
                        for i in 0..3 {
                            let row = (0..3)
                                .map(|j| format!("{:9.6e}", cov[(i, j)]))
                                .collect::<Vec<_>>()
                                .join(" ");
                            info!("    [ {} ]", row);
                        }

                        let std_x = cov[(0, 0)].sqrt();
                        let std_y = cov[(1, 1)].sqrt();
                        let std_theta = cov[(2, 2)].sqrt();

                        info!("  Standard deviations (1-sigma uncertainty):");
                        info!("    sigma_x     = {:.6} m", std_x);
                        info!("    sigma_y     = {:.6} m", std_y);
                        info!(
                            "    sigma_theta = {:.6} rad ({:.3} deg)",
                            std_theta,
                            std_theta.to_degrees()
                        );

                        info!("  95% confidence intervals (+-2 sigma):");
                        info!("    x:     +-{:.6} m", 2.0 * std_x);
                        info!("    y:     +-{:.6} m", 2.0 * std_y);
                        info!(
                            "    theta: +-{:.6} rad (+-{:.3} deg)",
                            2.0 * std_theta,
                            (2.0 * std_theta).to_degrees()
                        );
                    }
                }

                // The explicit API: same numbers, but computable at any point, with
                // a choice of algorithm, cross-covariance blocks, and a typed error
                // instead of a silent `None`.
                info!("Explicit Covariance API");
                match Covariance::compute(
                    CovarianceOptions::default(),
                    &problem,
                    &result.parameters,
                ) {
                    Ok(cov) => {
                        info!("Recomputed post-hoc via Covariance::compute");
                        if let Some(block) = cov.block(x1) {
                            info!(
                                "  x1 variances: [{:.6e}, {:.6e}, {:.6e}]",
                                block[(0, 0)],
                                block[(1, 1)],
                                block[(2, 2)]
                            );
                        }
                        // Cross-covariance is only available through this API.
                        if let Some(cross) = cov.block_pair(x1, x2) {
                            info!(
                                "  cov(x1, x2) leading entry: {:.6e} (correlation between neighbours)",
                                cross[(0, 0)]
                            );
                        }
                    }
                    Err(e) => warn!("Covariance::compute failed: {e}"),
                }

                // Gauge-free problems can still be handled explicitly, via the
                // pseudo-inverse, instead of failing.
                match Covariance::compute(
                    CovarianceOptions::new().with_algorithm(CovarianceAlgorithm::DenseSvd),
                    &problem,
                    &result.parameters,
                ) {
                    Ok(cov) => {
                        if let Some(block) = cov.block(x0) {
                            info!("  DenseSvd agrees on x0 variance: {:.6e}", block[(0, 0)]);
                        }
                    }
                    Err(e) => warn!("DenseSvd covariance failed: {e}"),
                }

                info!("Interpretation");
                info!("- Covariance matrices are in the tangent space (local coordinates)");
                info!("- Diagonal elements are variances; off-diagonal are correlations");
                info!("- Smaller values = higher confidence (less uncertainty)");
                info!("- First pose (x0) typically has smallest uncertainty (anchor)");
                info!("- Uncertainty typically grows with distance from constraints");
                info!("- Loop closures reduce uncertainty in the graph");
            } else {
                info!("No covariances computed!");
                info!("Make sure .with_compute_covariances(true) is set in config.");
            }
        }
        Err(e) => {
            warn!("Optimization failed: {:?}", e);
        }
    }

    info!("Usage Notes");
    info!("To enable covariance estimation in your code:");
    info!("  1. Set .with_compute_covariances(true) in optimizer config");
    info!("  2. Access result.covariances after optimization");
    info!("  3. Each variable gets its own tangent-space covariance matrix");
    info!("  4. For SE2: 3x3 matrix [x, y, theta]");
    info!("  5. For SE3: 6x6 matrix [trans_xyz, rot_xyz]");
    info!("  Note: Covariance computation adds ~10-20% overhead.");
}
