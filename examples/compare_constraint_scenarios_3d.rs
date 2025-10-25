use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::factors::{BetweenFactorSE3, PriorFactor};
use apex_solver::core::loss_functions::*;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt};
use nalgebra::{Quaternion, UnitQuaternion, Vector3, dvector};

/// Result from a single optimization run with specific dataset, optimizer, and constraint scenario
#[derive(Clone)]
struct ScenarioResult {
    dataset: String,
    optimizer: String,
    scenario: String, // "Neither", "Prior", "Fixed"
    vertices: usize,
    edges: usize,
    initial_cost: f64,
    final_cost: f64,
    cost_reduction: f64,
    improvement_percent: f64,
    iterations: usize,
    time_ms: u128,
    status: String,
    convergence_reason: String,
    // First vertex tracking
    x0_translation_movement: f64,
    x0_rotation_angle_change: f64,
}

/// Apply constraint scenario to the problem
fn apply_constraint_scenario(
    problem: &mut Problem,
    scenario: &str,
    first_vertex_id: usize,
    first_vertex_pose: &SE3,
) {
    let var_name = format!("x{}", first_vertex_id);

    match scenario {
        "Neither" => {
            // No constraints - gauge freedom
            println!("  Constraint: None (gauge freedom - x0 can drift)");
        }
        "Prior" => {
            // Add soft prior factor with HuberLoss
            let quat = first_vertex_pose.rotation_quaternion();
            let trans = first_vertex_pose.translation();
            let prior_value = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];

            let prior_factor = PriorFactor {
                data: prior_value.clone(),
            };

            let huber_loss = HuberLoss::new(1.0).expect("Failed to create HuberLoss");
            problem.add_residual_block(
                &[&var_name],
                Box::new(prior_factor),
                Some(Box::new(huber_loss)),
            );

            println!("  Constraint: Prior factor with HuberLoss (soft constraint on x0)");
        }
        "Fixed" => {
            // Fix all 6 DOF of first vertex (translation xyz + rotation xyz in tangent space)
            for dof in 0..6 {
                problem.fix_variable(&var_name, dof);
            }
            println!("  Constraint: Fixed all 6 DOF of x0 (hard constraint - zero movement)");
        }
        _ => panic!("Unknown scenario: {}", scenario),
    }
}

/// Extract SE3 pose from DVector representation
fn dvector_to_se3(vec: &nalgebra::DVector<f64>) -> SE3 {
    SE3::from_translation_quaternion(
        Vector3::new(vec[0], vec[1], vec[2]),
        Quaternion::new(vec[3], vec[4], vec[5], vec[6]),
    )
}

/// Compute angle between two unit quaternions in degrees
fn quaternion_angle_diff(q1: &UnitQuaternion<f64>, q2: &UnitQuaternion<f64>) -> f64 {
    let q_diff = q1.inverse() * q2;
    let angle_rad = 2.0 * q_diff.as_ref().w.acos();
    angle_rad.abs().to_degrees()
}

/// Run optimization for a specific dataset, optimizer, and constraint scenario
fn run_optimization(
    dataset_name: &str,
    optimizer_type: &str,
    scenario: &str,
) -> Result<ScenarioResult, Box<dyn std::error::Error>> {
    // Dataset-specific optimizations
    let (cost_tol, param_tol, max_iter) = match dataset_name {
        "grid3D" => (1e-1, 1e-1, 30),
        "rim" => (1e-3, 1e-3, 100),
        "torus3D" => (1e-5, 1e-5, 100),
        _ => (1e-4, 1e-4, 100),
    };

    // Load the G2O graph file
    let dataset_path = format!("data/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    if num_vertices == 0 {
        return Err(format!("No SE3 vertices found in dataset {}", dataset_name).into());
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE3 vertices as variables
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Get first vertex for tracking
    let first_vertex_id = *vertex_ids.first().unwrap();
    let first_vertex = graph.vertices_se3.get(&first_vertex_id).unwrap();
    let initial_x0_pose = first_vertex.pose.clone();
    let initial_x0_translation = initial_x0_pose.translation();
    let initial_x0_rotation = initial_x0_pose.rotation_quaternion();

    // Apply constraint scenario
    apply_constraint_scenario(&mut problem, scenario, first_vertex_id, &first_vertex.pose);

    // Add SE3 between factors (all use L2 loss for fair comparison)
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactorSE3::new(relative_pose);
        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), None);
    }

    // Compute initial cost
    let variables = problem.initialize_variables(&initial_values);
    let mut variable_name_to_col_idx_dict = HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
    sorted_vars.sort();
    for var_name in &sorted_vars {
        variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    let symbolic_structure = problem
        .build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset)
        .expect("Failed to build symbolic structure");

    let (residual, _jacobian) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .expect("Failed to compute residual and jacobian");

    let initial_cost = residual.as_ref().squared_norm_l2();

    // Run optimization
    let start_time = Instant::now();
    let result = match optimizer_type {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12)
                .with_verbose(false)
                .with_visualization(false);
            let mut solver = GaussNewton::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12)
                .with_verbose(false)
                .with_visualization(false);
            let mut solver = DogLeg::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        _ => {
            // Default to LM
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12)
                .with_verbose(false)
                .with_visualization(false);
            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
    };
    let duration = start_time.elapsed();

    // Determine convergence status
    let (status, convergence_reason) = match &result.status {
        apex_solver::optimizer::OptimizationStatus::Converged => {
            ("CONVERGED", "Converged".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::CostToleranceReached => {
            ("CONVERGED", "CostTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached => {
            ("CONVERGED", "ParameterTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::GradientToleranceReached => {
            ("CONVERGED", "GradientTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::TrustRegionRadiusTooSmall => {
            ("CONVERGED", "TrustRegionRadiusTooSmall".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::MinCostThresholdReached => {
            ("CONVERGED", "MinCostThreshold".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::MaxIterationsReached => {
            ("NOT_CONVERGED", "MaxIterations".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::Timeout => {
            ("NOT_CONVERGED", "Timeout".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::NumericalFailure => {
            ("NOT_CONVERGED", "NumericalFailure".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::IllConditionedJacobian => {
            ("NOT_CONVERGED", "IllConditionedJacobian".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::InvalidNumericalValues => {
            ("NOT_CONVERGED", "InvalidNumericalValues".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::UserTerminated => {
            ("NOT_CONVERGED", "UserTerminated".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::Failed(msg) => {
            ("NOT_CONVERGED", format!("Failed:{}", msg))
        }
    };

    // Track first vertex movement
    let var_name = format!("x{}", first_vertex_id);
    let final_var = result.parameters.get(&var_name).unwrap();
    let final_x0_vector = final_var.to_vector();
    let final_x0_pose = dvector_to_se3(&final_x0_vector);
    let final_x0_translation = final_x0_pose.translation();
    let final_x0_rotation = final_x0_pose.rotation_quaternion();

    let x0_translation_movement = (final_x0_translation - initial_x0_translation).norm();
    let x0_rotation_angle_change = quaternion_angle_diff(&initial_x0_rotation, &final_x0_rotation);

    let cost_reduction = initial_cost - result.final_cost;
    let improvement_percent = (cost_reduction / initial_cost) * 100.0;

    Ok(ScenarioResult {
        dataset: dataset_name.to_string(),
        optimizer: optimizer_type.to_string(),
        scenario: scenario.to_string(),
        vertices: num_vertices,
        edges: num_edges,
        initial_cost,
        final_cost: result.final_cost,
        cost_reduction,
        improvement_percent,
        iterations: result.iterations,
        time_ms: duration.as_millis(),
        status: status.to_string(),
        convergence_reason: convergence_reason.to_string(),
        x0_translation_movement,
        x0_rotation_angle_change,
    })
}

/// Print comprehensive summary table with all 54 results
fn print_complete_results_table(results: &[ScenarioResult]) {
    println!("\n{}", "=".repeat(180));
    println!("=== TABLE 1: COMPLETE RESULTS (All 54 Optimization Runs) ===");
    println!("{}", "=".repeat(180));

    // Header
    println!(
        "{:<16} | {:<10} | {:<10} | {:<7} | {:<6} | {:<12} | {:<12} | {:<12} | {:<8} | {:<5} | {:<8} | {:<10} | {:<10} | {:<15}",
        "Dataset",
        "Optimizer",
        "Scenario",
        "Vert",
        "Edges",
        "Init Cost",
        "Final Cost",
        "Cost Reduction",
        "Improv%",
        "Iters",
        "Time(ms)",
        "x0 Trans",
        "x0 Rot(°)",
        "Status"
    );
    println!("{}", "-".repeat(180));

    for result in results {
        println!(
            "{:<16} | {:<10} | {:<10} | {:<7} | {:<6} | {:<12.6e} | {:<12.6e} | {:<12.6e} | {:>7.2}% | {:<5} | {:<8} | {:<10.6} | {:>9.2} | {:<15}",
            result.dataset,
            result.optimizer,
            result.scenario,
            result.vertices,
            result.edges,
            result.initial_cost,
            result.final_cost,
            result.cost_reduction,
            result.improvement_percent,
            result.iterations,
            result.time_ms,
            result.x0_translation_movement,
            result.x0_rotation_angle_change,
            result.convergence_reason
        );
    }

    println!("{}", "-".repeat(180));

    // Summary statistics
    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    let total_count = results.len();
    println!(
        "\nSummary: {}/{} optimization runs converged successfully ({:.1}%)",
        converged_count,
        total_count,
        (converged_count as f64 / total_count as f64) * 100.0
    );
}

/// Print scenario comparison grouped by optimizer
fn print_scenario_comparison_by_optimizer(results: &[ScenarioResult]) {
    println!("\n{}", "=".repeat(160));
    println!("=== TABLE 2: SCENARIO COMPARISON BY OPTIMIZER ===");
    println!("{}", "=".repeat(160));

    let datasets = vec![
        "rim",
        "sphere2500",
        "parking-garage",
        "torus3D",
        "grid3D",
        "cubicle",
    ];
    let optimizers = vec!["LM", "GN", "DL"];

    for optimizer in &optimizers {
        println!("\n--- {} OPTIMIZER ---", optimizer);
        println!(
            "{:<16} | {:<25} | {:<25} | {:<25} | {:<12}",
            "Dataset", "Neither (x0 move)", "Prior (x0 move)", "Fixed (x0 move)", "Best Cost"
        );
        println!("{}", "-".repeat(110));

        for dataset in &datasets {
            let neither = results
                .iter()
                .find(|r| {
                    r.dataset == *dataset && r.optimizer == *optimizer && r.scenario == "Neither"
                })
                .unwrap();
            let prior = results
                .iter()
                .find(|r| {
                    r.dataset == *dataset && r.optimizer == *optimizer && r.scenario == "Prior"
                })
                .unwrap();
            let fixed = results
                .iter()
                .find(|r| {
                    r.dataset == *dataset && r.optimizer == *optimizer && r.scenario == "Fixed"
                })
                .unwrap();

            let best_scenario =
                if neither.final_cost < prior.final_cost && neither.final_cost < fixed.final_cost {
                    "Neither"
                } else if prior.final_cost < fixed.final_cost {
                    "Prior"
                } else {
                    "Fixed"
                };

            println!(
                "{:<16} | {:>6.2}% ({:>6.3}) | {:>6.2}% ({:>6.3}) | {:>6.2}% ({:>6.3}) | {:<12}",
                dataset,
                neither.improvement_percent,
                neither.x0_translation_movement,
                prior.improvement_percent,
                prior.x0_translation_movement,
                fixed.improvement_percent,
                fixed.x0_translation_movement,
                best_scenario
            );
        }
    }
}

/// Print optimizer comparison grouped by scenario
fn print_optimizer_comparison_by_scenario(results: &[ScenarioResult]) {
    println!("\n{}", "=".repeat(160));
    println!("=== TABLE 3: OPTIMIZER COMPARISON BY SCENARIO ===");
    println!("{}", "=".repeat(160));

    let datasets = vec![
        "rim",
        "sphere2500",
        "parking-garage",
        "torus3D",
        "grid3D",
        "cubicle",
    ];
    let scenarios = vec!["Neither", "Prior", "Fixed"];

    for scenario in &scenarios {
        println!("\n--- {} SCENARIO ---", scenario);
        println!(
            "{:<16} | {:<20} | {:<20} | {:<20} | {:<12}",
            "Dataset", "LM (cost/iters)", "GN (cost/iters)", "DL (cost/iters)", "Fastest"
        );
        println!("{}", "-".repeat(95));

        for dataset in &datasets {
            let lm = results
                .iter()
                .find(|r| r.dataset == *dataset && r.optimizer == "LM" && r.scenario == *scenario)
                .unwrap();
            let gn = results
                .iter()
                .find(|r| r.dataset == *dataset && r.optimizer == "GN" && r.scenario == *scenario)
                .unwrap();
            let dl = results
                .iter()
                .find(|r| r.dataset == *dataset && r.optimizer == "DL" && r.scenario == *scenario)
                .unwrap();

            let fastest = if lm.time_ms < gn.time_ms && lm.time_ms < dl.time_ms {
                "LM"
            } else if gn.time_ms < dl.time_ms {
                "GN"
            } else {
                "DL"
            };

            println!(
                "{:<16} | {:>9.3e}/{:<3} | {:>9.3e}/{:<3} | {:>9.3e}/{:<3} | {:<12}",
                dataset,
                lm.final_cost,
                lm.iterations,
                gn.final_cost,
                gn.iterations,
                dl.final_cost,
                dl.iterations,
                fastest
            );
        }
    }
}

/// Print key insights and verification
fn print_key_insights(results: &[ScenarioResult]) {
    println!("\n{}", "=".repeat(120));
    println!("=== TABLE 4: KEY INSIGHTS AND VERIFICATION ===");
    println!("{}", "=".repeat(120));

    // Verify fixed variables have exactly zero movement
    println!("\n✓ VERIFICATION: Fixed Variable Constraint");
    println!("{}", "-".repeat(80));
    let fixed_results: Vec<_> = results.iter().filter(|r| r.scenario == "Fixed").collect();

    let all_zero = fixed_results
        .iter()
        .all(|r| r.x0_translation_movement < 1e-10 && r.x0_rotation_angle_change < 1e-6);

    if all_zero {
        println!(
            "✅ SUCCESS: All {} 'Fixed' scenarios have exactly ZERO x0 movement!",
            fixed_results.len()
        );
        println!(
            "   Max translation movement: {:.3e}",
            fixed_results
                .iter()
                .map(|r| r.x0_translation_movement)
                .fold(0.0, f64::max)
        );
        println!(
            "   Max rotation change: {:.3e}°",
            fixed_results
                .iter()
                .map(|r| r.x0_rotation_angle_change)
                .fold(0.0, f64::max)
        );
    } else {
        println!("❌ WARNING: Some 'Fixed' scenarios show non-zero movement!");
        for r in fixed_results
            .iter()
            .filter(|r| r.x0_translation_movement > 1e-10 || r.x0_rotation_angle_change > 1e-6)
        {
            println!(
                "   {}-{}: trans={:.3e}, rot={:.3e}°",
                r.dataset, r.optimizer, r.x0_translation_movement, r.x0_rotation_angle_change
            );
        }
    }

    // Compare x0 movement across scenarios
    println!("\n📊 ANALYSIS: x0 Movement Comparison");
    println!("{}", "-".repeat(80));
    let datasets = vec![
        "rim",
        "sphere2500",
        "parking-garage",
        "torus3D",
        "grid3D",
        "cubicle",
    ];

    println!(
        "{:<16} | {:<12} | {:<12} | {:<12}",
        "Dataset", "Neither Avg", "Prior Avg", "Fixed Avg"
    );
    println!("{}", "-".repeat(60));

    for dataset in &datasets {
        let neither_avg: f64 = results
            .iter()
            .filter(|r| r.dataset == *dataset && r.scenario == "Neither")
            .map(|r| r.x0_translation_movement)
            .sum::<f64>()
            / 3.0;

        let prior_avg: f64 = results
            .iter()
            .filter(|r| r.dataset == *dataset && r.scenario == "Prior")
            .map(|r| r.x0_translation_movement)
            .sum::<f64>()
            / 3.0;

        let fixed_avg: f64 = results
            .iter()
            .filter(|r| r.dataset == *dataset && r.scenario == "Fixed")
            .map(|r| r.x0_translation_movement)
            .sum::<f64>()
            / 3.0;

        println!(
            "{:<16} | {:>11.3} | {:>11.3} | {:>11.3}",
            dataset, neither_avg, prior_avg, fixed_avg
        );
    }

    // Best optimizer per scenario
    println!("\n🏆 PERFORMANCE: Best Optimizer by Scenario");
    println!("{}", "-".repeat(80));

    for scenario in &["Neither", "Prior", "Fixed"] {
        let scenario_results: Vec<_> = results
            .iter()
            .filter(|r| r.scenario == *scenario && r.status == "CONVERGED")
            .collect();

        if scenario_results.is_empty() {
            continue;
        }

        let avg_time_lm: f64 = scenario_results
            .iter()
            .filter(|r| r.optimizer == "LM")
            .map(|r| r.time_ms as f64)
            .sum::<f64>()
            / scenario_results
                .iter()
                .filter(|r| r.optimizer == "LM")
                .count() as f64;

        let avg_time_gn: f64 = scenario_results
            .iter()
            .filter(|r| r.optimizer == "GN")
            .map(|r| r.time_ms as f64)
            .sum::<f64>()
            / scenario_results
                .iter()
                .filter(|r| r.optimizer == "GN")
                .count() as f64;

        let avg_time_dl: f64 = scenario_results
            .iter()
            .filter(|r| r.optimizer == "DL")
            .map(|r| r.time_ms as f64)
            .sum::<f64>()
            / scenario_results
                .iter()
                .filter(|r| r.optimizer == "DL")
                .count() as f64;

        let fastest = if avg_time_lm < avg_time_gn && avg_time_lm < avg_time_dl {
            "LM"
        } else if avg_time_gn < avg_time_dl {
            "GN"
        } else {
            "DL"
        };

        println!(
            "{:<10} scenario: LM={:.0}ms, GN={:.0}ms, DL={:.0}ms => Fastest: {}",
            scenario, avg_time_lm, avg_time_gn, avg_time_dl, fastest
        );
    }

    // Convergence rate by scenario
    println!("\n📈 CONVERGENCE: Success Rate by Scenario");
    println!("{}", "-".repeat(80));

    for scenario in &["Neither", "Prior", "Fixed"] {
        let scenario_results: Vec<_> = results.iter().filter(|r| r.scenario == *scenario).collect();
        let converged = scenario_results
            .iter()
            .filter(|r| r.status == "CONVERGED")
            .count();
        let total = scenario_results.len();
        println!(
            "{:<10} scenario: {}/{} converged ({:.1}%)",
            scenario,
            converged,
            total,
            (converged as f64 / total as f64) * 100.0
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(120));
    println!("=== COMPREHENSIVE 3D CONSTRAINT SCENARIO COMPARISON ===");
    println!("=== Testing: 6 Datasets × 3 Optimizers × 3 Scenarios = 54 Runs ===");
    println!("{}", "=".repeat(120));
    println!();
    println!("Datasets: rim, sphere2500, parking-garage, torus3D, grid3D, cubicle");
    println!("Optimizers: LM (Levenberg-Marquardt), GN (Gauss-Newton), DL (Dog Leg)");
    println!("Scenarios:");
    println!("  - Neither: No constraint (gauge freedom - x0 can drift)");
    println!("  - Prior: Soft constraint via PriorFactor with HuberLoss");
    println!("  - Fixed: Hard constraint - all 6 DOF fixed (x0 cannot move)");
    println!("Loss Function: L2 (squared error) for all runs");
    println!();
    println!("Starting comprehensive benchmark...");
    println!("{}", "=".repeat(120));

    let datasets = vec![
        "rim",
        "sphere2500",
        "parking-garage",
        "torus3D",
        "grid3D",
        "cubicle",
    ];
    let optimizers = vec!["LM", "GN", "DL"];
    let scenarios = vec!["Neither", "Prior", "Fixed"];

    let mut all_results = Vec::new();
    let mut run_count = 0;
    let total_runs = datasets.len() * optimizers.len() * scenarios.len();

    let overall_start = Instant::now();

    for dataset in &datasets {
        for optimizer in &optimizers {
            for scenario in &scenarios {
                run_count += 1;
                println!(
                    "\n[{}/{}] Testing: {} | {} | {}",
                    run_count, total_runs, dataset, optimizer, scenario
                );

                match run_optimization(dataset, optimizer, scenario) {
                    Ok(result) => {
                        println!(
                            "  ✓ {} | Cost: {:.3e} -> {:.3e} ({:.2}%) | Iters: {} | Time: {}ms | x0 move: {:.3}m",
                            result.convergence_reason,
                            result.initial_cost,
                            result.final_cost,
                            result.improvement_percent,
                            result.iterations,
                            result.time_ms,
                            result.x0_translation_movement
                        );
                        all_results.push(result);
                    }
                    Err(e) => {
                        eprintln!("  ✗ FAILED: {}", e);
                    }
                }
            }
        }
    }

    let overall_duration = overall_start.elapsed();

    println!("\n{}", "=".repeat(120));
    println!("=== BENCHMARK COMPLETED ===");
    println!(
        "Total runs: {} | Successful: {} | Failed: {} | Total time: {:.1}s",
        total_runs,
        all_results.len(),
        total_runs - all_results.len(),
        overall_duration.as_secs_f64()
    );
    println!("{}", "=".repeat(120));

    // Print comprehensive analysis tables
    print_complete_results_table(&all_results);
    print_scenario_comparison_by_optimizer(&all_results);
    print_optimizer_comparison_by_scenario(&all_results);
    print_key_insights(&all_results);

    println!("\n{}", "=".repeat(120));
    println!("=== FINAL SUMMARY ===");
    println!("{}", "=".repeat(120));
    println!(
        "✅ Benchmark completed successfully: {} optimization runs",
        all_results.len()
    );
    println!(
        "⏱️  Total execution time: {:.1} seconds",
        overall_duration.as_secs_f64()
    );
    println!(
        "📊 Average time per run: {:.1} ms",
        (overall_duration.as_millis() as f64) / (all_results.len() as f64)
    );

    Ok(())
}
