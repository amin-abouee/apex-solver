#include <ceres/ceres.h>
#include <Eigen/Core>

#include "../../common/include/benchmark_utils.h"
#include "../../common/include/read_g2o.h"
#include "../../common/include/unified_cost.h"
#include "../include/ceres_odometry.h"

// Cost functor classes moved to ceres_benchmark.h

// Benchmark SE2 dataset with Ceres
benchmark_utils::BenchmarkResult BenchmarkSE2(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE2";
    result.solver = "Ceres-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph2D graph;
    if (!g2o_reader::ReadG2oFile2D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Store initial graph for cost computation
    g2o_reader::Graph2D initial_graph = graph;

    // Compute initial cost using unified cost function (both metrics)
    auto initial_metrics = unified_cost::ComputeSE2CostMetrics(initial_graph);
    result.initial_chi2 = initial_metrics.chi2_cost;
    result.initial_cost = initial_metrics.unweighted_cost;

    // Create Ceres problem
    ceres::Problem problem;

    // Map to store pose parameters (use std::vector for stable addresses)
    std::map<int, std::vector<double>> pose_params;
    for (const auto& [id, pose] : graph.poses) {
        pose_params[id] = {pose.translation.x(), pose.translation.y(), pose.rotation};
    }

    // Add edge constraints first (so parameter blocks are registered)
    for (const auto& constraint : graph.constraints) {
        auto& pose_a = pose_params[constraint.id_begin];
        auto& pose_b = pose_params[constraint.id_end];

        // Pass identity matrix instead of sqrt(information matrix)
        Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

        ceres::CostFunction* cost_function = PoseGraph2DErrorTerm::Create(
            constraint.measurement.translation.x(),
            constraint.measurement.translation.y(),
            constraint.measurement.rotation,
            identity);

        problem.AddResidualBlock(cost_function, nullptr, pose_a.data(), pose_b.data());
    }

    // Add prior on first pose (gauge freedom) - after residual blocks are added
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        auto& first_pose = pose_params[first_id];
        problem.SetParameterBlockConstant(first_pose.data());
    }

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-3;
    options.parameter_tolerance = 1e-3;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // Solve
    ceres::Solver::Summary summary;
    Timer timer;
    ceres::Solve(options, &problem, &summary);
    result.time_ms = timer.elapsed_ms();

    // Extract optimized poses back into graph
    for (auto& [id, pose] : graph.poses) {
        const auto& params = pose_params[id];
        pose.translation.x() = params[0];
        pose.translation.y() = params[1];
        pose.rotation = params[2];
    }

    // Compute final cost using unified cost function (both metrics)
    auto final_metrics = unified_cost::ComputeSE2CostMetrics(graph);
    result.final_chi2 = final_metrics.chi2_cost;
    result.final_cost = final_metrics.unweighted_cost;

    // Extract results
    result.iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;
    result.chi2_improvement_pct = ((result.initial_chi2 - result.final_chi2) / result.initial_chi2) * 100.0;

    // Convergence check: Accept if >95% improvement OR (Ceres converged AND positive improvement)
    bool converged = (result.improvement_pct > 95.0) ||
                     ((summary.termination_type == ceres::CONVERGENCE) && (result.improvement_pct > 0.0));
    result.status = converged ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

// Benchmark SE3 dataset with Ceres
benchmark_utils::BenchmarkResult BenchmarkSE3(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE3";
    result.solver = "Ceres-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph3D graph;
    if (!g2o_reader::ReadG2oFile3D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Store initial graph for cost computation
    g2o_reader::Graph3D initial_graph = graph;

    // Compute initial cost using unified cost function (both metrics)
    auto initial_metrics = unified_cost::ComputeSE3CostMetrics(initial_graph);
    result.initial_chi2 = initial_metrics.chi2_cost;
    result.initial_cost = initial_metrics.unweighted_cost;

    // Create Ceres problem
    ceres::Problem problem;

    // Map to store pose parameters (quaternion [qw,qx,qy,qz] + translation [x,y,z])
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> pose_params;
    for (const auto& [id, pose] : graph.poses) {
        std::vector<double> quat = {pose.rotation.w(), pose.rotation.x(),
                                     pose.rotation.y(), pose.rotation.z()};
        std::vector<double> trans = {pose.translation.x(), pose.translation.y(),
                                      pose.translation.z()};
        pose_params[id] = {quat, trans};
    }

    // Use quaternion manifold (new Ceres API)
    ceres::Manifold* quaternion_manifold = new ceres::QuaternionManifold();

    // Add edge constraints first (so parameter blocks are registered)
    for (const auto& constraint : graph.constraints) {
        auto& pose_a = pose_params[constraint.id_begin];
        auto& pose_b = pose_params[constraint.id_end];

        // Pass identity matrix instead of sqrt(information matrix)
        Eigen::Matrix<double, 6, 6> identity = Eigen::Matrix<double, 6, 6>::Identity();

        ceres::CostFunction* cost_function = PoseGraph3DErrorTerm::Create(
            constraint.measurement, identity);

        problem.AddResidualBlock(cost_function, nullptr,
                                pose_a.first.data(), pose_a.second.data(),
                                pose_b.first.data(), pose_b.second.data());

        // Set quaternion manifold (new API: SetManifold instead of SetParameterization)
        problem.SetManifold(pose_a.first.data(), quaternion_manifold);
        problem.SetManifold(pose_b.first.data(), quaternion_manifold);
    }

    // Add prior on first pose (gauge freedom) - after residual blocks are added
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        problem.SetParameterBlockConstant(pose_params[first_id].first.data());   // quaternion
        problem.SetParameterBlockConstant(pose_params[first_id].second.data());  // translation
    }

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-3;
    options.parameter_tolerance = 1e-3;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // Solve
    ceres::Solver::Summary summary;
    Timer timer;
    ceres::Solve(options, &problem, &summary);
    result.time_ms = timer.elapsed_ms();

    // Extract optimized poses back into graph
    for (auto& [id, pose] : graph.poses) {
        const auto& params = pose_params[id];
        pose.rotation.w() = params.first[0];
        pose.rotation.x() = params.first[1];
        pose.rotation.y() = params.first[2];
        pose.rotation.z() = params.first[3];
        pose.translation.x() = params.second[0];
        pose.translation.y() = params.second[1];
        pose.translation.z() = params.second[2];
    }

    // Compute final cost using unified cost function (both metrics)
    auto final_metrics = unified_cost::ComputeSE3CostMetrics(graph);
    result.final_chi2 = final_metrics.chi2_cost;
    result.final_cost = final_metrics.unweighted_cost;

    // Extract results
    result.iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;
    result.chi2_improvement_pct = ((result.initial_chi2 - result.final_chi2) / result.initial_chi2) * 100.0;

    // Convergence check: Accept if >95% improvement OR (Ceres converged AND positive improvement)
    bool converged = (result.improvement_pct > 95.0) ||
                     ((summary.termination_type == ceres::CONVERGENCE) && (result.improvement_pct > 0.0));
    result.status = converged ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

int main(int argc, char** argv) {
    std::vector<benchmark_utils::BenchmarkResult> results;

    // SE3 datasets
    results.push_back(BenchmarkSE3("sphere2500", "../../../data/odometry/sphere2500.g2o"));
    results.push_back(BenchmarkSE3("parking-garage", "../../../data/odometry/parking-garage.g2o"));
    results.push_back(BenchmarkSE3("torus3D", "../../../data/odometry/torus3D.g2o"));
    results.push_back(BenchmarkSE3("cubicle", "../../../data/odometry/cubicle.g2o"));

    // SE2 datasets
    results.push_back(BenchmarkSE2("city10000", "../../../data/odometry/city10000.g2o"));
    results.push_back(BenchmarkSE2("mit", "../../../data/odometry/mit.g2o"));
    results.push_back(BenchmarkSE2("ring", "../../../data/odometry/ring.g2o"));
    results.push_back(BenchmarkSE2("M3500", "../../../data/odometry/M3500.g2o"));

    // Write to CSV
    std::string output_file = "ceres_odometry_benchmark_results.csv";
    benchmark_utils::WriteResultsToCSV(output_file, results);

    return 0;
}
