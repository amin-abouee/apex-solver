#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using gtsam::Symbol;

#include <iostream>
#include <vector>

#include "common/benchmark_utils.h"
#include "common/read_g2o.h"

// Benchmark SE2 dataset with GTSAM
benchmark_utils::BenchmarkResult BenchmarkSE2(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;
    using namespace gtsam;

    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE2";
    result.solver = "GTSAM-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph2D graph;
    if (!g2o_reader::ReadG2oFile2D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Create factor graph and initial values
    NonlinearFactorGraph graph_factors;
    Values initial_values;

    // Add poses to initial values
    for (const auto& [id, pose] : graph.poses) {
        Pose2 gtsam_pose(pose.translation.x(), pose.translation.y(), pose.rotation);
        initial_values.insert(Symbol('x', id), gtsam_pose);
    }

    // Add prior factor on first pose (gauge freedom)
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        const auto& first_pose = graph.poses.begin()->second;

        Pose2 prior_pose(first_pose.translation.x(), first_pose.translation.y(),
                        first_pose.rotation);

        // Strong prior with high information (low covariance)
        auto prior_noise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01));
        graph_factors.addPrior(Symbol('x', first_id), prior_pose, prior_noise);
    }

    // Add between factors
    for (const auto& constraint : graph.constraints) {
        Pose2 measurement(constraint.measurement.translation.x(),
                         constraint.measurement.translation.y(),
                         constraint.measurement.rotation);

        // Convert information matrix to noise model
        auto noise = noiseModel::Gaussian::Information(constraint.information);

        graph_factors.add(BetweenFactor<Pose2>(
            Symbol('x', constraint.id_begin),
            Symbol('x', constraint.id_end),
            measurement,
            noise));
    }

    // Compute initial error
    result.initial_cost = graph_factors.error(initial_values);

    // Configure optimizer
    LevenbergMarquardtParams params;
    params.setVerbosity("SILENT");
    params.setMaxIterations(100);
    params.setRelativeErrorTol(1e-3);
    params.setAbsoluteErrorTol(1e-3);

    // Optimize
    Timer timer;
    LevenbergMarquardtOptimizer optimizer(graph_factors, initial_values, params);
    Values optimized_values = optimizer.optimize();
    result.time_ms = timer.elapsed_ms();

    // Extract results
    result.final_cost = graph_factors.error(optimized_values);
    result.iterations = optimizer.iterations();
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;
    result.status = "CONVERGED";  // GTSAM doesn't provide explicit convergence status

    return result;
}

// Benchmark SE3 dataset with GTSAM
benchmark_utils::BenchmarkResult BenchmarkSE3(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;
    using namespace gtsam;

    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE3";
    result.solver = "GTSAM-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph3D graph;
    if (!g2o_reader::ReadG2oFile3D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Create factor graph and initial values
    NonlinearFactorGraph graph_factors;
    Values initial_values;

    // Add poses to initial values
    for (const auto& [id, pose] : graph.poses) {
        Rot3 rotation(Quaternion(pose.rotation.w(), pose.rotation.x(),
                                 pose.rotation.y(), pose.rotation.z()));
        Point3 translation(pose.translation.x(), pose.translation.y(), pose.translation.z());
        Pose3 gtsam_pose(rotation, translation);
        initial_values.insert(Symbol('x', id), gtsam_pose);
    }

    // Add prior factor on first pose (gauge freedom)
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        const auto& first_pose = graph.poses.begin()->second;

        Rot3 rotation(Quaternion(first_pose.rotation.w(), first_pose.rotation.x(),
                                 first_pose.rotation.y(), first_pose.rotation.z()));
        Point3 translation(first_pose.translation.x(), first_pose.translation.y(),
                          first_pose.translation.z());
        Pose3 prior_pose(rotation, translation);

        // Strong prior with high information (low covariance)
        auto prior_noise = noiseModel::Diagonal::Sigmas(
            (Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
        graph_factors.addPrior(Symbol('x', first_id), prior_pose, prior_noise);
    }

    // Add between factors
    for (const auto& constraint : graph.constraints) {
        Rot3 rotation(Quaternion(constraint.measurement.rotation.w(),
                                 constraint.measurement.rotation.x(),
                                 constraint.measurement.rotation.y(),
                                 constraint.measurement.rotation.z()));
        Point3 translation(constraint.measurement.translation.x(),
                          constraint.measurement.translation.y(),
                          constraint.measurement.translation.z());
        Pose3 measurement(rotation, translation);

        // Convert information matrix to noise model
        auto noise = noiseModel::Gaussian::Information(constraint.information);

        graph_factors.add(BetweenFactor<Pose3>(
            Symbol('x', constraint.id_begin),
            Symbol('x', constraint.id_end),
            measurement,
            noise));
    }

    // Compute initial error
    result.initial_cost = graph_factors.error(initial_values);

    // Configure optimizer
    LevenbergMarquardtParams params;
    params.setVerbosity("SILENT");
    params.setMaxIterations(100);
    params.setRelativeErrorTol(1e-3);
    params.setAbsoluteErrorTol(1e-3);

    // Optimize
    Timer timer;
    LevenbergMarquardtOptimizer optimizer(graph_factors, initial_values, params);
    Values optimized_values = optimizer.optimize();
    result.time_ms = timer.elapsed_ms();

    // Extract results
    result.final_cost = graph_factors.error(optimized_values);
    result.iterations = optimizer.iterations();
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;
    result.status = "CONVERGED";  // GTSAM doesn't provide explicit convergence status

    return result;
}

int main(int argc, char** argv) {
    std::cout << "=== GTSAM SOLVER BENCHMARK ===" << std::endl;

    std::vector<benchmark_utils::BenchmarkResult> results;

    // SE3 datasets
    results.push_back(BenchmarkSE3("sphere2500", "../../../data/sphere2500.g2o"));
    results.push_back(BenchmarkSE3("parking-garage", "../../../data/parking-garage.g2o"));
    results.push_back(BenchmarkSE3("torus3D", "../../../data/torus3D.g2o"));
    results.push_back(BenchmarkSE3("cubicle", "../../../data/cubicle.g2o"));

    // SE2 datasets
    results.push_back(BenchmarkSE2("intel", "../../../data/intel.g2o"));
    results.push_back(BenchmarkSE2("mit", "../../../data/mit.g2o"));
    results.push_back(BenchmarkSE2("ring", "../../../data/ring.g2o"));
    results.push_back(BenchmarkSE2("M3500", "../../../data/M3500.g2o"));

    // Print results
    benchmark_utils::PrintResults(results);

    // Write to CSV
    std::string output_file = "gtsam_benchmark_results.csv";
    if (benchmark_utils::WriteResultsToCSV(output_file, results)) {
        std::cout << "\nResults written to " << output_file << std::endl;
    }

    return 0;
}
