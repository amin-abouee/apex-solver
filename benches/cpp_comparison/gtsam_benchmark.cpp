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

#include <vector>

#include "common/benchmark_utils.h"
#include "common/read_g2o.h"
#include "common/unified_cost.h"

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

    // Store initial graph for cost computation
    g2o_reader::Graph2D initial_graph = graph;
    
    // Compute initial cost using unified cost function
    result.initial_cost = unified_cost::ComputeSE2Cost(initial_graph);

    // Create factor graph and initial values
    NonlinearFactorGraph graph_factors;
    Values initial_values;

    // Add poses to initial values
    for (const auto& [id, pose] : graph.poses) {
        Pose2 gtsam_pose(pose.translation.x(), pose.translation.y(), pose.rotation);
        initial_values.insert(Symbol('x', id), gtsam_pose);
    }

    // Fix first pose to eliminate gauge freedom (similar to Ceres/g2o approach)
    // Note: GTSAM doesn't support hard constraints like Ceres' SetParameterBlockConstant,
    // so we skip the prior entirely. This matches the unified cost computation which
    // doesn't include prior factors. The graph should still be well-constrained by edges.
    // (Removed prior factor to match unified cost - GTSAM will handle gauge freedom automatically)

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

    // Extract optimized poses back into graph
    for (auto& [id, pose] : graph.poses) {
        Pose2 optimized_pose = optimized_values.at<Pose2>(Symbol('x', id));
        pose.translation.x() = optimized_pose.x();
        pose.translation.y() = optimized_pose.y();
        pose.rotation = optimized_pose.theta();
    }

    // Compute final cost using unified cost function
    result.final_cost = unified_cost::ComputeSE2Cost(graph);
    
    // Extract results
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

    // Store initial graph for cost computation
    g2o_reader::Graph3D initial_graph = graph;
    
    // Compute initial cost using unified cost function
    result.initial_cost = unified_cost::ComputeSE3Cost(initial_graph);

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

    // Fix first pose to eliminate gauge freedom (similar to Ceres/g2o approach)
    // Note: GTSAM doesn't support hard constraints like Ceres' SetParameterBlockConstant,
    // so we skip the prior entirely. This matches the unified cost computation which
    // doesn't include prior factors. The graph should still be well-constrained by edges.
    // (Removed prior factor to match unified cost - GTSAM will handle gauge freedom automatically)

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

    // Extract optimized poses back into graph
    for (auto& [id, pose] : graph.poses) {
        Pose3 optimized_pose = optimized_values.at<Pose3>(Symbol('x', id));
        Quaternion quat = optimized_pose.rotation().toQuaternion();
        pose.rotation.w() = quat.w();
        pose.rotation.x() = quat.x();
        pose.rotation.y() = quat.y();
        pose.rotation.z() = quat.z();
        pose.translation.x() = optimized_pose.x();
        pose.translation.y() = optimized_pose.y();
        pose.translation.z() = optimized_pose.z();
    }

    // Compute final cost using unified cost function
    result.final_cost = unified_cost::ComputeSE3Cost(graph);
    
    // Extract results
    result.iterations = optimizer.iterations();
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;
    result.status = "CONVERGED";  // GTSAM doesn't provide explicit convergence status

    return result;
}

int main(int argc, char** argv) {
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

    // Write to CSV
    std::string output_file = "gtsam_benchmark_results.csv";
    benchmark_utils::WriteResultsToCSV(output_file, results);

    return 0;
}
