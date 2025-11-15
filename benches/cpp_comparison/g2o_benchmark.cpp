#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <iostream>
#include <vector>

#include "common/benchmark_utils.h"
#include "common/read_g2o.h"

using namespace g2o;

// Benchmark SE2 dataset with g2o
benchmark_utils::BenchmarkResult BenchmarkSE2(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;

    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE2";
    result.solver = "g2o-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph2D graph;
    if (!g2o_reader::ReadG2oFile2D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Create g2o optimizer
    SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    // Configure linear solver
    typedef BlockSolver<BlockSolverTraits<-1, -1>> BlockSolverType;
    typedef LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    auto algorithm = new OptimizationAlgorithmLevenberg(std::move(blockSolver));

    optimizer.setAlgorithm(algorithm);

    // Add vertices
    for (const auto& [id, pose] : graph.poses) {
        VertexSE2* vertex = new VertexSE2();
        vertex->setId(id);
        vertex->setEstimate(SE2(pose.translation.x(), pose.translation.y(), pose.rotation));
        optimizer.addVertex(vertex);
    }

    // Fix first vertex (gauge freedom)
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        optimizer.vertex(first_id)->setFixed(true);
    }

    // Add edges
    for (const auto& constraint : graph.constraints) {
        EdgeSE2* edge = new EdgeSE2();
        edge->setVertex(0, optimizer.vertex(constraint.id_begin));
        edge->setVertex(1, optimizer.vertex(constraint.id_end));

        SE2 measurement(constraint.measurement.translation.x(),
                       constraint.measurement.translation.y(),
                       constraint.measurement.rotation);
        edge->setMeasurement(measurement);
        edge->setInformation(constraint.information);

        optimizer.addEdge(edge);
    }

    // Initialize optimizer first (required for chi2() to compute error correctly)
    optimizer.initializeOptimization();

    // Compute initial error and cache it before optimization
    optimizer.computeActiveErrors();
    result.initial_cost = optimizer.chi2();

    // Optimize
    Timer timer;
    int iterations = optimizer.optimize(100);
    result.time_ms = timer.elapsed_ms();

    // Extract results
    result.final_cost = optimizer.chi2();
    result.iterations = iterations;
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;

    // Check convergence based on final improvement
    result.status = (iterations < 100 || result.improvement_pct > 90.0) ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

// Benchmark SE3 dataset with g2o
benchmark_utils::BenchmarkResult BenchmarkSE3(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;

    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE3";
    result.solver = "g2o-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph3D graph;
    if (!g2o_reader::ReadG2oFile3D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Create g2o optimizer
    SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    // Configure linear solver
    typedef BlockSolver<BlockSolverTraits<-1, -1>> BlockSolverType;
    typedef LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    auto algorithm = new OptimizationAlgorithmLevenberg(std::move(blockSolver));

    optimizer.setAlgorithm(algorithm);

    // Add vertices
    for (const auto& [id, pose] : graph.poses) {
        VertexSE3* vertex = new VertexSE3();
        vertex->setId(id);

        Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
        isometry.linear() = pose.rotation.toRotationMatrix();
        isometry.translation() = pose.translation;

        vertex->setEstimate(isometry);
        optimizer.addVertex(vertex);
    }

    // Fix first vertex (gauge freedom)
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        optimizer.vertex(first_id)->setFixed(true);
    }

    // Add edges
    for (const auto& constraint : graph.constraints) {
        EdgeSE3* edge = new EdgeSE3();
        edge->setVertex(0, optimizer.vertex(constraint.id_begin));
        edge->setVertex(1, optimizer.vertex(constraint.id_end));

        Eigen::Isometry3d measurement = Eigen::Isometry3d::Identity();
        measurement.linear() = constraint.measurement.rotation.toRotationMatrix();
        measurement.translation() = constraint.measurement.translation;

        edge->setMeasurement(measurement);
        edge->setInformation(constraint.information);

        optimizer.addEdge(edge);
    }

    // Initialize optimizer first (required for chi2() to compute error correctly)
    optimizer.initializeOptimization();

    // Compute initial error and cache it before optimization
    optimizer.computeActiveErrors();
    result.initial_cost = optimizer.chi2();

    // Optimize
    Timer timer;
    int iterations = optimizer.optimize(100);
    result.time_ms = timer.elapsed_ms();

    // Extract results
    result.final_cost = optimizer.chi2();
    result.iterations = iterations;
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;

    // Check convergence based on final improvement
    result.status = (iterations < 100 || result.improvement_pct > 90.0) ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

int main(int argc, char** argv) {
    std::cout << "=== G2O SOLVER BENCHMARK ===" << std::endl;

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
    std::string output_file = "g2o_benchmark_results.csv";
    if (benchmark_utils::WriteResultsToCSV(output_file, results)) {
        std::cout << "\nResults written to " << output_file << std::endl;
    }

    return 0;
}
