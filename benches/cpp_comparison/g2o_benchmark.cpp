#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <vector>

#include "common/include/benchmark_utils.h"
#include "common/include/read_g2o.h"
#include "common/include/unified_cost.h"

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

    // Store initial graph for unified cost computation
    g2o_reader::Graph2D initial_graph = graph;

    // Compute initial cost using unified cost function
    result.initial_cost = unified_cost::ComputeSE2Cost(initial_graph);

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

        // USER REQUEST: Use unweighted cost (ignore information matrix)
        edge->setInformation(Eigen::Matrix3d::Identity());

        optimizer.addEdge(edge);
    }

    // Initialize optimizer first
    optimizer.initializeOptimization();

    // Optimize
    Timer timer;
    int iterations = optimizer.optimize(100);
    result.time_ms = timer.elapsed_ms();

    // Extract optimized poses back into graph structure
    g2o_reader::Graph2D optimized_graph = initial_graph;  // Copy structure with constraints
    for (auto& [id, pose] : optimized_graph.poses) {
        VertexSE2* vertex = static_cast<VertexSE2*>(optimizer.vertex(id));
        if (vertex) {
            SE2 estimate = vertex->estimate();
            pose.translation.x() = estimate[0];
            pose.translation.y() = estimate[1];
            pose.rotation = estimate[2];
        }
    }

    // Compute final cost using unified cost function
    result.final_cost = unified_cost::ComputeSE2Cost(optimized_graph);
    result.iterations = iterations;
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;

    // Convergence check: Accept if >95% improvement OR (positive improvement and didn't hit max iterations)
    bool converged = (result.improvement_pct > 95.0) ||
                     ((result.improvement_pct > 0.0) && (iterations < 100));
    result.status = converged ? "CONVERGED" : "NOT_CONVERGED";

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

    // Store initial graph for unified cost computation
    g2o_reader::Graph3D initial_graph = graph;

    // Compute initial cost using unified cost function
    result.initial_cost = unified_cost::ComputeSE3Cost(initial_graph);

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

        // USER REQUEST: Use unweighted cost (ignore information matrix)
        edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());

        optimizer.addEdge(edge);
    }

    // Initialize optimizer first
    optimizer.initializeOptimization();

    // Optimize
    Timer timer;
    int iterations = optimizer.optimize(100);
    result.time_ms = timer.elapsed_ms();

    // Extract optimized poses back into graph structure
    g2o_reader::Graph3D optimized_graph = initial_graph;  // Copy structure with constraints
    for (auto& [id, pose] : optimized_graph.poses) {
        VertexSE3* vertex = static_cast<VertexSE3*>(optimizer.vertex(id));
        if (vertex) {
            Eigen::Isometry3d estimate = vertex->estimate();
            pose.translation = estimate.translation();
            pose.rotation = Eigen::Quaterniond(estimate.linear());
        }
    }

    // Compute final cost using unified cost function
    result.final_cost = unified_cost::ComputeSE3Cost(optimized_graph);
    result.iterations = iterations;
    result.improvement_pct = ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0;

    // Convergence check: Accept if >95% improvement OR (positive improvement and didn't hit max iterations)
    bool converged = (result.improvement_pct > 95.0) ||
                     ((result.improvement_pct > 0.0) && (iterations < 100));
    result.status = converged ? "CONVERGED" : "NOT_CONVERGED";

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
    std::string output_file = "g2o_benchmark_results.csv";
    benchmark_utils::WriteResultsToCSV(output_file, results);

    return 0;
}
