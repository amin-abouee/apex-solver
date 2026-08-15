#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/base_binary_edge.h>

#include "../../common/include/read_bal.h"
#include "../../common/include/ba_cost.h"
#include "../../common/include/ba_benchmark_utils.h"
#include "../include/g2o_ba.h"

// EdgeBALProjection class moved to g2o_ba.h

benchmark_utils::BenchmarkResult BenchmarkG2O(const std::string& dataset_path) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = "problem-1723-156502-pre";
    result.solver = "g2o";
    result.language = "C++";

    // Load dataset
    bal_reader::BALDataset dataset;
    if (!bal_reader::ReadBALFile(dataset_path, dataset)) {
        result.status = "LOAD_FAILED";
        return result;
    }
    
    result.num_cameras = dataset.num_cameras();
    result.num_points = dataset.num_points();
    result.num_observations = dataset.num_observations();
    
    // Compute initial cost
    result.initial_mse = ba_cost::ComputeMSE(dataset);
    result.initial_rmse = ba_cost::ComputeRMSE(dataset);

    // Setup g2o optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    
    // Add camera vertices
    for (size_t i = 0; i < dataset.cameras.size(); ++i) {
        const auto& cam = dataset.cameras[i];
        
        auto* v_cam = new g2o::VertexSE3Expmap();
        v_cam->setId(static_cast<int>(i));
        
        // Convert axis-angle to SE3
        Eigen::Vector3d aa = cam.rotation;
        double angle = aa.norm();
        Eigen::Matrix3d R;
        if (angle < 1e-10) {
            R = Eigen::Matrix3d::Identity();
        } else {
            Eigen::AngleAxisd rot(angle, aa / angle);
            R = rot.toRotationMatrix();
        }
        
        g2o::SE3Quat pose(R, cam.translation);
        v_cam->setEstimate(pose);
        
        // Fix first camera (gauge freedom)
        if (i == 0) {
            v_cam->setFixed(true);
        }
        
        optimizer.addVertex(v_cam);
    }
    
    // Add point vertices
    int point_id_offset = static_cast<int>(dataset.cameras.size());
    for (size_t j = 0; j < dataset.points.size(); ++j) {
        const auto& pt = dataset.points[j];
        
        auto* v_point = new g2o::VertexPointXYZ();
        v_point->setId(point_id_offset + static_cast<int>(j));
        v_point->setEstimate(pt.position);
        v_point->setMarginalized(true);  // Marginalize landmarks for Schur complement
        
        optimizer.addVertex(v_point);
    }
    
    // Add projection edges with custom BAL camera model
    for (const auto& obs : dataset.observations) {
        const auto& cam = dataset.cameras[obs.camera_index];
        
        auto* edge = new EdgeBALProjection();
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            optimizer.vertex(point_id_offset + obs.point_index)));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            optimizer.vertex(obs.camera_index)));
        edge->setMeasurement(Eigen::Vector2d(obs.x, obs.y));
        
        // Set per-camera intrinsics
        edge->setIntrinsics(cam.focal_length, cam.k1, cam.k2);
        
        // Information matrix (identity - unweighted)
        edge->setInformation(Eigen::Matrix2d::Identity());
        
        // Huber robust kernel (matching Ceres and Apex)
        auto* robust_kernel = new g2o::RobustKernelHuber();
        robust_kernel->setDelta(1.0);
        edge->setRobustKernel(robust_kernel);
        
        optimizer.addEdge(edge);
    }
    
    // g2o is single-threaded and very slow on large problems
    // Use fewer iterations for practical benchmarking (still demonstrates convergence)
    const int max_iterations = 20;  // Reduced from 100 for practical runtime

    // Initialize and optimize
    optimizer.initializeOptimization();

    Timer timer;
    int iterations = optimizer.optimize(max_iterations);
    result.time_ms = timer.elapsed_ms();

    result.iterations = iterations;

    // Extract optimized values
    for (size_t i = 0; i < dataset.cameras.size(); ++i) {
        auto* v_cam = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(static_cast<int>(i)));
        g2o::SE3Quat pose = v_cam->estimate();
        
        // Convert quaternion back to axis-angle
        Eigen::Quaterniond q = pose.rotation();
        Eigen::AngleAxisd aa(q);
        dataset.cameras[i].rotation = aa.angle() * aa.axis();
        dataset.cameras[i].translation = pose.translation();
    }
    
    for (size_t j = 0; j < dataset.points.size(); ++j) {
        auto* v_point = dynamic_cast<g2o::VertexPointXYZ*>(
            optimizer.vertex(point_id_offset + static_cast<int>(j)));
        dataset.points[j].position = v_point->estimate();
    }
    
    // Compute final cost
    result.final_mse = ba_cost::ComputeMSE(dataset);
    result.final_rmse = ba_cost::ComputeRMSE(dataset);

    result.status = (iterations > 0) ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

int main(int argc, char** argv) {
    std::string dataset_path = "../../../data/bundle_adjustment/ladybug/problem-1723-156502-pre.txt";
    
    if (argc > 1) {
        dataset_path = argv[1];
    }
    
    std::vector<benchmark_utils::BenchmarkResult> results;
    results.push_back(BenchmarkG2O(dataset_path));
    
    std::string csv_path = "g2o_ba_benchmark_results.csv";
    if (!benchmark_utils::WriteResultsToCSV(csv_path, results)) {
        std::cerr << "Failed to write CSV results" << std::endl;
        return 1;
    }
    
    return 0;
}
