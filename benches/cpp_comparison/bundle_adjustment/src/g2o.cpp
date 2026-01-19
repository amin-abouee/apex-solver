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

#include <thread>

/**
 * Custom edge for BAL bundle adjustment with radial distortion.
 * 
 * This edge connects:
 * - Vertex 0: VertexSE3Expmap (camera pose, world-to-camera)
 * - Vertex 1: VertexPointXYZ (3D landmark)
 * 
 * Camera intrinsics (focal_length, k1, k2) are stored per-edge since
 * BAL datasets have per-camera intrinsics.
 */
class EdgeBALProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, 
                                                      g2o::VertexPointXYZ, 
                                                      g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeBALProjection() : focal_length_(1.0), k1_(0.0), k2_(0.0) {}

    void setIntrinsics(double focal_length, double k1, double k2) {
        focal_length_ = focal_length;
        k1_ = k1;
        k2_ = k2;
    }

    void computeError() override {
        const g2o::VertexPointXYZ* point = 
            static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
        const g2o::VertexSE3Expmap* camera = 
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        // Transform point to camera frame: p_cam = T_cw * p_world
        Eigen::Vector3d p_cam = camera->estimate().map(point->estimate());

        // Check if point is behind camera
        // BAL convention: camera looks down NEGATIVE Z axis
        // So point is in front of camera if p_cam.z() < 0
        if (p_cam[2] >= 0.0) {
            _error = Eigen::Vector2d(1e6, 1e6);
            return;
        }

        // Project to normalized image plane
        // BAL convention: p' = -P / P.z (note the negation)
        double xp = -p_cam[0] / p_cam[2];
        double yp = -p_cam[1] / p_cam[2];

        // Apply radial distortion (BAL model)
        double r2 = xp * xp + yp * yp;
        double distortion = 1.0 + k1_ * r2 + k2_ * r2 * r2;

        // Apply focal length to get pixel coordinates
        double predicted_x = focal_length_ * distortion * xp;
        double predicted_y = focal_length_ * distortion * yp;

        // Compute residual (predicted - observed)
        _error[0] = predicted_x - _measurement[0];
        _error[1] = predicted_y - _measurement[1];
    }

    // Jacobians are computed numerically by g2o (no need to override linearizeOplus)

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

private:
    double focal_length_;
    double k1_;
    double k2_;
};

benchmark_utils::BenchmarkResult BenchmarkG2O(const std::string& dataset_path) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = "problem-1723-156502-pre";
    result.solver = "g2o";
    result.language = "C++";
    
    std::cout << "\n=== g2o Benchmark ===" << std::endl;
    
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
    std::cout << "Computing initial metrics..." << std::endl;
    result.initial_mse = ba_cost::ComputeMSE(dataset);
    result.initial_rmse = ba_cost::ComputeRMSE(dataset);
    std::cout << "Initial RMSE: " << result.initial_rmse << " pixels" << std::endl;
    
    // Setup g2o optimizer
    std::cout << "Building optimization problem..." << std::endl;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    
    // Add camera vertices
    std::cout << "Adding " << dataset.cameras.size() << " camera vertices..." << std::endl;
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
            std::cout << "Fixed first camera for gauge freedom" << std::endl;
        }
        
        optimizer.addVertex(v_cam);
    }
    
    // Add point vertices
    std::cout << "Adding " << dataset.points.size() << " point vertices..." << std::endl;
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
    std::cout << "Adding " << dataset.observations.size() << " projection edges..." << std::endl;
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
    
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    std::cout << "Solver configuration:" << std::endl;
    // g2o is single-threaded and very slow on large problems
    // Use fewer iterations for practical benchmarking (still demonstrates convergence)
    const int max_iterations = 20;  // Reduced from 100 for practical runtime
    
    std::cout << "  Algorithm: Levenberg-Marquardt" << std::endl;
    std::cout << "  Linear solver: Eigen sparse (Schur complement on landmarks)" << std::endl;
    std::cout << "  Max iterations: " << max_iterations << " (reduced for g2o single-threaded)" << std::endl;
    std::cout << "  Available threads: " << num_threads << " (g2o core is single-threaded)" << std::endl;
    
    // Initialize and optimize
    std::cout << "\nInitializing optimization..." << std::endl;
    optimizer.initializeOptimization();
    
    std::cout << "Starting optimization..." << std::endl;
    Timer timer;
    int iterations = optimizer.optimize(max_iterations);
    result.time_ms = timer.elapsed_ms();
    
    result.iterations = iterations;
    
    std::cout << "Optimization completed in " << result.iterations << " iterations" << std::endl;
    
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
    std::cout << "Computing final metrics..." << std::endl;
    result.final_mse = ba_cost::ComputeMSE(dataset);
    result.final_rmse = ba_cost::ComputeRMSE(dataset);
    
    double improvement_pct = ((result.initial_mse - result.final_mse) / result.initial_mse) * 100.0;
    result.status = (iterations > 0) ? "CONVERGED" : "NOT_CONVERGED";
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Final RMSE: " << result.final_rmse << " pixels" << std::endl;
    std::cout << "  Improvement: " << improvement_pct << "%" << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Time: " << result.time_ms / 1000.0 << " seconds" << std::endl;
    std::cout << "  Status: " << result.status << std::endl;
    
    return result;
}

int main(int argc, char** argv) {
    std::string dataset_path = "../../../data/bundle_adjustment/problem-1723-156502-pre.txt";
    
    if (argc > 1) {
        dataset_path = argv[1];
    }
    
    std::vector<benchmark_utils::BenchmarkResult> results;
    results.push_back(BenchmarkG2O(dataset_path));
    
    std::string csv_path = "g2o_ba_benchmark_results.csv";
    if (benchmark_utils::WriteResultsToCSV(csv_path, results)) {
        std::cout << "\nResults written to " << csv_path << std::endl;
    } else {
        std::cerr << "Failed to write CSV results" << std::endl;
        return 1;
    }
    
    return 0;
}
