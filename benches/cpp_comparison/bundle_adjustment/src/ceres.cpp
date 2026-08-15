#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <thread>
#include "../../common/include/read_bal.h"
#include "../../common/include/ba_cost.h"
#include "../../common/include/ba_benchmark_utils.h"
#include "../include/ceres_ba.h"

// Cost functor struct moved to ceres_ba.h

benchmark_utils::BenchmarkResult BenchmarkCeres(const std::string& dataset_path) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = "problem-1723-156502-pre";
    result.solver = "Ceres";
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

    // Setup Ceres problem
    ceres::Problem problem;
    
    // Convert cameras to parameter blocks (9 params each)
    std::vector<std::vector<double>> camera_params;
    camera_params.reserve(dataset.cameras.size());
    for (const auto& cam : dataset.cameras) {
        std::vector<double> params = {
            cam.rotation[0], cam.rotation[1], cam.rotation[2],
            cam.translation[0], cam.translation[1], cam.translation[2],
            cam.focal_length, cam.k1, cam.k2
        };
        camera_params.push_back(params);
    }
    
    // Convert points to parameter blocks (3 params each)
    std::vector<std::vector<double>> point_params;
    point_params.reserve(dataset.points.size());
    for (const auto& pt : dataset.points) {
        std::vector<double> params = {pt.position[0], pt.position[1], pt.position[2]};
        point_params.push_back(params);
    }
    
    // Add residual blocks
    for (const auto& obs : dataset.observations) {
        ceres::CostFunction* cost_function = 
            BALReprojectionError::Create(obs.x, obs.y);
        
        // Use Huber loss (matching user requirement)
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
        
        problem.AddResidualBlock(
            cost_function,
            loss_function,
            camera_params[obs.camera_index].data(),
            point_params[obs.point_index].data());
    }
    
    // Fix first camera (gauge freedom)
    problem.SetParameterBlockConstant(camera_params[0].data());
    
    // Configure solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.use_inner_iterations = false;  // Disabled for fair comparison with other solvers
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance = 1e-14;
    options.minimizer_progress_to_stdout = false;

    // Solve
    ceres::Solver::Summary summary;
    Timer timer;
    ceres::Solve(options, &problem, &summary);
    result.time_ms = timer.elapsed_ms();

    // Extract optimized values back to dataset
    for (size_t i = 0; i < dataset.cameras.size(); ++i) {
        const auto& params = camera_params[i];
        dataset.cameras[i].rotation = Eigen::Vector3d(params[0], params[1], params[2]);
        dataset.cameras[i].translation = Eigen::Vector3d(params[3], params[4], params[5]);
        dataset.cameras[i].focal_length = params[6];
        dataset.cameras[i].k1 = params[7];
        dataset.cameras[i].k2 = params[8];
    }
    
    for (size_t i = 0; i < dataset.points.size(); ++i) {
        const auto& params = point_params[i];
        dataset.points[i].position = Eigen::Vector3d(params[0], params[1], params[2]);
    }
    
    // Compute final cost
    result.final_mse = ba_cost::ComputeMSE(dataset);
    result.final_rmse = ba_cost::ComputeRMSE(dataset);

    result.iterations = summary.iterations.size();
    result.status = (summary.termination_type == ceres::CONVERGENCE) ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_minloglevel = google::GLOG_FATAL;  // Silence Ceres' internal glog output

    std::string dataset_path = "../../../data/bundle_adjustment/ladybug/problem-1723-156502-pre.txt";
    
    if (argc > 1) {
        dataset_path = argv[1];
    }
    
    std::vector<benchmark_utils::BenchmarkResult> results;
    results.push_back(BenchmarkCeres(dataset_path));
    
    std::string csv_path = "ceres_ba_benchmark_results.csv";
    if (!benchmark_utils::WriteResultsToCSV(csv_path, results)) {
        std::cerr << "Failed to write CSV results" << std::endl;
        return 1;
    }

    return 0;
}
