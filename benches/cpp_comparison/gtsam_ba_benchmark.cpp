// GTSAM Bundle Adjustment Benchmark using native GTSAM BAL reader
// Uses SfmData and GeneralSFMFactor for proper BAL camera model handling

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3Bundler.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/SfmData.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/inference/Symbol.h>

#include "common/include/ba_benchmark_utils.h"

#include <thread>
#include <fstream>

using namespace gtsam;
using symbol_shorthand::C;  // Camera 
using symbol_shorthand::P;  // 3D points

// Define the SfM camera type (Pose + Cal3Bundler)
using SfmCamera = PinholeCamera<Cal3Bundler>;

benchmark_utils::BenchmarkResult BenchmarkGTSAM(const std::string& dataset_path) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = "problem-1723-156502-pre";
    result.solver = "GTSAM";
    result.language = "C++";
    
    std::cout << "\n=== GTSAM Benchmark ===" << std::endl;
    
    // Use GTSAM's native BAL reader which handles camera model conventions correctly
    std::cout << "Loading BAL dataset using GTSAM native reader..." << std::endl;
    SfmData sfm_data;
    
    try {
        sfm_data = SfmData::FromBalFile(dataset_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load BAL file: " << e.what() << std::endl;
        result.status = "LOAD_FAILED";
        return result;
    }
    
    result.num_cameras = sfm_data.numberCameras();
    result.num_points = sfm_data.numberTracks();
    
    // Count total observations
    size_t total_obs = 0;
    for (size_t j = 0; j < sfm_data.numberTracks(); ++j) {
        total_obs += sfm_data.tracks[j].numberMeasurements();
    }
    result.num_observations = static_cast<int>(total_obs);
    
    std::cout << "Loaded: " << result.num_cameras << " cameras, " 
              << result.num_points << " points, "
              << result.num_observations << " observations" << std::endl;
    
    // Build factor graph using GeneralSFMFactor (optimizes full camera including calibration)
    std::cout << "Building optimization problem..." << std::endl;
    NonlinearFactorGraph graph;
    Values initial;
    
    // Add camera variables
    std::cout << "Adding " << sfm_data.numberCameras() << " cameras..." << std::endl;
    for (size_t i = 0; i < sfm_data.numberCameras(); ++i) {
        initial.insert(C(i), sfm_data.cameras[i]);
    }
    
    // Add point variables and projection factors
    std::cout << "Adding " << sfm_data.numberTracks() << " 3D points and projection factors..." << std::endl;
    
    // Measurement noise with Huber robust loss
    auto noise = noiseModel::Robust::Create(
        noiseModel::mEstimator::Huber::Create(1.0),
        noiseModel::Isotropic::Sigma(2, 1.0));
    
    // Define the factor type
    using SfmFactor = GeneralSFMFactor<SfmCamera, Point3>;
    
    for (size_t j = 0; j < sfm_data.numberTracks(); ++j) {
        const SfmTrack& track = sfm_data.tracks[j];
        
        // Add 3D point
        initial.insert(P(j), track.point3());
        
        // Add projection factors for each measurement
        for (size_t k = 0; k < track.numberMeasurements(); ++k) {
            const auto& measurement = track.measurements[k];
            size_t camera_idx = measurement.first;
            Point2 uv = measurement.second;
            
            graph.emplace_shared<SfmFactor>(uv, noise, C(camera_idx), P(j));
        }
    }
    
    // Fix first camera for gauge freedom
    std::cout << "Fixing first camera for gauge freedom..." << std::endl;
    // Use a strong prior on the first camera (all 9 DOF: 6 pose + 3 calibration)
    auto prior_noise = noiseModel::Diagonal::Sigmas(
        (Vector(9) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
    graph.addPrior(C(0), sfm_data.cameras[0], prior_noise);
    
    // Also add prior on first point to fix scale
    auto point_prior_noise = noiseModel::Isotropic::Sigma(3, 1e-6);
    graph.addPrior(P(0), sfm_data.tracks[0].point3(), point_prior_noise);
    
    // Configure optimizer
    // Use more conservative settings to handle potential numerical issues
    LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");
    params.setMaxIterations(100);
    params.setAbsoluteErrorTol(1e-6);
    params.setRelativeErrorTol(1e-6);
    // Start with higher damping to take smaller, safer steps
    params.setlambdaInitial(1.0);
    params.setlambdaFactor(2.0);  // More gradual lambda changes
    params.setlambdaUpperBound(1e10);
    params.setlambdaLowerBound(1e-10);
    // Use diagonal damping for better conditioning
    params.setDiagonalDamping(true);
    
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    std::cout << "Solver configuration:" << std::endl;
    std::cout << "  Algorithm: Levenberg-Marquardt" << std::endl;
    std::cout << "  Max iterations: " << params.getMaxIterations() << std::endl;
    std::cout << "  Tolerance: " << params.getAbsoluteErrorTol() << std::endl;
    std::cout << "  Available threads: " << num_threads << " (GTSAM uses TBB if available)" << std::endl;
    std::cout << "  Note: Optimizing full cameras (pose + calibration)" << std::endl;
    
    // Compute initial error
    double initial_error = graph.error(initial);
    result.initial_mse = initial_error / result.num_observations;
    result.initial_rmse = std::sqrt(result.initial_mse);
    std::cout << "Initial error: " << initial_error << std::endl;
    std::cout << "Initial RMSE: " << result.initial_rmse << " pixels" << std::endl;
    
    // Optimize
    std::cout << "\nStarting optimization..." << std::endl;
    Timer timer;
    
    try {
        LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        Values optimized = optimizer.optimize();
        result.time_ms = timer.elapsed_ms();
        result.iterations = optimizer.iterations();
        
        // Compute final error
        double final_error = graph.error(optimized);
        result.final_mse = final_error / result.num_observations;
        result.final_rmse = std::sqrt(result.final_mse);
        
        result.status = "CONVERGED";
        
        std::cout << "\nOptimization completed in " << result.iterations << " iterations" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Optimization failed: " << e.what() << std::endl;
        result.time_ms = timer.elapsed_ms();
        result.status = "FAILED";
        result.final_mse = result.initial_mse;
        result.final_rmse = result.initial_rmse;
    }
    
    double improvement_pct = ((result.initial_mse - result.final_mse) / result.initial_mse) * 100.0;
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Initial RMSE: " << result.initial_rmse << " pixels" << std::endl;
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
    results.push_back(BenchmarkGTSAM(dataset_path));
    
    std::string csv_path = "gtsam_ba_benchmark_results.csv";
    if (benchmark_utils::WriteResultsToCSV(csv_path, results)) {
        std::cout << "\nResults written to " << csv_path << std::endl;
    } else {
        std::cerr << "Failed to write CSV results" << std::endl;
        return 1;
    }
    
    return 0;
}
