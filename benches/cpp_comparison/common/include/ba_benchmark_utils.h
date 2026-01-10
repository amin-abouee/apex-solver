#pragma once

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

namespace benchmark_utils {

/// High-resolution timer
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

/// Benchmark result structure
struct BenchmarkResult {
    std::string dataset;
    std::string solver;
    std::string language;
    int num_cameras;
    int num_points;
    int num_observations;
    double initial_mse;
    double final_mse;
    double initial_rmse;
    double final_rmse;
    double time_ms;
    int iterations;
    std::string status;
};

/// Write results to CSV file
inline bool WriteResultsToCSV(const std::string& filename, 
                               const std::vector<BenchmarkResult>& results) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        return false;
    }
    
    // Write header
    out << "dataset,solver,language,num_cameras,num_points,num_observations,"
        << "initial_mse,final_mse,initial_rmse,final_rmse,time_ms,iterations,status\n";
    
    // Write data
    for (const auto& r : results) {
        out << r.dataset << ","
            << r.solver << ","
            << r.language << ","
            << r.num_cameras << ","
            << r.num_points << ","
            << r.num_observations << ","
            << r.initial_mse << ","
            << r.final_mse << ","
            << r.initial_rmse << ","
            << r.final_rmse << ","
            << r.time_ms << ","
            << r.iterations << ","
            << r.status << "\n";
    }
    
    out.close();
    return true;
}

}  // namespace benchmark_utils
