#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace benchmark_utils {

// Timer utility
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Benchmark result for pose graph optimization
struct BenchmarkResult {
    std::string dataset;
    std::string solver;
    std::string language;
    std::string manifold;
    int vertices = 0;
    int edges = 0;
    
    // Dual metrics: both chi-squared (information-weighted) and unweighted
    double initial_chi2 = 0.0;    // Initial chi-squared cost (r^T * Omega * r)
    double final_chi2 = 0.0;      // Final chi-squared cost
    double initial_cost = 0.0;    // Initial unweighted cost (0.5 * ||r||^2)
    double final_cost = 0.0;      // Final unweighted cost
    double improvement_pct = 0.0; // Improvement based on unweighted cost
    double chi2_improvement_pct = 0.0; // Improvement based on chi-squared
    
    double time_ms = 0.0;
    int iterations = 0;
    std::string status;
};

// Print benchmark result to console
inline void print_result(const BenchmarkResult& result) {
    std::cout << "\n=== Benchmark Result ===" << std::endl;
    std::cout << "Dataset:          " << result.dataset << std::endl;
    std::cout << "Solver:           " << result.solver << std::endl;
    std::cout << "Language:         " << result.language << std::endl;
    std::cout << "Manifold:         " << result.manifold << std::endl;
    std::cout << "Vertices:         " << result.vertices << std::endl;
    std::cout << "Edges:            " << result.edges << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial chi2:     " << result.initial_chi2 << std::endl;
    std::cout << "Final chi2:       " << result.final_chi2 << std::endl;
    std::cout << "Chi2 improvement: " << result.chi2_improvement_pct << "%" << std::endl;
    std::cout << "Initial cost:     " << result.initial_cost << std::endl;
    std::cout << "Final cost:       " << result.final_cost << std::endl;
    std::cout << "Improvement:      " << result.improvement_pct << "%" << std::endl;
    std::cout << "Time:             " << result.time_ms << " ms" << std::endl;
    std::cout << "Iterations:       " << result.iterations << std::endl;
    std::cout << "Status:           " << result.status << std::endl;
    std::cout << "========================\n" << std::endl;
}

// Write multiple benchmark results to CSV with both metrics
inline bool WriteResultsToCSV(const std::string& output_file, 
                               const std::vector<BenchmarkResult>& results) {
    std::ofstream ofs(output_file);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    // Write header with both metrics
    ofs << "dataset,manifold,solver,language,vertices,edges,"
        << "initial_chi2,final_chi2,chi2_improvement_pct,"
        << "initial_cost,final_cost,improvement_pct,"
        << "iterations,time_ms,status\n";

    // Write each result
    for (const auto& result : results) {
        ofs << result.dataset << ","
            << result.manifold << ","
            << result.solver << ","
            << result.language << ","
            << result.vertices << ","
            << result.edges << ","
            << std::fixed << std::setprecision(6) 
            << result.initial_chi2 << ","
            << result.final_chi2 << ","
            << result.chi2_improvement_pct << ","
            << result.initial_cost << ","
            << result.final_cost << ","
            << result.improvement_pct << ","
            << result.iterations << ","
            << result.time_ms << ","
            << result.status << "\n";
    }

    ofs.close();
    std::cout << "Results written to: " << output_file << std::endl;
    return true;
}

}  // namespace benchmark_utils
