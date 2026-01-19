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
    int vertices;
    int edges;
    double initial_cost;
    double final_cost;
    double improvement_pct;
    double time_ms;
    int iterations;
    std::string status;
};

// Write benchmark result to CSV
inline void write_result_csv(const BenchmarkResult& result, const std::string& output_file) {
    std::ofstream ofs(output_file);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    // Write header
    ofs << "dataset,solver,language,manifold,vertices,edges,initial_cost,final_cost,"
        << "improvement_pct,time_ms,iterations,status\n";

    // Write data
    ofs << result.dataset << ","
        << result.solver << ","
        << result.language << ","
        << result.manifold << ","
        << result.vertices << ","
        << result.edges << ","
        << std::fixed << std::setprecision(6) << result.initial_cost << ","
        << result.final_cost << ","
        << result.improvement_pct << ","
        << result.time_ms << ","
        << result.iterations << ","
        << result.status << "\n";

    ofs.close();
    std::cout << "Results written to: " << output_file << std::endl;
}

// Print benchmark result to console
inline void print_result(const BenchmarkResult& result) {
    std::cout << "\n=== Benchmark Result ===\n";
    std::cout << "Dataset:        " << result.dataset << "\n";
    std::cout << "Solver:         " << result.solver << "\n";
    std::cout << "Language:       " << result.language << "\n";
    std::cout << "Manifold:       " << result.manifold << "\n";
    std::cout << "Vertices:       " << result.vertices << "\n";
    std::cout << "Edges:          " << result.edges << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial cost:   " << result.initial_cost << "\n";
    std::cout << "Final cost:     " << result.final_cost << "\n";
    std::cout << "Improvement:    " << result.improvement_pct << "%\n";
    std::cout << "Time:           " << result.time_ms << " ms\n";
    std::cout << "Iterations:     " << result.iterations << "\n";
    std::cout << "Status:         " << result.status << "\n";
    std::cout << "========================\n\n";
}

// Write multiple benchmark results to CSV
inline void WriteResultsToCSV(const std::string& output_file, 
                               const std::vector<BenchmarkResult>& results) {
    std::ofstream ofs(output_file);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    // Write header
    ofs << "dataset,solver,language,manifold,vertices,edges,initial_cost,final_cost,"
        << "improvement_pct,time_ms,iterations,status\n";

    // Write each result
    for (const auto& result : results) {
        ofs << result.dataset << ","
            << result.solver << ","
            << result.language << ","
            << result.manifold << ","
            << result.vertices << ","
            << result.edges << ","
            << std::fixed << std::setprecision(6) << result.initial_cost << ","
            << result.final_cost << ","
            << result.improvement_pct << ","
            << result.time_ms << ","
            << result.iterations << ","
            << result.status << "\n";
    }

    ofs.close();
    std::cout << "Results written to: " << output_file << std::endl;
}

}  // namespace benchmark_utils
