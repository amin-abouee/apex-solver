#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace benchmark_utils {

// Result structure for a single benchmark run
struct BenchmarkResult {
    std::string dataset;
    std::string manifold;
    std::string solver;
    std::string language;
    int vertices;
    int edges;
    double initial_cost;
    double final_cost;
    double improvement_pct;
    int iterations;
    long time_ms;
    std::string status;

    BenchmarkResult()
        : dataset(""),
          manifold(""),
          solver(""),
          language("C++"),
          vertices(0),
          edges(0),
          initial_cost(0.0),
          final_cost(0.0),
          improvement_pct(0.0),
          iterations(0),
          time_ms(0),
          status("UNKNOWN") {}
};

// Simple timer class
class Timer {
public:
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    long elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time_);
        return duration.count();
    }

    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Write results to CSV file
inline bool WriteResultsToCSV(const std::string& filename,
                              const std::vector<BenchmarkResult>& results,
                              bool append = false) {
    std::ofstream outfile;
    if (append) {
        outfile.open(filename, std::ios::app);
    } else {
        outfile.open(filename);
        // Write header
        outfile << "dataset,manifold,solver,language,vertices,edges,"
                << "init_cost,final_cost,improvement_pct,iterations,time_ms,status\n";
    }

    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return false;
    }

    for (const auto& result : results) {
        outfile << result.dataset << ","
                << result.manifold << ","
                << result.solver << ","
                << result.language << ","
                << result.vertices << ","
                << result.edges << ","
                << result.initial_cost << ","
                << result.final_cost << ","
                << result.improvement_pct << ","
                << result.iterations << ","
                << result.time_ms << ","
                << result.status << "\n";
    }

    outfile.close();
    return true;
}

// Print results to console in a formatted table
inline void PrintResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(150, '=') << "\n";
    std::cout << "=== BENCHMARK RESULTS ===\n\n";

    // Header
    printf("%-12s | %-8s | %-10s | %-8s | %-6s | %-12s | %-12s | %-11s | %-5s | %-9s | %-10s\n",
           "Dataset", "Manifold", "Solver", "Vertices", "Edges", 
           "Init Cost", "Final Cost", "Improvement", "Iters", "Time(ms)", "Status");
    std::cout << std::string(150, '-') << "\n";

    for (const auto& r : results) {
        printf("%-12s | %-8s | %-10s | %-8d | %-6d | %-12.6e | %-12.6e | %10.2f%% | %-5d | %-9ld | %-10s\n",
               r.dataset.c_str(),
               r.manifold.c_str(),
               r.solver.c_str(),
               r.vertices,
               r.edges,
               r.initial_cost,
               r.final_cost,
               r.improvement_pct,
               r.iterations,
               r.time_ms,
               r.status.c_str());
    }

    std::cout << std::string(150, '-') << "\n";
}

}  // namespace benchmark_utils
