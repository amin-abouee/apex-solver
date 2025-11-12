#include <iostream>
#include <string>
#include <vector>

#include "common/benchmark_utils.h"
#include "common/read_g2o.h"

// Forward declarations for benchmark functions from individual files
// Note: For the unified build, you'd include implementations here or link them

int main(int argc, char** argv) {
    std::cout << "=== APEX-SOLVER C++ BENCHMARK SUITE ===" << std::endl;
    std::cout << "Comparing Ceres, GTSAM, and g2o on pose graph optimization\n" << std::endl;

    // Dataset paths
    struct Dataset {
        std::string name;
        std::string path;
        std::string type;  // "SE2" or "SE3"
    };

    std::vector<Dataset> datasets = {
        {"rim", "../data/rim.g2o", "SE3"},
        {"cubicle", "../data/cubicle.g2o", "SE3"},
        {"intel", "../data/intel.g2o", "SE2"},
        {"mit", "../data/mit.g2o", "SE2"}
    };

    std::cout << "To run individual benchmarks:\n";
    std::cout << "  ./ceres_benchmark   - Run Ceres Solver benchmarks\n";
    std::cout << "  ./gtsam_benchmark   - Run GTSAM benchmarks\n";
    std::cout << "  ./g2o_benchmark     - Run g2o benchmarks\n";
    std::cout << "\nOr use the automation script:\n";
    std::cout << "  bash run_all_benchmarks.sh\n";
    std::cout << "\nDatasets to benchmark:\n";
    
    for (const auto& dataset : datasets) {
        std::cout << "  - " << dataset.name << " (" << dataset.type << "): " 
                  << dataset.path << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Please run individual benchmark executables or the automation script." << std::endl;

    return 0;
}
