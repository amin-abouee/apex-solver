#pragma once

#include <string>

#include "../../common/include/benchmark_utils.h"

// Function declarations for GTSAM benchmark functions
benchmark_utils::BenchmarkResult BenchmarkSE2(const std::string& dataset_name,
                                             const std::string& filepath);

benchmark_utils::BenchmarkResult BenchmarkSE3(const std::string& dataset_name,
                                             const std::string& filepath);
