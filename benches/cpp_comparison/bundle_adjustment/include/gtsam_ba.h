#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3Bundler.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/inference/Symbol.h>
#include <string>

#include "../../common/include/ba_benchmark_utils.h"

// GTSAM namespaces and symbols
using namespace gtsam;
using symbol_shorthand::C;  // Camera 
using symbol_shorthand::P;  // 3D points

// Define the SfM camera type (Pose + Cal3Bundler)
using SfmCamera = PinholeCamera<Cal3Bundler>;

// Function declaration for GTSAM bundle adjustment benchmark
benchmark_utils::BenchmarkResult BenchmarkGTSAM(const std::string& dataset_path);
