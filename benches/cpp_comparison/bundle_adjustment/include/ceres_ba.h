#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <string>

#include "../../common/include/ba_benchmark_utils.h"

// Ceres cost function for BAL reprojection error
// Must be in header due to C++ template requirements for automatic differentiation
struct BALReprojectionError {
    BALReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2]: axis-angle rotation
        // camera[3,4,5]: translation
        // camera[6]: focal length
        // camera[7,8]: radial distortion k1, k2
        
        // Rotate point
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        
        // Add translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // Project to normalized image plane
        // BAL convention: camera looks down NEGATIVE Z axis
        // Projection: p' = -P / P.z (note the negation)
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];
        
        // Apply radial distortion
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + camera[7] * r2 + camera[8] * r2 * r2;
        
        // Apply focal length
        T predicted_x = camera[6] * distortion * xp;
        T predicted_y = camera[6] * distortion * yp;
        
        // Residual
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        
        return true;
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y) {
        return new ceres::AutoDiffCostFunction<BALReprojectionError, 2, 9, 3>(
            new BALReprojectionError(observed_x, observed_y));
    }

    double observed_x;
    double observed_y;
};

// Function declaration for Ceres bundle adjustment benchmark
benchmark_utils::BenchmarkResult BenchmarkCeres(const std::string& dataset_path);
