#pragma once

#include "read_bal.h"
#include <Eigen/Core>

namespace ba_cost {

/// Compute reprojection error for a single observation
/// Uses BAL camera model: axis-angle rotation, radial distortion (k1, k2)
/// @param valid Optional output flag, set to false if point is behind camera
Eigen::Vector2d ComputeReprojectionError(
    const bal_reader::BALCamera& camera,
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& observation,
    bool* valid = nullptr);

/// Compute Mean Squared Error (MSE) across all observations
/// MSE = (1/N) * sum(||residual_i||^2) in pixels^2
double ComputeMSE(const bal_reader::BALDataset& dataset);

/// Compute Root Mean Squared Error (RMSE)
/// RMSE = sqrt(MSE) in pixels
double ComputeRMSE(const bal_reader::BALDataset& dataset);

}  // namespace ba_cost
