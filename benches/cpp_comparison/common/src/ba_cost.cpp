#include "ba_cost.h"
#include <cmath>
#include <Eigen/Geometry>

namespace ba_cost {

Eigen::Vector2d ComputeReprojectionError(
    const bal_reader::BALCamera& camera,
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& observation,
    bool* valid) {
    
    // 1. Convert axis-angle to rotation matrix
    Eigen::Vector3d aa = camera.rotation;
    double angle = aa.norm();
    Eigen::Matrix3d R;
    
    if (angle < 1e-10) {
        R = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = aa / angle;
        R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }
    
    // 2. Transform point to camera frame (world-to-camera)
    Eigen::Vector3d p_cam = R * point + camera.translation;
    
    // 3. Check if point is behind camera (invalid projection)
    // BAL convention: camera looks down NEGATIVE Z axis
    // So point is in front of camera if p_cam.z() < 0
    if (p_cam.z() >= 0.0) {
        if (valid) *valid = false;
        return Eigen::Vector2d(0, 0);  // Return zero, mark invalid
    }
    
    if (valid) *valid = true;
    
    // 4. Project to normalized image plane (BAL uses -P/P.z due to -Z convention)
    double x = -p_cam.x() / p_cam.z();
    double y = -p_cam.y() / p_cam.z();
    
    // 5. Apply radial distortion (BAL camera model)
    double r2 = x * x + y * y;
    double distortion = 1.0 + camera.k1 * r2 + camera.k2 * r2 * r2;
    double x_distorted = x * distortion;
    double y_distorted = y * distortion;
    
    // 6. Apply focal length
    double u = camera.focal_length * x_distorted;
    double v = camera.focal_length * y_distorted;
    
    // 7. Compute residual (predicted - observed)
    return Eigen::Vector2d(u - observation.x(), v - observation.y());
}

double ComputeMSE(const bal_reader::BALDataset& dataset) {
    double total_squared_error = 0.0;
    int valid_count = 0;
    
    for (const auto& obs : dataset.observations) {
        const auto& cam = dataset.cameras[obs.camera_index];
        const auto& pt = dataset.points[obs.point_index];
        
        bool valid = false;
        Eigen::Vector2d residual = ComputeReprojectionError(
            cam, pt.position, Eigen::Vector2d(obs.x, obs.y), &valid);
        
        if (valid) {
            total_squared_error += residual.squaredNorm();
            valid_count++;
        }
    }
    
    if (valid_count == 0) return 0.0;
    return total_squared_error / valid_count;
}

double ComputeRMSE(const bal_reader::BALDataset& dataset) {
    return std::sqrt(ComputeMSE(dataset));
}

}  // namespace ba_cost
