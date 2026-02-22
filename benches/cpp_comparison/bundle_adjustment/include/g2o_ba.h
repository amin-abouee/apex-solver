#pragma once

#include <g2o/core/base_binary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <Eigen/Core>
#include <string>

#include "../../common/include/ba_benchmark_utils.h"

/**
 * Custom edge for BAL bundle adjustment with radial distortion.
 * 
 * This edge connects:
 * - Vertex 0: VertexPointXYZ (3D landmark)
 * - Vertex 1: VertexSE3Expmap (camera pose, world-to-camera)
 * 
 * Camera intrinsics (focal_length, k1, k2) are stored per-edge since
 * BAL datasets have per-camera intrinsics.
 * 
 * Must be in header because g2o edge classes need full definition for polymorphic vertex handling.
 */
class EdgeBALProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, 
                                                      g2o::VertexPointXYZ, 
                                                      g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeBALProjection() : focal_length_(1.0), k1_(0.0), k2_(0.0) {}

    void setIntrinsics(double focal_length, double k1, double k2) {
        focal_length_ = focal_length;
        k1_ = k1;
        k2_ = k2;
    }

    void computeError() override {
        const g2o::VertexPointXYZ* point = 
            static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
        const g2o::VertexSE3Expmap* camera = 
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        // Transform point to camera frame: p_cam = T_cw * p_world
        Eigen::Vector3d p_cam = camera->estimate().map(point->estimate());

        // Check if point is behind camera
        // BAL convention: camera looks down NEGATIVE Z axis
        // So point is in front of camera if p_cam.z() < 0
        if (p_cam[2] >= 0.0) {
            _error = Eigen::Vector2d(1e6, 1e6);
            return;
        }

        // Project to normalized image plane
        // BAL convention: p' = -P / P.z (note the negation)
        double xp = -p_cam[0] / p_cam[2];
        double yp = -p_cam[1] / p_cam[2];

        // Apply radial distortion (BAL model)
        double r2 = xp * xp + yp * yp;
        double distortion = 1.0 + k1_ * r2 + k2_ * r2 * r2;

        // Apply focal length to get pixel coordinates
        double predicted_x = focal_length_ * distortion * xp;
        double predicted_y = focal_length_ * distortion * yp;

        // Compute residual (predicted - observed)
        _error[0] = predicted_x - _measurement[0];
        _error[1] = predicted_y - _measurement[1];
    }

    // Jacobians are computed numerically by g2o (no need to override linearizeOplus)

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

private:
    double focal_length_;
    double k1_;
    double k2_;
};

// Function declaration for g2o bundle adjustment benchmark
benchmark_utils::BenchmarkResult BenchmarkG2O(const std::string& dataset_path);
