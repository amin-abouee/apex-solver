#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>

#include "../../common/include/benchmark_utils.h"
#include "../../common/include/read_g2o.h"

// Templated cost functor for SE2 pose graph optimization
// Must be in header due to C++ template requirements
class PoseGraph2DErrorTerm {
public:
    PoseGraph2DErrorTerm(double x, double y, double theta, const Eigen::Matrix3d& sqrt_information)
        : x_(x), y_(y), theta_(theta), sqrt_information_(sqrt_information) {}

    template <typename T>
    bool operator()(const T* const pose_a, const T* const pose_b, T* residuals_ptr) const {
        // Extract pose_a (x, y, theta)
        const T cos_theta_a = ceres::cos(pose_a[2]);
        const T sin_theta_a = ceres::sin(pose_a[2]);

        T delta_x = pose_b[0] - pose_a[0];
        T delta_y = pose_b[1] - pose_a[1];

        T h_x = cos_theta_a * delta_x + sin_theta_a * delta_y;
        T h_y = -sin_theta_a * delta_x + cos_theta_a * delta_y;
        T h_theta = pose_b[2] - pose_a[2];

        // Normalize theta to [-pi, pi]
        h_theta = ceres::atan2(ceres::sin(h_theta), ceres::cos(h_theta));

        // Compute residual
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals[0] = h_x - T(x_);
        residuals[1] = h_y - T(y_);
        residuals[2] = h_theta - T(theta_);

        // Apply sqrt information
        residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* Create(double x, double y, double theta,
                                       const Eigen::Matrix3d& sqrt_information) {
        return new ceres::AutoDiffCostFunction<PoseGraph2DErrorTerm, 3, 3, 3>(
            new PoseGraph2DErrorTerm(x, y, theta, sqrt_information));
    }

private:
    const double x_;
    const double y_;
    const double theta_;
    const Eigen::Matrix3d sqrt_information_;
};

// Templated cost functor for SE3 pose graph optimization
// Must be in header due to C++ template requirements
class PoseGraph3DErrorTerm {
public:
    PoseGraph3DErrorTerm(const g2o_reader::Pose3D& measurement,
                        const Eigen::Matrix<double, 6, 6>& sqrt_information)
        : measurement_(measurement), sqrt_information_(sqrt_information) {}

    template <typename T>
    bool operator()(const T* const pose_a_quat, const T* const pose_a_trans,
                   const T* const pose_b_quat, const T* const pose_b_trans,
                   T* residuals_ptr) const {
        // Map quaternions and translations
        Eigen::Map<const Eigen::Quaternion<T>> q_a(pose_a_quat);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_a(pose_a_trans);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(pose_b_quat);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_b(pose_b_trans);

        // Compute relative transformation
        // T_ab = T_a^{-1} * T_b
        Eigen::Quaternion<T> q_a_inv = q_a.conjugate();
        Eigen::Quaternion<T> q_ab = q_a_inv * q_b;
        Eigen::Matrix<T, 3, 1> t_ab = q_a_inv * (t_b - t_a);

        // Expected measurement
        Eigen::Quaternion<T> q_ab_measured = measurement_.rotation.template cast<T>();
        Eigen::Matrix<T, 3, 1> t_ab_measured = measurement_.translation.template cast<T>();

        // Compute residuals
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);

        // Quaternion error: q_error = q_ab_measured^{-1} * q_ab
        Eigen::Quaternion<T> q_error = q_ab_measured.conjugate() * q_ab;
        Eigen::Matrix<T, 3, 1> t_error = t_ab - t_ab_measured;

        // Convert quaternion error to angle-axis (3D vector)
        // For small rotations: angle_axis ≈ 2 * [qx, qy, qz]
        // For larger rotations, use proper conversion
        T qw = q_error.w();

        // Clamp to avoid numerical issues
        if (ceres::abs(qw) > T(0.999999)) {
            // Very small rotation, use linearization
            residuals[0] = T(2.0) * q_error.x();
            residuals[1] = T(2.0) * q_error.y();
            residuals[2] = T(2.0) * q_error.z();
        } else {
            // Proper angle-axis conversion
            T theta = T(2.0) * ceres::acos(qw);
            T sin_half_theta = ceres::sqrt(q_error.x() * q_error.x() +
                                          q_error.y() * q_error.y() +
                                          q_error.z() * q_error.z());

            // Angle-axis = (theta / sin(theta/2)) * [qx, qy, qz]
            // For numerical stability, check if sin_half_theta is too small
            if (ceres::abs(sin_half_theta) < T(1e-7)) {
                // Use Taylor expansion for small angles
                T angle_sq = theta * theta;
                // theta / sin(theta/2) ≈ 2 + theta^2 / 12
                T coef = T(2.0) + angle_sq / T(12.0);
                residuals[0] = coef * q_error.x();
                residuals[1] = coef * q_error.y();
                residuals[2] = coef * q_error.z();
            } else {
                // Standard formula
                T theta = T(2.0) * ceres::acos(qw);
                T sin_theta = ceres::sin(theta);
                T cos_theta = ceres::cos(theta);

                // Compute theta / (2 * sin(theta/2)) = theta / (2 * sqrt(1 - cos^2(theta/2)))
                // Using half-angle: sin(theta/2) = sqrt((1 - cos(theta)) / 2)
                // So: 2 * sin(theta/2) = sqrt(2 * (1 - cos(theta)))
                T coef = theta / (T(2.0) * sin_half_theta);

                residuals[0] = coef * q_error.x();
                residuals[1] = coef * q_error.y();
                residuals[2] = coef * q_error.z();
            }
        }

        // Translation residual
        Eigen::Matrix<T, 3, 1> translation_residual = t_error;
        residuals[3] = translation_residual[0];
        residuals[4] = translation_residual[1];
        residuals[5] = translation_residual[2];

        // Apply sqrt information
        residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* Create(const g2o_reader::Pose3D& measurement,
                                       const Eigen::Matrix<double, 6, 6>& sqrt_information) {
        return new ceres::AutoDiffCostFunction<PoseGraph3DErrorTerm, 6, 4, 3, 4, 3>(
            new PoseGraph3DErrorTerm(measurement, sqrt_information));
    }

private:
    const g2o_reader::Pose3D measurement_;
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

// Function declarations for benchmark functions
benchmark_utils::BenchmarkResult BenchmarkSE2(const std::string& dataset_name,
                                             const std::string& filepath);

benchmark_utils::BenchmarkResult BenchmarkSE3(const std::string& dataset_name,
                                             const std::string& filepath);
