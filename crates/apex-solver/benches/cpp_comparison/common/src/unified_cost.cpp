#include "unified_cost.h"
#include <cmath>
#include <Eigen/Dense>

namespace unified_cost {

namespace {

/// Normalize angle to [-π, π]
double NormalizeAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

/// Skew-symmetric matrix from 3D vector
Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return m;
}

/// SE3 log map for SO(3) component (quaternion to rotation vector)
/// Uses full Rodriguez formula to match Rust implementation
Eigen::Vector3d SO3LogMap(const Eigen::Quaterniond& q) {
    // Clamp qw to [-1, 1] to avoid numerical issues with acos
    double qw = std::clamp(q.w(), -1.0, 1.0);

    // If w is close to 1 (angle close to 0)
    if (qw > 1.0 - 1e-10) {
        // Small angle approximation: 2 * vec
        return 2.0 * q.vec();
    }

    double theta = 2.0 * std::acos(qw);
    double sin_half_theta = std::sqrt(1.0 - qw * qw);

    if (sin_half_theta < 1e-10) {
        return 2.0 * q.vec();
    }

    return (theta / sin_half_theta) * q.vec();
}

/// Compute inverse left Jacobian of SO(3)
Eigen::Matrix3d ComputeSO3LeftJacobianInverse(const Eigen::Vector3d& theta_vec) {
    double angle_sq = theta_vec.squaredNorm();
    Eigen::Matrix3d theta_skew = Skew(theta_vec);

    if (angle_sq < 1e-10) {
        // Small angle approximation: I - 0.5 * [theta]_x
        return Eigen::Matrix3d::Identity() - 0.5 * theta_skew;
    }

    double theta = std::sqrt(angle_sq);
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);

    double coef = 1.0 / angle_sq - (1.0 + cos_theta) / (2.0 * theta * sin_theta);

    return Eigen::Matrix3d::Identity() - 0.5 * theta_skew + coef * (theta_skew * theta_skew);
}

/// Compute SE2 residual and return both chi2 and unweighted contributions
std::pair<double, double> ComputeSE2ResidualCosts(
    const Eigen::Vector3d& residual,
    const Eigen::Matrix3d& information) {
    
    // Chi-squared: r^T * Omega * r
    double chi2 = residual.transpose() * information * residual;
    
    // Unweighted: 0.5 * ||r||^2
    double unweighted = 0.5 * residual.squaredNorm();
    
    return {chi2, unweighted};
}

/// Compute SE3 residual and return both chi2 and unweighted contributions
std::pair<double, double> ComputeSE3ResidualCosts(
    const Eigen::Matrix<double, 6, 1>& residual,
    const Eigen::Matrix<double, 6, 6>& information) {
    
    // Chi-squared: r^T * Omega * r
    double chi2 = residual.transpose() * information * residual;
    
    // Unweighted: 0.5 * ||r||^2
    double unweighted = 0.5 * residual.squaredNorm();
    
    return {chi2, unweighted};
}

}  // anonymous namespace

CostMetrics ComputeSE2CostMetrics(const g2o_reader::Graph2D& graph) {
    CostMetrics metrics{0.0, 0.0};
    
    if (graph.constraints.empty()) {
        return metrics;
    }

    for (const auto& constraint : graph.constraints) {
        // Get the two poses
        auto pose_i_it = graph.poses.find(constraint.id_begin);
        auto pose_j_it = graph.poses.find(constraint.id_end);

        // Skip if either vertex is missing
        if (pose_i_it == graph.poses.end() || pose_j_it == graph.poses.end()) {
            continue;
        }

        const auto& pose_i = pose_i_it->second;
        const auto& pose_j = pose_j_it->second;

        // Compute actual relative transformation: T_i^{-1} * T_j
        
        // T_i^{-1}
        double cos_i = std::cos(pose_i.rotation);
        double sin_i = std::sin(pose_i.rotation);
        Eigen::Matrix2d R_i_inv;
        R_i_inv << cos_i, sin_i,
                  -sin_i, cos_i;
        Eigen::Vector2d t_i_inv = -R_i_inv * pose_i.translation;

        // T_j
        double cos_j = std::cos(pose_j.rotation);
        double sin_j = std::sin(pose_j.rotation);
        Eigen::Matrix2d R_j;
        R_j << cos_j, -sin_j,
               sin_j,  cos_j;

        // Actual relative: T_i^{-1} * T_j
        Eigen::Matrix2d R_actual = R_i_inv * R_j;
        Eigen::Vector2d t_actual = R_i_inv * pose_j.translation + t_i_inv;
        double theta_actual = std::atan2(R_actual(1, 0), R_actual(0, 0));

        // Measurement inverse
        double cos_m = std::cos(constraint.measurement.rotation);
        double sin_m = std::sin(constraint.measurement.rotation);
        Eigen::Matrix2d R_m_inv;
        R_m_inv << cos_m, sin_m,
                  -sin_m, cos_m;
        Eigen::Vector2d t_m_inv = -R_m_inv * constraint.measurement.translation;

        // Error: measurement^{-1} * actual_relative
        Eigen::Matrix2d R_error = R_m_inv * R_actual;
        Eigen::Vector2d t_error = R_m_inv * t_actual + t_m_inv;
        double theta_error = std::atan2(R_error(1, 0), R_error(0, 0));

        // Normalize angle to [-π, π]
        theta_error = NormalizeAngle(theta_error);

        // Convert to tangent space using exact SE2 log map
        double theta = theta_error;
        double theta_sq = theta * theta;
        double a, b;

        if (theta_sq < 1e-10) {
            // Taylor approximation
            a = 1.0 - theta_sq / 6.0;
            b = 0.5 * theta - theta * theta_sq / 24.0;
        } else {
            // Exact formula
            a = std::sin(theta) / theta;
            b = (1.0 - std::cos(theta)) / theta;
        }

        double det = a * a + b * b;
        double a_scaled = a / det;
        double b_scaled = b / det;

        // V^{-1} = [a_scaled, b_scaled; -b_scaled, a_scaled]
        double x_tangent = a_scaled * t_error.x() + b_scaled * t_error.y();
        double y_tangent = -b_scaled * t_error.x() + a_scaled * t_error.y();

        // Tangent vector: [dx, dy, dtheta]
        Eigen::Vector3d residual;
        residual << x_tangent, y_tangent, theta_error;

        // Compute both metrics
        auto [chi2, unweighted] = ComputeSE2ResidualCosts(residual, constraint.information);
        metrics.chi2_cost += chi2;
        metrics.unweighted_cost += unweighted;
    }

    return metrics;
}

CostMetrics ComputeSE3CostMetrics(const g2o_reader::Graph3D& graph) {
    CostMetrics metrics{0.0, 0.0};
    
    if (graph.constraints.empty()) {
        return metrics;
    }

    for (const auto& constraint : graph.constraints) {
        // Get the two poses
        auto pose_i_it = graph.poses.find(constraint.id_begin);
        auto pose_j_it = graph.poses.find(constraint.id_end);

        // Skip if either vertex is missing
        if (pose_i_it == graph.poses.end() || pose_j_it == graph.poses.end()) {
            continue;
        }

        const auto& pose_i = pose_i_it->second;
        const auto& pose_j = pose_j_it->second;

        // Compute actual relative transformation: T_i^{-1} * T_j

        // T_i^{-1}: inverse quaternion and transformed translation
        Eigen::Quaterniond q_i_inv = pose_i.rotation.conjugate();
        Eigen::Vector3d t_i_inv = -(q_i_inv * pose_i.translation);

        // Actual relative transformation
        Eigen::Quaterniond q_actual = q_i_inv * pose_j.rotation;
        Eigen::Vector3d t_actual = q_i_inv * pose_j.translation + t_i_inv;

        // Measurement inverse
        Eigen::Quaterniond q_m_inv = constraint.measurement.rotation.conjugate();
        Eigen::Vector3d t_m_inv = -(q_m_inv * constraint.measurement.translation);

        // Error: measurement^{-1} * actual_relative
        Eigen::Quaterniond q_error = q_m_inv * q_actual;
        Eigen::Vector3d t_error = q_m_inv * t_actual + t_m_inv;

        // Convert to tangent space (6D vector via SE3 log map)

        // Rotation component using SO3 log map
        Eigen::Vector3d residual_rotation = SO3LogMap(q_error);

        // Translation component using J_l^{-1}
        Eigen::Matrix3d J_l_inv = ComputeSO3LeftJacobianInverse(residual_rotation);
        Eigen::Vector3d residual_translation = J_l_inv * t_error;

        // Construct 6D residual vector: [translation, rotation]
        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = residual_translation;
        residual.tail<3>() = residual_rotation;

        // Compute both metrics
        auto [chi2, unweighted] = ComputeSE3ResidualCosts(residual, constraint.information);
        metrics.chi2_cost += chi2;
        metrics.unweighted_cost += unweighted;
    }

    return metrics;
}

// Legacy functions (for backward compatibility)
double ComputeSE2Cost(const g2o_reader::Graph2D& graph) {
    return ComputeSE2CostMetrics(graph).unweighted_cost;
}

double ComputeSE3Cost(const g2o_reader::Graph3D& graph) {
    return ComputeSE3CostMetrics(graph).unweighted_cost;
}

}  // namespace unified_cost
