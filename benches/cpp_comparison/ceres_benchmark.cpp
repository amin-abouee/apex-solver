#include <ceres/ceres.h>
#include <Eigen/Core>
#include <iostream>
#include <map>
#include <vector>

#include "common/benchmark_utils.h"
#include "common/read_g2o.h"

// SE2 residual (3 DOF: x, y, theta)
class PoseGraph2DErrorTerm {
public:
    PoseGraph2DErrorTerm(double x, double y, double theta, const Eigen::Matrix3d& sqrt_information)
        : x_(x), y_(y), theta_(theta), sqrt_information_(sqrt_information) {}

    template <typename T>
    bool operator()(const T* const pose_a, const T* const pose_b, T* residuals_ptr) const {
        // Compute relative transformation
        const T cos_theta_a = ceres::cos(pose_a[2]);
        const T sin_theta_a = ceres::sin(pose_a[2]);

        T delta_x = pose_b[0] - pose_a[0];
        T delta_y = pose_b[1] - pose_a[1];

        T h_x = cos_theta_a * delta_x + sin_theta_a * delta_y;
        T h_y = -sin_theta_a * delta_x + cos_theta_a * delta_y;
        T h_theta = pose_b[2] - pose_a[2];

        // Normalize angle to [-pi, pi]
        h_theta = ceres::atan2(ceres::sin(h_theta), ceres::cos(h_theta));

        // Compute error
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals(0) = h_x - T(x_);
        residuals(1) = h_y - T(y_);
        residuals(2) = h_theta - T(theta_);

        // Apply information matrix (sqrt for Ceres)
        residuals = sqrt_information_.template cast<T>() * residuals;

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

// SE3 residual (6 DOF: rotation quaternion + translation)
class PoseGraph3DErrorTerm {
public:
    PoseGraph3DErrorTerm(const g2o_reader::Pose3D& measurement,
                         const Eigen::Matrix<double, 6, 6>& sqrt_information)
        : measurement_(measurement), sqrt_information_(sqrt_information) {}

    template <typename T>
    bool operator()(const T* const pose_a_quat, const T* const pose_a_trans,
                    const T* const pose_b_quat, const T* const pose_b_trans,
                    T* residuals_ptr) const {
        // Quaternions: [qw, qx, qy, qz]
        Eigen::Map<const Eigen::Quaternion<T>> q_a(pose_a_quat);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_a(pose_a_trans);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(pose_b_quat);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_b(pose_b_trans);

        // Compute relative transformation: T_ab_measured vs T_ab_current
        // T_ab_current = T_a^{-1} * T_b
        Eigen::Quaternion<T> q_a_inv = q_a.conjugate();
        Eigen::Quaternion<T> q_ab = q_a_inv * q_b;
        Eigen::Matrix<T, 3, 1> t_ab = q_a_inv * (t_b - t_a);

        // Measurement
        Eigen::Quaternion<T> q_ab_measured = measurement_.rotation.template cast<T>();
        Eigen::Matrix<T, 3, 1> t_ab_measured = measurement_.translation.template cast<T>();

        // Compute residuals
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);

        // Translation error
        residuals.template head<3>() = t_ab - t_ab_measured;

        // Rotation error (quaternion difference)
        Eigen::Quaternion<T> q_error = q_ab_measured.conjugate() * q_ab;
        residuals.template tail<3>() = T(2.0) * q_error.vec();

        // Apply information matrix
        residuals = sqrt_information_.template cast<T>() * residuals;

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

// Benchmark SE2 dataset with Ceres
benchmark_utils::BenchmarkResult BenchmarkSE2(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE2";
    result.solver = "Ceres-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph2D graph;
    if (!g2o_reader::ReadG2oFile2D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Create Ceres problem
    ceres::Problem problem;

    // Map to store pose parameters (use std::vector for stable addresses)
    std::map<int, std::vector<double>> pose_params;
    for (const auto& [id, pose] : graph.poses) {
        pose_params[id] = {pose.translation.x(), pose.translation.y(), pose.rotation};
    }

    // Add edge constraints first (so parameter blocks are registered)
    for (const auto& constraint : graph.constraints) {
        auto& pose_a = pose_params[constraint.id_begin];
        auto& pose_b = pose_params[constraint.id_end];

        Eigen::Matrix3d sqrt_info = constraint.information.llt().matrixL();

        ceres::CostFunction* cost_function = PoseGraph2DErrorTerm::Create(
            constraint.measurement.translation.x(),
            constraint.measurement.translation.y(),
            constraint.measurement.rotation,
            sqrt_info);

        problem.AddResidualBlock(cost_function, nullptr, pose_a.data(), pose_b.data());
    }

    // Add prior on first pose (gauge freedom) - after residual blocks are added
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        auto& first_pose = pose_params[first_id];
        problem.SetParameterBlockConstant(first_pose.data());
    }

    // Compute initial cost
    double initial_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &initial_cost, nullptr, nullptr, nullptr);
    result.initial_cost = initial_cost;

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-3;
    options.parameter_tolerance = 1e-3;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // Solve
    ceres::Solver::Summary summary;
    Timer timer;
    ceres::Solve(options, &problem, &summary);
    result.time_ms = timer.elapsed_ms();

    // Extract results
    result.final_cost = summary.final_cost;
    result.iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    result.improvement_pct = ((initial_cost - summary.final_cost) / initial_cost) * 100.0;
    result.status = (summary.termination_type == ceres::CONVERGENCE) ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

// Benchmark SE3 dataset with Ceres
benchmark_utils::BenchmarkResult BenchmarkSE3(const std::string& dataset_name,
                                             const std::string& filepath) {
    using namespace benchmark_utils;
    BenchmarkResult result;
    result.dataset = dataset_name;
    result.manifold = "SE3";
    result.solver = "Ceres-LM";
    result.language = "C++";

    // Load graph
    g2o_reader::Graph3D graph;
    if (!g2o_reader::ReadG2oFile3D(filepath, graph)) {
        result.status = "LOAD_FAILED";
        return result;
    }

    result.vertices = graph.poses.size();
    result.edges = graph.constraints.size();

    // Create Ceres problem
    ceres::Problem problem;

    // Map to store pose parameters (quaternion [qw,qx,qy,qz] + translation [x,y,z])
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> pose_params;
    for (const auto& [id, pose] : graph.poses) {
        std::vector<double> quat = {pose.rotation.w(), pose.rotation.x(),
                                     pose.rotation.y(), pose.rotation.z()};
        std::vector<double> trans = {pose.translation.x(), pose.translation.y(),
                                      pose.translation.z()};
        pose_params[id] = {quat, trans};
    }

    // Use quaternion manifold (new Ceres API)
    ceres::Manifold* quaternion_manifold = new ceres::QuaternionManifold();

    // Add edge constraints first (so parameter blocks are registered)
    for (const auto& constraint : graph.constraints) {
        auto& pose_a = pose_params[constraint.id_begin];
        auto& pose_b = pose_params[constraint.id_end];

        Eigen::Matrix<double, 6, 6> sqrt_info = constraint.information.llt().matrixL();

        ceres::CostFunction* cost_function = PoseGraph3DErrorTerm::Create(
            constraint.measurement, sqrt_info);

        problem.AddResidualBlock(cost_function, nullptr,
                                pose_a.first.data(), pose_a.second.data(),
                                pose_b.first.data(), pose_b.second.data());

        // Set quaternion manifold (new API: SetManifold instead of SetParameterization)
        problem.SetManifold(pose_a.first.data(), quaternion_manifold);
        problem.SetManifold(pose_b.first.data(), quaternion_manifold);
    }

    // Add prior on first pose (gauge freedom) - after residual blocks are added
    if (!graph.poses.empty()) {
        int first_id = graph.poses.begin()->first;
        problem.SetParameterBlockConstant(pose_params[first_id].first.data());   // quaternion
        problem.SetParameterBlockConstant(pose_params[first_id].second.data());  // translation
    }

    // Compute initial cost
    double initial_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &initial_cost, nullptr, nullptr, nullptr);
    result.initial_cost = initial_cost;

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-3;
    options.parameter_tolerance = 1e-3;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // Solve
    ceres::Solver::Summary summary;
    Timer timer;
    ceres::Solve(options, &problem, &summary);
    result.time_ms = timer.elapsed_ms();

    // Extract results
    result.final_cost = summary.final_cost;
    result.iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    result.improvement_pct = ((initial_cost - summary.final_cost) / initial_cost) * 100.0;
    result.status = (summary.termination_type == ceres::CONVERGENCE) ? "CONVERGED" : "NOT_CONVERGED";

    return result;
}

int main(int argc, char** argv) {
    std::vector<benchmark_utils::BenchmarkResult> results;

    // SE3 datasets
    results.push_back(BenchmarkSE3("sphere2500", "../../../data/sphere2500.g2o"));
    results.push_back(BenchmarkSE3("parking-garage", "../../../data/parking-garage.g2o"));
    results.push_back(BenchmarkSE3("torus3D", "../../../data/torus3D.g2o"));
    results.push_back(BenchmarkSE3("cubicle", "../../../data/cubicle.g2o"));

    // SE2 datasets
    results.push_back(BenchmarkSE2("intel", "../../../data/intel.g2o"));
    results.push_back(BenchmarkSE2("mit", "../../../data/mit.g2o"));
    results.push_back(BenchmarkSE2("ring", "../../../data/ring.g2o"));
    results.push_back(BenchmarkSE2("M3500", "../../../data/M3500.g2o"));

    // Write to CSV
    std::string output_file = "ceres_benchmark_results.csv";
    benchmark_utils::WriteResultsToCSV(output_file, results);

    return 0;
}
