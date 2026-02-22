#pragma once

#include "read_g2o.h"

namespace unified_cost {

/// Struct to hold both cost metrics
struct CostMetrics {
    double chi2_cost;       ///< Information-weighted chi-squared: sum_i r_i^T * Omega_i * r_i
    double unweighted_cost; ///< Unweighted squared norm: 0.5 * sum_i ||r_i||^2
};

/// Compute both SE2 cost metrics from graph data
///
/// Chi-squared formula: chi2 = sum_i r_i^T * Omega_i * r_i (information-weighted)
/// Unweighted formula: cost = 0.5 * sum_i ||r_i||^2
///
/// @param graph The SE2 graph containing poses and constraints
/// @return CostMetrics struct with both metrics
CostMetrics ComputeSE2CostMetrics(const g2o_reader::Graph2D& graph);

/// Compute both SE3 cost metrics from graph data
///
/// Chi-squared formula: chi2 = sum_i r_i^T * Omega_i * r_i (information-weighted)
/// Unweighted formula: cost = 0.5 * sum_i ||r_i||^2
///
/// @param graph The SE3 graph containing poses and constraints
/// @return CostMetrics struct with both metrics
CostMetrics ComputeSE3CostMetrics(const g2o_reader::Graph3D& graph);

// Legacy functions (for backward compatibility, return unweighted cost)
double ComputeSE2Cost(const g2o_reader::Graph2D& graph);
double ComputeSE3Cost(const g2o_reader::Graph3D& graph);

}  // namespace unified_cost
