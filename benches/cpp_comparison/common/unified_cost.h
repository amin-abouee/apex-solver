#pragma once

#include "read_g2o.h"

namespace unified_cost {

/// Compute SE2 cost from graph data using unified formula
///
/// Formula: cost = 0.5 * sum_i ||r_i||²_Σ
/// where ||r||²_Σ = r^T * Σ^(-1) * r (information-weighted squared norm)
///
/// This function computes cost directly from G2O graph data, independent of
/// solver internals, for fair benchmarking across all solvers.
///
/// @param graph The SE2 graph containing poses and constraints
/// @return Total cost value
double ComputeSE2Cost(const g2o_reader::Graph2D& graph);

/// Compute SE3 cost from graph data using unified formula
///
/// Formula: cost = 0.5 * sum_i ||r_i||²_Σ
/// where ||r||²_Σ = r^T * Σ^(-1) * r (information-weighted squared norm)
///
/// This function computes cost directly from G2O graph data, independent of
/// solver internals, for fair benchmarking across all solvers.
///
/// @param graph The SE3 graph containing poses and constraints
/// @return Total cost value
double ComputeSE3Cost(const g2o_reader::Graph3D& graph);

}  // namespace unified_cost
