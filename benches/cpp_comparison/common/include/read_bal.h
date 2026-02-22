#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace bal_reader {

struct BALCamera {
    Eigen::Vector3d rotation;     // Axis-angle representation
    Eigen::Vector3d translation;
    double focal_length;
    double k1;                    // First radial distortion coefficient
    double k2;                    // Second radial distortion coefficient
};

struct BALPoint {
    Eigen::Vector3d position;
};

struct BALObservation {
    int camera_index;
    int point_index;
    double x;  // Pixel coordinate
    double y;
};

struct BALDataset {
    std::vector<BALCamera> cameras;
    std::vector<BALPoint> points;
    std::vector<BALObservation> observations;
    
    int num_cameras() const { return static_cast<int>(cameras.size()); }
    int num_points() const { return static_cast<int>(points.size()); }
    int num_observations() const { return static_cast<int>(observations.size()); }
};

/// Read BAL file with multi-threaded parsing
/// Returns true on success, false on error
bool ReadBALFile(const std::string& filepath, BALDataset& dataset);

}  // namespace bal_reader
