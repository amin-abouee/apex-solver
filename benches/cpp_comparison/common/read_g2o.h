#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace g2o_reader {

// SE2 pose representation (x, y, theta)
struct Pose2D {
    Eigen::Vector2d translation;
    double rotation;  // angle in radians

    Pose2D() : translation(Eigen::Vector2d::Zero()), rotation(0.0) {}
    Pose2D(double x, double y, double theta) 
        : translation(x, y), rotation(theta) {}
};

// SE3 pose representation (quaternion + translation)
struct Pose3D {
    Eigen::Quaterniond rotation;
    Eigen::Vector3d translation;

    Pose3D() 
        : rotation(Eigen::Quaterniond::Identity()), 
          translation(Eigen::Vector3d::Zero()) {}
    
    Pose3D(const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
        : rotation(q), translation(t) {}
};

// SE2 edge/constraint
struct Constraint2D {
    int id_begin;
    int id_end;
    Pose2D measurement;
    Eigen::Matrix3d information;  // 3x3 information matrix

    Constraint2D() : id_begin(0), id_end(0), information(Eigen::Matrix3d::Identity()) {}
};

// SE3 edge/constraint
struct Constraint3D {
    int id_begin;
    int id_end;
    Pose3D measurement;
    Eigen::Matrix<double, 6, 6> information;  // 6x6 information matrix

    Constraint3D() : id_begin(0), id_end(0), information(Eigen::Matrix<double, 6, 6>::Identity()) {}
};

// Result structure for SE2 graphs
struct Graph2D {
    std::map<int, Pose2D> poses;
    std::vector<Constraint2D> constraints;
};

// Result structure for SE3 graphs
struct Graph3D {
    std::map<int, Pose3D> poses;
    std::vector<Constraint3D> constraints;
};

// Load SE2 g2o file
bool ReadG2oFile2D(const std::string& filename, Graph2D& graph);

// Load SE3 g2o file
bool ReadG2oFile3D(const std::string& filename, Graph3D& graph);

// Auto-detect 2D or 3D based on file content
enum class GraphType { SE2, SE3, UNKNOWN };
GraphType DetectGraphType(const std::string& filename);

}  // namespace g2o_reader
