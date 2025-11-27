#include "read_g2o.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace g2o_reader {

GraphType DetectGraphType(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return GraphType::UNKNOWN;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "VERTEX_SE2" || tag == "EDGE_SE2") {
            return GraphType::SE2;
        } else if (tag == "VERTEX_SE3:QUAT" || tag == "EDGE_SE3:QUAT") {
            return GraphType::SE3;
        }
    }

    return GraphType::UNKNOWN;
}

bool ReadG2oFile2D(const std::string& filename, Graph2D& graph) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    graph.poses.clear();
    graph.constraints.clear();

    std::string line;
    int line_number = 0;

    while (std::getline(infile, line)) {
        line_number++;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "VERTEX_SE2") {
            int id;
            double x, y, theta;
            if (!(iss >> id >> x >> y >> theta)) {
                std::cerr << "Error parsing VERTEX_SE2 at line " << line_number << std::endl;
                continue;
            }
            graph.poses[id] = Pose2D(x, y, theta);
        }
        else if (tag == "EDGE_SE2") {
            Constraint2D constraint;
            double x, y, theta;

            if (!(iss >> constraint.id_begin >> constraint.id_end >> x >> y >> theta)) {
                std::cerr << "Error parsing EDGE_SE2 at line " << line_number << std::endl;
                continue;
            }

            constraint.measurement = Pose2D(x, y, theta);

            // Read information matrix (upper triangular: 6 values for 3x3 symmetric)
            double info[6];
            for (int i = 0; i < 6; ++i) {
                if (!(iss >> info[i])) {
                    std::cerr << "Error reading information matrix at line " << line_number << std::endl;
                    continue;
                }
            }

            // Fill symmetric information matrix
            constraint.information <<
                info[0], info[1], info[2],
                info[1], info[3], info[4],
                info[2], info[4], info[5];

            graph.constraints.push_back(constraint);
        }
    }

    return true;
}

bool ReadG2oFile3D(const std::string& filename, Graph3D& graph) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    graph.poses.clear();
    graph.constraints.clear();

    std::string line;
    int line_number = 0;

    while (std::getline(infile, line)) {
        line_number++;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "VERTEX_SE3:QUAT") {
            int id;
            double tx, ty, tz, qx, qy, qz, qw;

            if (!(iss >> id >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
                std::cerr << "Error parsing VERTEX_SE3:QUAT at line " << line_number << std::endl;
                continue;
            }

            Eigen::Quaterniond q(qw, qx, qy, qz);
            q.normalize();
            Eigen::Vector3d t(tx, ty, tz);
            graph.poses[id] = Pose3D(q, t);
        }
        else if (tag == "EDGE_SE3:QUAT") {
            Constraint3D constraint;
            double tx, ty, tz, qx, qy, qz, qw;

            if (!(iss >> constraint.id_begin >> constraint.id_end
                     >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
                std::cerr << "Error parsing EDGE_SE3:QUAT at line " << line_number << std::endl;
                continue;
            }

            Eigen::Quaterniond q(qw, qx, qy, qz);
            q.normalize();
            Eigen::Vector3d t(tx, ty, tz);
            constraint.measurement = Pose3D(q, t);

            // Read information matrix (upper triangular: 21 values for 6x6 symmetric)
            double info[21];
            for (int i = 0; i < 21; ++i) {
                if (!(iss >> info[i])) {
                    std::cerr << "Error reading information matrix at line " << line_number << std::endl;
                    break;
                }
            }

            // Fill symmetric 6x6 information matrix from upper triangular
            int idx = 0;
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    constraint.information(i, j) = info[idx];
                    constraint.information(j, i) = info[idx];
                    idx++;
                }
            }

            graph.constraints.push_back(constraint);
        }
    }

   return true;
}

}  // namespace g2o_reader
