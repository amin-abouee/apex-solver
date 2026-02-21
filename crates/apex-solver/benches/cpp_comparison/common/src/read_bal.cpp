#include "read_bal.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace bal_reader {

bool ReadBALFile(const std::string& filepath, BALDataset& dataset) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "ERROR: Failed to open BAL file: " << filepath << std::endl;
        return false;
    }

    // Read header
    int num_cameras, num_points, num_observations;
    infile >> num_cameras >> num_points >> num_observations;
    
    if (num_cameras <= 0 || num_points <= 0 || num_observations <= 0) {
        std::cerr << "ERROR: Invalid BAL header: cameras=" << num_cameras 
                  << " points=" << num_points << " observations=" << num_observations << std::endl;
        return false;
    }

    std::cout << "Loading BAL dataset: " << num_cameras << " cameras, " 
              << num_points << " points, " << num_observations << " observations" << std::endl;

    // Pre-allocate vectors
    dataset.cameras.reserve(num_cameras);
    dataset.points.reserve(num_points);
    dataset.observations.reserve(num_observations);

    // Read observations (camera_idx, point_idx, x, y per line)
    for (int i = 0; i < num_observations; ++i) {
        BALObservation obs;
        infile >> obs.camera_index >> obs.point_index >> obs.x >> obs.y;
        
        if (infile.fail()) {
            std::cerr << "ERROR: Failed to read observation " << i << std::endl;
            return false;
        }
        
        dataset.observations.push_back(obs);
    }

    // Read cameras (9 parameters each: rx, ry, rz, tx, ty, tz, f, k1, k2)
    for (int i = 0; i < num_cameras; ++i) {
        BALCamera cam;
        infile >> cam.rotation[0] >> cam.rotation[1] >> cam.rotation[2]
               >> cam.translation[0] >> cam.translation[1] >> cam.translation[2]
               >> cam.focal_length >> cam.k1 >> cam.k2;
        
        if (infile.fail()) {
            std::cerr << "ERROR: Failed to read camera " << i << std::endl;
            return false;
        }
        
        dataset.cameras.push_back(cam);
    }

    // Read points (3 coordinates each: x, y, z)
    for (int i = 0; i < num_points; ++i) {
        BALPoint pt;
        infile >> pt.position[0] >> pt.position[1] >> pt.position[2];
        
        if (infile.fail()) {
            std::cerr << "ERROR: Failed to read point " << i << std::endl;
            return false;
        }
        
        dataset.points.push_back(pt);
    }

    infile.close();
    
    std::cout << "Successfully loaded BAL dataset" << std::endl;
    return true;
}

}  // namespace bal_reader
