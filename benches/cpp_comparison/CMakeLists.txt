cmake_minimum_required(VERSION 3.15)
project(apex_cpp_benchmarks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release build by default for fair benchmarking
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Strategy: Use Eigen 5.0.x for Ceres and g2o, Eigen@3 (3.4.x) for GTSAM
# We need to explicitly set paths to avoid CMake finding the wrong version

# First, find Eigen 5.0.x for Ceres and g2o
set(Eigen3_DIR "/opt/homebrew/Cellar/eigen/5.0.1/share/eigen3/cmake" CACHE PATH "Path to Eigen 5.x CMake config")
find_package(Eigen3 5.0 QUIET NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen 5.x: ${EIGEN3_INCLUDE_DIRS}")
    message(STATUS "Eigen 5.x version: ${Eigen3_VERSION}")
    set(EIGEN5_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
    set(EIGEN5_VERSION ${Eigen3_VERSION})
    set(EIGEN5_AVAILABLE TRUE)
else()
    message(WARNING "Eigen 5.x not found")
    set(EIGEN5_AVAILABLE FALSE)
endif()

# Find Ceres (needs Eigen 5.0.x)
if(EIGEN5_AVAILABLE)
    find_package(Ceres QUIET)
    if(Ceres_FOUND)
        message(STATUS "Found Ceres: ${CERES_VERSION}")
        set(BUILD_CERES_BENCHMARK TRUE)
    else()
        message(WARNING "Ceres not found - skipping Ceres benchmark")
        message(WARNING "To install: brew install ceres-solver")
        set(BUILD_CERES_BENCHMARK FALSE)
    endif()
else()
    message(WARNING "Eigen 5.x not available - skipping Ceres benchmark")
    set(BUILD_CERES_BENCHMARK FALSE)
endif()

# Now switch to Eigen@3 for GTSAM only
unset(Eigen3_DIR CACHE)
set(Eigen3_DIR "/opt/homebrew/opt/eigen@3/share/eigen3/cmake" CACHE PATH "Path to Eigen3 CMake config")
unset(Eigen3_FOUND)
unset(EIGEN3_INCLUDE_DIRS)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen@3: ${EIGEN3_INCLUDE_DIRS}")
    message(STATUS "Eigen@3 version: ${Eigen3_VERSION}")
    set(EIGEN3_INCLUDE_DIRS_34 ${EIGEN3_INCLUDE_DIRS})
    set(EIGEN3_VERSION_34 ${Eigen3_VERSION})
    set(EIGEN3_AVAILABLE TRUE)
else()
    message(WARNING "Eigen@3 not found - GTSAM benchmark may not build")
    set(EIGEN3_AVAILABLE FALSE)
endif()

# Find GTSAM (needs Eigen@3)
if(EIGEN3_AVAILABLE)
    find_package(GTSAM QUIET)
    if(GTSAM_FOUND)
        message(STATUS "Found GTSAM: ${GTSAM_VERSION}")
        set(BUILD_GTSAM_BENCHMARK TRUE)
    else()
        message(WARNING "GTSAM not found - skipping GTSAM benchmark")
        message(WARNING "To install: brew install gtsam")
        set(BUILD_GTSAM_BENCHMARK FALSE)
    endif()
else()
    message(WARNING "Eigen@3 not available - skipping GTSAM benchmark")
    set(BUILD_GTSAM_BENCHMARK FALSE)
endif()

# Find g2o (needs Eigen 5.0.x)
if(EIGEN5_AVAILABLE)
    find_package(g2o QUIET)
    if(g2o_FOUND)
        message(STATUS "Found g2o")
        set(BUILD_G2O_BENCHMARK TRUE)
    else()
        message(WARNING "g2o not found - skipping g2o benchmark")
        message(WARNING "To install: brew install g2o")
        set(BUILD_G2O_BENCHMARK FALSE)
    endif()
else()
    message(WARNING "Eigen 5.x not available - skipping g2o benchmark")
    set(BUILD_G2O_BENCHMARK FALSE)
endif()

# Common utilities library for Ceres and g2o (use Eigen 5.0.x)
add_library(bench_common STATIC
    common/read_g2o.cpp
)

target_include_directories(bench_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN5_INCLUDE_DIRS}
)

target_link_libraries(bench_common PUBLIC
    Eigen3::Eigen
)

# Ceres benchmark executable (uses Eigen 5.0.x via bench_common)
if(BUILD_CERES_BENCHMARK)
    add_executable(ceres_benchmark ceres_benchmark.cpp)

    target_link_libraries(ceres_benchmark PRIVATE
        bench_common
        Ceres::ceres
    )

    # Add compile definition to help debug
    target_compile_definitions(ceres_benchmark PRIVATE EIGEN_VERSION_STRING="${EIGEN5_VERSION}")
endif()

# GTSAM benchmark executable (uses Eigen@3, builds read_g2o.cpp separately)
if(BUILD_GTSAM_BENCHMARK)
    add_executable(gtsam_benchmark gtsam_benchmark.cpp common/read_g2o.cpp)

    # Use Eigen 3.4 include dirs explicitly
    target_include_directories(gtsam_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIRS_34}
    )

    target_link_libraries(gtsam_benchmark PRIVATE
        gtsam
        Eigen3::Eigen
    )
endif()

# g2o benchmark executable (uses Eigen 5.0.x via bench_common)
if(BUILD_G2O_BENCHMARK)
    add_executable(g2o_benchmark g2o_benchmark.cpp)

    target_link_libraries(g2o_benchmark PRIVATE
        bench_common
        Eigen3::Eigen
        g2o::core
        g2o::stuff
        g2o::types_slam2d
        g2o::types_slam3d
        g2o::solver_eigen
    )

    # Add compile definition to help debug
    target_compile_definitions(g2o_benchmark PRIVATE EIGEN_VERSION_STRING="${EIGEN5_VERSION}")
endif()

# Check if at least one benchmark can be built
if(NOT BUILD_CERES_BENCHMARK AND NOT BUILD_GTSAM_BENCHMARK AND NOT BUILD_G2O_BENCHMARK)
    message(FATAL_ERROR "No C++ optimization libraries found!
        Please install at least one of: ceres-solver, gtsam, g2o")
endif()

# Print summary
message(STATUS "")
message(STATUS "=== APEX C++ Benchmark Suite Configuration ===")
if(BUILD_CERES_BENCHMARK)
    message(STATUS "  Ceres version: ${CERES_VERSION} (with Eigen ${EIGEN5_VERSION})")
else()
    message(STATUS "  Ceres: NOT AVAILABLE")
endif()
if(BUILD_GTSAM_BENCHMARK)
    message(STATUS "  GTSAM version: ${GTSAM_VERSION} (with Eigen ${EIGEN3_VERSION_34})")
else()
    message(STATUS "  GTSAM: NOT AVAILABLE")
endif()
if(BUILD_G2O_BENCHMARK)
    message(STATUS "  g2o: AVAILABLE (with Eigen ${EIGEN5_VERSION})")
else()
    message(STATUS "  g2o: NOT AVAILABLE")
endif()
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Executables to build:")
if(BUILD_CERES_BENCHMARK)
    message(STATUS "  - ceres_benchmark")
endif()
if(BUILD_GTSAM_BENCHMARK)
    message(STATUS "  - gtsam_benchmark")
endif()
if(BUILD_G2O_BENCHMARK)
    message(STATUS "  - g2o_benchmark")
endif()
message(STATUS "")
