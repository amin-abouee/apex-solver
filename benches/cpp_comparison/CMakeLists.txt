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

# =============================================================================
# Eigen Configuration
# Strategy: Use Eigen 5.0.x for Ceres and g2o, Eigen@3 (3.4.x) for GTSAM
# =============================================================================

# First, find Eigen 5.0.x for Ceres and g2o
set(Eigen3_DIR "/opt/homebrew/opt/eigen/share/eigen3/cmake" CACHE PATH "Path to Eigen 5.x CMake config")
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

# =============================================================================
# Common Libraries
# =============================================================================

# Common utilities library for Ceres and g2o (use Eigen 5.0.x)
# Contains: read_g2o, unified_cost for odometry benchmarks
add_library(odometry_common STATIC
    common/src/read_g2o.cpp
    common/src/unified_cost.cpp
)

target_include_directories(odometry_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${EIGEN5_INCLUDE_DIRS}
)

target_link_libraries(odometry_common PUBLIC
    Eigen3::Eigen
)

# Bundle Adjustment common utilities library (Eigen 5.0.x)
# Contains: read_bal, ba_cost for bundle adjustment benchmarks
add_library(ba_common STATIC
    common/src/read_bal.cpp
    common/src/ba_cost.cpp
)

target_include_directories(ba_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${EIGEN5_INCLUDE_DIRS}
)

target_link_libraries(ba_common PUBLIC
    Eigen3::Eigen
)

# =============================================================================
# Ceres Benchmarks
# =============================================================================
if(BUILD_CERES_BENCHMARK)
    # Ceres Odometry Benchmark
    add_executable(ceres_odometry_benchmark odometry/src/ceres_odometry.cpp)
    target_include_directories(ceres_odometry_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/odometry/include
    )
    target_link_libraries(ceres_odometry_benchmark PRIVATE
        odometry_common
        Ceres::ceres
    )
    target_compile_definitions(ceres_odometry_benchmark PRIVATE EIGEN_VERSION_STRING="${EIGEN5_VERSION}")

    # Ceres Bundle Adjustment Benchmark
    add_executable(ceres_ba_benchmark bundle_adjustment/src/ceres.cpp)
    target_include_directories(ceres_ba_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment/include
    )
    target_link_libraries(ceres_ba_benchmark PRIVATE
        ba_common
        Ceres::ceres
    )

    # OpenMP for multi-threading
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ceres_ba_benchmark PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "Ceres BA: OpenMP enabled")
    else()
        message(WARNING "OpenMP not found - Ceres BA will be single-threaded")
    endif()

    # Optimization flags
    target_compile_options(ceres_ba_benchmark PRIVATE -O3 -march=native)
endif()

# =============================================================================
# GTSAM Benchmarks
# =============================================================================
if(BUILD_GTSAM_BENCHMARK)
    # GTSAM Odometry Benchmark (uses Eigen@3, builds common sources separately)
    add_executable(gtsam_odometry_benchmark
        odometry/src/gtsam_odometry.cpp
        common/src/read_g2o.cpp
        common/src/unified_cost.cpp
    )
    target_include_directories(gtsam_odometry_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/odometry/include
        ${EIGEN3_INCLUDE_DIRS_34}
    )
    target_link_libraries(gtsam_odometry_benchmark PRIVATE
        gtsam
        Eigen3::Eigen
    )

    # GTSAM Bundle Adjustment Benchmark
    add_executable(gtsam_ba_benchmark
        bundle_adjustment/src/gtsam.cpp
        common/src/read_bal.cpp
        common/src/ba_cost.cpp
    )
    target_include_directories(gtsam_ba_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment/include
        ${EIGEN3_INCLUDE_DIRS_34}
    )
    target_link_libraries(gtsam_ba_benchmark PRIVATE
        gtsam
        Eigen3::Eigen
    )

    # Intel OneTBB for GTSAM
    find_package(TBB)
    if(TBB_FOUND)
        target_link_libraries(gtsam_ba_benchmark PRIVATE TBB::tbb)
        message(STATUS "GTSAM BA: Intel OneTBB enabled")
    else()
        message(WARNING "Intel OneTBB not found - GTSAM BA may be single-threaded")
        message(WARNING "To install: brew install tbb")
    endif()

    # Optimization flags
    target_compile_options(gtsam_ba_benchmark PRIVATE -O3 -march=native)
endif()

# =============================================================================
# g2o Benchmarks
# =============================================================================
if(BUILD_G2O_BENCHMARK)
    # g2o Odometry Benchmark
    add_executable(g2o_odometry_benchmark odometry/src/g2o_odometry.cpp)
    target_include_directories(g2o_odometry_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/odometry/include
    )
    target_link_libraries(g2o_odometry_benchmark PRIVATE
        odometry_common
        Eigen3::Eigen
        g2o::core
        g2o::stuff
        g2o::types_slam2d
        g2o::types_slam3d
        g2o::solver_eigen
    )
    target_compile_definitions(g2o_odometry_benchmark PRIVATE EIGEN_VERSION_STRING="${EIGEN5_VERSION}")

    # g2o Bundle Adjustment Benchmark
    add_executable(g2o_ba_benchmark bundle_adjustment/src/g2o.cpp)
    target_include_directories(g2o_ba_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment/include
    )
    target_link_libraries(g2o_ba_benchmark PRIVATE
        ba_common
        Eigen3::Eigen
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::solver_eigen
    )

    # OpenMP for multi-threading
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(g2o_ba_benchmark PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "g2o BA: OpenMP enabled")
    else()
        message(WARNING "OpenMP not found - g2o BA will be single-threaded")
    endif()

    # Optimization flags
    target_compile_options(g2o_ba_benchmark PRIVATE -O3 -march=native)
endif()

# =============================================================================
# Validation
# =============================================================================
if(NOT BUILD_CERES_BENCHMARK AND NOT BUILD_GTSAM_BENCHMARK AND NOT BUILD_G2O_BENCHMARK)
    message(FATAL_ERROR "No C++ optimization libraries found!
        Please install at least one of: ceres-solver, gtsam, g2o")
endif()

# =============================================================================
# Summary
# =============================================================================
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
message(STATUS "  Odometry Benchmarks:")
if(BUILD_CERES_BENCHMARK)
    message(STATUS "    - ceres_odometry_benchmark")
endif()
if(BUILD_GTSAM_BENCHMARK)
    message(STATUS "    - gtsam_odometry_benchmark")
endif()
if(BUILD_G2O_BENCHMARK)
    message(STATUS "    - g2o_odometry_benchmark")
endif()
message(STATUS "  Bundle Adjustment Benchmarks:")
if(BUILD_CERES_BENCHMARK)
    message(STATUS "    - ceres_ba_benchmark (with OpenMP)")
endif()
if(BUILD_GTSAM_BENCHMARK)
    message(STATUS "    - gtsam_ba_benchmark (with OneTBB)")
endif()
if(BUILD_G2O_BENCHMARK)
    message(STATUS "    - g2o_ba_benchmark (with OpenMP)")
endif()
message(STATUS "")
