cmake_minimum_required(VERSION 3.15)
project(apex_cpp_benchmarks CXX)
# CMP0167 (use upstream BoostConfig.cmake) only exists from CMake 3.30; Ubuntu
# noble ships 3.28.3, where cmake_policy(SET) on an unknown policy errors out.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release build by default for fair benchmarking
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Install hint using the host platform's package manager. Called from each
# find_package() failure branch below; four call sites make a helper worth it.
function(apex_install_hint brew_cmd apt_pkg)
    if(APPLE)
        message(WARNING "To install: ${brew_cmd}")
    else()
        message(WARNING "To install: sudo apt install ${apt_pkg}")
    endif()
endfunction()

find_package(Eigen3 QUIET NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen: ${Eigen3_DIR}")
    message(STATUS "Eigen version: ${Eigen3_VERSION}")
    set(EIGEN_VERSION ${Eigen3_VERSION})
    set(EIGEN_AVAILABLE TRUE)
else()
    message(WARNING "Eigen not found")
    set(EIGEN_AVAILABLE FALSE)
endif()

# Find Ceres (needs Eigen)
if(EIGEN_AVAILABLE)
    find_package(Ceres QUIET)
    if(Ceres_FOUND)
        message(STATUS "Found Ceres: ${CERES_VERSION}")
        set(BUILD_CERES_BENCHMARK TRUE)
    else()
        message(WARNING "Ceres not found - skipping Ceres benchmark")
        apex_install_hint("brew install ceres-solver" "libceres-dev")
        set(BUILD_CERES_BENCHMARK FALSE)
    endif()
else()
    message(WARNING "Eigen not available - skipping Ceres benchmark")
    set(BUILD_CERES_BENCHMARK FALSE)
endif()

# Find GTSAM (needs Eigen, and >= 4.3)
if(EIGEN_AVAILABLE)
    find_package(GTSAM 4.3 QUIET)
    if(GTSAM_FOUND)
        message(STATUS "Found GTSAM: ${GTSAM_VERSION}")
        set(BUILD_GTSAM_BENCHMARK TRUE)
    else()
        message(WARNING "GTSAM not found - skipping GTSAM benchmark")
        if(APPLE)
            message(WARNING "To install: brew install gtsam")
        else()
            # Ubuntu's libgtsam-dev is 4.2.0, below this project's 4.3 floor.
            message(WARNING "To install: build GTSAM >= 4.3 from source (https://github.com/borglab/gtsam); apt's libgtsam-dev is only 4.2.0")
        endif()
        set(BUILD_GTSAM_BENCHMARK FALSE)
    endif()
else()
    message(WARNING "Eigen not available - skipping GTSAM benchmark")
    set(BUILD_GTSAM_BENCHMARK FALSE)
endif()

# Find g2o (needs Eigen)
if(EIGEN_AVAILABLE)
    find_package(g2o QUIET)
    if(g2o_FOUND)
        message(STATUS "Found g2o")
        set(BUILD_G2O_BENCHMARK TRUE)
    else()
        message(WARNING "g2o not found - skipping g2o benchmark")
        apex_install_hint("brew install g2o" "libg2o-dev")
        set(BUILD_G2O_BENCHMARK FALSE)
    endif()
else()
    message(WARNING "Eigen not available - skipping g2o benchmark")
    set(BUILD_G2O_BENCHMARK FALSE)
endif()

# =============================================================================
# Common Libraries
# =============================================================================

# Common utilities library for Ceres and g2o (uses Eigen)
# Contains: read_g2o, unified_cost for odometry benchmarks
add_library(odometry_common STATIC
    common/src/read_g2o.cpp
    common/src/unified_cost.cpp
)

target_include_directories(odometry_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
)

target_link_libraries(odometry_common PUBLIC
    Eigen3::Eigen
)

# Bundle Adjustment common utilities library (uses Eigen)
# Contains: read_bal, ba_cost for bundle adjustment benchmarks
add_library(ba_common STATIC
    common/src/read_bal.cpp
    common/src/ba_cost.cpp
)

target_include_directories(ba_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
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
    target_compile_definitions(ceres_odometry_benchmark PRIVATE EIGEN_VERSION_STRING="${EIGEN_VERSION}")

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
    if(APPLE)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES omp)
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            include_directories("${LIBOMP_PREFIX}/include")
        endif()
    endif()
    find_package(OpenMP QUIET)
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
    # GTSAM Odometry Benchmark (builds common sources separately)
    add_executable(gtsam_odometry_benchmark
        odometry/src/gtsam_odometry.cpp
        common/src/read_g2o.cpp
        common/src/unified_cost.cpp
    )
    target_include_directories(gtsam_odometry_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/odometry/include
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
        apex_install_hint("brew install tbb" "libtbb-dev")
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
    target_compile_definitions(g2o_odometry_benchmark PRIVATE EIGEN_VERSION_STRING="${EIGEN_VERSION}")

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
    if(APPLE)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES omp)
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            include_directories("${LIBOMP_PREFIX}/include")
        endif()
    endif()
    find_package(OpenMP QUIET)
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
    if(APPLE)
        message(FATAL_ERROR "No C++ optimization libraries found!
        Please install at least one of: ceres-solver, gtsam, g2o")
    else()
        message(FATAL_ERROR "No C++ optimization libraries found!
        Please install at least one of: libceres-dev, libg2o-dev, or GTSAM >= 4.3")
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== APEX C++ Benchmark Suite Configuration ===")
if(BUILD_CERES_BENCHMARK)
    message(STATUS "  Ceres version: ${CERES_VERSION} (with Eigen ${EIGEN_VERSION})")
else()
    message(STATUS "  Ceres: NOT AVAILABLE")
endif()
if(BUILD_GTSAM_BENCHMARK)
    message(STATUS "  GTSAM version: ${GTSAM_VERSION} (with Eigen ${EIGEN_VERSION})")
else()
    message(STATUS "  GTSAM: NOT AVAILABLE")
endif()
if(BUILD_G2O_BENCHMARK)
    message(STATUS "  g2o: AVAILABLE (with Eigen ${EIGEN_VERSION})")
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
