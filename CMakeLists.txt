cmake_minimum_required( VERSION 3.10 )

project( slam_solver )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# In order to test kalman filter, add imu model from 'Sensor_Model'.
set( SENSOR_MODEL_PATH ${PROJECT_SOURCE_DIR}/../Sensor_Model/src )
if ( NOT TARGET lib_imu_model )
	add_subdirectory( ${SENSOR_MODEL_PATH}/imu ${PROJECT_SOURCE_DIR}/build/lib_imu_model )
endif()

# Add general graph optimizor.
set( GENERAL_GRAPH_OPTIMIZOR_PATH ${PROJECT_SOURCE_DIR}/src/general_graph_optimizor )
if ( NOT TARGET lib_general_graph_optimizor )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH} ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor )
endif()

# Add kalman filter.
set( KALMAN_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/kalman_filter/kalman_filter )
if ( NOT TARGET lib_kalman_filter )
    add_subdirectory( ${KALMAN_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_kalman_filter )
endif()

# Add error kalman filter.
set( ERROR_KALMAN_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/kalman_filter/error_kalman_filter )
if ( NOT TARGET lib_error_kalman_filter )
    add_subdirectory( ${ERROR_KALMAN_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_error_kalman_filter )
endif()

# Add square root kalman filter.
set( SQUARE_ROOT_KALMAN_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/kalman_filter/square_root_kalman_filter )
if ( NOT TARGET lib_square_root_kalman_filter )
    add_subdirectory( ${SQUARE_ROOT_KALMAN_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_square_root_kalman_filter )
endif()

# Add test_general_graph_optimizor_polynomial_problem.
add_executable( test_general_graph_optimizor_polynomial_problem
    test/test_general_graph_optimizor_polynomial_problem.cpp
)
target_link_libraries( test_general_graph_optimizor_polynomial_problem
    lib_general_graph_optimizor
)

# Add test_general_graph_optimizor_bundle_adjustment.
add_executable( test_general_graph_optimizor_bundle_adjustment
    test/test_general_graph_optimizor_bundle_adjustment.cpp
)
target_link_libraries( test_general_graph_optimizor_bundle_adjustment
    lib_general_graph_optimizor
)

# Add test_general_graph_optimizor_marginalization.
add_executable( test_general_graph_optimizor_marginalization
    test/test_general_graph_optimizor_marginalization.cpp
)
target_link_libraries( test_general_graph_optimizor_marginalization
    lib_general_graph_optimizor
)

# Add test_kalman_filter_constant.
add_executable( test_kalman_filter_constant
    test/test_kalman_filter_constant.cpp
)
target_link_libraries( test_kalman_filter_constant
    lib_kalman_filter
    lib_error_kalman_filter
    lib_square_root_kalman_filter
)

# Add test_kalman_filter_imu_attitude.
add_executable( test_kalman_filter_imu_attitude
    test/test_kalman_filter_imu_attitude.cpp
)
target_link_libraries( test_kalman_filter_imu_attitude
    lib_kalman_filter
    lib_error_kalman_filter
    lib_square_root_kalman_filter

    lib_imu_model
)
