cmake_minimum_required( VERSION 3.10 )

project( slam_solver )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 20 )

set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
# Add all relative components of Slam_Utility。
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
# Add all relative components of slam utility data type.
if ( NOT TARGET lib_line_segment )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/line_segment ${PROJECT_SOURCE_DIR}/build/lib_line_segment )
endif()
if ( NOT TARGET lib_plane )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/plane ${PROJECT_SOURCE_DIR}/build/lib_plane )
endif()
# Add all 3rd libraries.
if ( NOT TARGET lib_stack_backward )
    add_subdirectory( ${SLAM_UTILITY_PATH}/3rd_libraries/backward ${PROJECT_SOURCE_DIR}/build/lib_stack_backward )
endif()
# In order to test kalman filter, add imu model from 'Sensor_Model'.
set( SENSOR_MODEL_PATH ${PROJECT_SOURCE_DIR}/../Sensor_Model )
if ( NOT TARGET lib_imu_model )
	add_subdirectory( ${SENSOR_MODEL_PATH}/src/imu ${PROJECT_SOURCE_DIR}/build/lib_imu_model )
endif()
# Add image painter.
set( IMAGE_PAINTER_PATH ${PROJECT_SOURCE_DIR}/../Image_Painter )
if ( NOT TARGET lib_image_painter )
    add_subdirectory( ${IMAGE_PAINTER_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_image_painter )
endif()
# Add visualizor.
set( VISUALIZOR_2D_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if ( NOT TARGET lib_2d_visualizor )
    add_subdirectory( ${VISUALIZOR_2D_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_2d_visualizor )
endif()
set( VISUALIZOR_3D_PATH ${PROJECT_SOURCE_DIR}/../Visualizor3D )
if ( NOT TARGET lib_3d_visualizor )
    add_subdirectory( ${VISUALIZOR_3D_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_3d_visualizor )
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
# Add polynomial solver.
set( POLYNOMIAL_SOLVER_PATH ${PROJECT_SOURCE_DIR}/src/polynomial_solver )
if ( NOT TARGET lib_polynomial_solver )
    add_subdirectory( ${POLYNOMIAL_SOLVER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_polynomial_solver )
endif()
# Add pose graph optimizor.
set( POSE_GRAPH_OPTIMIZOR ${PROJECT_SOURCE_DIR}/src/pose_graph_optimizor )
if ( NOT TARGET lib_pose_graph_optimizor )
    add_subdirectory( ${POSE_GRAPH_OPTIMIZOR} ${PROJECT_SOURCE_DIR}/build/lib_pose_graph_optimizor )
endif()

# Add test_general_graph_optimizor_polynomial_problem.
add_executable( test_general_graph_optimizor_polynomial_problem
    test/test_general_graph_optimizor_polynomial_problem.cpp
)
target_link_libraries( test_general_graph_optimizor_polynomial_problem
    lib_general_graph_optimizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_pos_norm_plane_BA.
add_executable( test_general_graph_optimizor_pos_norm_plane_BA
    test/test_general_graph_optimizor_pos_norm_plane_BA.cpp
)
target_link_libraries( test_general_graph_optimizor_pos_norm_plane_BA
    lib_general_graph_optimizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_invdep_norm_plane_BA.
add_executable( test_general_graph_optimizor_invdep_norm_plane_BA
    test/test_general_graph_optimizor_invdep_norm_plane_BA.cpp
)
target_link_libraries( test_general_graph_optimizor_invdep_norm_plane_BA
    lib_general_graph_optimizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_pos_unit_sphere_BA.
add_executable( test_general_graph_optimizor_pos_unit_sphere_BA
    test/test_general_graph_optimizor_pos_unit_sphere_BA.cpp
)
target_link_libraries( test_general_graph_optimizor_pos_unit_sphere_BA
    lib_general_graph_optimizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_invdep_unit_sphere_BA.
add_executable( test_general_graph_optimizor_invdep_unit_sphere_BA
    test/test_general_graph_optimizor_invdep_unit_sphere_BA.cpp
)
target_link_libraries( test_general_graph_optimizor_invdep_unit_sphere_BA
    lib_general_graph_optimizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_invdep_one_cam_two_frame.
add_executable( test_general_graph_optimizor_invdep_one_cam_two_frame
    test/test_general_graph_optimizor_invdep_one_cam_two_frame.cpp
)
target_link_libraries( test_general_graph_optimizor_invdep_one_cam_two_frame
    lib_general_graph_optimizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_marginalization.
add_executable( test_general_graph_optimizor_marginalization
    test/test_general_graph_optimizor_marginalization.cpp
)
target_link_libraries( test_general_graph_optimizor_marginalization
    lib_general_graph_optimizor
    lib_stack_backward
    lib_2d_visualizor
    lib_image_painter
)

# Add test_prior_operation.
add_executable( test_prior_operation
    test/test_prior_operation.cpp
)
target_link_libraries( test_prior_operation
    lib_general_graph_optimizor
    lib_stack_backward
    lib_2d_visualizor
    lib_image_painter
)

# Add test_kalman_filter_constant.
add_executable( test_kalman_filter_constant
    test/test_kalman_filter_constant.cpp
)
target_link_libraries( test_kalman_filter_constant
    lib_kalman_filter
    lib_error_kalman_filter
    lib_square_root_kalman_filter
    lib_stack_backward
    lib_slam_utility_log
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
    lib_stack_backward
)

# Add test_pose_graph_optimizor.
add_executable( test_pose_graph_optimizor
    test/test_pose_graph_optimizor.cpp
)
target_link_libraries( test_pose_graph_optimizor
    lib_general_graph_optimizor
    lib_pose_graph_optimizor
    lib_3d_visualizor
    lib_stack_backward
)

# Add test_general_graph_optimizor_line_BA.
add_executable( test_general_graph_optimizor_line_BA
    test/test_general_graph_optimizor_line_BA.cpp
)
target_link_libraries( test_general_graph_optimizor_line_BA
    lib_general_graph_optimizor
    lib_stack_backward
    lib_line_segment
    lib_plane
    lib_2d_visualizor
    lib_3d_visualizor
)
