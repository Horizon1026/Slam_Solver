cmake_minimum_required( VERSION 3.10 )

project( slam_solver )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 20 )

set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
# Add all relative components of Slam_Utility。
if( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
# Add all relative components of slam utility data type.
if( NOT TARGET lib_line_segment )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/line_segment ${PROJECT_SOURCE_DIR}/build/lib_line_segment )
endif()
if( NOT TARGET lib_plane )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/plane ${PROJECT_SOURCE_DIR}/build/lib_plane )
endif()
# Add all 3rd libraries.
if( NOT TARGET lib_stack_backward )
    add_subdirectory( ${SLAM_UTILITY_PATH}/3rd_libraries/backward ${PROJECT_SOURCE_DIR}/build/lib_stack_backward )
endif()
# In order to test kalman filter, add imu model from 'Sensor_Model'.
set( SENSOR_MODEL_PATH ${PROJECT_SOURCE_DIR}/../Sensor_Model )
if( NOT TARGET lib_imu_model )
	add_subdirectory( ${SENSOR_MODEL_PATH}/src/imu ${PROJECT_SOURCE_DIR}/build/lib_imu_model )
endif()
# Add image painter.
set( IMAGE_PAINTER_PATH ${PROJECT_SOURCE_DIR}/../Image_Painter )
if( NOT TARGET lib_image_painter )
    add_subdirectory( ${IMAGE_PAINTER_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_image_painter )
endif()
# Add visualizor.
set( VISUALIZOR_2D_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if( NOT TARGET lib_2d_visualizor )
    add_subdirectory( ${VISUALIZOR_2D_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_2d_visualizor )
endif()
set( VISUALIZOR_3D_PATH ${PROJECT_SOURCE_DIR}/../Visualizor3D )
if( NOT TARGET lib_3d_visualizor )
    add_subdirectory( ${VISUALIZOR_3D_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_3d_visualizor )
endif()

# Add general graph optimizor.
set( GENERAL_GRAPH_OPTIMIZOR_PATH ${PROJECT_SOURCE_DIR}/src/general_graph_optimizor )
if( NOT TARGET lib_general_graph_optimizor )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH} ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor )
endif()
# Add kalman filter.
set( KALMAN_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/state_filter/kalman_filter )
if( NOT TARGET lib_kalman_filter )
    add_subdirectory( ${KALMAN_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_kalman_filter )
endif()
# Add error kalman filter.
set( ERROR_KALMAN_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/state_filter/error_kalman_filter )
if( NOT TARGET lib_error_kalman_filter )
    add_subdirectory( ${ERROR_KALMAN_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_error_kalman_filter )
endif()
# Add square root kalman filter.
set( SQUARE_ROOT_KALMAN_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/state_filter/square_root_kalman_filter )
if( NOT TARGET lib_square_root_kalman_filter )
    add_subdirectory( ${SQUARE_ROOT_KALMAN_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_square_root_kalman_filter )
endif()
# Add information filter.
set( INFORMAION_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/state_filter/information_filter )
if( NOT TARGET lib_information_filter )
    add_subdirectory( ${INFORMAION_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_information_filter )
endif()
# Add error information filter.
set( ERROR_INFORMAION_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/state_filter/error_information_filter )
if( NOT TARGET lib_error_information_filter )
    add_subdirectory( ${ERROR_INFORMAION_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_error_information_filter )
endif()
# Add square root information filter.
set( SQUARE_ROOT_INFORMAION_FILTER_PATH ${PROJECT_SOURCE_DIR}/src/state_filter/square_root_information_filter )
if( NOT TARGET lib_square_root_information_filter )
    add_subdirectory( ${SQUARE_ROOT_INFORMAION_FILTER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_square_root_information_filter )
endif()
# Add polynomial solver.
set( POLYNOMIAL_SOLVER_PATH ${PROJECT_SOURCE_DIR}/src/polynomial_solver )
if( NOT TARGET lib_polynomial_solver )
    add_subdirectory( ${POLYNOMIAL_SOLVER_PATH} ${PROJECT_SOURCE_DIR}/build/lib_polynomial_solver )
endif()
# Add pose graph optimizor.
set( POSE_GRAPH_OPTIMIZOR ${PROJECT_SOURCE_DIR}/src/pose_graph_optimizor )
if( NOT TARGET lib_pose_graph_optimizor )
    add_subdirectory( ${POSE_GRAPH_OPTIMIZOR} ${PROJECT_SOURCE_DIR}/build/lib_pose_graph_optimizor )
endif()

# Compile targets based on general graph optimizor.
set( UNIT_TEST_OF_GENERAL_GRAPH_OPTIMIZOR_LIST
    test_general_graph_optimizor_polynomial_problem
    test_general_graph_optimizor_pos_norm_plane_BA
    test_general_graph_optimizor_invdep_norm_plane_BA
    test_general_graph_optimizor_pos_unit_sphere_BA
    test_general_graph_optimizor_invdep_unit_sphere_BA
    test_general_graph_optimizor_invdep_one_cam_two_frame
    test_general_graph_optimizor_marginalization
    test_prior_operation
    test_general_graph_optimizor_line_BA
)
foreach( unit_test_src ${UNIT_TEST_OF_GENERAL_GRAPH_OPTIMIZOR_LIST} )
    add_executable( ${unit_test_src}
        test/${unit_test_src}.cpp
    )
    target_link_libraries( ${unit_test_src}
        lib_general_graph_optimizor
        lib_stack_backward
        lib_line_segment
        lib_plane
        lib_image_painter
        lib_2d_visualizor
        lib_3d_visualizor
    )
endforeach()

# Compile targets based on state filter.
set( UNIT_TEST_OF_STATE_FILTER_LIST
    test_state_filter_constant
    test_state_filter_imu_attitude
)
foreach( unit_test_src ${UNIT_TEST_OF_STATE_FILTER_LIST} )
    add_executable( ${unit_test_src}
        test/${unit_test_src}.cpp
    )
    target_link_libraries( ${unit_test_src}
        lib_kalman_filter
        lib_error_kalman_filter
        lib_square_root_kalman_filter
        lib_information_filter
        lib_error_information_filter
        lib_square_root_information_filter

        lib_imu_model
        lib_stack_backward

        lib_slam_utility_basic_type
        lib_slam_utility_math
        lib_slam_utility_operate
        lib_slam_utility_log
    )
endforeach()

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
