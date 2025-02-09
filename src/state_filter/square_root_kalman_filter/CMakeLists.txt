aux_source_directory( . AUX_SRC_SQUARE_ROOT_KALMAN_FILTER )

# Add all relative components of general graph optimizor.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()

# Create library.
add_library( lib_square_root_kalman_filter ${AUX_SRC_SQUARE_ROOT_KALMAN_FILTER} )
target_include_directories( lib_square_root_kalman_filter PUBLIC
    .
    ..
)
target_link_libraries( lib_square_root_kalman_filter
    lib_slam_utility_basic_type
)
