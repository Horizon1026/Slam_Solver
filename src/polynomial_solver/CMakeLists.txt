aux_source_directory( . AUX_SRC_GENERAL_POLYNOMIAL_SOLVER )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

# Create library.
add_library( lib_polynomial_solver ${AUX_SRC_GENERAL_POLYNOMIAL_SOLVER} )
target_include_directories( lib_polynomial_solver PUBLIC . )
target_link_libraries( lib_polynomial_solver
    lib_slam_utility_datatype
    lib_slam_utility_math
)
