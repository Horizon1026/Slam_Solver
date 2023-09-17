aux_source_directory( . AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_OPTIMIZOR )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()

# Add graph.
set( GENERAL_GRAPH_OPTIMIZOR_PATH ${PROJECT_SOURCE_DIR}/src/general_graph_optimizor )
if ( NOT TARGET lib_general_graph_optimizor_graph )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH}/graph ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor_graph )
endif()

# Create library.
add_library( lib_general_graph_optimizor_optimizor ${AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_OPTIMIZOR} )
target_include_directories( lib_general_graph_optimizor_optimizor PUBLIC . )
target_link_libraries( lib_general_graph_optimizor_optimizor
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_log
    lib_slam_utility_operate
    lib_slam_utility_tick_tock

    lib_general_graph_optimizor_graph
)
