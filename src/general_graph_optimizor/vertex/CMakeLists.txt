aux_source_directory( . AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_VERTEX )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()

# Create library.
add_library( lib_general_graph_optimizor_vertex ${AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_VERTEX} )
target_include_directories( lib_general_graph_optimizor_vertex PUBLIC . )
target_link_libraries( lib_general_graph_optimizor_vertex
    lib_slam_utility_datatype
    lib_slam_utility_math
)
