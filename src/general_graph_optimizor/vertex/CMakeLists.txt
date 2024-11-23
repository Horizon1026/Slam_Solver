aux_source_directory( . AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_VERTEX )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()

# Create library.
add_library( lib_general_graph_optimizor_vertex ${AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_VERTEX} )
target_include_directories( lib_general_graph_optimizor_vertex PUBLIC . )
target_link_libraries( lib_general_graph_optimizor_vertex
    lib_slam_utility_basic_type
)
