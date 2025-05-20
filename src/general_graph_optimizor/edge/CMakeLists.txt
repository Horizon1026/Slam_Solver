aux_source_directory( . AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_EDGE )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add vertex and kernel.
set( GENERAL_GRAPH_OPTIMIZOR_PATH ${PROJECT_SOURCE_DIR}/src/general_graph_optimizor )
if( NOT TARGET lib_general_graph_optimizor_vertex )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH}/vertex ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor_vertex )
endif()
if( NOT TARGET lib_general_graph_optimizor_kernel )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH}/kernel ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor_kernel )
endif()

# Create library.
add_library( lib_general_graph_optimizor_edge ${AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_EDGE} )
target_include_directories( lib_general_graph_optimizor_edge PUBLIC . )
target_link_libraries( lib_general_graph_optimizor_edge
    lib_slam_utility_basic_type
    lib_slam_utility_log

    lib_general_graph_optimizor_vertex
    lib_general_graph_optimizor_kernel
)
