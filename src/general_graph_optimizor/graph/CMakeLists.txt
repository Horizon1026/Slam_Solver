aux_source_directory( . AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_GRAPH )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

# Add vertex and edge.
set( GENERAL_GRAPH_OPTIMIZOR_PATH ${PROJECT_SOURCE_DIR}/src/general_graph_optimizor )
if( NOT TARGET lib_general_graph_optimizor_vertex )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH}/vertex ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor_vertex )
endif()
if( NOT TARGET lib_general_graph_optimizor_edge )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH}/edge ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor_edge )
endif()

# Create library.
add_library( lib_general_graph_optimizor_graph ${AUX_SRC_GENERAL_GRAPH_OPTIMIZOR_GRAPH} )
target_include_directories( lib_general_graph_optimizor_graph PUBLIC . )
target_link_libraries( lib_general_graph_optimizor_graph
    lib_slam_utility_basic_type
    lib_slam_utility_log
    lib_slam_utility_operate

    lib_general_graph_optimizor_vertex
    lib_general_graph_optimizor_edge
)

# Try to find intel tbb.
find_package( TBB )
if( TBB_FOUND )
    message( ">> Find package tbb. Graph will use it." )
    target_link_libraries( lib_general_graph_optimizor_graph
        TBB::tbb
    )
    add_definitions( -DENABLE_TBB_PARALLEL )
endif()
