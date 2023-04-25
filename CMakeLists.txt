cmake_minimum_required( VERSION 3.10 )

project( slam_solver )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# Add general graph optimizor.
set( GENERAL_GRAPH_OPTIMIZOR_PATH ${PROJECT_SOURCE_DIR}/src/general_graph_optimizor )
if ( NOT TARGET lib_general_graph_optimizor )
    add_subdirectory( ${GENERAL_GRAPH_OPTIMIZOR_PATH} ${PROJECT_SOURCE_DIR}/build/lib_general_graph_optimizor )
endif()

# Add test_general_graph_optimizor.
add_executable( test_general_graph_optimizor
    test/test_general_graph_optimizor.cpp
)
target_link_libraries( test_general_graph_optimizor
    lib_general_graph_optimizor
)
