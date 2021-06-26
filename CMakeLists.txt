cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_BUILD_TYPE Debug)
project(CoSM_ICP)

#find_package(Eigen3 REQUIRED)
find_package(PCL 1.9 REQUIRED PATHS external_libraries/pcl_1_9/install_dir/share/pcl-1.9 NO_DEFAULT_PATH )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (CoSM_ICP_demo_Viewer CoSM_ICP_demo_Viewer.cpp)
target_link_libraries (CoSM_ICP_demo_Viewer ${PCL_LIBRARIES})

add_executable (CoSM_Results_Collection CoSM_Results_Collection.cpp)
target_link_libraries (CoSM_Results_Collection ${PCL_LIBRARIES})

