cmake_minimum_required(VERSION 3.26)
project(Visual-Odometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(modules)

add_executable(kitti_odometry examples/kitti_odometry.cpp)
target_link_libraries(kitti_odometry PRIVATE
    visual_odometry_module
)

add_executable(testing examples/testing.cpp)
target_link_libraries(testing PRIVATE
    visual_odometry_module
)