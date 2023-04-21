cmake_minimum_required(VERSION 3.16)
project(Visual-Odometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VISUAL_ODOMETRY_SOURCE_FILES
    src/Feature.cpp
    src/MotionEstimation.cpp
    src/Triangulation.cpp
    src/LocalMap.cpp
)

add_library(visual_odometry_module
    ${VISUAL_ODOMETRY_SOURCE_FILES}
)

find_package(OpenCV REQUIRED HINTS
    #${CMAKE_SOURCE_DIR}/thirdparty/opencv/install/lib/cmake/opencv4
)
if(OpenCV_FOUND)
    message(STATUS "OpenCV Found! - ${OpenCV_DIR}")
endif()

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 Found! - ${Eigen3_DIR}")
    set(Eigen3_LIBS Eigen3::Eigen)
endif()

find_package(Ceres REQUIRED HINTS
    ${CMAKE_SOURCE_DIR}/thirdparty/ceres/install/lib/cmake/Ceres
)
if(Ceres_FOUND)
    message(STATUS "Ceres Found! - ${Ceres_DIR}")
    set(CERES_LIBS Ceres::ceres)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP Found!")
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
endif()

target_include_directories(visual_odometry_module PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(visual_odometry_module PUBLIC
    ${OpenCV_LIBS}
    ${Eigen3_LIBS}
    ${CERES_LIBS}
    ${OpenMP_LIBS}
)
