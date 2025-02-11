cmake_minimum_required(VERSION 3.16)
project(conv LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

# Add the executable
add_executable(conv conv.cu)

# Set the target properties
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set_target_properties(conv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conv ${OpenCV_LIBS})