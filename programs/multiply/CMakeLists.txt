cmake_minimum_required(VERSION 3.18)
project(hello_cuda LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

# Add the executable
add_executable(matrix-matrix matrix-matrix-multiply.cu)

# Set the target properties
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set_target_properties(matrix-matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(matrix-matrix ${OpenCV_LIBS})