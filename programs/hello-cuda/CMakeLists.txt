cmake_minimum_required(VERSION 3.18)
project(hello_cuda LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

# Add the executable
add_executable(hello_cuda vec-add.cu)

# Set the target properties
set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)