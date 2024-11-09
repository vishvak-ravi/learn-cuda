# Chapter 3: Multidimensional Grids and Data
## Basics
- dim3 datatype for grid and block sizes vecAddKernel<<<grid, thread>>> and set other components to 1 if unused
- also use integers for vecAddKernel<<<i, j>>> for simple syntax (uses C++ default params)
- max gridDim.x is 2^32 - 1 othera re 2^16 - 1
- max blockIdx is denoted by blockDims
- block (1,0) designates block.y = 1, block.x = 0 (e.g. higher dim comes first)
- CUDA uses row-major layout (e.g. rows are contiguous units placed one-after another)
- Mat[j][i] denotes the jth row ith col
