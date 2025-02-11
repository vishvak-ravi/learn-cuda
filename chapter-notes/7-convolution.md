# Convolution
## Basics
- you know what conv is
- ```__constant__``` keyword to declare constant mem (global across SMs)
    - ```cudaMemcpyToSymbol(dest, src, amt)``` to initialize without any pointers
- recall main hope is to maximize ops/glob-mem-access
## algos
- tiled
    - inputTile > outputTile size so tiling can launch some threads to load for a tile but not compute (block size == inputTile size) or launch some threads to doubly load and all compute (blockSize == outputTile size)
- insight: this method causes LOTS of overlap between input tiles of varying threads
    - these are likely to reside in L2 cache anyway so each thread may simply store in shared memory input tile corresponding to sie of output tile
    - this implementation requires careful handling by retrieving from N when not in bounds of N_s
- the upper bound ops/byte moved per block when output tile >> filter is (FILTER_RADIUS * 2 + 1)^2 * 2 / 4
## exercises
8. check conv.cu