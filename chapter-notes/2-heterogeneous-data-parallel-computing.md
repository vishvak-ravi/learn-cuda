# Chapter 2: Heterogeneous Data Parallel Computing
## Basics
- Each kernel launches a grid
- Each grid contains up to 1024 thread blocks (keep it multiple of 3)
    - BlockDim.xyz
    - global thread index = blockIdx.x * blockDim.x * i + threadIdx.x
- Call kernel function require indicating number of blocks and thread/block
    - given n elements and t threads, launch ceil(n / t) blocks
    - requires handling of non-existent data in kernel code 
    - check unique thread identifier i as above and check (i < n)
### Functions
- cudaMalloc(float *devicePointer, int size)
- cudaMemcpy(float *destinationPointer, float *sourcePointer, int size, cudaDeviceToHost)
    - last arg determines directions
- cudaFree(float deviceArray)
- function keywords

| keyword | caller | executed on | executed by|
| -------- | -------- | -------- | -------- |
| \_\_host\_\_ | host | host | host thread |
| \_\_global\_\_ | host or device | device | threads from new grid|
| \_\_device\_\_ | device | device | caller device thread |

## Answers to selected exercises
1. C: it's in the notes...
2. C: each thread processes 2x as much data, so first element data index is twice what it would've been from 1
3. D: each block accesses twice as much data, but thread offset remains 1 since sections are computed one after another