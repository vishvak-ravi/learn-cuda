# Chapter 4: Compute Architecture and Scheduling
## Basics
- Warps are smallest unit of SM handling: 32 contiguous threads decomposed from a block in row-major order 
- Control divergence occurs when individual threads in a block execute different instructions (e.g. data-bounds checking)
    - CD requires a "pass" per path, making some threads in the warp inactivate for those not taking the path
- Zero-overhead scheduling allows for a warp to remain in register file while waiting for a long-latency instruction to finish executing while another warp takes over
    - Different from context-switching in a CPU since state to save (e.g. PC does not need to be written to memory)
- Dynamic block partitioning allows for an SM to hold varying block-sizes. Each SM limits the number of blocks and threads—this implies a good ratio is required to enusre the occupancy (threads assigned / max threads) is high...
    - e.g. given 2048 threads and 32 blocks on A100 SM (max), block and thread size of 32 each hits block limit, but yields 50% SM occupancy
- Also if the block size is not divisible by the max block size... only a fraction of a block size is available on an SM and goes unused
- Maximum registers 65,636 on A100 SM (65,536 / 2048) ~ 32 registers/thread MAX
    - if goes over, occupancy decreases or register spilling is used increasing runtime

## Selected Exercises
1. Consider a CUDA kernel
    - a) Number of warps per block?
        - 128/32 = 4
    - b) Number of warps in grid?
        - int((1024 + 128 -1) / 128) blocks * 4 warps/block = 32 warps
    - c) For line 04
        - Active warps in grid: 3 warps/block * 8 warps/block = 24 warps
        - Divergent warps in grid: 2 warps/block * 8 blocks = 16 warps
        - SIMD efficiency of warp 0, block 0: 100% since all take branch
        - SIMD efficient of warp 1 in block 0: 50% since two passes are required
        - SIMD efficiency of warp 2 in block 0: 100% since all take branch
    - d) For line 07
        - Activer warps in grid: ... I got tired of arithmetic...

2. 2000 required threads/ 512 threads/block-> 4 blocks * 512 threads/block =2048 threads in grid
3. 1 has divergence since only one could possibly saddle the bound\]
4. More arithmetic...
5. Though it is true the block will be assigned a single warp, if a barrier synchronization is required, inter-thread data dependency must be maintined. If control flow exists, the ordering of the "passes" is not guaranteed and it is possible some threads perform operations in different order from the programmed one due to control flow.