# Chapter 5: Memory architecture and data locality
## Basics
- arithmetic intsenity / computational intensity = number of FLOPs performed / byte accessed from global memory within some part of a program
### memory hierarchy
- local mem: per-thread R/W
    - placed in global with similar access but is not shared
    - not in registers... statically allocated arrays, spilled registers and other thread call stack items
- registers and shared mem are on-chip memories
    - very fast and parallelizable
- shared mem: per-block R/W
    - on chip and very fast for sharing input data and intermediate results
- global mem: per-grid R/W
    - off-chip and MUCH slower (two magnitudes slower than on-chip and an order magnitude more power utilization)
- registers > shared > global bandwidth
### tiled matrix multiplication
- enforce each block to compute a square tile of entries over phases
    - each phase denotes a tile of M, N being loaded into shared memory to compute a cumulative dot product
    - enforce bounds checking only on memory access instances—some threads supply data but do not store values
- TILE_WIDTH^2 * 2 * 4b / TILE_WIDTH^2 threads -> 8b/thread storage required
    - A100 SM has 168Kb of shared memory with 2048 max threads so we are not memory constrained
    - However, if a thread block uses too much memory, it will lead to low occupancy

## Selected exercises
- 1) No. Assuming each thread computes one output value (I'm not even sure this assumption is required), there is no redundancy in data reads.
- 3) If the first isn't placed, some threads will attempt multiplication on possibly stale values. If the second isn't placed, some threads will begin the next iteration, overwriting values that may need to be read by those still multiplying.
- 5) Before tiling, each thread required reading ```width * 2``` elements to compute a vector dot product. With tiling, ```width * 2 / TILE_WIDTH``` is read per thread since the rest is accumulated from others in the block. Thus, a reduction is by a factor of ```1/32```.
- 6) With 1000 blocks with 512 threads each, if a local var is declared, 1000 * 500 copies exist since it is private to each thread.
- 7) Given the above scenario, if a shared var is declared, 1000 copies are made to allow for sharing within a block
- 10) 
    - a) Definitely for size of 1...
    - b) A __threadSynchronize() must be placed between copying from the intermediate to the destination to prevent corrupting other thread reads or overwriting other thread writes