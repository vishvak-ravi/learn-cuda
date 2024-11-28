# Memory Coalescing
## Basics
- DRAM bursts (e.g. for global memory) allows for consecutive address accesses to be much faster than truly random accesses
- Ensure burst accesses are contained within a warp that operates instructions for all threads within it
- e.g. MM with thread-to-element 1-1 mapping has consecutive threads with right matrix accessing consecutive elements within a row PER iteration ASSUMING ROW-MAJOR ORDERING IN MEMORY
- Transpose of a matrix requires access of a row-major matrix in memory VIA colmn-major indexing in C-style array storage
- Memory coalescing ensures adjacent threads access adjacent memory addresses to leverage DRAM bursts
- Memory latency hiding is more natural (e.g. not influenced by programmer) due to the data distribution (e.g. allowing a contiguous array to be physically spread out across multiple DRAM banks to ensure a channel has minimal bank conflicts)
    - Tiled matrix multiplication allows threads within a block/tile to utilize DRAM bursts within a single phase
- Thread coarsening allows for a thread to handle an amount of processing greater than the smallest unit of processing
    - Used when the overhead of parallelism is too high (e.g. blocks become serialized due to hardware constraints)
    - Specifically, smallest granularity of tiled matrix multiplication allows a single thread to process a single element. Thus, threads outputting values in the same row in different thread blocks must both store in memory the row of A in A*B. If these blocks run in parallel, this load redundancy is worth—however, if they schedule sequentially, you may as well have had one thread do the work of both and shave off a load from the row of A
### DRAM
- Each processor is equipped with a few channels—each channel has several banks to access
- Per access...
    - Issuing
    - ...
    - Decoder cells
    - Sense amplifier
    - Burst data through bus
- The DRAM latency from issuing to reception allows time for channel to initiate multiple transactions to multiple DRAM banks -> hiding latency
- DDR - double data rate implies data transaction on both rising and falling edge of clock

# Selected Exercises
- 1. See corner tuned implementation of A * B^T as tiledRowMajorColumnMajor() in matrix-matrix-multiply.cu
- 2. In tiled MM, BLOCK_SIZES that are multiples of the DRAM burst size (number of banks/channel) and the warp size (32) are minimally required
- 4. 
    - a) (N + N)/ 2N : N multiplies and N adds for 2N reads
    - b) (N + N) / (2N/K) : N multiplies and N adds for reads split amongst tile
    - c) (4(N + N) / (1 + 4N/K)) : 4 times as much compute, but only three more column reads of B instead of 7