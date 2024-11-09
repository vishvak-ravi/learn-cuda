#ifdef MISC
#define MISC

#define CUDA_CHECK(call)
do {
    cudaError_t err = call;
    if (cudaError_t != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

#endif