#include <iostream>
#include <cuda_runtime.h>
#include "misc.cuh"

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = B[i] + A[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d; // declare pointers (no need for dynamic mem)
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int threadCount = 12;
    vecAddKernel<<<ceil(n * 1.0 / threadCount), threadCount>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char** argv) {
    int n = 4; // size of vector
    float* A_h = new float[n];
    float* B_h = new float[n];
    // Initialize A_h with the first n odd numbers
    for (int i = 0; i < n; i++) {
        A_h[i] = 2 * i + 1;
    }

    // Initialize B_h with the first n even numbers
    for (int i = 0; i < n; i++) {
        B_h[i] = 2 * i;
    }
    float* C_h = new float[n];

    // for (int i = 0; i < n; i++) {
    //     vec_C_h[i] = vec_A_h[i] + vec_B_h[i];
    // } sequential version

    vecAdd(A_h, B_h, C_h, n);

    //print C
    std::cout << "C: ";
    for (int i = 0; i < n; i++) {
        std::cout << C_h[i];
    }
    std::cout << std::endl;

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    return 1;
}