#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            printf("%s in %s at line %d\n",                     \
                   cudaGetErrorString(err), __FILE__, __LINE__);\
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Dimensions of the matrices
#define WIDTH 1024

void matrix_multiply(const int* m, const int* n, int* result) {
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            result[i * WIDTH + j] = 0;
            for (int k = 0; k < WIDTH; k++) {
                result[i * WIDTH + j] += m[i * WIDTH + k] * n[k * WIDTH + j];
            }
        }
    }
}

void print_matrix(const char* name, int matrix[WIDTH][WIDTH]) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void validate_result(int *result, int *expected, int rows, int cols) {
    int match = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i * cols + j;
            if (result[index] != expected[index]) {
                printf("Mismatch at element [%d][%d]: result = %d, expected = %d\n", i, j, result[index], expected[index]);
                match = 0;
            }
        }
    }
    if (match) {
        printf("The result matches the expected output.\n");
    } else {
        printf("The result does not match the expected output.\n");
    }
}

typedef enum {
    ELEMENT,
    ROW,
    COL,
    TILE,
    AB_T,
} MultiplyType;

#define TILE_WIDTH 64 //asumes width is divisible by TILE_WIDTH
__global__ 
void tiledMatrixMatrixKernel(int* A, int* B, int* C, int width) {
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int P_val = 0;
    for (int ph = 0; ph < (width + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (row < width && ph * TILE_WIDTH + tx < width) {
            Mds[ty][tx] = A[row * width + ph * TILE_WIDTH + tx];
        }
        else {
            Mds[ty][tx] = 0;
        }
        if (col < width && ph * TILE_WIDTH + ty < width) {
            Nds[ty][tx] = B[col + (ph * TILE_WIDTH + ty) * width];
        }
        else {
            Nds[ty][tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            P_val += Mds[ty][i] * Nds[i][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width) {
        C[col + row * width] = P_val;
    }
}

__global__
void tiledMatrixMatrixTransposeKernel(int* A, int* B, int*C, int width) {
    // note B is assumed to stored in column major order (e.g. row-major B is same as B^T stored in column-major)
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int row = ty + TILE_WIDTH * by;
    int col = tx + TILE_WIDTH * bx;

    int P_val = 0;
    for (int ph = 0; ph < (width + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // load shared for this phase
        if (row < width && TILE_WIDTH * ph + tx < width) {
            Mds[ty][tx] = A[width * row  + (TILE_WIDTH * ph + tx)];
        }
        else {
            Mds[ty][tx] = 0;
        }
        // if (row < width && (ph * TILE_WIDTH + ty) < width) {   
        //     // adapting for column-major without coalescing
        //     Nds[ty][tx] = B[col * width + (TILE_WIDTH * ph + ty)];
        // }
        if (row < width && (ph * TILE_WIDTH + tx) < width) {   
            Nds[tx][ty] = B[row * width + (ph * TILE_WIDTH + tx)]; // this changed—that's all
        }
        else {
            Nds[ty][tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            P_val += Mds[ty][i] * Nds[i][tx];
        }
        __syncthreads();
        //accumulate into pval
    }
    C[row * width + col] = P_val;
}

__global__
void elementMatrixMatrixKernel(int* A, int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        int cum = 0;
        for (int i = 0; i < width; i++) {
            cum += A[width * row + i] * B[col + width * i];
        }
        C[row * width + col] = cum;
    }
}

__global__
void rowMatrixMatrixKernel(int* A, int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < width) {
        for (int col = 0; col < width; col++) {
            int sum = 0;
            for (int j = 0; j < width; j++) {
                sum += A[row * width + j] * B[col + width * j];
            }
            C[row * width + col] = sum;
        }
    }
}

__global__
void columnMatrixMatrixKernel(int* A, int*B, int* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        for (int row = 0; row < width; row++) {
            int sum = 0;
            for (int i = 0; i < width; i++) {
                sum += A[row * width + i] * B[col + width * i];
            }
            C[row * width + col] = sum;
        }
    }
}

void matrixMatrix(int* A_h, int* B_h, int* C_h, int width, MultiplyType mode) {
    int *A_d, *B_d, *C_d;
    int size = WIDTH * WIDTH * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **) &A_d, size));
    CUDA_CHECK(cudaMalloc((void **) &B_d, size));
    CUDA_CHECK(cudaMalloc((void **) &C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice));

    int threadCount = 16;
    cudaEvent_t start, stop;
    float elapsedTime;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    switch (mode) {
        case ELEMENT: {
            dim3 blockShape(threadCount, threadCount);
            dim3 gridShape(ceil(1.0 * width / threadCount), ceil(1.0 * width / threadCount));
            CUDA_CHECK(cudaEventRecord(start, 0));
            elementMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("ELEMENT kernel execution time: %f ms\n", elapsedTime);
            break;   
        }
        case ROW: {
            dim3 blockShape(1, threadCount, 1);
            dim3 gridShape(1, ceil(1.0 * width / threadCount), 1);
            CUDA_CHECK(cudaEventRecord(start, 0));
            rowMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("ROW kernel execution time: %f ms\n", elapsedTime);
            break;
        }
        case COL: {
            dim3 blockShape(threadCount);
            dim3 gridShape(ceil(1.0 * width / threadCount));
            CUDA_CHECK(cudaEventRecord(start, 0));
            columnMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("COL kernel execution time: %f ms\n", elapsedTime);
            break;
        }
        case TILE: {
            threadCount =  TILE_WIDTH;
            dim3 blockShape(threadCount, threadCount);
            dim3 gridShape(ceil(1.0 * width / threadCount), ceil(1.0 * width / threadCount));
            CUDA_CHECK(cudaEventRecord(start, 0));
            tiledMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("TILE kernel execution time: %f ms\n", elapsedTime);
            break;
        }
        case AB_T: {
            threadCount =  TILE_WIDTH;
            dim3 blockShape(threadCount, threadCount);
            dim3 gridShape(ceil(1.0 * width / threadCount), ceil(1.0 * width / threadCount));
            CUDA_CHECK(cudaEventRecord(start, 0));
            tiledMatrixMatrixTransposeKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("AB_T kernel execution time: %f ms\n", elapsedTime);
            break;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // Initialize two matrices
    srand((unsigned int)time(NULL));

    // Initialize two matrices with random values
    int *m = (int *)malloc(WIDTH * WIDTH * sizeof(int));
    int *n = (int *)malloc(WIDTH * WIDTH * sizeof(int));

    // Fill the matrices with random values
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            m[i * WIDTH + j] = rand() % 100; // Random values between 0 and 99
            n[i * WIDTH + j] = rand() % 100; // Random values between 0 and 99
        }
    }

    int *seq_result = (int *)malloc(WIDTH * WIDTH * sizeof(int));
    int *result = (int *)malloc(WIDTH * WIDTH * sizeof(int));

    // Perform matrix multiplication
    matrix_multiply(m, n, seq_result);

    // for transpose calculation
    int *seq_result_T = (int *)malloc(WIDTH * WIDTH * sizeof(int));
    int *n_T = (int *)malloc(WIDTH * WIDTH * sizeof(int));
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            n_T[i * WIDTH + j] = n[j * WIDTH + i];
        }
    }
    matrix_multiply(m, n_T, seq_result_T);

    // Print matrices
    //print_matrix("m", m);
    //print_matrix("n", n);
    //print_matrix("sequential result (m * n)", seq_result);
    //print_matrix("parallale result (m * n)", result);

    

    // Validate the result matrix against the expected matrix
    printf("Checking ELEMENT\n");
    matrixMatrix( m,  n,result, WIDTH, ELEMENT);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    printf("Checking ROW\n");
    matrixMatrix(m, n, result, WIDTH, ROW);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    printf("Checking COL\n");
    matrixMatrix(m, n, result, WIDTH, COL);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    printf("Checking TILE\n");
    matrixMatrix(m, n, result, WIDTH, TILE);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    // check A* B^T
    printf("Checking AB_T\n");
    matrixMatrix(m, n_T, result, WIDTH, AB_T); // force column-major ordering of n
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    return 0;
}