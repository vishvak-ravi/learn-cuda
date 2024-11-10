#include <stdio.h>
#include <stdlib.h>
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
#define WIDTH 32

void matrix_multiply(int m[WIDTH][WIDTH], int n[WIDTH][WIDTH], int result[WIDTH][WIDTH]) {
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            result[i][j] = 0;
            for (int k = 0; k < WIDTH; k++) {
                result[i][j] += m[i][k] * n[k][j];
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
    COL
} MultiplyType;

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
    switch (mode) {
        case ELEMENT: {
            dim3 blockShape(threadCount, threadCount);
            dim3 gridShape(ceil(1.0 * width / threadCount), ceil(1.0 * width / threadCount));
            elementMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            break;   
        }
        case ROW: {
            dim3 blockShape(1, threadCount, 1);
            dim3 gridShape(1, ceil(1.0 * width / threadCount), 1);
            rowMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
            break;
        }
        case COL: {
            dim3 blockShape(threadCount);
            dim3 gridShape(ceil(1.0 * width / threadCount));
            columnMatrixMatrixKernel<<<gridShape, blockShape>>>(A_d, B_d, C_d, width);
        }
    }

    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // Initialize two matrices
    srand((unsigned int)time(NULL));

    // Initialize two matrices with random values
    int m[WIDTH][WIDTH];
    int n[WIDTH][WIDTH];

    // Fill the matrices with random values
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            m[i][j] = rand() % 100; // Random values between 0 and 99
            n[i][j] = rand() % 100; // Random values between 0 and 99
        }
    }
    int seq_result[WIDTH][WIDTH] = {0};
    int result[WIDTH][WIDTH] = {0};

    // Perform matrix multiplication
    matrix_multiply(m, n, seq_result);

    // Print matrices
    //print_matrix("m", m);
    //print_matrix("n", n);
    //print_matrix("sequential result (m * n)", seq_result);
    //print_matrix("parallale result (m * n)", result);

    

    // Validate the result matrix against the expected matrix
    printf("Checking ELEMENT\n");
    matrixMatrix((int *) &m, (int *) &n, (int *) &result, WIDTH, ELEMENT);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    printf("Checking ROW\n");
    matrixMatrix((int *) &m, (int *) &n, (int *) &result, WIDTH, ROW);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);
    printf("Checking COL\n");
    matrixMatrix((int *) &m, (int *) &n, (int *) &result, WIDTH, COL);
    validate_result((int*)result, (int*)seq_result, WIDTH, WIDTH);

    return 0;
}