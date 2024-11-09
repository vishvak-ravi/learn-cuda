#include <stdio.h>
#include <stdlib.h>

// Dimensions of the matrices
#define ROWS 3
#define COLS 3

void matrix_multiply(int m[ROWS][COLS], int n[ROWS][COLS], int result[ROWS][COLS]) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            result[i][j] = 0;
            for (int k = 0; k < COLS; k++) {
                result[i][j] += m[i][k] * n[k][j];
            }
        }
    }
}

void print_matrix(const char* name, int matrix[ROWS][COLS]) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
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

int main() {
    // Initialize two matrices
    int m[ROWS][COLS] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int n[ROWS][COLS] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };
    int result[ROWS][COLS] = {0};
    int expected[ROWS][COLS] = {
        {30, 24, 18},
        {84, 69, 54},
        {138, 114, 90}
    }; // Example expected result for validation

    // Perform matrix multiplication
    matrix_multiply(m, n, result);

    // Print matrices
    print_matrix("m", m);
    print_matrix("n", n);
    print_matrix("result (m * n)", result);

    // Validate the result matrix against the expected matrix
    validate_result((int*)result, (int*)expected, ROWS, COLS);

    return 0;
}