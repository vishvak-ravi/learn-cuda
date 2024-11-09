#include <iostream>

int main(int argc, char** argv) {
    // sequential implementation

    int n = 4; // size of vector
    int* vec_A_h = new int[n];
    int* vec_B_h = new int[n];
    int* vec_C_h = new int[n];
    for (int i = 0; i < n; i++) {
        vec_C_h[i] = vec_A_h[i] + vec_B_h[i];
    }

    //print C
    std::cout << "C: ";
    for (int i = 0; i < n; i++) {
        std::cout << vec_C_h[i];
    }
    std::cout << std::endl;

    delete[] vec_A_h;
    delete[] vec_B_h;
    delete[] vec_C_h;

    return 1;
}