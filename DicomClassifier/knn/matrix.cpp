#include "matrix.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <new> // For std::bad_alloc

// Constructor that initializes the matrix with the given number of rows and columns
matrix_base::matrix_base(size_t rows, size_t cols)
    : rows(rows), cols(cols), data(nullptr) {
    try {
        data = new double[rows * cols](); // Value-initialization sets all values to 0
        std::cout << "matrix: allocated double: " << rows << " * " << cols << " at " << static_cast<void*>(data) << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cerr << "matrix: failed to allocate memory: " << e.what() << std::endl;
        throw; // Rethrow exception after logging
    }
}

// Destructor that frees the allocated memory
matrix_base::~matrix_base() {
    std::cout << "matrix: deleting: " << rows << " x " << cols << " at " << static_cast<void*>(data) << std::endl;
    delete[] data;
}

// Clears the matrix by setting all values to 0
void matrix_base::clear() {
    if (data) {
        std::fill(data, data + (rows * cols), 0.0);
    }
}

// Output the matrix to std::cout
void outputMatrix(const matrix_base& matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            std::cout << const_cast<matrix_base&>(matrix).pos(i, j) << " "; // Use const_cast here
        }
        std::cout << std::endl;
    }
}

