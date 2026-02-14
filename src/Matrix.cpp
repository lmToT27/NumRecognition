#include "Matrix.h"
#include <cassert>
#include <omp.h>

Matrix::Matrix(int rows, int cols, bool rand) {
    this -> rows = rows;
    this -> cols = cols;
    data.resize(rows * cols);
    if (rand) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                (*this)(i, j) = -0.1 + static_cast <double> (std::rand()) / RAND_MAX * 0.2;
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                (*this)(i, j) = 0.0;
            }
        }
    }
}

Matrix::Matrix(int rows, int cols, const std::vector <double> &values) {
    this -> rows = rows;
    this -> cols = cols;
    data = values;
}

double &Matrix::operator()(int row, int col) {
    return this -> data[row * cols + col];
}

const double &Matrix::operator()(int row, int col) const {
    return this -> data[row * cols + col];
}

Matrix Matrix::operator=(const Matrix &other) {
    this -> rows = other.rows;
    this -> cols = other.cols;
    this -> data = other.data;
    return *this;
}

Matrix Matrix::Flatten(int axis) const {
    if (axis == 0) {
        Matrix res = *this;
        res.rows = 1;
        res.cols = rows * cols;
        return res;
    } else {
        Matrix res = *this;
        res.rows = rows * cols;
        res.cols = 1;
        return res;
    }
}

Matrix Matrix::operator+(const Matrix &other) const {
    assert(rows == other.rows && cols == other.cols && "Matrix dimensions must match for addition.");
    Matrix res(rows, cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &other) const {
    assert(cols == other.rows && "Matrix dimensions are not compatible for multiplication.");
    Matrix res(rows, other.cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0.0;
            #pragma omp simd
            for (int k = 0; k < cols; k++) {
                sum += (*this)(i, k) * other(k, j);
            }
            res(i, j) = sum;
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &other) const {
    assert(rows == other.rows && cols == other.cols && "Matrix dimensions must match for subtraction.");
    Matrix res(rows, cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return res;
}

void Matrix::AddInPlace(const Matrix &other) {
    assert(rows == other.rows && cols == other.cols && "Matrix dimensions must match for addition.");
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*this)(i, j) += other(i, j);
        }
    }
}

void Matrix::Fill(double value) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        data[i] = value;
    }
}

Matrix Matrix::HadamardMul(const Matrix &other) const {
    assert(rows == other.rows && cols == other.cols && "Matrix dimensions must match for Hadamard multiplication.");
    Matrix res(rows, cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    return res;
}

Matrix Matrix::ScalarMul(double scalar) const {
    Matrix res(rows, cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) * scalar;
        }
    }
    return res;
}

Matrix Matrix::Transpose() const {
    Matrix res(cols, rows);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(j, i) = (*this)(i, j);
        }
    }
    return res;
}

void Matrix::ApplySigmoid() {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        data[i] = Sigmoid(data[i]);
    }
}

void Matrix::ApplySigmoidDerivative() {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        data[i] = SigmoidDerivative(data[i]);
    }
}

void Matrix::ApplyReLU() {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        data[i] = ReLU(data[i]);
    }
}

void Matrix::ApplyReLUDerivative() {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        data[i] = ReLUDerivative(data[i]);
    }
}

void Matrix::ApplySoftmax() {
    std::vector <double> vec = Softmax(data);
    data = vec;
}

void Matrix::Print() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

size_t Matrix::GetRows() const {
    return rows;
}

size_t Matrix::GetCols() const {
    return cols;
}