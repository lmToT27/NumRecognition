#include "Matrix.h"
#include "header/Math.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

Matrix::Matrix(int rows, int cols, bool rand) {
    this -> rows = rows;
    this -> cols = cols;
    data.resize(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (rand) (*this)(i, j) = static_cast <double> (std::rand()) / RAND_MAX;
            else (*this)(i, j) = 0.0;
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

Matrix Matrix::operator+(const Matrix &other) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &other) const {
    Matrix res(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            for (int k = 0; k < cols; k++) {
                res(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &other) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return res;
}

Matrix Matrix::HadamardMul(const Matrix &other) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    return res;
}

Matrix Matrix::ScalarMul(double scalar) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = (*this)(i, j) * scalar;
        }
    }
    return res;
}

Matrix Matrix::Transpose() const {
    Matrix res(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(j, i) = (*this)(i, j);
        }
    }
    return res;
}

void Matrix::ApplySigmoid() {
    for (double &val : data) {
        val = Sigmoid(val);
    }
}

void Matrix::ApplySigmoidDerivative() {
    for (double &val : data) {
        val = SigmoidDerivative(val);
    }
}

void Matrix::ApplyReLU() {
    for (double &val : data) {
        val = ReLU(val);
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