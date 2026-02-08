#pragma once

#include <vector>

class Matrix {
private:
    std::vector <double> data;
    size_t rows; size_t cols;

public:
    Matrix(int rows, int cols, bool rand = false);
    double &operator()(int row, int col);
    const double &operator()(int row, int col) const;
    Matrix operator+(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator-() const;
    Matrix HadamardMul(const Matrix &other) const;
    Matrix ScalarMul(double scalar) const;
    Matrix Transpose() const;

    void Print();
};