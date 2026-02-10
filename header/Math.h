#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

double Sigmoid(double x);
double SigmoidDerivative(double y);
double Random();
double ReLU(double x);
double ReLUDerivative(double x);
std::vector <double> Softmax(const std::vector <double> &vec);