#pragma once
#include <vector>

double Sigmoid(double x);
double SigmoidDerivative(double y);
double Random();
double ReLU(double x);
double ReLUDerivative(double x);
std::vector <double> Softmax(const std::vector <double> &vec);