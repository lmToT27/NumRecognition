#include "Math.h"

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double SigmoidDerivative(double y) {
    return y * (1.0 - y);
}

double Random() {
    return static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
}

double ReLU(double x) {
    return x > 0.0 ? x : 0.0;
}

double ReLUDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

std::vector <double> Softmax(const std::vector <double> &u) {
    double mx = *std::max_element(u.begin(), u.end());
    std::vector <double> res;
    double sum = 0.0;
    for (const double &x : u) {
        double val = std::exp(x - mx);
        res.push_back(val);
        sum += val;        
    }
    for (double &val : res) {
        val /= sum;
    }
    return res;
}