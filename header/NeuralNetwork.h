#pragma once

#include <vector>
#include "Matrix.h"

class NeuralNetwork {
private:
    std::vector <Matrix> weights;
    std::vector <Matrix> biases;
    std::vector <int> layer_sizes;
    std::vector <Matrix> layer_outputs;
public:
    NeuralNetwork(const std::vector <int> &layer_sizes);
    Matrix FeedForward(const Matrix &input);
    Matrix FeedForward(const std::vector <double> &input);
    // Backpropagation and training methods can be added here
    void BackPropagate(const Matrix &input, const Matrix &target, double learning_rate);
};