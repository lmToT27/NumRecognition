#pragma once

#include <vector>
#include "Matrix.h"
#include <string>
#include <fstream>
#include <cassert>

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
    void BackPropagate(const Matrix &input, const Matrix &target, double learning_rate);
    void SaveModel(const std::string &filepath);
    void LoadModel(const std::string &filepath);
};