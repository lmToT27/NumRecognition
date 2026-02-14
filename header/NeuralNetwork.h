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

    Matrix FeedForwardWithCache(const Matrix &input, std::vector <Matrix> &cache) const;
    void InitGradientBuffers(std::vector <Matrix> &weight_grads, std::vector <Matrix> &bias_grads) const;
    void ComputeGradients(const Matrix &input, const Matrix &target,
                          std::vector <Matrix> &weight_grads, std::vector <Matrix> &bias_grads) const;
public:
    NeuralNetwork(const std::vector <int> &layer_sizes);
    Matrix FeedForward(const Matrix &input);
    Matrix FeedForward(const std::vector <double> &input);
    void BackPropagate(const Matrix &input, const Matrix &target, double learning_rate);
    void BackPropagateBatch(const std::vector <Matrix> &inputs, const std::vector <Matrix> &targets, double learning_rate);
    void SaveModel(const std::string &filepath);
    void LoadModel(const std::string &filepath);
};