#include "NeuralNetwork.h"
#include "header/Math.h"

NeuralNetwork::NeuralNetwork(const std::vector <int> &layer_sizes) {
    this -> layer_sizes = layer_sizes;
    for (size_t i = 0; i < (int)layer_sizes.size() - 1; i++) {
        int from = layer_sizes[i];
        int to = layer_sizes[i + 1];
        Matrix w(from, to, true);
        Matrix b(1, to, true);
        weights.push_back(w);
        biases.push_back(b);
    }
}

Matrix NeuralNetwork::FeedForward(const Matrix &input) {
    Matrix res = input;
    layer_outputs.clear();
    layer_outputs.push_back(res);
    for (int i = 0; i < weights.size(); i++) {
        res = (res * weights[i]) + biases[i];
        if (i != (int)weights.size() - 1) res.ApplySigmoid();
        else res.ApplySoftmax();
        layer_outputs.push_back(res);   
    }
    return res;
}

Matrix NeuralNetwork::FeedForward(const std::vector <double> &input) {
    Matrix input_matrix(1, input.size(), input);
    return FeedForward(input_matrix);
}

void NeuralNetwork::BackPropagate(const Matrix &input, const Matrix &target, double learning_rate) {
    Matrix output = FeedForward(input);
    Matrix delta = output - target;

    for (int layer = (int)weights.size() - 1; layer >= 0; layer--) {
        Matrix prev_activation = layer_outputs[layer];
        
        Matrix weight_gradient = prev_activation.Transpose() * delta;
        Matrix bias_gradient = delta;
        
        weights[layer] = weights[layer] - weight_gradient.ScalarMul(learning_rate);
        biases[layer] = biases[layer] - bias_gradient.ScalarMul(learning_rate);

        if (layer > 0) {
            Matrix prev_derivative = prev_activation;
            prev_derivative.ApplySigmoidDerivative();
            delta = (delta * (weights[layer].Transpose())).HadamardMul(prev_derivative);
        }
    }
}