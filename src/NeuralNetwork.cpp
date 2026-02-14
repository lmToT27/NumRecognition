#include "NeuralNetwork.h"

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
        if (i != (int)weights.size() - 1) res.ApplyReLU();
        else res.ApplySoftmax();
        layer_outputs.push_back(res);   
    }
    return res;
}

Matrix NeuralNetwork::FeedForwardWithCache(const Matrix &input, std::vector <Matrix> &cache) const {
    Matrix res = input;
    cache.clear();
    cache.push_back(res);
    for (int i = 0; i < weights.size(); i++) {
        res = (res * weights[i]) + biases[i];
        if (i != (int)weights.size() - 1) res.ApplyReLU();
        else res.ApplySoftmax();
        cache.push_back(res);
    }
    return res;
}

Matrix NeuralNetwork::FeedForward(const std::vector <double> &input) {
    Matrix input_matrix(1, input.size(), input);
    return FeedForward(input_matrix);
}

void NeuralNetwork::BackPropagate(const Matrix &input, const Matrix &target, double learning_rate) {
    std::vector <Matrix> weight_grads;
    std::vector <Matrix> bias_grads;
    ComputeGradients(input, target, weight_grads, bias_grads);
    for (int layer = (int)weights.size() - 1; layer >= 0; layer--) {
        weights[layer] = weights[layer] - weight_grads[layer].ScalarMul(learning_rate);
        biases[layer] = biases[layer] - bias_grads[layer].ScalarMul(learning_rate);
    }
}

void NeuralNetwork::InitGradientBuffers(std::vector <Matrix> &weight_grads, std::vector <Matrix> &bias_grads) const {
    weight_grads.clear();
    bias_grads.clear();
    weight_grads.reserve(weights.size());
    bias_grads.reserve(biases.size());
    for (size_t i = 0; i < weights.size(); i++) {
        weight_grads.emplace_back((int)weights[i].GetRows(), (int)weights[i].GetCols());
        bias_grads.emplace_back((int)biases[i].GetRows(), (int)biases[i].GetCols());
    }
}

void NeuralNetwork::ComputeGradients(const Matrix &input, const Matrix &target,
                                     std::vector <Matrix> &weight_grads, std::vector <Matrix> &bias_grads) const {
    if (weight_grads.empty() || bias_grads.empty()) {
        InitGradientBuffers(weight_grads, bias_grads);
    }

    std::vector <Matrix> cache;
    Matrix output = FeedForwardWithCache(input, cache);
    Matrix delta = output - target;

    for (int layer = (int)weights.size() - 1; layer >= 0; layer--) {
        Matrix prev_activation = cache[layer];

        weight_grads[layer] = prev_activation.Transpose() * delta;
        bias_grads[layer] = delta;

        if (layer > 0) {
            Matrix prev_derivative = prev_activation;
            prev_derivative.ApplyReLUDerivative();
            delta = (delta * (weights[layer].Transpose())).HadamardMul(prev_derivative);
        }
    }
}

void NeuralNetwork::BackPropagateBatch(const std::vector <Matrix> &inputs, const std::vector <Matrix> &targets, double learning_rate) {
    if (inputs.empty()) return;
    assert(inputs.size() == targets.size() && "Inputs and targets must be the same size.");

    std::vector <Matrix> weight_grads_sum;
    std::vector <Matrix> bias_grads_sum;
    InitGradientBuffers(weight_grads_sum, bias_grads_sum);

    #pragma omp parallel
    {
        std::vector <Matrix> local_w;
        std::vector <Matrix> local_b;
        std::vector <Matrix> sample_w;
        std::vector <Matrix> sample_b;
        InitGradientBuffers(local_w, local_b);
        InitGradientBuffers(sample_w, sample_b);

        #pragma omp for schedule(static)
        for (int i = 0; i < (int)inputs.size(); i++) {
            ComputeGradients(inputs[i], targets[i], sample_w, sample_b);
            for (size_t layer = 0; layer < local_w.size(); layer++) {
                local_w[layer].AddInPlace(sample_w[layer]);
                local_b[layer].AddInPlace(sample_b[layer]);
            }
        }

        #pragma omp critical
        {
            for (size_t layer = 0; layer < weight_grads_sum.size(); layer++) {
                weight_grads_sum[layer].AddInPlace(local_w[layer]);
                bias_grads_sum[layer].AddInPlace(local_b[layer]);
            }
        }
    }

    double scale = learning_rate / static_cast<double>(inputs.size());
    for (int layer = (int)weights.size() - 1; layer >= 0; layer--) {
        weights[layer] = weights[layer] - weight_grads_sum[layer].ScalarMul(scale);
        biases[layer] = biases[layer] - bias_grads_sum[layer].ScalarMul(scale);
    }
}

void NeuralNetwork::SaveModel(const std::string &filepath) {
    std::ofstream file(filepath, std::ios::binary);
    assert(file.is_open() && ("Failed to open file " + filepath).c_str());
    int num_layers = layer_sizes.size();
    file.write((char*)&num_layers, sizeof(num_layers));
    for (int size : layer_sizes) {
        file.write((char*)&size, sizeof(size));
    }
    for (const Matrix &w : weights) {
        for (size_t i = 0; i < w.GetRows(); i++) {
            for (size_t j = 0; j < w.GetCols(); j++) {
                double val = w(i, j);
                file.write((char*)&val, sizeof(val));
            }
        }
    }
    for (const Matrix &b : biases) {
        for (size_t i = 0; i < b.GetRows(); i++) {
            for (size_t j = 0; j < b.GetCols(); j++) {
                double val = b(i, j);
                file.write((char*)&val, sizeof(val));
            }
        }
    }
    file.close();
}

void NeuralNetwork::LoadModel(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    assert(file.is_open() && ("Failed to open file " + filepath).c_str());
    int num_layers;
    file.read((char*)&num_layers, sizeof(num_layers));
    layer_sizes.clear();
    for (int i = 0; i < num_layers; i++) {
        int size;
        file.read((char*)&size, sizeof(size));
        layer_sizes.push_back(size);
    }
    weights.clear();
    biases.clear();
    for (int i = 0; i < num_layers - 1; i++) {
        int from = layer_sizes[i];
        int to = layer_sizes[i + 1];
        Matrix w(from, to);
        for (size_t r = 0; r < from; r++) {
            for (size_t c = 0; c < to; c++) {
                double val;
                file.read((char*)&val, sizeof(val));
                w(r, c) = val;
            }
        }
        weights.push_back(w);
    }
    for (int i = 0; i < num_layers - 1; i++) {
        int to = layer_sizes[i + 1];
        Matrix b(1, to);
        for (size_t c = 0; c < to; c++) {
            double val;
            file.read((char*)&val, sizeof(val));
            b(0, c) = val;
        }
        biases.push_back(b);
    }
    file.close();
}