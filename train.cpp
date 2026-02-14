#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <random>
#include "MNISTReader.h"
#include "NeuralNetwork.h"

std::vector <double> GetTargetVector(int label) {
    std::vector <double> v(10, 0.0);
    v[label] = 1.0;
    return v;
}

int main() {
	std::srand(static_cast <unsigned int> (std::time(0)));

	std::string train_img_path = "dataset/train-images.idx3-ubyte";
	std::string train_lbl_path = "dataset/train-labels.idx1-ubyte";

	std::vector <Matrix> train_images = ReadImages(train_img_path);
	std::vector <int> train_labels = ReadLabels(train_lbl_path);
	std::vector <Matrix> train_inputs;
	train_inputs.reserve(train_images.size());
	for (const Matrix &img : train_images) {
		train_inputs.push_back(img.Flatten(0));
	}

	NeuralNetwork nn({ 784, 128, 64, 10 });
	double learning_rate = 0.01;
	int epochs = 15;
	int batch_size = 64;
	std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
	std::vector <size_t> order(train_inputs.size());
	std::iota(order.begin(), order.end(), 0);

	printf("Training started...\n");

	for (int epoch = 0; epoch < epochs; epoch++) {
		std::shuffle(order.begin(), order.end(), rng);
		for (size_t start = 0; start < train_inputs.size(); start += batch_size) {
			size_t end = std::min(start + (size_t)batch_size, train_inputs.size());
			std::vector <Matrix> batch_inputs;
			std::vector <Matrix> batch_targets;
			batch_inputs.reserve(end - start);
			batch_targets.reserve(end - start);
			for (size_t idx = start; idx < end; idx++) {
				size_t sample = order[idx];
				batch_inputs.push_back(train_inputs[sample]);
				batch_targets.emplace_back(1, 10, GetTargetVector(train_labels[sample]));
			}
			nn.BackPropagateBatch(batch_inputs, batch_targets, learning_rate);
			if (((start / (size_t)batch_size) + 1) % 10 == 0) {
				printf("\rEpoch %02d/%d - Batch %zu/%zu", epoch + 1, epochs,
					((start / (size_t)batch_size) + 1),
					((train_inputs.size() + batch_size - 1) / (size_t)batch_size));
			}
		}
		printf("     Epoch %02d completed.\n", epoch + 1);
	}

	nn.SaveModel("mnist_model.dat");
	return 0;
}