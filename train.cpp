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

	NeuralNetwork nn({ 784, 64, 10 });
	double learning_rate = 0.01;
	int epochs = 15;

	printf("Training started...\n");

	for (int epoch = 0; epoch < epochs; epoch++) {
		for (size_t i = 0; i < train_images.size(); i++) {
			Matrix input = train_images[i].Flatten(0);
			Matrix target_matrix(1, 10, GetTargetVector(train_labels[i]));
			nn.BackPropagate(input, target_matrix, learning_rate);
			printf("\rEpoch %02d/%d - Sample %zu/%zu", epoch + 1, epochs, i + 1, train_images.size());
		}
		printf("     Epoch %02d completed.\n", epoch + 1);
	}

	nn.SaveModel("mnist_model.dat");
	return 0;
}