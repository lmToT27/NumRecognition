#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "MNISTReader.h"
#include "NeuralNetwork.h"

std::vector<double> GetTargetVector(int label) {
    std::vector<double> v(10, 0.0);
    v[label] = 1.0;
    return v;
}

int main() {
	std::srand(static_cast<unsigned int>(std::time(0)));
	std::string train_img_path = "dataset/train-images.idx3-ubyte";
	std::string train_lbl_path = "dataset/train-labels.idx1-ubyte";
	std::string test_img_path = "dataset/t10k-images.idx3-ubyte";
	std::string test_lbl_path = "dataset/t10k-labels.idx1-ubyte";
	std::vector <Matrix> train_images = ReadImages(train_img_path);
	std::vector <int> train_labels = ReadLabels(train_lbl_path);

	NeuralNetwork nn({ 784, 128, 64, 10 });
	double learning_rate = 0.01;
	int epochs = 5;

	printf("Training started...\n");

	for (int epoch = 0; epoch < epochs; epoch++) {
		for (size_t i = 0; i < train_images.size(); i++) {
			Matrix input = train_images[i].Flatten(0);
			Matrix target(10, 1, GetTargetVector(train_labels[i]));
			nn.BackPropagate(input, target, learning_rate);
			printf("\rEpoch %d/%d - Sample %zu/%zu", epoch + 1, epochs, i + 1, train_images.size());
		}
		std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
	}

	std::vector <Matrix> test_images = ReadImages(test_img_path);
	std::vector <int> test_labels = ReadLabels(test_lbl_path);
	int correct = 0;
	for (size_t i = 0; i < test_images.size(); i++) {
		Matrix input = test_images[i].Flatten(0);
		Matrix output = nn.FeedForward(input);
		int predicted_label = 0;
		double max_value = output(0, 0);
		for (int j = 1; j < output.GetRows(); j++) {
			if (output(j, 0) > max_value) {
				max_value = output(j, 0);
				predicted_label = j;
			}
		}
		if (predicted_label == test_labels[i]) {
			correct++;
		}
	}

	std::cout << "Accuracy: " << static_cast<double>(correct) / test_images.size() * 100 << "%" << std::endl;

	return 0;
}