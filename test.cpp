#include "NeuralNetwork.h"
#include "MNISTReader.h"

int main() {
    NeuralNetwork nn({ 784, 64, 10 });
    nn.LoadModel("mnist_model.dat");

    std::vector <Matrix> test_images = ReadImages("dataset/t10k-images.idx3-ubyte");
    std::vector <int> test_labels = ReadLabels("dataset/t10k-labels.idx1-ubyte");
    int correct = 0;
    for (size_t i = 0; i < test_images.size(); i++) {
        Matrix input = test_images[i].Flatten(0);
        Matrix output = nn.FeedForward(input);
        int predicted_label = 0;
        double max_value = output(0, 0);
        for (int j = 1; j < output.GetCols(); j++) {
            if (output(0, j) > max_value) {
                max_value = output(0, j);
                predicted_label = j;
            }
        }
        if (predicted_label == test_labels[i]) {
            correct++;
        }
    }

    std::cout << correct << " out of " << test_images.size() << " correct." << std::endl;
    std::cout << "Accuracy: " << (static_cast <double> (correct) / test_images.size()) * 100.0 << "%" << std::endl;
    return 0;
}