# NumRecognition

## How number recognition works in this code

This project is a small, from-scratch neural network for MNIST digits. The core flow is:

1) Load images and labels from the IDX files into matrices.
2) Flatten each $28 \times 28$ image into a $1 \times 784$ row vector.
3) Run a feed-forward pass through a fully connected network.
4) During training, compute gradients with backpropagation and update weights.
5) During testing, pick the class with the largest softmax probability.

The main entry points are:

- Training loop: [train.cpp](train.cpp)
- Evaluation loop: [test.cpp](test.cpp)
- Network implementation: [src/NeuralNetwork.cpp](src/NeuralNetwork.cpp)
- Matrix and activation ops: [src/Matrix.cpp](src/Matrix.cpp) and [src/Math.cpp](src/Math.cpp)
- MNIST IDX reader: [src/MNISTReader.cpp](src/MNISTReader.cpp)

## Model architecture

The model is a simple 3-layer multilayer perceptron defined in [train.cpp](train.cpp):

$$784 \rightarrow 64 \rightarrow 10$$

- Input layer: 784 features (flattened pixels).
- Hidden layer: 64 units with ReLU activation.
- Output layer: 10 units with softmax.

## Training flow

The training loop in [train.cpp](train.cpp) processes each sample one at a time (stochastic gradient descent):

- Read a single image, flatten it to $1 \times 784$.
- Create a one-hot target vector of size $10$.
- Call `BackPropagate(input, target, learning_rate)`.

Inside backpropagation ([src/NeuralNetwork.cpp](src/NeuralNetwork.cpp)):

- A forward pass stores each layer output.
- The output error is computed as `output - target`, which matches softmax + cross-entropy.
- Weight and bias gradients are computed with matrix multiplication.
- Parameters are updated with simple gradient descent.

## Inference flow

The test loop in [test.cpp](test.cpp) runs forward propagation and picks the class index with the maximum softmax value.

## Additional math details

For a more formal, math-first derivation of the backpropagation used here, see [BackPropagation.md](BackPropagation.md).

## Documentation links

- English backprop math: [BackPropagation.md](BackPropagation.md)
- Vietnamese overview: [README.vi.md](README.vi.md)
- Vietnamese backprop math: [BackPropagation.vi.md](BackPropagation.vi.md)