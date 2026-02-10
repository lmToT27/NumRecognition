# Backpropagation (Math View)

This file explains the math behind the training step implemented in [src/NeuralNetwork.cpp](src/NeuralNetwork.cpp). The code uses a small MLP for MNIST with ReLU in hidden layers and softmax at the output.

## Notation and shapes

Single-sample (batch size 1) notation:

- $a^0$ is the input row vector (flattened image).
- For each layer $l = 1, \dots, L$:
  - $W^l$ is the weight matrix.
  - $b^l$ is the bias row vector.
  - $z^l = a^{l-1} W^l + b^l$ is the pre-activation.
  - $a^l$ is the activation.

Default network in [train.cpp](train.cpp):

- $a^0$ is $1 \times 784$
- $W^1$ is $784 \times 64$, $b^1$ is $1 \times 64$
- $W^2$ is $64 \times 10$, $b^2$ is $1 \times 10$

## Forward pass

Hidden layers use ReLU, output uses softmax:

$$
\begin{aligned}
z^l &= a^{l-1} W^l + b^l \\
  ext{for } l < L: \quad a^l &= \mathrm{ReLU}(z^l) = \max(0, z^l) \\
  ext{for } l = L: \quad a^L &= \mathrm{softmax}(z^L)
\end{aligned}
$$

Softmax for a row vector $u$:

$$
\mathrm{softmax}(u)_i = \frac{e^{u_i}}{\sum_j e^{u_j}}
$$

## Loss function

For one-hot targets $y$, the cross-entropy loss is:

$$
\mathcal{L} = -\sum_i y_i \log(a^L_i)
$$

With softmax, the output-layer error simplifies to:

$$
\delta^L = a^L - y
$$

## Backpropagation

For each layer $l$ from $L$ down to $1$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^l} &= (a^{l-1})^T \delta^l \\
\frac{\partial \mathcal{L}}{\partial b^l} &= \delta^l
\end{aligned}
$$

For hidden layers, the error propagates through weights and ReLU:

$$
\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(a^{l-1})
$$

The code applies $\mathrm{ReLU}'$ to the activation $a^{l-1}$, which works because $a^{l-1} > 0$ iff $z^{l-1} > 0$.

## Parameter update

The implementation uses plain gradient descent per sample:

$$
\begin{aligned}
W^l &\leftarrow W^l - \eta \frac{\partial \mathcal{L}}{\partial W^l} \\
b^l &\leftarrow b^l - \eta \frac{\partial \mathcal{L}}{\partial b^l}
\end{aligned}
$$

where $\eta$ is the learning rate.

## Mapping to the implementation

- Forward pass and stored activations: [src/NeuralNetwork.cpp](src/NeuralNetwork.cpp)
- Matrix ops (multiply, transpose, Hadamard): [src/Matrix.cpp](src/Matrix.cpp)
- ReLU and softmax: [src/Math.cpp](src/Math.cpp)

## Documentation links

- Project overview: [README.md](README.md)
- Vietnamese overview: [README.vi.md](README.vi.md)
- Vietnamese backprop math: [BackPropagation.vi.md](BackPropagation.vi.md)