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
\text{for } l < L: \quad a^l &= \mathrm{ReLU}(z^l) = \max(0, z^l) \\
\text{for } l = L: \quad a^L &= \mathrm{softmax}(z^L)
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

### Understanding delta (δ): The error signal

**Delta** is the core of backpropagation. It represents how much each neuron contributed to the final error and flows backward through the network.

#### Step 1: Initialize delta at output layer

```cpp
Matrix delta = output - target;
```

Mathematically: $\delta^L = a^L - y$

**What it means:**
- If output is `[0.1, 0.8, 0.1]` and target is `[0, 1, 0]`
- Delta is `[0.1, -0.2, 0.1]`
- The negative value at index 1 means "this neuron undershot, increase its inputs"
- Positive values mean "this neuron overshot, decrease its inputs"

**Why it's this simple:** The softmax + cross-entropy derivative conveniently cancels to just $(a^L - y)$. This is one of the main reasons we use this combination.

#### Step 2: Compute gradients for current layer

```cpp
Matrix weight_gradient = prev_activation.Transpose() * delta;
Matrix bias_gradient = delta;
```

Mathematically:
- $\frac{\partial \mathcal{L}}{\partial W^l} = (a^{l-1})^T \delta^l$
- $\frac{\partial \mathcal{L}}{\partial b^l} = \delta^l$

**Why transpose?** To get the right matrix dimensions:
- `prev_activation` is $1 \times n$ (row vector of previous layer activations)
- `delta` is $1 \times m$ (row vector of current layer errors)
- Transpose makes it $(n \times 1) \times (1 \times m) = n \times m$, matching $W^l$ shape

**Interpretation of weight gradient:**
$$\frac{\partial \mathcal{L}}{\partial W^l_{ij}} = a^{l-1}_i \cdot \delta^l_j$$

- If neuron $i$ in the previous layer was **inactive** ($a^{l-1}_i = 0$), that weight doesn't need updating
- If the current neuron $j$ has **no error** ($\delta^l_j = 0$), that weight doesn't need updating
- The gradient is the **product** of input strength and error magnitude

#### Step 3: Propagate delta backward (for hidden layers only)

```cpp
if (layer > 0) {
    Matrix prev_derivative = prev_activation;
    prev_derivative.ApplyReLUDerivative();
    delta = (delta * (weights[layer].Transpose())).HadamardMul(prev_derivative);
}
```

Mathematically: $\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(a^{l-1})$

**Breaking it down:**

1. **Route error through weights:** `delta * weights[layer].Transpose()`
   - Takes errors from the next layer and distributes them backward
   - Each weight acts like a "pipe" carrying error proportional to its strength
   - Mathematically: $\delta^l (W^l)^T$ where shapes are $(1 \times m) \times (m \times n) = 1 \times n$

2. **Gate by activation derivative:** `.HadamardMul(prev_derivative)`
   - Element-wise multiplication with $\mathrm{ReLU}'(a^{l-1})$
   - For ReLU: $\mathrm{ReLU}'(x) = 1$ if $x > 0$, else $0$
   - **Dead neurons** (where $a^{l-1}_i = 0$) block error flow completely
   - Active neurons (where $a^{l-1}_i > 0$) let error pass through

**Why check `layer > 0`?** 
- We don't need to compute delta for the input layer (layer 0)
- There are no weights before the input, so no gradients to compute there

#### Step 4: Update parameters

```cpp
weights[layer] = weights[layer] - weight_gradient.ScalarMul(learning_rate);
biases[layer] = biases[layer] - bias_gradient.ScalarMul(learning_rate);
```

Mathematically:
- $W^l \leftarrow W^l - \eta \frac{\partial \mathcal{L}}{\partial W^l}$
- $b^l \leftarrow b^l - \eta \frac{\partial \mathcal{L}}{\partial b^l}$

### Complete example: 2-layer network (784→64→10)

**Forward pass (saves activations):**
```cpp
layer_outputs[0] = input;           // 1×784
layer_outputs[1] = ReLU(z¹);        // 1×64
layer_outputs[2] = softmax(z²);     // 1×10 (output)
```

**Backward pass:**

**Iteration 1 (layer = 1, output layer):**
```cpp
delta = output - target;                              // 1×10
prev_activation = layer_outputs[1];                   // 1×64
weight_gradient = prev_activation.T() * delta;        // 64×10
bias_gradient = delta;                                // 1×10
// Propagate delta backward:
delta = (delta * weights[1].T()) ⊙ ReLU'(layer_outputs[1]);  // 1×64
// Update W² and b²
```

**Iteration 2 (layer = 0, hidden layer):**
```cpp
prev_activation = layer_outputs[0];                   // 1×784
weight_gradient = prev_activation.T() * delta;        // 784×64
bias_gradient = delta;                                // 1×64
// Skip delta propagation (layer == 0)
// Update W¹ and b¹
```

**Key insight:** Delta flows backward like a river:
1. Starts as $(output - target)$ at the final layer
2. At each layer, it computes gradients for that layer's weights
3. Then transforms itself to become the error signal for the previous layer
4. The transformation involves two operations: weight-based routing + activation gating

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