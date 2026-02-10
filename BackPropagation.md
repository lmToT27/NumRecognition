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

For one-hot targets $y = [0, \dots, 1, \dots, 0]$, the cross-entropy loss is:

$$
\mathcal{L} = -\sum_{i=1}^K y_i \log(a^L_i)
$$

Since only one element $y_c = 1$ (correct class), this becomes:

$$
\mathcal{L} = -\log(a^L_c)
$$

### Mathematical derivation: Why $\delta^L = a^L - y$

We need to compute $\frac{\partial \mathcal{L}}{\partial z^L_j}$ where $z^L$ is the pre-activation (logits) before softmax.

**Step 1: Softmax derivative**

Recall: $a^L_i = \mathrm{softmax}(z^L)_i = \frac{e^{z^L_i}}{\sum_k e^{z^L_k}}$

The derivative of softmax has two cases:

$$
\frac{\partial a^L_i}{\partial z^L_j} = 
\begin{cases}
a^L_i(1 - a^L_i) & \text{if } i = j \\
-a^L_i a^L_j & \text{if } i \neq j
\end{cases}
$$

**Proof:** For $i = j$:
$$
\frac{\partial a^L_i}{\partial z^L_i} = \frac{\partial}{\partial z^L_i}\left(\frac{e^{z^L_i}}{\sum_k e^{z^L_k}}\right) = \frac{e^{z^L_i}\sum_k e^{z^L_k} - e^{z^L_i}e^{z^L_i}}{(\sum_k e^{z^L_k})^2} = a^L_i(1-a^L_i)
$$

For $i \neq j$:
$$
\frac{\partial a^L_i}{\partial z^L_j} = \frac{-e^{z^L_i}e^{z^L_j}}{(\sum_k e^{z^L_k})^2} = -a^L_i a^L_j
$$

**Step 2: Apply chain rule**

For the correct class $c$ (where $y_c = 1$):

$$
\frac{\partial \mathcal{L}}{\partial z^L_j} = \frac{\partial}{\partial z^L_j}(-\log a^L_c) = -\frac{1}{a^L_c} \frac{\partial a^L_c}{\partial z^L_j}
$$

**Case 1:** $j = c$ (the correct class)
$$
\frac{\partial \mathcal{L}}{\partial z^L_c} = -\frac{1}{a^L_c} \cdot a^L_c(1-a^L_c) = -(1-a^L_c) = a^L_c - 1
$$

**Case 2:** $j \neq c$ (incorrect class)
$$
\frac{\partial \mathcal{L}}{\partial z^L_j} = -\frac{1}{a^L_c} \cdot (-a^L_c a^L_j) = a^L_j
$$

**Step 3: Combine with one-hot vector**

Since $y_c = 1$ and $y_j = 0$ for $j \neq c$:

$$
\frac{\partial \mathcal{L}}{\partial z^L_j} = a^L_j - y_j
$$

In vector form:

$$
\boxed{\delta^L = \frac{\partial \mathcal{L}}{\partial z^L} = a^L - y}
$$

This beautiful simplification is why we pair softmax with cross-entropy.

## Backpropagation

### Mathematical definition of delta

Define $\delta^l$ as the error signal at layer $l$:

$$
\delta^l \equiv \frac{\partial \mathcal{L}}{\partial z^l}
$$

where $z^l$ is the pre-activation (before applying ReLU or softmax).

### Derivation 1: Weight gradient formula

**Goal:** Compute $\frac{\partial \mathcal{L}}{\partial W^l_{ij}}$ (gradient for weight from neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$).

**Forward equation:** $z^l_j = \sum_i W^l_{ij} a^{l-1}_i + b^l_j$

By chain rule:

$$
\frac{\partial \mathcal{L}}{\partial W^l_{ij}} = \frac{\partial \mathcal{L}}{\partial z^l_j} \frac{\partial z^l_j}{\partial W^l_{ij}}
$$

Since $\frac{\partial z^l_j}{\partial W^l_{ij}} = a^{l-1}_i$:

$$
\frac{\partial \mathcal{L}}{\partial W^l_{ij}} = \delta^l_j \cdot a^{l-1}_i
$$

In matrix form (outer product):

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W^l} = (a^{l-1})^T \delta^l}
$$

**Shapes:** $(n \times 1) \times (1 \times m) = n \times m$ matching $W^l$.

### Derivation 2: Bias gradient formula

**Forward equation:** $z^l_j = \sum_i W^l_{ij} a^{l-1}_i + b^l_j$

By chain rule:

$$
\frac{\partial \mathcal{L}}{\partial b^l_j} = \frac{\partial \mathcal{L}}{\partial z^l_j} \frac{\partial z^l_j}{\partial b^l_j}
$$

Since $\frac{\partial z^l_j}{\partial b^l_j} = 1$:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial b^l} = \delta^l}
$$

### Derivation 3: Delta propagation formula

**Goal:** Compute $\delta^{l-1} = \frac{\partial \mathcal{L}}{\partial z^{l-1}}$ given $\delta^l = \frac{\partial \mathcal{L}}{\partial z^l}$.

**Forward equations:**
- $z^l = a^{l-1} W^l + b^l$ (matrix form)
- $a^{l-1} = \mathrm{ReLU}(z^{l-1})$ (element-wise)

Apply chain rule for element $i$ in layer $l-1$:

$$
\frac{\partial \mathcal{L}}{\partial z^{l-1}_i} = \sum_j \frac{\partial \mathcal{L}}{\partial z^l_j} \frac{\partial z^l_j}{\partial a^{l-1}_i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_i}
$$

**Step 1:** Since $z^l_j = \sum_k W^l_{kj} a^{l-1}_k + b^l_j$:

$$
\frac{\partial z^l_j}{\partial a^{l-1}_i} = W^l_{ij}
$$

**Step 2:** Since $a^{l-1}_i = \mathrm{ReLU}(z^{l-1}_i)$:

$$
\frac{\partial a^{l-1}_i}{\partial z^{l-1}_i} = \mathrm{ReLU}'(z^{l-1}_i) = 
\begin{cases}
1 & \text{if } z^{l-1}_i > 0 \\
0 & \text{otherwise}
\end{cases}
$$

**Step 3:** Combine:

$$
\delta^{l-1}_i = \sum_j \delta^l_j W^l_{ij} \cdot \mathrm{ReLU}'(z^{l-1}_i)
$$

In matrix form:

$$
\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(z^{l-1})
$$

**Implementation note:** Since $\mathrm{ReLU}'(z^{l-1}_i) = 1$ iff $z^{l-1}_i > 0$ iff $a^{l-1}_i > 0$, we can equivalently write:

$$
\boxed{\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(a^{l-1})}
$$

This saves memory since we already stored $a^{l-1}$ during forward pass.

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