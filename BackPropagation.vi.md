# Backpropagation (Góc nhìn Toán học)

Tài liệu này giải thích cơ sở toán học của bước train được triển khai trong [src/NeuralNetwork.cpp](src/NeuralNetwork.cpp). Code đang dùng MLP nhỏ cho MNIST với ReLU ở lớp ẩn và softmax ở lớp ra.

## Ký hiệu và kích thước

Ký hiệu với batch size = 1:

- $a^0$ là vector hàng đầu vào (ảnh đã làm phẳng).
- Với mỗi lớp $l = 1, \dots, L$:
  - $W^l$ là ma trận trọng số.
  - $b^l$ là vector bias (hàng).
  - $z^l = a^{l-1} W^l + b^l$ là pre-activation.
  - $a^l$ là activation.

Mạng mặc định trong [train.cpp](train.cpp):

- $a^0$ là $1 \times 784$
- $W^1$ là $784 \times 64$, $b^1$ là $1 \times 64$
- $W^2$ là $64 \times 10$, $b^2$ là $1 \times 10$

## Forward pass

Lớp ẩn dùng ReLU, lớp ra dùng softmax:

$$
\begin{aligned}
z^l &= a^{l-1} W^l + b^l \\
\text{với } l < L: \quad a^l &= \mathrm{ReLU}(z^l) = \max(0, z^l) \\
\text{với } l = L: \quad a^L &= \mathrm{softmax}(z^L)
\end{aligned}
$$

Softmax với vector hàng $u$:

$$
\mathrm{softmax}(u)_i = \frac{e^{u_i}}{\sum_j e^{u_j}}
$$

## Hàm mất mát

Với target one-hot $y = [0, \dots, 1, \dots, 0]$, cross-entropy là:

$$
\mathcal{L} = -\sum_{i=1}^K y_i \log(a^L_i)
$$

Vì chỉ có một phần tử $y_c = 1$ (lớp đúng), công thức trở thành:

$$
\mathcal{L} = -\log(a^L_c)
$$

### Chứng minh toán học: Tại sao $\delta^L = a^L - y$

Ta cần tính $\frac{\partial \mathcal{L}}{\partial z^L_j}$ với $z^L$ là pre-activation (logits) trước softmax.

**Bước 1: Đạo hàm của softmax**

Nhớ lại: $a^L_i = \mathrm{softmax}(z^L)_i = \frac{e^{z^L_i}}{\sum_k e^{z^L_k}}$

Đạo hàm của softmax có hai trường hợp:

$$
\frac{\partial a^L_i}{\partial z^L_j} = 
\begin{cases}
a^L_i(1 - a^L_i) & \text{nếu } i = j \\
-a^L_i a^L_j & \text{nếu } i \neq j
\end{cases}
$$

**Chứng minh:** Với $i = j$:
$$
\frac{\partial a^L_i}{\partial z^L_i} = \frac{\partial}{\partial z^L_i}\left(\frac{e^{z^L_i}}{\sum_k e^{z^L_k}}\right) = \frac{e^{z^L_i}\sum_k e^{z^L_k} - e^{z^L_i}e^{z^L_i}}{(\sum_k e^{z^L_k})^2} = a^L_i(1-a^L_i)
$$

Với $i \neq j$:
$$
\frac{\partial a^L_i}{\partial z^L_j} = \frac{-e^{z^L_i}e^{z^L_j}}{(\sum_k e^{z^L_k})^2} = -a^L_i a^L_j
$$

**Bước 2: Áp dụng quy tắc chuỗi**

Với lớp đúng $c$ (khi $y_c = 1$):

$$
\frac{\partial \mathcal{L}}{\partial z^L_j} = \frac{\partial}{\partial z^L_j}(-\log a^L_c) = -\frac{1}{a^L_c} \frac{\partial a^L_c}{\partial z^L_j}
$$

**Trường hợp 1:** $j = c$ (lớp đúng)
$$
\frac{\partial \mathcal{L}}{\partial z^L_c} = -\frac{1}{a^L_c} \cdot a^L_c(1-a^L_c) = -(1-a^L_c) = a^L_c - 1
$$

**Trường hợp 2:** $j \neq c$ (lớp sai)
$$
\frac{\partial \mathcal{L}}{\partial z^L_j} = -\frac{1}{a^L_c} \cdot (-a^L_c a^L_j) = a^L_j
$$

**Bước 3: Kết hợp với vector one-hot**

Vì $y_c = 1$ và $y_j = 0$ với $j \neq c$:

$$
\frac{\partial \mathcal{L}}{\partial z^L_j} = a^L_j - y_j
$$

Dạng vector:

$$
\boxed{\delta^L = \frac{\partial \mathcal{L}}{\partial z^L} = a^L - y}
$$

Sự rút gọn tuyệt đẹp này là lý do ta ghép softmax với cross-entropy.

## Backpropagation

### Định nghĩa toán học của delta

Định nghĩa $\delta^l$ là tín hiệu lỗi tại lớp $l$:

$$
\delta^l \equiv \frac{\partial \mathcal{L}}{\partial z^l}
$$

trong đó $z^l$ là pre-activation (trước khi áp dụng ReLU hay softmax).

### Chứng minh 1: Công thức gradient của trọng số

**Mục tiêu:** Tính $\frac{\partial \mathcal{L}}{\partial W^l_{ij}}$ (gradient cho trọng số từ nơ-ron $i$ ở lớp $l-1$ đến nơ-ron $j$ ở lớp $l$).

**Phương trình forward:** $z^l_j = \sum_i W^l_{ij} a^{l-1}_i + b^l_j$

Theo quy tắc chuỗi:

$$
\frac{\partial \mathcal{L}}{\partial W^l_{ij}} = \frac{\partial \mathcal{L}}{\partial z^l_j} \frac{\partial z^l_j}{\partial W^l_{ij}}
$$

Vì $\frac{\partial z^l_j}{\partial W^l_{ij}} = a^{l-1}_i$:

$$
\frac{\partial \mathcal{L}}{\partial W^l_{ij}} = \delta^l_j \cdot a^{l-1}_i
$$

Dạng ma trận (tích ngoài):

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W^l} = (a^{l-1})^T \delta^l}
$$

**Kích thước:** $(n \times 1) \times (1 \times m) = n \times m$ khớp với $W^l$.

### Chứng minh 2: Công thức gradient của bias

**Phương trình forward:** $z^l_j = \sum_i W^l_{ij} a^{l-1}_i + b^l_j$

Theo quy tắc chuỗi:

$$
\frac{\partial \mathcal{L}}{\partial b^l_j} = \frac{\partial \mathcal{L}}{\partial z^l_j} \frac{\partial z^l_j}{\partial b^l_j}
$$

Vì $\frac{\partial z^l_j}{\partial b^l_j} = 1$:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial b^l} = \delta^l}
$$

### Chứng minh 3: Công thức lan truyền delta

**Mục tiêu:** Tính $\delta^{l-1} = \frac{\partial \mathcal{L}}{\partial z^{l-1}}$ khi biết $\delta^l = \frac{\partial \mathcal{L}}{\partial z^l}$.

**Phương trình forward:**
- $z^l = a^{l-1} W^l + b^l$ (dạng ma trận)
- $a^{l-1} = \mathrm{ReLU}(z^{l-1})$ (từng phần tử)

Áp dụng quy tắc chuỗi cho phần tử $i$ ở lớp $l-1$:

$$
\frac{\partial \mathcal{L}}{\partial z^{l-1}_i} = \sum_j \frac{\partial \mathcal{L}}{\partial z^l_j} \frac{\partial z^l_j}{\partial a^{l-1}_i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_i}
$$

**Bước 1:** Vì $z^l_j = \sum_k W^l_{kj} a^{l-1}_k + b^l_j$:

$$
\frac{\partial z^l_j}{\partial a^{l-1}_i} = W^l_{ij}
$$

**Bước 2:** Vì $a^{l-1}_i = \mathrm{ReLU}(z^{l-1}_i)$:

$$
\frac{\partial a^{l-1}_i}{\partial z^{l-1}_i} = \mathrm{ReLU}'(z^{l-1}_i) = 
\begin{cases}
1 & \text{nếu } z^{l-1}_i > 0 \\
0 & \text{ngược lại}
\end{cases}
$$

**Bước 3:** Kết hợp:

$$
\delta^{l-1}_i = \sum_j \delta^l_j W^l_{ij} \cdot \mathrm{ReLU}'(z^{l-1}_i)
$$

Dạng ma trận:

$$
\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(z^{l-1})
$$

**Lưu ý triển khai:** Vì $\mathrm{ReLU}'(z^{l-1}_i) = 1$ khi và chỉ khi $z^{l-1}_i > 0$ khi và chỉ khi $a^{l-1}_i > 0$, ta có thể viết tương đương:

$$
\boxed{\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(a^{l-1})}
$$

Điều này tiết kiệm bộ nhớ vì ta đã lưu $a^{l-1}$ trong forward pass.

### Hiểu về delta (δ): Tín hiệu sai số

**Delta** là trung tâm của backpropagation. Nó đại diện cho mức độ mỗi nơ-ron đóng góp vào sai số cuối cùng và chảy ngược qua mạng.

#### Bước 1: Khởi tạo delta tại lớp đầu ra

```cpp
Matrix delta = output - target;
```

Toán học: $\delta^L = a^L - y$

**Ý nghĩa:**
- Nếu output là `[0.1, 0.8, 0.1]` và target là `[0, 1, 0]`
- Delta là `[0.1, -0.2, 0.1]`
- Giá trị âm tại chỉ số 1 nghĩa là "nơ-ron này dự đoán thấp, cần tăng đầu vào"
- Giá trị dương nghĩa là "nơ-ron này dự đoán cao, cần giảm đầu vào"

**Tại sao đơn giản thế này?** Đạo hàm của softmax + cross-entropy triệt tiêu lẫn nhau, chỉ còn $(a^L - y)$. Đây là lý do chính ta dùng cặp này.

#### Bước 2: Tính gradient cho lớp hiện tại

```cpp
Matrix weight_gradient = prev_activation.Transpose() * delta;
Matrix bias_gradient = delta;
```

Toán học:
- $\frac{\partial \mathcal{L}}{\partial W^l} = (a^{l-1})^T \delta^l$
- $\frac{\partial \mathcal{L}}{\partial b^l} = \delta^l$

**Tại sao transpose?** Để có đúng kích thước ma trận:
- `prev_activation` là $1 \times n$ (vector hàng của activation lớp trước)
- `delta` là $1 \times m$ (vector hàng sai số lớp hiện tại)
- Transpose biến thành $(n \times 1) \times (1 \times m) = n \times m$, khớp với kích thước $W^l$

**Diễn giải gradient của trọng số:**
$$\frac{\partial \mathcal{L}}{\partial W^l_{ij}} = a^{l-1}_i \cdot \delta^l_j$$

- Nếu nơ-ron $i$ ở lớp trước **không hoạt động** ($a^{l-1}_i = 0$), trọng số đó không cần cập nhật
- Nếu nơ-ron $j$ hiện tại **không có lỗi** ($\delta^l_j = 0$), trọng số đó không cần cập nhật
- Gradient là **tích** của cường độ đầu vào và độ lớn sai số

#### Bước 3: Lan truyền delta ngược (chỉ với lớp ẩn)

```cpp
if (layer > 0) {
    Matrix prev_derivative = prev_activation;
    prev_derivative.ApplyReLUDerivative();
    delta = (delta * (weights[layer].Transpose())).HadamardMul(prev_derivative);
}
```

Toán học: $\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(a^{l-1})$

**Phân tích chi tiết:**

1. **Định tuyến lỗi qua trọng số:** `delta * weights[layer].Transpose()`
   - Lấy lỗi từ lớp tiếp theo và phân phối ngược lại
   - Mỗi trọng số như một "ống dẫn" chở lỗi tỷ lệ với độ mạnh của nó
   - Toán học: $\delta^l (W^l)^T$ với kích thước $(1 \times m) \times (m \times n) = 1 \times n$

2. **Cổng qua đạo hàm activation:** `.HadamardMul(prev_derivative)`
   - Nhân từng phần tử với $\mathrm{ReLU}'(a^{l-1})$
   - Với ReLU: $\mathrm{ReLU}'(x) = 1$ nếu $x > 0$, ngược lại $0$
   - **Nơ-ron chết** (khi $a^{l-1}_i = 0$) chặn hoàn toàn dòng lỗi
   - Nơ-ron hoạt động (khi $a^{l-1}_i > 0$) cho lỗi đi qua

**Tại sao kiểm tra `layer > 0`?**
- Không cần tính delta cho lớp đầu vào (layer 0)
- Không có trọng số trước đầu vào, nên không có gradient để tính

#### Bước 4: Cập nhật tham số

```cpp
weights[layer] = weights[layer] - weight_gradient.ScalarMul(learning_rate);
biases[layer] = biases[layer] - bias_gradient.ScalarMul(learning_rate);
```

Toán học:
- $W^l \leftarrow W^l - \eta \frac{\partial \mathcal{L}}{\partial W^l}$
- $b^l \leftarrow b^l - \eta \frac{\partial \mathcal{L}}{\partial b^l}$

### Ví dụ đầy đủ: mạng 2 lớp (784→64→10)

**Forward pass (lưu activation):**
```cpp
layer_outputs[0] = input;           // 1×784
layer_outputs[1] = ReLU(z¹);        // 1×64
layer_outputs[2] = softmax(z²);     // 1×10 (output)
```

**Backward pass:**

**Vòng lặp 1 (layer = 1, lớp đầu ra):**
```cpp
delta = output - target;                              // 1×10
prev_activation = layer_outputs[1];                   // 1×64
weight_gradient = prev_activation.T() * delta;        // 64×10
bias_gradient = delta;                                // 1×10
// Lan truyền delta ngược:
delta = (delta * weights[1].T()) ⊙ ReLU'(layer_outputs[1]);  // 1×64
// Cập nhật W² và b²
```

**Vòng lặp 2 (layer = 0, lớp ẩn):**
```cpp
prev_activation = layer_outputs[0];                   // 1×784
weight_gradient = prev_activation.T() * delta;        // 784×64
bias_gradient = delta;                                // 1×64
// Bỏ qua lan truyền delta (layer == 0)
// Cập nhật W¹ và b¹
```

**Điểm mấu chốt:** Delta chảy ngược như dòng sông:
1. Bắt đầu là $(output - target)$ tại lớp cuối
2. Tại mỗi lớp, nó tính gradient cho trọng số của lớp đó
3. Sau đó biến đổi chính nó thành tín hiệu lỗi cho lớp trước
4. Phép biến đổi gồm hai thao tác: định tuyến qua trọng số + cổng qua activation

## Cập nhật tham số

Triển khai dùng gradient descent mỗi mẫu:

$$
\begin{aligned}
W^l &\leftarrow W^l - \eta \frac{\partial \mathcal{L}}{\partial W^l} \\
b^l &\leftarrow b^l - \eta \frac{\partial \mathcal{L}}{\partial b^l}
\end{aligned}
$$

Trong đó $\eta$ là learning rate.

## Liên hệ với code

- Forward pass và lưu activation: [src/NeuralNetwork.cpp](src/NeuralNetwork.cpp)
- Phép toán ma trận (nhân, transpose, Hadamard): [src/Matrix.cpp](src/Matrix.cpp)
- ReLU và softmax: [src/Math.cpp](src/Math.cpp)

## Liên kết tài liệu

- Tổng quan (English): [README.md](README.md)
- Backprop (English): [BackPropagation.md](BackPropagation.md)
- Tổng quan (Tiếng Việt): [README.vi.md](README.vi.md)
