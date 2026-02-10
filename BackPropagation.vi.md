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
	ext{với } l < L: \quad a^l &= \mathrm{ReLU}(z^l) = \max(0, z^l) \\
	ext{với } l = L: \quad a^L &= \mathrm{softmax}(z^L)
\end{aligned}
$$

Softmax với vector hàng $u$:

$$
\mathrm{softmax}(u)_i = \frac{e^{u_i}}{\sum_j e^{u_j}}
$$

## Hàm mất mát

Với target one-hot $y$, cross-entropy là:

$$
\mathcal{L} = -\sum_i y_i \log(a^L_i)
$$

Kết hợp với softmax, sai số lớp cuối rút gọn thành:

$$
\delta^L = a^L - y
$$

## Backpropagation

Với mỗi lớp $l$ từ $L$ về $1$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^l} &= (a^{l-1})^T \delta^l \\
\frac{\partial \mathcal{L}}{\partial b^l} &= \delta^l
\end{aligned}
$$

Với lớp ẩn, sai số lan truyền ngược qua trọng số và ReLU:

$$
\delta^{l-1} = \left(\delta^l (W^l)^T\right) \odot \mathrm{ReLU}'(a^{l-1})
$$

Code áp dụng $\mathrm{ReLU}'$ trên activation $a^{l-1}$, hợp lệ vì $a^{l-1} > 0$ khi và chỉ khi $z^{l-1} > 0$.

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
