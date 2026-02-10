# NumRecognition

## Cách nhận dạng chữ số hoạt động

Dự án này là một mạng nơ-ron tự xây dựng cho bộ dữ liệu MNIST. Quy trình chính:

1) Đọc ảnh và nhãn từ các file IDX vào ma trận.
2) Làm phẳng mỗi ảnh $28 \times 28$ thành vector hàng $1 \times 784$.
3) Chạy forward qua mạng fully connected.
4) Khi train, tính gradient bằng backpropagation và cập nhật trọng số.
5) Khi test, chọn lớp có xác suất softmax lớn nhất.

Các điểm vào chính:

- Vòng lặp train: [train.cpp](train.cpp)
- Vòng lặp test: [test.cpp](test.cpp)
- Mô hình mạng: [src/NeuralNetwork.cpp](src/NeuralNetwork.cpp)
- Phép toán ma trận và activation: [src/Matrix.cpp](src/Matrix.cpp) và [src/Math.cpp](src/Math.cpp)
- Đọc MNIST IDX: [src/MNISTReader.cpp](src/MNISTReader.cpp)

## Kiến trúc mô hình

Mô hình là MLP 3 lớp trong [train.cpp](train.cpp):

$$784 \rightarrow 64 \rightarrow 10$$

- Lớp vào: 784 đặc trưng (pixel đã làm phẳng).
- Lớp ẩn: 64 nút với ReLU.
- Lớp ra: 10 nút với softmax.

## Dòng chảy train

Vòng lặp train trong [train.cpp](train.cpp) xử lý từng mẫu (stochastic gradient descent):

- Đọc một ảnh, làm phẳng về $1 \times 784$.
- Tạo vector one-hot kích thước $10$.
- Gọi `BackPropagate(input, target, learning_rate)`.

Trong backpropagation ([src/NeuralNetwork.cpp](src/NeuralNetwork.cpp)):

- Forward pass lưu output từng lớp.
- Sai số đầu ra tính theo `output - target`, phù hợp softmax + cross-entropy.
- Gradient của weight và bias tính bằng nhân ma trận.
- Cập nhật tham số bằng gradient descent.

## Dòng chảy suy luận

Vòng lặp test trong [test.cpp](test.cpp) chạy forward và chọn chỉ số có giá trị softmax lớn nhất.

## Chi tiết toán học

Nếu cần mô tả chính xác hơn về backpropagation trong dự án, xem [BackPropagation.vi.md](BackPropagation.vi.md).

## Liên kết tài liệu

- Tổng quan (English): [README.md](README.md)
- Backprop (English): [BackPropagation.md](BackPropagation.md)
- Backprop (Tiếng Việt): [BackPropagation.vi.md](BackPropagation.vi.md)
