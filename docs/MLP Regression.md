# Multilayer Perceptron Regression (`MLP Regression`)

## 1. Purpose
**Multilayer Perceptron Regression (`MLP Regression`)** là mô hình mạng nơ-ron nhân tạo truyền thẳng đa tầng (*Feedforward Artificial Neural Network*) đóng vai trò là thuật toán trọng tâm của **Bài thực hành Lab 04 (Chương 5: Artificial Neural Networks)**.

Mục đích của mô hình là khắc phục hạn chế tuyến tính của một Perceptron đơn tầng bằng cách xếp chồng hai tầng ẩn kết nối đầy đủ (*Fully Connected / Dense Layers*) sử dụng hàm kích hoạt phi tuyến **ReLU** kết hợp điều chuẩn **$L_2$ (`l2_lambda = 0.001`)**, **`Dropout` (`dropout_rate = 0.05`)** và tầng đầu ra **Linear**, giúp mô hình tự động học được các tương tác phi tuyến bậc cao giữa 13 đặc trưng đô thị (`LSTAT`, `RM`, `DIS`, `NOX`, `TAX`,...) để dự đoán chính xác giá nhà trung vị `MEDV` (`$1,000 USD`).

## 2. Input-Output

### Input
| Input | Mô tả chi tiết |
| :--- | :--- |
| `X_train_sub` `(323, 13)`, `y_train_sub` `(323,)` | Dữ liệu đặc trưng đã chuẩn hóa Z-Score và nhãn giá nhà dùng để cập nhật trọng số qua Backpropagation |
| `X_val_scaled` `(81, 13)`, `y_val` `(81,)` | Tập kiểm định độc lập dùng để theo dõi `val_loss`, `val_mae`, `val_mape` và kích hoạt `EarlyStopping` |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Tập kiểm tra độc lập dùng để đánh giá hiệu năng tổng quát hóa cuối cùng |
| `config.yaml -> mlp_model` | Kiến trúc `13 -> 32 -> 16 -> 1`, `optimizer = adam (lr = 0.005)`, `batch_size = 32`, `epochs = 100`, `patience = 15`, `l2_lambda = 0.001`, `dropout_rate = 0.05` |

### Output
| Output | Mô tả chi tiết |
| :--- | :--- |
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N 	imes 1}$) | Vector giá trị nhà dự đoán liên tục (đơn vị: `$1,000 USD`) |
| `models/mlp_regression.keras` | Tệp mô hình Keras lưu trữ toàn bộ kiến trúc và **`993` tham số** tối ưu tại `Best Epoch = 91` |
| `models/mlp_training_history.csv` | Bảng lịch sử hội tụ `loss`, `mae`, `mape`, `val_loss`, `val_mae`, `val_mape` qua `100` epochs |
| `reports/metrics/mlp_architecture_tuning.csv` | Bảng đối sánh 4 kiến trúc MLP (`MLP_Small`, `MLP_Baseline`, `MLP_Wide`, `MLP_Deep`) |

### Cấu trúc Tham số Mô hình (`Model Parameters` — `993` tham số)
| Tầng mạng (Layer) | Kích thước Trọng số `W` | Kích thước Độ lệch `b` | Hàm Kích hoạt / Điều chuẩn | Số lượng Tham số |
| :--- | :--- | :--- | :--- | :---: |
| **Input Layer** | — | — | Chuẩn hóa Z-Score `(None, 13)` | `0` |
| **Hidden Layer 1 (`Dense 32`)** | `(13, 32)` (`416` weights) | `(32,)` (`32` biases) | `ReLU` — $\max(0, z)$, `L2(0.001)` | **`448`** |
| **Dropout 1 (`Dropout`)** | — | — | Tắt ngẫu nhiên `5%` nơ-ron (`rate = 0.05`) khi train | `0` |
| **Hidden Layer 2 (`Dense 16`)** | `(32, 16)` (`512` weights) | `(16,)` (`16` biases) | `ReLU` — $\max(0, z)$, `L2(0.001)` | **`528`** |
| **Dropout 2 (`Dropout`)** | — | — | Tắt ngẫu nhiên `5%` nơ-ron (`rate = 0.05`) khi train | `0` |
| **Output Layer (`Dense 1`)** | `(16, 1)` (`16` weights) | `(1,)` (`1` bias) | `Linear` — $f(z) = z$, `L2(0.001)` | **`17`** |
| **Tổng cộng (`Total Trainable Parameters`)** | — | — | — | **`993`** |

## 3. How to do
1. **Lan truyền thuận (Forward Propagation):**
   $$\mathbf{z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \qquad \mathbf{a}^{(1)} = 	ext{Dropout}_{0.05}\!\left(\max\left(0, \mathbf{z}^{(1)}
ight)
ight) \in \mathbb{R}^{N 	imes 32}$$
   $$\mathbf{z}^{(2)} = \mathbf{a}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}, \qquad \mathbf{a}^{(2)} = 	ext{Dropout}_{0.05}\!\left(\max\left(0, \mathbf{z}^{(2)}
ight)
ight) \in \mathbb{R}^{N 	imes 16}$$
   $$\hat{\mathbf{y}} = \mathbf{z}^{(3)} = \mathbf{a}^{(2)}\mathbf{W}^{(3)} + \mathbf{b}^{(3)} \in \mathbb{R}^{N 	imes 1}$$
2. **Hàm mất mát MSE kết hợp Điều chuẩn $L_2$ (`Weight Decay`):**
   $$\mathcal{L}(	heta) = rac{1}{B}\sum_{i=1}^{B}(y_i - \hat{y}_i)^2 + \lambda \sum_{l=1}^{3}\|\mathbf{W}^{(l)}\|_F^2, \qquad \lambda = 0.001$$
3. **Lan truyền ngược (Backpropagation) & Cập nhật Adam:**
   $$\delta^{(3)} = rac{2}{B}(\hat{\mathbf{y}} - \mathbf{y}), \qquad \delta^{(2)} = \left(\delta^{(3)}(\mathbf{W}^{(3)})^T
ight) \odot \mathbb{I}(\mathbf{z}^{(2)} > 0), \qquad \delta^{(1)} = \left(\delta^{(2)}(\mathbf{W}^{(2)})^T
ight) \odot \mathbb{I}(\mathbf{z}^{(1)} > 0)$$
   Bộ tối ưu **Adam** ($\eta = 0.005$) cập nhật trọng số trên `11` mini-batches/epoch.
4. **Giám sát Kiểm định & Khôi phục Trọng số (`EarlyStopping`):**
   - Chạy `100/100` epochs, hàm mất mát huấn luyện giảm từ `578.2240` (Epoch 1) xuống `12.1656` (Epoch 100). Điểm cực tiểu kiểm định đạt tại **Epoch `91`** (`Train MSE = 11.7010`, `Best Val MSE = 14.0724`, `Best Val MAE = 2.6805`), không còn hiện tượng Overfitting.

## 4. Evaluation metrics
| Nhóm Đánh giá | Metric | Giá trị Đạt được | Nhận xét & Ý nghĩa Học thuật |
| :--- | :--- | :---: | :--- |
| **Hội tụ Huấn luyện (`Train_sub` 323 mẫu)** | `Epoch 1 -> Epoch 100 Train MSE` | `578.2240` $
ightarrow$ **`12.1656`** | Tại Best Epoch `91`: `Train MSE = 11.7010`, bám sát `Val MSE = 14.0724` |
| **Kiểm định (`Validation` 81 mẫu)** | `Best Epoch` / `Best Val MSE` / `MAE` | **Epoch `91`** / **`14.0724`** / **`$2.6805k`** | Điều chuẩn `L2 (0.001)` + `Dropout (0.05)` giúp thu hẹp khoảng cách Train–Val |
| **Kiểm tra Độc lập (`Test Set` 102 mẫu)** | **Test MSE / RMSE** | **`12.5605`** / **`$3.5441k`** | **Giảm 48.29% MSE** so với `Linear Regression` (`24.2911`) |
| **Kiểm tra Độc lập (`Test Set` 102 mẫu)** | **Test MAE / MAPE (%)** | **`$2.2868k`** / **`11.71%`** | Tốt hơn cả `Decision Tree` (`MAE = $2.5192k`, `MAPE = 13.88%`) và `Linear` (`16.87%`) |
| **Kiểm tra Độc lập (`Test Set` 102 mẫu)** | **Test $R^2$ Score** | **`0.8287` (`82.87%`)** | Giải thích `82.87%` phương sai giá nhà trên tập Test |
| **Đánh giá 5-Fold Cross-Validation** | **CV MSE / CV $R^2$ (Mean ± Std)** | **`11.4700 ± 3.8966`** / **`0.8607 ± 0.0459`** | **Đứng Hạng #1 toàn bảng 5-Fold CV**, vượt cả `Random Forest` (`0.8459`) và `Decision Tree` (`0.7832`) |
| **Mở rộng Kiến trúc (`MLP_Deep`)** | `13 -> 64 -> 32 -> 16 -> 1` (`3,521` params) | `Test MSE = 10.0503`, **`Test R² = 0.8630`** | Vượt qua toàn bộ các mô hình cây trên tập Test (`MAPE = 10.56%`) |
| **Chẩn đoán Phần dư (`Model Summary`)** | `Mean Residual` / `Within ±$3k` / `DW` | **`+0.1149`** / **`75.49%`** / `2.0540` | Độ chệch trung bình gần `0` nhất; không có tự tương quan phần dư ($DW pprox 2$) |

![Actual vs Predicted MLP](../reports/figures/actual_vs_predicted_mlp.png)

## 5. Used data
| Tập dữ liệu | Số mẫu | Kích thước | Vai trò đối với `MLP Regression` |
| :--- | :---: | :---: | :--- |
| `X_train_sub`, `y_train_sub` | `323` | `(323, 13)` | Chia thành `11` mini-batches (`batch_size = 32`) để cập nhật `993` tham số ở mỗi epoch |
| `X_val_scaled`, `y_val` | `81` | `(81, 13)` | Giám sát `val_loss` cuối mỗi epoch để chọn ra `Best Epoch = 91` |
| `X_train_scaled`, `y_train` | `404` | `(404, 13)` | Đánh giá 5-Fold Cross-Validation (`CV R² = 0.8607 ± 0.0459`) và trích xuất `model_summary.csv` |
| `X_test_scaled`, `y_test` | `102` | `(102, 13)` | Đánh giá độc lập 5 chỉ số hồi quy và phân tích phần dư sau khi đã khóa trọng số |
