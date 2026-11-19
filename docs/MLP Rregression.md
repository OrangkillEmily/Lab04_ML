# Multilayer Perceptron Regression (MLP Regression — PyTorch Edition)

## 1. Purpose — Mục đích

**Multilayer Perceptron Regression (`BostonMLPNet` — PyTorch `nn.Module`)** là **mô hình trọng tâm** của bài thực hành Lab 04 (`LAB04_MLP_Regression`).

Khác với phiên bản Keras trước đây, hệ thống hiện tại triển khai mạng nơ-ron đa tầng hoàn toàn bằng **PyTorch (`torch.nn.Module`)** kết hợp `BostonHousingDataset`, `DataLoader(batch_size=32, shuffle=True)`, chạy tăng tốc phần cứng **GPU CUDA** (tự động fallback sang CPU nếu không có GPU). Mô hình học các ánh xạ phi tuyến từ 13 đặc trưng đô thị Boston để dự đoán chính xác giá nhà trung vị `MEDV` (`$1,000 USD`), đồng thời được đối chiếu chặt chẽ với 3 mô hình cơ sở (`Linear Regression`, `Decision Tree`, `Random Forest`) qua Hold-out Test, 5-Fold Cross-Validation và Architecture Tuning.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_sub` `(323, 13)`, `y_train_sub` `(323,)` | Dữ liệu huấn luyện cập nhật trọng số qua `DataLoader(batch_size=32, shuffle=True)` (`80%` của `Train` tổng `404` mẫu) |
| `X_val_scaled` `(81, 13)`, `y_val` `(81,)` | Dữ liệu kiểm định nội bộ dùng cho `EarlyStopping` ở chế độ `model.eval()` + `torch.no_grad()` (`20%` của `Train` tổng) |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Tập Kiểm tra độc lập (`Hold-out Test`) chỉ dùng một lần ở bước đánh giá cuối |
| `config.yaml -> mlp_model` | `hidden_layers: [32, 16]`, `learning_rate: 0.005`, `l2_lambda: 0.001` (`weight_decay`), `dropout_rate: 0.05`, `epochs: 100`, `batch_size: 32`, `patience: 15` |

### Output

| Output | Description |
|---|---|
| `models/mlp_regression.pth` | Checkpoint PyTorch lưu `state_dict` tốt nhất (tại **Epoch 41**) cùng siêu tham số kiến trúc |
| `models/mlp_training_history.json` & `.csv` | Lịch sử `loss`, `mae`, `mape`, `val_loss`, `val_mae`, `val_mape` qua `56` epochs |
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá nhà dự đoán liên tục (`$1,000 USD`) khi chạy suy luận trên GPU/CPU |

### Cấu trúc Chi tiết Từng tầng (`BostonMLPNet` — Khớp 100% với Cell 26 & Cell 63 `lab04.ipynb`)

| Layer (type) | Shape | Activation | Param # | Weight Mean | Weight Std | Weight L2 Norm | Khởi tạo (Init) |
|---|---|---|---:|---:|---:|---:|---|
| `hidden_layer_1` (`nn.Linear`) | `(13 -> 32)` | `relu` | `448` | `-0.0539` | `0.4498` | `9.2408` | He Normal |
| `dropout_1` (`nn.Dropout`) | `(p=0.05)` | `none` | `0` | `0.0000` | `0.0000` | `0.0000` | — |
| `hidden_layer_2` (`nn.Linear`) | `(32 -> 16)` | `relu` | `528` | `0.0224` | `0.2890` | `6.5591` | He Normal |
| `dropout_2` (`nn.Dropout`) | `(p=0.05)` | `none` | `0` | `0.0000` | `0.0000` | `0.0000` | — |
| `output_layer` (`nn.Linear`) | `(16 -> 1)` | `linear` | `17` | `0.1482` | `0.4480` | `1.8875` | Xavier Uniform |
| **Tổng cộng (Total Trainable)** | `13 -> 32 -> 16 -> 1` | — | **`993`** | — | — | — | — |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### PyTorch MLP Regression workflow

```text
      Train_sub (323, 13) + Val (81, 13) [StandardScaler fit on 404 Train]
                          |
                          v
      BostonHousingDataset -> DataLoader(batch_size=32, shuffle=True)
                          |
                          v
      Forward Pass on CUDA GPU:
      Linear(13,32)->ReLU->Dropout(0.05)->Linear(32,16)->ReLU->Dropout(0.05)->Linear(16,1)
                          |
                          v
      Loss: nn.MSELoss() + L2 Weight Decay (weight_decay=0.001)
      Backward & Step: torch.optim.Adam(lr=0.005)
                          |
                          v
      Eval on Val (model.eval() + torch.no_grad()) -> EarlyStopping (patience=15)
                          |
                          v
      Restore Best state_dict (Epoch 41) -> Save models/mlp_regression.pth
```

### Công thức Toán học Cốt lõi

1. **Lan truyền thuận (Forward Propagation)**:
   $$\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \quad \mathbf{a}^{(2)} = \text{ReLU}(\mathbf{a}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}), \quad \hat{\mathbf{y}} = \mathbf{a}^{(2)}\mathbf{W}^{(3)} + \mathbf{b}^{(3)}$$
2. **Hàm mất mát với Điều chuẩn $L_2$ (`weight_decay`)**:
   $$\mathcal{L}_{\text{total}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \lambda \sum_{l} \|\mathbf{W}^{(l)}\|_F^2, \qquad \lambda = 0.001$$
3. **Chỉ số MAPE (%)**:
   $$\text{MAPE}(\%) = \frac{100\%}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

## 4. Kết quả Thực nghiệm & Đối chiếu (`lab04.ipynb`)

### 4.1. Diễn biến Huấn luyện (`56/100` Epochs, Best Epoch = `41`, Thời gian = `6.1319s` — Cell 31 & 34)

| Giai đoạn | Train MSE | Train MAE | Train MAPE (%) | Val MSE | Val MAE | Val MAPE (%) |
|---|---:|---:|---:|---:|---:|---:|
| **Epoch 1 (Khởi đầu)** | `511.1486` | `19.9696` | `82.51%` | `488.0780` | `20.0626` | `83.47%` |
| **Epoch 5 (Giảm nhanh)** | `54.6194` | `5.9047` | `28.69%` | `54.0678` | `5.4365` | `28.78%` |
| **Best Validation (Epoch 41)** | **`8.4487 `**| **`2.0542`** | **`10.24%`** | **`15.8257`** | **`2.7935`** | **`13.43%`** |
| **Epoch 56 (Dừng EarlyStopping)** | `8.1125` | `2.0539` | `9.75%` | `17.0391` | `2.9352` | `14.19%` |

### 4.2. Hiệu suất Tập Test (`102` mẫu) & 5-Fold Cross-Validation (`506` mẫu)

| Nhóm Đánh giá | MSE | RMSE | MAE | MAPE (%) | $R^2$ | Max Error |
|---|---:|---:|---:|---:|---:|---:|
| **Hold-out Test (`102` mẫu — Cell 46, 49)** | `13.5348` | `3.6790` | `2.2902` | `12.0913%` | `0.8154` | `26.2101` |
| **5-Fold CV (`506` mẫu — Cell 52)** | **`12.4888 ± 2.7870`** | **`3.5177`** | `2.4428` | `11.9853%` | **`0.8483 ± 0.0352`** | `17.6804 (Max 24.3206)` | `4.0215s/fold (Tổng 20.1073s)`|

### 4.3. Đối chiếu 4 Kiến trúc PyTorch MLP (Architecture Tuning — Cell 55 & 63)

| Kiến trúc | Hidden Units | Params | Best Epoch | Train MSE | Val MSE | Test MSE | Test RMSE | Test MAE | Test MAPE (%) | Test $R^2$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`MLP_Small`** (`13->16->8->1`) | `[16, 8]` | `369` | `93/100` | `8.4158` | `14.6431` | **`12.4992`** | **`3.5354`** | **`2.2884`** | **`11.8373%`** | **`0.8296`** |
| **`MLP_Baseline`** (`13->32->16->1`) | `[32, 16]` | `993` | `41/56` | `8.4487` | `15.8257` | `13.5348` | `3.6790` | `2.2902` | `12.0913%` | `0.8154` |
| **`MLP_Wide`** (`13->64->32->1`) | `[64, 32]` | `3,009` | `34/49` | `8.3452` | `14.6707` | `13.7687` | `3.7106` | `2.4422` | `12.9155%` | `0.8122` |
| **`MLP_Deep`** (`13->64->32->16->1`) | `[64, 32, 16]` | `3,521` | `31/46` | `7.4476` | **`12.8325`** | `13.7648` | `3.7101` | `2.6371` | `13.2612%` | `0.8123` |

### 4.4. Thống kê Chẩn đoán `Model Summary` & Suy luận Mẫu (Cell 60, 63, 64)

- **Chẩn đoán OLS-style (`Train = 404`)**: `Train R² = 0.8857`, `Log-Likelihood = -1036.909`, `AIC = 2101.817`, `BIC = 2157.837`, `HQIC = 2123.993`, `F-statistic = 232.517`, `Residual Variance = 10.2841`.
- **Kiểm định Phần dư**: `Durbin-Watson = 2.002` (rất gần `2.0`, không tự tương quan), `Ljung-Box Q = 0.01 (p = 0.92)`, `Heteroskedasticity H = 0.84 (p = 0.20)`, `Jarque-Bera = 613.89` (Skew `0.57`, Kurtosis `8.93`).
- **Phân tích Phần dư Test (`102` mẫu)**: `Mean_Residual = 0.0121` (gần `0` nhất trong 4 mô hình — hầu như không chệch hệ thống), `Std_Residual = 3.6789`, `Within ±$3k USD = 76.47%`.
- **Thử nghiệm Suy luận Thực tế (Căn hộ mẫu `#0` — Cell 64)**: Nạp lại `models/standard_scaler.pkl` và `models/mlp_regression.pth`, giá thực tế `$24.00k ($24,000 USD)` vs. giá dự đoán `$27.46k ($27,465 USD)` (`AE = $3.46k`, `APE = 14.44%`).
