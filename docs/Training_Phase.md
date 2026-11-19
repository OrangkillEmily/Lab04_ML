# Training Phase — Huấn luyện PyTorch MLP & 3 Mô hình Đối chiếu (`docs/Training_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Model Training** (`src/model.py` và `src/train.py`) khởi tạo và huấn luyện mô hình trọng tâm **PyTorch MLP Regression (`BostonMLPNet`)** trên **GPU CUDA / CPU** với `DataLoader`, điều chuẩn `L2 (weight_decay)` + `Dropout` và `EarlyStopping`, đồng thời huấn luyện 3 mô hình đối chiếu (`Linear Regression`, `Decision Tree`, `Random Forest`) bằng `scikit-learn` trên cùng bộ dữ liệu đã chuẩn hóa.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `X_train_sub` `(323, 13)`, `y_train_sub` `(323,)` | Dữ liệu huấn luyện theo mini-batch (`batch_size = 32`, `shuffle = True`) cho PyTorch MLP |
| `X_val_scaled` `(81, 13)`, `y_val` `(81,)` | Dữ liệu kiểm định nội bộ dùng cho `EarlyStopping(patience = 15, restore_best_weights = True)` |
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Dữ liệu huấn luyện cho 3 mô hình `scikit-learn` |
| `config.yaml -> mlp_model, baseline_models` | `hidden_layers: [32, 16]`, `lr: 0.005`, `weight_decay: 0.001`, `dropout: 0.05`, `epochs: 100` |

### Output

| Thành phần | Mô tả |
|---|---|
| `models/mlp_regression.pth` | Checkpoint PyTorch lưu `state_dict` tốt nhất (tại **Epoch 41**) |
| `models/mlp_training_history.json` & `.csv` | Lịch sử `loss`, `mae`, `mape`, `val_loss`, `val_mae`, `val_mape` qua `56` epochs |
| `models/linear_regression.pkl`, `decision_tree.pkl`, `random_forest.pkl` | Các mô hình đối chiếu đã huấn luyện |
| `models/training_metadata.json` | Siêu dữ liệu kiến trúc, `best_epoch`, thời gian huấn luyện và thiết bị (`CUDA`) |
| `reports/figures/mlp_training_loss_curve.png`, `mlp_train_vs_val_loss.png`, `training_time_comparison.png` | Biểu đồ quá trình hội tụ và so sánh thời gian huấn luyện |

## 3. How to do — Quy trình thực hiện

```text
  [1] Seed Lock (seed=42: Python, NumPy, PyTorch) + Detect CUDA GPU
                                  |
        +-------------------------+-------------------------+
        |                                                   |
        v                                                   v
  [2a] PyTorch MLP (src/train.py)                  [2b] 3 Baselines (sklearn)
  BostonHousingDataset -> DataLoader(bs=32)        Fit on X_train_scaled (404):
  Linear(13,32)->ReLU->Dropout(0.05)               - LinearRegression()
  ->Linear(32,16)->ReLU->Dropout(0.05)->Linear(1)  - DecisionTree(max_depth=6)
  Adam(lr=0.005, weight_decay=0.001) + MSELoss     - RandomForest(n=100, d=10)
  EarlyStopping(patience=15) -> Best Epoch 41             |
        |                                                 |
        v                                                 v
  Save models/mlp_regression.pth                   Save models/*.pkl
```

## 4. Evaluation metrics — Chỉ số & Kết quả Huấn luyện (`lab04.ipynb` Cell 26, 31, 34, 63)

| Mô hình | Số tham số / Cấu hình | Thời gian Train | Diễn biến / Độ khớp Train |
|---|---|---:|---|
| **PyTorch MLP (`13->32->16->1`)** | `993` params (`448 + 528 + 17`) | `6.1319s` (CUDA) | Dừng tại Epoch `56/100`, **Best Epoch `41`**: `Train MSE = 8.4487`, `Train MAE = 2.0542`, `Train MAPE = 10.24%`, `Val MSE = 15.8257`, `Val MAE = 2.8341`, `Val MAPE = 13.85%`; `Train R² = 0.8857` |
| **Linear Regression (OLS)** | `14` params (`1` intercept + `13` coef) | `0.0083s` | `Train R² = 0.7509` |
| **Decision Tree (CART)** | `max_depth=6`, `34` leaves | `0.0073s` | `Train R² = 0.9229` |
| **Random Forest (Bagging)** | `100` trees, `max_depth=10` | `0.1655s` | `Train R² = 0.9526` |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Kích thước | Mô hình sử dụng |
|---|---|---|
| `data/processed/X_train_sub.pkl` & `y_train_sub.pkl` | `(323, 13)` | Cập nhật trọng số PyTorch MLP (`DataLoader`) |
| `data/processed/X_val_scaled.pkl` & `y_val.pkl` | `(81, 13)` | Đánh giá `model.eval()` + `torch.no_grad()` cho `EarlyStopping` |
| `data/processed/X_train_scaled.pkl` & `y_train.pkl` | `(404, 13)` | Huấn luyện `Linear Regression`, `Decision Tree`, `Random Forest` |
