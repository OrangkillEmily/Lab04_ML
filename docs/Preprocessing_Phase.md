# Preprocessing Phase — Tiền xử lý & Chống Rò rỉ Dữ liệu (`docs/Preprocessing_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Preprocessing** (`src/preprocess.py`) thực hiện tách biệt ma trận đặc trưng `X` và biến mục tiêu `y`, phân chia các phân vùng `Train / Train_sub / Validation / Test` với `random_state = 42`, và chuẩn hóa Z-Score bằng `StandardScaler` theo quy tắc **Anti-Data-Leakage**: chỉ `fit` bộ chuẩn hóa trên tập `Train` (`404` mẫu) trước khi `transform` sang `Validation` và `Test`.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `X_raw` `(506, 13)`, `y_raw` `(506,)` | Ma trận 13 đặc trưng và vector `MEDV` trích xuất từ `data/raw/boston_housing.csv` |
| `config.yaml -> data, preprocessing` | `test_size: 0.2`, `val_size: 0.2`, `random_state: 42`, cấu hình `StandardScaler(with_mean=True, with_std=True)` |

### Output

| Thành phần | Mô tả |
|---|---|
| `models/standard_scaler.pkl` | Bộ chuẩn hóa `StandardScaler` đã `fit` duy nhất trên `404` mẫu `Train` |
| `data/processed/X_train_scaled.pkl`, `y_train.pkl` | Tập Train tổng `(404, 13)` dùng cho 3 mô hình Baseline và kiểm toán |
| `data/processed/X_train_sub.pkl`, `y_train_sub.pkl` | Tập Train_sub `(323, 13)` dùng cập nhật gradient cho PyTorch MLP |
| `data/processed/X_val_scaled.pkl`, `y_val.pkl` | Tập Validation `(81, 13)` dùng cho `EarlyStopping` của PyTorch MLP |
| `data/processed/X_test_scaled.pkl`, `y_test.pkl` | Tập Test Hold-out `(102, 13)` tách biệt hoàn toàn cho đánh giá cuối |
| `reports/figures/train_val_test_split.png` | Biểu đồ đối chiếu kích thước và phân phối `MEDV` giữa các phân vùng |

## 3. How to do — Quy trình thực hiện

```text
        X_raw (506, 13), y_raw (506,)
                      |
                      v
    [1] Hold-out Split (80% / 20%, seed=42):
        Train (404)          |  Test (102)
            |                |      |
            v                |      |
    [2] Fit StandardScaler   |      |
        ONLY on Train (404)  |      |
            |                |      |
            +--> transform --+------+--> X_test_scaled (102, 13)
            |
            v
        X_train_scaled (404, 13)
            |
            v
    [3] MLP Internal Split (80% / 20%, seed=42):
        Train_sub (323, 13)  |  Validation (81, 13)
```

## 4. Evaluation metrics — Chỉ số Kiểm toán Chống Rò rỉ (`lab04.ipynb` Cell 17, 20, 23)

- **Leakage Audit (`leakage_free_verified = True`)**:
  - `scaler_samples_seen`: `404` (chỉ khớp đúng kích thước tập Train, không chạm vào `102` mẫu Test).
  - `max_diff_scaler_mean_vs_train_raw_mean`: `0.0`.
  - `max_abs_train_scaled_mean`: `8.52e-16` ($\approx 0.0000$).
  - `max_abs_train_scaled_std_error`: `2.22e-16` ($\text{Std} = 1.0000$ trên toàn bộ 13 cột Train).
  - `test_scaled_mean_l2_norm`: `0.460298` (trung bình các cột Test dao động quanh `0`, từ `-0.2188` ở `RM` đến `+0.3058` ở `PTRATIO`, chứng minh không dùng thống kê Test khi chuẩn hóa).
- **Độ đồng đều phân phối `MEDV`**: `Train` (`Mean = $22.80k`, `Std = $9.32k`), `Train_sub` (`$22.79k ± $9.53k`), `Validation` (`$22.84k ± $8.43k`), `Test` (`$21.49k ± $8.56k`).

## 5. Used data — Dữ liệu sử dụng

| Phân vùng | Số mẫu | Tỷ lệ (%) | Mục đích sử dụng |
|---|---:|---:|---|
| `Train` tổng | `404` | `79.84%` | Fit `StandardScaler` và huấn luyện 3 Baseline (`Linear`, `Decision Tree`, `Random Forest`) |
| `Train_sub` | `323` | `63.83%` | Huấn luyện cập nhật trọng số mạng PyTorch MLP |
| `Validation` | `81` | `16.01%` | Giám sát `val_loss` và kích hoạt `EarlyStopping` |
| `Test` (Hold-out) | `102` | `20.16%` | Đánh giá độc lập cuối cùng |
