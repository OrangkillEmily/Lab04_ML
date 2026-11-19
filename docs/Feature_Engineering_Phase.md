# Feature Engineering Phase — Chuẩn hóa & Trích xuất Đặc trưng (`docs/Feature_Engineering_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Feature Engineering** giải quyết bài toán chênh lệch thang đo giữa `13` biến đầu vào của Boston Housing (ví dụ `NOX` $\in [0.385, 0.871]$ trong khi `TAX` $\in [187, 711]$ và `B` $\in [0.32, 396.90]$) bằng chuẩn hóa **Z-Score (`StandardScaler`)**, đồng thời phân tích vai trò trích xuất đặc trưng phi tuyến của các tầng ẩn `ReLU` trong **PyTorch `BostonMLPNet`** so với trọng số tuyến tính `OLS` và mức độ quan trọng đặc trưng (`feature_importances_`) của mô hình cây.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `13` biến đặc trưng gốc | `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT` |
| `models/standard_scaler.pkl` | Tham số $\mu_{\text{train}}$ và $\sigma_{\text{train}}$ học từ `404` mẫu `Train` |

### Output

| Thành phần | Mô tả |
|---|---|
| Ma trận đặc trưng chuẩn hóa | `X_train_scaled` `(404, 13)`, `X_train_sub` `(323, 13)`, `X_val_scaled` `(81, 13)`, `X_test_scaled` `(102, 13)` |
| Biểu diễn ẩn phi tuyến (`BostonMLPNet`) | Không gian ẩn `32` chiều (`hidden_layer_1`) và `16` chiều (`hidden_layer_2`) học tự động qua `ReLU` |
| Bảng xếp hạng đặc trưng | Hệ số `coef_` (`Linear Regression`) và `feature_importances_` (`Decision Tree`, `Random Forest`) |

## 3. How to do — Quy trình thực hiện

```text
  13 Raw Features (Mixed Scales: NOX ~0.55 vs TAX ~404)
                        |
                        v
    [1] Z-Score Standardization (Zero-mean, Unit-variance):
        x_scaled = (x - mu_train) / sigma_train
                        |
        +---------------+---------------+
        |                               |
        v                               v
  [2a] PyTorch MLP Hidden Layers   [2b] Tree / Linear Feature Weighting
  Linear(13->32, ReLU) -> (32->16)  OLS coef_ & CART Impurity Reduction
```

- **Lý do không xóa hay cắt biến ngoại lai**: Giữ nguyên toàn bộ `13` đặc trưng và `506` bản ghi để bảo toàn tín hiệu của phân khúc nhà cao cấp (`MEDV = 50.0`) và đảm bảo so sánh công bằng giữa các mô hình.

## 4. Evaluation metrics — Đánh giá Tầm quan trọng Đặc trưng (`lab04.ipynb` Cell 23, 37, 40, 43)

- **Hiệu quả chuẩn hóa `StandardScaler`**: Đưa toàn bộ `13` đặc trưng tập `Train` về `Mean = 0.0000`, `Std = 1.0000`, giúp bộ tối ưu `Adam (lr=0.005)` hội tụ nhanh ngay từ 5 epoch đầu (`Train MSE` giảm từ `511.15` xuống `54.62`).
- **Xếp hạng đặc trưng có tác động mạnh nhất**:
  - **Linear Regression (`|coef_|`)**: `LSTAT` (`-3.6117`), `RM` (`+3.1452`), `DIS` (`-3.0819`), `RAD` (`+2.2514`), `PTRATIO` (`-2.0378`).
  - **Decision Tree (`feature_importances_`)**: `RM` (`0.6278` — `62.78%`) và `LSTAT` (`0.1957` — `19.57%`) chiếm tổng cộng **`82.35%`**.
  - **Random Forest (`feature_importances_`)**: `RM` (`0.2955`), `LSTAT` (`0.2324`), `NOX` (`0.0888`), `INDUS` (`0.0754`), `PTRATIO` (`0.0673`).

## 5. Used data — Dữ liệu sử dụng

| Dữ liệu | Kích thước | Vai trò |
|---|---|---|
| `data/processed/X_train_scaled.pkl` | `(404, 13)` | Tính $\mu_{\text{train}}, \sigma_{\text{train}}$ và phân tích trọng số đặc trưng |
| `data/processed/X_test_scaled.pkl` | `(102, 13)` | Kiểm tra phân phối đặc trưng sau biến đổi (`Test_Scaled_Mean` quanh `0`) |
