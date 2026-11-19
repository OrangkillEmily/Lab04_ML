# Linear Regression — Hồi quy Tuyến tính (Baseline 1)

## 1. Purpose — Mục đích

**Linear Regression (`Ordinary Least Squares — OLS`)** đóng vai trò là **Mô hình Đối chiếu Cơ sở Số 1 (`Baseline Model 1`)** trong dự án Lab 04 (`LAB04_MLP_Regression — PyTorch Edition`).

Về mặt kiến trúc học máy, `Linear Regression` tương đương với một **mạng nơ-ron tối giản chỉ có 1 nơ-ron đầu ra tuyến tính và không có tầng ẩn** ($\hat{y} = \mathbf{w}^T \mathbf{x} + b$). Việc đối chiếu `PyTorch MLP Regression` (`13 -> 32 -> 16 -> 1`) với `Linear Regression` giúp chứng minh bằng thực nghiệm giá trị của các tầng ẩn phi tuyến `ReLU` kết hợp điều chuẩn `L2 + Dropout` trong việc mô hình hóa các quan hệ cong phức tạp (như mối quan hệ giữa `LSTAT`, `RM`, `DIS` và giá nhà `MEDV`).

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Ma trận 13 đặc trưng đã chuẩn hóa Z-Score (`StandardScaler` chỉ `fit` trên tập Train, `leakage_free_verified = True`) và vector giá nhà `MEDV` |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Ma trận đặc trưng và nhãn giá nhà của tập Kiểm tra độc lập (Hold-out Test) |
| `config.yaml -> baseline_models.linear_regression` | Cấu hình `fit_intercept: true` (sử dụng `sklearn.linear_model.LinearRegression`) |

### Output

| Output | Description |
|---|---|
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá trị nhà dự đoán theo siêu phẳng tuyến tính toàn cục |
| `models/linear_regression.pkl` | Tệp lưu trữ vector trọng số `coef_` (`13` phần tử) và hệ số chặn `intercept_ = 22.7965` |

### Bảng Trọng số Hồi quy Tuyến tính (`coef_` & Thống kê OLS — Khớp 100% với Cell 37 & Cell 63 `lab04.ipynb`)

| Hạng | Feature | Weight ($w_i$) | $|w_i|$ | Std Err | $t$-stat | $P > \|t\|$ | Ý nghĩa Kinh tế - Đô thị |
|---:|---|---:|---:|---:|---:|---:|---|
| — | **`const` (Intercept $b$)** | **`22.7965`** | `22.7965` | `0.236` | `96.774` | `0.000` | Giá nhà trung bình kỳ vọng khi mọi đặc trưng chuẩn hóa bằng `0` |
| 1 | **LSTAT** | **`-3.6117`** | `3.6117` | `0.395` | `-9.133` | `0.000` | Tỷ lệ thu nhập thấp tăng làm giá nhà giảm mạnh nhất |
| 2 | **RM** | **`+3.1452`** | `3.1452` | `0.329` | `9.567` | `0.000` | Số phòng trung bình tăng kéo giá nhà tăng mạnh nhất |
| 3 | **DIS** | **`-3.0819`** | `3.0819` | `0.481` | `-6.408` | `0.000` | Khoảng cách tới trung tâm việc làm tăng làm giảm giá nhà (khi cố định các biến khác) |
| 4 | **RAD** | **`+2.2514`** | `2.2514` | `0.652` | `3.454` | `0.001` | Khả năng tiếp cận trục cao tốc hướng tâm |
| 5 | **PTRATIO** | **`-2.0378`** | `2.0378` | `0.321` | `-6.357` | `0.000` | Tỷ lệ học sinh/giáo viên cao làm giảm giá trị khu dân cư |
| 6 | **NOX** | **`-2.0223`** | `2.0223` | `0.498` | `-4.061` | `0.000` | Ô nhiễm khí Nitơ Oxit làm giảm giá nhà |
| 7 | **TAX** | **`-1.7670`** | `1.7670` | `0.704` | `-2.508` | `0.013` | Thuế suất bất động sản cao gây áp lực giảm giá |
| 8 | **B** | **`+1.1296`** | `1.1296` | `0.271` | `4.166` | `0.000` | Chỉ số nhân khẩu học lịch sử năm 1978 |
| 9 | **CRIM** | **`-1.0021`** | `1.0021` | `0.308` | `-3.250` | `0.001` | Tỷ lệ tội phạm bình quân đầu người |
| 10 | **CHAS** | **`+0.7187`** | `0.7187` | `0.247` | `2.914` | `0.004` | Khu vực giáp sông Charles có giá cao hơn |
| 11 | **ZN** | `+0.6963` | `0.6963` | `0.370` | `1.882` | `0.061` | Tỷ lệ đất ở quy hoạch lô lớn |
| 12 | **INDUS** | `+0.2781` | `0.2781` | `0.464` | `0.599` | `0.549` | Tỷ lệ diện tích công nghiệp phi bán lẻ (không có ý nghĩa thống kê ở mức 5%) |
| 13 | **AGE** | `-0.1760` | `0.1760` | `0.407` | `-0.432` | `0.666` | Tỷ lệ nhà xây trước năm 1940 (không có ý nghĩa thống kê ở mức 5%) |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Linear Regression workflow

```text
      X_train_scaled (404, 13) + y_train (404,)
                          |
                          v
      Add Bias Column x_0 = 1 -> Design Matrix X_tilde (404, 14)
                          |
                          v
      Minimize Residual Sum of Squares (OLS Objective):
      L(w, b) = (1 / N) * || y - (X * w + b) ||_2^2
                          |
                          v
      Closed-Form Normal Equation / SVD Solver:
      theta_hat = (X_tilde^T * X_tilde)^(-1) * X_tilde^T * y
                          |
                          v
      Save models/linear_regression.pkl -> Predict on Test (102)
```

### Công thức Toán học Cốt lõi

1. **Hàm dự đoán tuyến tính**:
   $$\hat{y}_i = b + \sum_{j=1}^{13} w_j x_{i,j}^{\text{scaled}} = \mathbf{w}^T \mathbf{x}_i^{\text{scaled}} + b$$
2. **Hàm mất mát OLS**:
   $$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i^{\text{scaled}} - b)^2$$
   Vì không có hàm kích hoạt phi tuyến, mô hình chỉ có thể biểu diễn **một siêu phẳng duy nhất** trong không gian $\mathbb{R}^{13}$.

## 4. Kết quả Thực nghiệm & Đối chiếu (`lab04.ipynb`)

### 4.1. Hiệu suất trên Tập Test (`102` mẫu — Cell 37, 46, 49) & 5-Fold Cross-Validation (`506` mẫu — Cell 52)

| Nhóm Đánh giá | MSE | RMSE | MAE | MAPE (%) | $R^2$ | Max Error | Thời gian Huấn luyện |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hold-out Test (`102` mẫu)** | `24.2911` | `4.9286` | `3.1891` | `16.8664%` | `0.6688` | `25.2604` | `0.0083s` |
| **5-Fold CV (`506` mẫu, Mean ± Std)** | `23.4886 ± 2.0601` | `4.8428` | `3.3885` | `17.0105%` | `0.7152 ± 0.0419` | — | — |

### 4.2. Thống kê Chẩn đoán `Model Summary` (Cell 63)

| Chỉ số Chẩn đoán | Giá trị Thực nghiệm | Nhận xét |
|---|---:|---|
| **Train $R^2$** | `0.7509` | Phản ánh hiện tượng **Underfitting** do giả định tuyến tính |
| **Log-Likelihood / AIC / BIC** | `-1194.322` / `2416.644` / `2472.664` | Độ phù hợp thấp nhất trong 4 mô hình (AIC/BIC cao nhất) |
| **F-statistic** | `90.427` ($p = 0.0$) | Mô hình tổng thể có ý nghĩa thống kê |
| **Durbin-Watson / Ljung-Box $Q$** | `2.114` / `1.33` ($p = 0.25$) | Phần dư không có tự tương quan bậc 1 |
| **Jarque-Bera (Skew / Kurtosis)** | `579.82` (`1.38` / `8.18`) | Phần dư lệch phải do sai số ở phân khúc nhà giá cao (`MEDV = 50.0`) |
| **Within ±$3k USD (Test Set — Cell 60)** | `58.82%` | Thấp nhất trong 4 mô hình (`MLP Regression` đạt `76.47%`) |

### 4.3. Kết luận So sánh với `PyTorch MLP Regression`
- **Ưu điểm**: Huấn luyện cực nhanh (`0.0083s`), nghiệm giải tích chính xác và dễ giải thích trực tiếp dấu/độ lớn từng hệ số `coef_`.
- **Hạn chế**: Do không thể học đường cong phi tuyến (như `LSTAT` giảm theo hàm mũ đối với `MEDV`), `Linear Regression` có `Test MSE = 24.2911` cao gần gấp đôi `PyTorch MLP Regression` (`13.5348`) và `Test R² = 0.6688` thấp hơn rõ rệt (`MLP` đạt `0.8154`, `5-Fold CV R²` đạt `0.8483`).
