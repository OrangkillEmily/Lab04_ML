# Linear Regression — Hồi quy Tuyến tính (Baseline 1)

## 1. Purpose — Mục đích

**Linear Regression (Hồi quy Tuyến tính Bình phương Tối thiểu — OLS)** đóng vai trò là **Mô hình Đối chiếu Cơ sở Số 1 (`Baseline Model 1`)** trong dự án Lab 04.

Về mặt toán học của Chương 5 (Artificial Neural Networks), `Linear Regression` tương đương với **một nơ-ron nhân tạo đơn lẻ không có tầng ẩn và sử dụng hàm kích hoạt tuyến tính** ($\hat{y} = \mathbf{w}^T\mathbf{x} + b$). Mục đích đưa `Linear Regression` vào thực nghiệm là để định lượng trực tiếp giá trị gia tăng của các tầng ẩn phi tuyến `ReLU` trong mạng `MLP Regression` khi giải quyết bài toán dự đoán giá nhà `MEDV`.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Ma trận 13 đặc trưng đã chuẩn hóa Z-Score ($\mu = 0, \sigma = 1$) và vector giá nhà của tập Huấn luyện |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Ma trận đặc trưng và nhãn giá nhà của tập Kiểm tra độc lập |
| `config.yaml -> baseline_models.linear_regression` | Cấu hình `fit_intercept = true` |

### Output

| Output | Description |
|---|---|
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá trị nhà dự đoán theo tổ hợp tuyến tính của 13 đặc trưng |
| `models/linear_regression.pkl` | Tệp lưu trữ vector hệ số góc `coef_` ($\mathbf{w} \in \mathbb{R}^{13}$) và hệ số chặn `intercept_` ($b = 22.7965$) |

### Bảng Trọng số Đã Học (`coef_` & `intercept_` — Khớp 100% với Cell 13 `lab04.ipynb`)

- **Hệ số chặn (`Intercept b`):** **`22.7965`** (`$22,796.5 USD` — giá nhà dự đoán cơ sở khi cả 13 đặc trưng ở mức trung bình).
- **13 Trọng số hồi quy (`Weight w_i`) sắp xếp theo độ lớn ảnh hưởng tuyệt đối $|w_i|$:**

| Hạng | Feature | Weight ($w_i$) | Abs_Weight ($|w_i|$) | Diễn giải Kinh tế (khi đặc trưng tăng $+1\sigma$) |
|---:|---|---:|---:|---|
| 1 | **LSTAT** | `-3.6117` | `3.6117` | Tỷ lệ hộ thu nhập thấp tăng làm giá nhà giảm mạnh nhất (`-$3,611.7 USD`) |
| 2 | **RM** | `+3.1452` | `3.1452` | Số phòng trung bình tăng làm giá nhà tăng mạnh nhất (`+$3,145.2 USD`) |
| 3 | **DIS** | `-3.0819` | `3.0819` | Khoảng cách tới trung tâm việc làm xa hơn làm giảm giá nhà |
| 4 | **RAD** | `+2.2514` | `2.2514` | Khả năng tiếp cận cao tốc hướng tâm tốt hơn làm tăng giá trị |
| 5 | **PTRATIO** | `-2.0378` | `2.0378` | Tỷ lệ học sinh/giáo viên cao làm giảm giá nhà khu vực |
| 6 | **NOX** | `-2.0223` | `2.0223` | Nồng độ khí thải Nitơ Oxit cao làm giảm giá nhà |
| 7 | **TAX** | `-1.7670` | `1.7670` | Thuế suất bất động sản cao làm giảm giá nhà |
| 8 | **B** | `+1.1296` | `1.1296` | Chỉ số thống kê dân cư lịch sử năm 1978 |
| 9 | **CRIM** | `-1.0021` | `1.0021` | Tỷ lệ tội phạm bình quân đầu người cao làm giảm giá nhà |
| 10 | **CHAS** | `+0.7187` | `0.7187` | Khu vực giáp sông Charles có mức giá cộng thêm |
| 11 | **ZN** | `+0.6963` | `0.6963` | Tỷ lệ đất quy hoạch lô lớn tăng làm tăng giá nhà |
| 12 | **INDUS** | `+0.2781` | `0.2781` | Tỷ lệ diện tích công nghiệp phi bán lẻ |
| 13 | **AGE** | `-0.1760` | `0.1760` | Tỷ lệ nhà cũ xây trước năm 1940 cao làm giảm nhẹ giá nhà |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Linear Regression workflow

```text
      X_train_scaled (404, 13) + y_train (404,)
                          |
                          v
        Formulate Design Matrix X_b = [1, X_train_scaled]
                          |
                          v
        Solve Ordinary Least Squares (OLS Objective):
        min_{w, b} || X_train_scaled * w + b - y_train ||_2^2
                          |
                          v
        Extract Learned Parameters:
        b = 22.7965, w = [-1.0021, ..., -3.6117]^T
                          |
                          v
        Save Model -> models/linear_regression.pkl
                          |
                          v
        Predict on Test Set (102 samples):
        y_hat_test = X_test_scaled * w + b
```

### Các bước thực hiện

**Step 1 — Biểu diễn siêu phẳng tuyến tính**

Mô hình giả định giá nhà $\hat{y}_i$ là một hàm tuyến tính thuần túy của 13 đặc trưng đầu vào:

$$\hat{y}_i = b + \sum_{j=1}^{13} w_j x_{i,j}^{\text{scaled}} = \mathbf{w}^T \mathbf{x}_i^{\text{scaled}} + b$$

**Step 2 — Tối ưu hóa Bình phương Tối thiểu (Ordinary Least Squares — OLS)**

Tìm bộ tham số $(\mathbf{w}^*, b^*)$ cực tiểu hóa tổng bình phương sai số trên `404` mẫu huấn luyện:

$$(\mathbf{w}^*, b^*) = \arg\min_{\mathbf{w}, b} \frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}} \left(y_i - (\mathbf{w}^T \mathbf{x}_i^{\text{scaled}} + b)\right)^2$$

Nghiệm đóng giải tích được tính trực tiếp qua phép phân rã ma trận giả nghịch đảo Moore-Penrose $\hat{\theta} = (\mathbf{X}_b^T\mathbf{X}_b)^+\mathbf{X}_b^T\mathbf{y}_{\text{train}}$ trong thời gian chỉ `0.0083` giây.

**Step 3 — Lưu mô hình & Đánh giá trên Tập Test**

Lưu đối tượng vào `models/linear_regression.pkl` và tính dự đoán trên `102` mẫu `X_test_scaled`.

## 4. Evaluation metrics
| Metric | Giá trị trên Tập Test (`102` mẫu) | Giá trị trên Tập Train (`404` mẫu) | Nhận xét & Đối chiếu với `MLP Regression` |
| :--- | :---: | :---: | :--- |
| **MSE (↓)** | **`24.2911`** | `21.6414` | Cao gần gấp đôi `MLP Regression` (`12.5605`) do bị Underfitting |
| **RMSE (↓)** | **`$4.9286k`** | `$4.6520k` | Lệch trung bình gần `$5,000 USD` mỗi căn hộ (so với `$3.5441k` của MLP) |
| **MAE (↓)** | **`$3.1891k`** | `$3.3148k` | Kém hơn `MLP Regression` (`$2.2868k`) tới `$902 USD` trên mỗi căn nhà |
| **MAPE (↓)** | **`16.8664%` (`16.87%`)** | `16.5749%` | Sai số phần trăm cao nhất trong 4 mô hình (MLP chỉ `11.71%`) |
| **$R^2$ Score (↑)** | **`0.6688` (`66.88%`)** | `0.7509` | Bỏ sót hơn **33.12%** phương sai giá nhà do không bắt được độ cong phi tuyến |
| **5-Fold Cross-Validation** | `CV MSE = 23.4886 ± 2.0601` | `CV R² = 0.7152 ± 0.0419` | Xếp hạng 4/4 trên cả Hold-out Test và 5-Fold CV |
| **Within $\pm\$3,000	ext{ USD}$ (↑)** | **`58.82%`** | — | Thấp hơn nhiều so với `MLP Regression` (`75.49%`) và `Random Forest` (`81.37%`) |
| **TrainingTime / AIC / BIC** | `0.0016s` | `AIC = 2416.64`, `BIC = 2472.66` | Huấn luyện nhanh nhất nhưng chất lượng khớp dữ liệu kém nhất |

## 5. Used data
| Tập dữ liệu | Số mẫu | Kích thước | Vai trò đối với `Linear Regression` |
| :--- | :---: | :---: | :--- |
| `X_train_scaled`, `y_train` | `404` | `(404, 13)` | Ước lượng trực tiếp 13 hệ số góc `coef_`, hệ số chặn `intercept_` (`14` tham số) và đánh giá 5-Fold CV |
| `X_test_scaled`, `y_test` | `102` | `(102, 13)` | Đánh giá độc lập 5 chỉ số hồi quy và đối sánh với mạng MLP |

## Code References

| File | Responsibility |
|---|---|
| `src/model.py` (`build_baseline_models`, `LinearRegressionCompat`) | Khởi tạo mô hình `LinearRegression` |
| `src/train.py` | Huấn luyện trên `X_train_scaled` và lưu `models/linear_regression.pkl` |
| `src/evaluate.py` | Đánh giá 5 chỉ số của `Linear Regression` trên tập Train và tập Test |
| `notebooks/lab04.ipynb` (Mục 13, 16, 17, 19) | Trích xuất bảng 13 trọng số `w_i`, `intercept_` và phân tích hiện tượng Underfitting |
