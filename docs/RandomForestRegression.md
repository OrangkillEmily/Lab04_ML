# Random Forest Regression — Rừng Ngẫu nhiên Hồi quy (Baseline 3)

## 1. Purpose — Mục đích

**Random Forest Regression (`RandomForestRegressor` — Học tập hợp Bagging)** đóng vai trò là **Mô hình Đối chiếu Cơ sở Số 3 (`Baseline Model 3`)** trong dự án Lab 04 (`LAB04_MLP_Regression — PyTorch Edition`).

Khắc phục nhược điểm phương sai lớn (*high variance*) của một cây quyết định đơn lẻ (`Decision Tree`), `Random Forest` huấn luyện song song **100 cây quyết định độc lập (`n_estimators = 100`)** trên các tập con lấy mẫu có hoàn lại (*Bootstrap Sampling*) và chọn ngẫu nhiên tập con đặc trưng tại mỗi điểm chia (`max_features = 'sqrt'`), sau đó lấy **trung bình cộng** dự đoán của cả 100 cây để tạo ra đường biên dự đoán mượt và ổn định hơn.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Ma trận 13 đặc trưng đã chuẩn hóa (`StandardScaler` chỉ `fit` trên tập Train) và vector giá nhà của tập Huấn luyện |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Ma trận đặc trưng và nhãn giá nhà của tập Kiểm tra độc lập (`Hold-out Test`) |
| `config.yaml -> baseline_models.random_forest` | Siêu tham số tập hợp: `n_estimators = 100`, `max_depth = 10`, `max_features = "sqrt"`, `random_state = 42` |

### Output

| Output | Description |
|---|---|
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá trị nhà dự đoán bằng trung bình cộng đầu ra của 100 cây quyết định |
| `models/random_forest.pkl` | Tệp lưu trữ 100 cây quyết định (`Mean Depth = 10.0`, `Mean Leaves/Tree = 90.3`) và `feature_importances_` |

### Bảng Mức độ Quan trọng Đặc trưng (`feature_importances_` — Khớp 100% với Cell 43 `lab04.ipynb`)

| Hạng | Feature | Feature_Importance | Tỷ trọng Đóng góp (%) | So sánh với `Decision Tree` |
|---:|---|---:|---:|---|
| 1 | **RM** | **`0.2955`** | `29.55%` | Phân tán trọng số hợp lý hơn (giảm từ `62.78%` nhờ `max_features='sqrt'`) |
| 2 | **LSTAT** | **`0.2324`** | `23.24%` | Đặc trưng kinh tế-xã hội quan trọng thứ hai (`23.24%` vs `19.57%`) |
| 3 | **NOX** | **`0.0888`** | `8.88%` | Nồng độ khí Nitơ Oxit được khai thác sâu hơn ở nhiều cây |
| 4 | **INDUS** | **`0.0754`** | `7.54%` | Tỷ lệ diện tích công nghiệp phi bán lẻ |
| 5 | **PTRATIO** | **`0.0673`** | `6.73%` | Tỷ lệ học sinh / giáo viên |
| 6 | **CRIM** | `0.0658` | `6.58%` | Tỷ lệ tội phạm bình quân đầu người |
| 7 | **DIS** | `0.0588` | `5.88%` | Khoảng cách tới 5 trung tâm việc làm tại Boston |
| 8 | **TAX** | `0.0353` | `3.53%` | Thuế suất bất động sản |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Random Forest Regression workflow

```text
      X_train_scaled (404, 13) + y_train (404,)
                          |
        +-----------------+-----------------+
        v                 v                 v
  Bootstrap_1       Bootstrap_2  ...  Bootstrap_100
        |                 |                 |
        v                 v                 v
  Tree_1 (sqrt)     Tree_2 (sqrt) ... Tree_100 (sqrt)
  (max_depth=10)    (max_depth=10)    (max_depth=10)
        |                 |                 |
        +-----------------+-----------------+
                          |
                          v
      Average Predictions: y_hat = (1/100) * Sum(T_b(x))
                          |
                          v
      Save models/random_forest.pkl -> Predict on Test (102)
```

### Công thức Toán học Cốt lõi

1. **Hàm dự đoán tập hợp (Bagging Ensemble)**:
   $$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}), \qquad B = 100$$
2. **Giảm phương sai nhờ lấy trung bình $B$ cây có tương quan $\rho$**:
   $$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x})\right) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
   Cơ chế chọn ngẫu nhiên `sqrt(13)` đặc trưng tại mỗi nút chia giúp làm giảm hệ số tương quan $\rho$ giữa các cây, từ đó giảm mạnh phương sai dự đoán trên dữ liệu mới.

## 4. Kết quả Thực nghiệm & Đối chiếu (`lab04.ipynb`)

### 4.1. Hiệu suất trên Tập Test (`102` mẫu — Cell 43, 46, 49) & 5-Fold Cross-Validation (`506` mẫu — Cell 52)

| Nhóm Đánh giá | MSE | RMSE | MAE | MAPE (%) | $R^2$ | Max Error | Thời gian Huấn luyện |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hold-out Test (`102` mẫu)** | `11.5205` | `3.3942` | `2.0994` | `10.8235%` | `0.8429` | `22.6003` | `0.1655s` |
| **5-Fold CV (`506` mẫu, Mean ± Std)** | `12.5518 ± 1.9864` | `3.5339` | `2.3361` | `11.6827%` | `0.8459 ± 0.0395` | — | — |

### 4.2. Thống kê Chẩn đoán `Model Summary` (Cell 63)

| Chỉ số Chẩn đoán | Giá trị Thực nghiệm | Nhận xét |
|---|---:|---|
| **Cấu trúc Tập hợp** | `100` cây, `Depth 10.0 / 10`, `90.3` lá/cây | Độ phức tạp tập hợp cao nhất trong 3 Baseline |
| **Train $R^2$** | `0.9526` | Khớp dữ liệu huấn luyện rất cao |
| **Log-Likelihood / AIC / BIC** | `-859.252` / `1746.505` / `1802.525` | AIC/BIC thấp nhất trên tập Train |
| **F-statistic** | `602.577` ($p = 0.0$) | Rất có ý nghĩa thống kê |
| **Durbin-Watson / Ljung-Box $Q$** | `2.069` / `0.50` ($p = 0.48$) | Phần dư độc lập, không có tự tương quan |
| **Jarque-Bera (Skew / Kurtosis)** | `2170.26` (`1.86` / `13.73`) | Có một số điểm ngoại lai lớn ở nhóm nhà giá cao (`Max_Error = 22.6003`) |
| **Within ±$3k USD (Test Set — Cell 60)** | `81.37%` | Cao nhất trong 4 mô hình (nhỉnh hơn `MLP Regression` ở mức `76.47%`) |

### 4.3. Kết luận So sánh với `PyTorch MLP Regression`
- **Trên tập Hold-out Test (`102` mẫu)**: `Random Forest` dẫn đầu về sai số tuyệt đối (`MAE = 2.0994`, `MAPE = 10.82%`, `Within ±$3k = 81.37%`) và đạt `Test MSE = 11.5205`, `R² = 0.8429` (nhỉnh hơn `MLP Regression` với `MSE = 13.5348`, `MAPE = 12.09%`, `R² = 0.8154`).
- **Trên 5-Fold Cross-Validation (`506` mẫu)**: `PyTorch MLP Regression` (`CV_MSE = 12.4888 ± 2.7870`, `CV_R² = 0.8483 ± 0.0352`) và `Random Forest` (`CV_MSE = 12.5518 ± 1.9864`, `CV_R² = 0.8459 ± 0.0395`) đạt hiệu năng **tương đương nhau**, trong đó `MLP` nhỉnh hơn nhẹ về `CV_MSE` và `CV_R²`, còn `Random Forest` có độ lệch chuẩn `CV_MSE` thấp hơn (`1.9864`) và tốc độ dự đoán `Test` (`0.0109s`) chậm hơn khoảng `5.4` lần so với `MLP` trên GPU CUDA (`0.0020s`).
