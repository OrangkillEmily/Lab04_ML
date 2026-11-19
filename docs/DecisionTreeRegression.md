# Decision Tree Regression — Cây Quyết định Hồi quy (Baseline 2)

## 1. Purpose — Mục đích

**Decision Tree Regression (`DecisionTreeRegressor` — Thuật toán CART)** đóng vai trò là **Mô hình Đối chiếu Cơ sở Số 2 (`Baseline Model 2`)** trong dự án Lab 04 (`LAB04_MLP_Regression — PyTorch Edition`).

Khác với `Linear Regression` (dùng siêu phẳng toàn cục) và `PyTorch MLP Regression` (dùng mặt cong trơn tạo bởi các tầng ẩn `ReLU`), `Decision Tree Regression` tiếp cận bài toán dự đoán giá nhà bằng cách **phân hoạch đệ quy không gian 13 đặc trưng thành các hình hộp chữ nhật rời rạc** (*axis-aligned regions*) và dự đoán giá nhà tại mỗi nút lá bằng giá trị trung bình của các mẫu rơi vào vùng đó.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Ma trận 13 đặc trưng đã chuẩn hóa (`StandardScaler` chỉ `fit` trên tập Train) và vector giá nhà của tập Huấn luyện |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Ma trận đặc trưng và nhãn giá nhà của tập Kiểm tra độc lập (`Hold-out Test`) |
| `config.yaml -> baseline_models.decision_tree` | Siêu tham số kiểm soát độ sâu: `max_depth = 6`, `min_samples_split = 5`, `min_samples_leaf = 2`, `random_state = 42` |

### Output

| Output | Description |
|---|---|
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá trị nhà dự đoán theo hàm bậc thang từng khúc tại các nút lá (`34` nút lá) |
| `models/decision_tree.pkl` | Tệp lưu trữ cấu trúc cây nhị phân đã học và vector mức độ quan trọng đặc trưng `feature_importances_` |

### Bảng Mức độ Quan trọng Đặc trưng (`feature_importances_` — Khớp 100% với Cell 40 `lab04.ipynb`)

| Hạng | Feature | Feature_Importance | Tỷ trọng Đóng góp (%) | Ý nghĩa trong Cấu trúc Phân nhánh CART |
|---:|---|---:|---:|---|
| 1 | **RM** | **`0.6278`** | `62.78%` | Biến phân nhánh gốc quan trọng nhất quyết định phân khúc nhà diện tích lớn/nhỏ |
| 2 | **LSTAT** | **`0.1957`** | `19.57%` | Biến phân nhánh quan trọng thứ hai tách biệt khu vực thu nhập cao/thấp |
| 3 | **DIS** | **`0.0680`** | `6.80%` | Khoảng cách tới 5 trung tâm việc làm tại Boston |
| 4 | **CRIM** | **`0.0574`** | `5.74%` | Tỷ lệ tội phạm bình quân đầu người tại địa phương |
| 5 | **PTRATIO** | `0.0159` | `1.59%` | Tỷ lệ học sinh / giáo viên |
| 6 | **AGE** | `0.0109` | `1.09%` | Tỷ lệ căn hộ xây dựng trước năm 1940 |
| 7 | **TAX** | `0.0095` | `0.95%` | Thuế suất bất động sản |
| 8 | **NOX** | `0.0058` | `0.58%` | Nồng độ khí Nitơ Oxit |

*(Riêng hai đặc trưng `RM` và `LSTAT` đã chiếm tới **82.35%** tổng lượng giảm phương sai của toàn bộ cây quyết định).*

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Decision Tree Regression workflow

```text
      X_train_scaled (404, 13) + y_train (404,)
                          |
                          v
        Start at Root Node P (404 samples)
                          |
                          v
        Search All Features j in {1..13} & Thresholds t:
        Maximize MSE Reduction:
        Delta_MSE = MSE(P) - [ (N_L/N_P)*MSE(L) + (N_R/N_P)*MSE(R) ]
                          |
                          v
        Split Node if depth < 6, N_P >= 5, N_L >= 2, N_R >= 2
                          |
                          v
        Assign Leaf Value = Mean(y_i) in Region R_m (34 Leaves)
                          |
                          v
      Save models/decision_tree.pkl -> Predict on Test (102)
```

### Công thức Toán học Cốt lõi

1. **Tiêu chuẩn phân nhánh CART (Giảm phương sai MSE)**:
   $$\Delta \text{MSE}(j, t) = \text{MSE}(P) - \left(\frac{N_L}{N_P}\text{MSE}(L) + \frac{N_R}{N_P}\text{MSE}(R)\right)$$
2. **Hàm dự đoán tại nút lá $R_m$**:
   $$\hat{y}(\mathbf{x}) = \sum_{m=1}^{34} c_m \cdot \mathbb{I}(\mathbf{x} \in R_m), \qquad c_m = \frac{1}{|R_m|}\sum_{\mathbf{x}_i \in R_m} y_i$$

## 4. Kết quả Thực nghiệm & Đối chiếu (`lab04.ipynb`)

### 4.1. Hiệu suất trên Tập Test (`102` mẫu — Cell 40, 46, 49) & 5-Fold Cross-Validation (`506` mẫu — Cell 52)

| Nhóm Đánh giá | MSE | RMSE | MAE | MAPE (%) | $R^2$ | Max Error | Thời gian Huấn luyện |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hold-out Test (`102` mẫu)** | `10.5431` | `3.2470` | `2.5192` | `13.8809%` | `0.8562` | `10.6500` | `0.0073s` |
| **5-Fold CV (`506` mẫu, Mean ± Std)** | `18.3829 ± 11.6161` | `4.1529` | `2.8782` | `14.5435%` | `0.7832 ± 0.1212` | — | — |

### 4.2. Thống kê Chẩn đoán `Model Summary` (Cell 63)

| Chỉ số Chẩn đoán | Giá trị Thực nghiệm | Nhận xét |
|---|---:|---|
| **Độ sâu thực tế / Số nút lá** | `6` / `34` | Tuân thủ đúng giới hạn `max_depth = 6` |
| **Train $R^2$** | `0.9229` | Học rất khớp trên tập Train (`404` mẫu) |
| **Log-Likelihood / AIC / BIC** | `-957.309` / `1942.619` / `1998.639` | Tốt hơn `Linear Regression` và `MLP` trên mẫu Train |
| **F-statistic** | `359.308` ($p = 0.0$) | Có ý nghĩa thống kê cao |
| **Durbin-Watson / Ljung-Box $Q$** | `1.960` / `0.12` ($p = 0.73$) | Phần dư không có tự tương quan |
| **Jarque-Bera (Skew / Kurtosis)** | `457.53` (`-0.83` / `7.94`) | Độ nhọn cao do cấu trúc dự đoán bậc thang |
| **Within ±$3k USD (Test Set — Cell 60)** | `68.63%` | Thấp hơn `MLP Regression` (`76.47%`) và `Random Forest` (`81.37%`) |

### 4.3. Phân tích Chuyên sâu: Nghịch lý Hold-out Test vs. 5-Fold Cross-Validation
- **Trên Hold-out Test (`102` mẫu)**: `Decision Tree` đạt `Test MSE = 10.5431` và `Test R² = 0.8562` (tốt nhất theo `MSE` và `R²` trên phép chia này, với `Max_Error = 10.6500`), nhưng `MAPE = 13.88%` và `Within ±$3k = 68.63%` lại kém hơn `MLP Regression` (`MAPE = 12.09%`, `Within ±$3k = 76.47%`).
- **Trên 5-Fold Cross-Validation (`506` mẫu — Cell 52)**: Độ bất ổn của một cây đơn lẻ bộc lộ rõ với `CV_MSE = 18.3829 ± 11.6161` và `CV_R² = 0.7832 ± 0.1212` (độ lệch chuẩn lớn gấp `4.2` lần `PyTorch MLP` và `5.8` lần `Random Forest`). Điều này chứng minh `Decision Tree` đơn lẻ rất nhạy với cách chia dữ liệu, và đó là lý do cần kiểm định chéo thay vì chỉ dựa vào một tập Test cố định.
