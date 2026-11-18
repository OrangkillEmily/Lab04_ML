# Decision Tree Regression — Cây Quyết định Hồi quy (Baseline 2)

## 1. Purpose — Mục đích

**Decision Tree Regression (`DecisionTreeRegressor` — Thuật toán CART)** đóng vai trò là **Mô hình Đối chiếu Cơ sở Số 2 (`Baseline Model 2`)** trong dự án Lab 04.

Khác với `Linear Regression` (dùng siêu phẳng toàn cục) và `MLP Regression` (dùng mặt cong trơn tạo bởi các tầng ẩn `ReLU`), `Decision Tree Regression` tiếp cận bài toán dự đoán giá nhà bằng cách **phân hoạch đệ quy không gian 13 đặc trưng thành các hình hộp chữ nhật rời rạc** (axis-aligned regions) và dự đoán giá nhà tại mỗi nút lá bằng giá trị trung bình của các mẫu rơi vào vùng đó.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Ma trận 13 đặc trưng và vector giá nhà của tập Huấn luyện |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Ma trận đặc trưng và nhãn giá nhà của tập Kiểm tra độc lập |
| `config.yaml -> baseline_models.decision_tree` | Siêu tham số kiểm soát độ sâu: `max_depth = 6`, `min_samples_split = 5`, `min_samples_leaf = 2`, `random_state = 42` |

### Output

| Output | Description |
|---|---|
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá trị nhà dự đoán theo hàm bậc thang từng khúc tại các nút lá |
| `models/decision_tree.pkl` | Tệp lưu trữ cấu trúc cây nhị phân đã học và vector mức độ quan trọng đặc trưng `feature_importances_` |

### Bảng Mức độ Quan trọng Đặc trưng (`feature_importances_` — Khớp 100% với Cell 14 `lab04.ipynb`)

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
        Assign Leaf Value = Mean(y_i) in Region R_m
                          |
                          v
        Save Model -> models/decision_tree.pkl
                          |
                          v
        Route Test Samples (102) Down the Tree -> Predict y_hat
```

### Các bước thực hiện

**Step 1 — Tiêu chuẩn Phân nhánh Giảm Phương sai (CART Variance / MSE Reduction)**

Tại mỗi nút cha $P$ chứa tập mẫu $S_P$, thuật toán duyệt qua từng đặc trưng $j \in \{1, \dots, 13\}$ và điểm cắt $t$ để chia $S_P$ thành nhánh trái $S_L = \{i \in S_P : x_{i,j} \le t\}$ và nhánh phải $S_R = \{i \in S_P : x_{i,j} > t\}$ sao cho tổng bình phương sai số sau khi chia giảm nhiều nhất:

$$\Delta \text{MSE}(j, t) = \text{MSE}(S_P) - \left(\frac{|S_L|}{|S_P|}\text{MSE}(S_L) + \frac{|S_R|}{|S_P|}\text{MSE}(S_R)\right)$$

trong đó $\text{MSE}(S) = \frac{1}{|S|}\sum_{i \in S}(y_i - \bar{y}_S)^2$.

**Step 2 — Điều kiện Dừng & Cắt tỉa Trước (Pre-pruning Hyperparameters)**

Để ngăn cây học thuộc lòng nhiễu (Overfitting), quá trình chia nhánh dừng lại khi chạm một trong các ngưỡng từ `config.yaml`: chiều sâu đạt `max_depth = 6`, số mẫu tại nút nhỏ hơn `min_samples_split = 5`, hoặc nhánh con tạo ra có ít hơn `min_samples_leaf = 2` mẫu.

**Step 3 — Dự đoán tại Nút Lá (Leaf Prediction)**

Khi toàn bộ không gian được chia thành $M$ vùng lá $R_1, R_2, \dots, R_M$, giá trị dự đoán cho bất kỳ mẫu $\mathbf{x}$ nào rơi vào vùng $R_m$ là trung bình cộng nhãn của các mẫu huấn luyện trong vùng đó:

$$\hat{f}(\mathbf{x}) = \sum_{m=1}^{M} c_m \mathbb{I}(\mathbf{x} \in R_m), \qquad c_m = \frac{1}{|R_m|}\sum_{i \in R_m} y_i$$

## 4. Evaluation metrics
| Metric | Giá trị trên Tập Test (`102` mẫu) | Giá trị trên Tập Train (`404` mẫu) | Nhận xét & Đối chiếu với `MLP Regression` |
| :--- | :---: | :---: | :--- |
| **MSE (↓)** | **`10.5431` (Best Hold-out)** | `6.6944` | Đạt MSE thấp nhất trên phép chia Hold-out `random_state=42` nhờ kiểm soát tốt `Max_Error` (`10.65`) |
| **RMSE (↓)** | **`$3.2470k` (Best Hold-out)** | `$2.5874k` | Nhỉnh hơn trên 102 mẫu Test nhưng kém ổn định khi đổi tập chia |
| **MAE (↓)** | **`$2.5192k`** | `$1.9563k` | **Kém hơn `MLP Regression` (`$2.2868k`)** và `Random Forest` (`$2.0994k`) do dự đoán bậc thang rời rạc |
| **MAPE (↓)** | **`13.8809%` (`13.88%`)** | `9.6020%` | **Kém hơn `MLP Regression` (`11.71%`)** tới `2.17` điểm phần trăm và kém hơn `Random Forest` (`10.82%`) |
| **$R^2$ Score (↑)** | **`0.8562` (`85.62%`)** | `0.9229` | Giải thích `85.62%` phương sai trên tập Hold-out Test |
| **5-Fold Cross-Validation** | `CV MSE = 18.3829 ± 11.6161` | `CV R² = 0.7832 ± 0.1212` | **Phương sai rất lớn Across Folds** — kém xa `MLP Regression` (`CV R² = 0.8607 ± 0.0459`) |
| **Within $\pm\$3,000	ext{ USD}$ (↑)** | **`68.63%`** | — | **Thấp hơn `MLP Regression` (`75.49%`)** và `Random Forest` (`81.37%`) |
| **TrainingTime** | **`0.0047s`** | — | Tốc độ huấn luyện rất nhanh trên dữ liệu bảng nhỏ |

> **Phân tích học thuật quan trọng:** Trên phép chia Hold-out đơn lẻ (`random_state=42`), `Decision Tree` có `MSE` thấp (`10.5431`) nhưng `MAE` (`2.5192`) và `MAPE` (`13.88%`) đều kém hơn `MLP Regression` (`MAE = 2.2868`, `MAPE = 11.71%`). Đặc biệt, khi kiểm định bằng **5-Fold Cross-Validation**, `Decision Tree` tụt xuống hạng 3 với `CV MSE = 18.3829 ± 11.6161` (`CV R² = 0.7832`), chứng minh một cây đơn lẻ rất nhạy với cách chia dữ liệu, trong khi **`MLP Regression` ổn định và dẫn đầu 5-Fold CV (`CV R² = 0.8607 ± 0.0459`)**.

## 5. Used data
| Tập dữ liệu | Số mẫu | Kích thước | Vai trò đối với `Decision Tree Regression` |
| :--- | :---: | :---: | :--- |
| `X_train_scaled`, `y_train` | `404` | `(404, 13)` | Tìm các ngưỡng phân nhánh $(j, t)$, tính giá trị trung bình tại `34` nút lá và đánh giá 5-Fold CV |
| `X_test_scaled`, `y_test` | `102` | `(102, 13)` | Đánh giá độc lập 5 chỉ số hồi quy trên dữ liệu chưa từng thấy |

## Code References

| File | Responsibility |
|---|---|
| `src/model.py` (`build_baseline_models`, `DecisionTreeRegressorCompat`) | Khởi tạo `DecisionTreeRegressor` với cấu hình từ `config.yaml` |
| `src/train.py` | Huấn luyện cây quyết định và lưu `models/decision_tree.pkl` |
| `src/evaluate.py` | Đánh giá 5 chỉ số của `Decision Tree` trên tập Train và tập Test |
| `notebooks/lab04.ipynb` (Mục 14, 16, 17, 19) | Trích xuất `feature_importances_` và đối sánh chi tiết với `MLP Regression` |
