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

## 4. Evaluation Metrics — Các chỉ số đánh giá thực nghiệm

Kết quả đánh giá chính thức của `Decision Tree Regression` trên tập Kiểm tra (`102` mẫu — khớp 100% với **Cells 14, 16, 17 & 19 của `lab04.ipynb`**):

| Metric | Giá trị trên Tập Test (`102` mẫu) | Giá trị trên Tập Train (`404` mẫu) | Nhận xét & Đối chiếu với `MLP Regression` |
|---|---:|---:|---|
| **MSE (↓)** | **`10.5431` (Best)** | `5.4820` | Đạt MSE thấp nhất nhờ bắt trúng các ngưỡng nhảy giá theo `RM` và `LSTAT`, hạn chế được sai số cực đại |
| **RMSE (↓)** | **`$3.2470k` (Best)** | `$2.3414k` | Nhỉnh hơn một chút so với `Random Forest` (`$3.3942k`) và `MLP Regression` (`$3.4004k`) |
| **MAE (↓)** | **`$2.5192k`** | `$1.7125k` | **Kém hơn `MLP Regression` (`$2.3160k`)** và `Random Forest` (`$2.0994k`) do dự đoán bậc thang rời rạc |
| **MAPE (↓)** | **`13.8809%` (`13.88%`)** | `8.6412%` | **Kém hơn `MLP Regression` (`12.22%`)** tới `1.66` điểm phần trăm và kém hơn `Random Forest` (`10.82%`) |
| **$R^2$ Score (↑)** | **`0.8562` (`85.62%` — Best)** | `0.9369` | Giải thích được `85.62%` phương sai trên tập Test |
| **Max_Error (↓)** | **`$10.6500k` (Best)** | `$10.2500k` | Sai số cực đại thấp nhất trong 4 mô hình |
| **Within $\pm\$3,000\text{ USD}$ (↑)** | **`68.63%`** | — | **Thấp hơn `MLP Regression` (`72.55%`)** và `Random Forest` (`81.37%`) |
| **Training / Prediction Time** | **`0.0071s` (Best)** / `0.000574s` | — | Huấn luyện nhanh nhất trong cả 4 mô hình |

> **Phân tích học thuật quan trọng (Cell 14 & Cell 17 `lab04.ipynb`):** Tại sao `Decision Tree` có `MSE` thấp nhất (`10.5431`) nhưng `MAE` (`2.5192`) và `MAPE` (`13.88%`) lại kém hơn `MLP Regression` (`MAE = 2.3160`, `MAPE = 12.22%`)? Vì `MSE` bình phương sai số nên rất nhạy với 1–2 điểm ngoại lai (`Max_Error` của Decision Tree chỉ là `10.65`, trong khi các mô hình khác có 1 căn nhà dị biệt lệch ~`20k`). Ngược lại, trên đa số các căn nhà thông thường trong tập Test, bề mặt dự đoán liên tục và trơn tru của **`MLP Regression` bám sát giá thực tế hơn**, giúp `MLP` thắng `Decision Tree` cả về `MAE`, `MAPE (%)` lẫn tỷ lệ dự đoán trong biên độ $\pm\$3,000\text{ USD}$ (`72.55%` vs `68.63%`).

## 5. Used Data — Dữ liệu sử dụng

| Tập dữ liệu | Số mẫu | Kích thước | Vai trò đối với `Decision Tree Regression` |
|---|---:|---|---|
| `X_train_scaled`, `y_train` | `404` | `(404, 13)` | Dùng để tìm các ngưỡng phân nhánh $(j, t)$ và tính giá trị trung bình tại các nút lá |
| `X_test_scaled`, `y_test` | `102` | `(102, 13)` | Dùng để đánh giá độc lập 5 chỉ số hồi quy trên dữ liệu chưa từng thấy |

## Code References

| File | Responsibility |
|---|---|
| `src/model.py` (`build_baseline_models`, `DecisionTreeRegressorCompat`) | Khởi tạo `DecisionTreeRegressor` với cấu hình từ `config.yaml` |
| `src/train.py` | Huấn luyện cây quyết định và lưu `models/decision_tree.pkl` |
| `src/evaluate.py` | Đánh giá 5 chỉ số của `Decision Tree` trên tập Train và tập Test |
| `notebooks/lab04.ipynb` (Mục 14, 16, 17, 19) | Trích xuất `feature_importances_` và đối sánh chi tiết với `MLP Regression` |
