# Random Forest Regression — Rừng Ngẫu nhiên Hồi quy (Baseline 3)

## 1. Purpose — Mục đích

**Random Forest Regression (`RandomForestRegressor` — Leo Breiman, 2001)** đóng vai trò là **Mô hình Đối chiếu Cơ sở Số 3 (`Baseline Model 3`)** đại diện cho phương pháp Học máy Tổ hợp (*Ensemble Learning — Bootstrap Aggregating / Bagging*) trong dự án Lab 04.

Mục đích của `Random Forest` là khắc phục nhược điểm phương sai cao (*high variance*) và hàm dự đoán bậc thang thô của một cây quyết định `Decision Tree` đơn lẻ bằng cách huấn luyện song song **$B = 100$ cây quyết định độc lập** trên các mẫu Bootstrap khác nhau kết hợp lấy mẫu ngẫu nhiên tập con đặc trưng tại mỗi nút phân nhánh (`max_features = 'sqrt'`), sau đó lấy trung bình cộng dự đoán của cả 100 cây.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train_scaled` `(404, 13)`, `y_train` `(404,)` | Ma trận 13 đặc trưng và vector giá nhà của tập Huấn luyện (`404` mẫu) |
| `X_test_scaled` `(102, 13)`, `y_test` `(102,)` | Ma trận đặc trưng và nhãn giá nhà của tập Kiểm tra độc lập (`102` mẫu) |
| `config.yaml -> baseline_models.random_forest` | Cấu hình tổ hợp: `n_estimators = 100`, `max_depth = 10`, `min_samples_split = 4`, `min_samples_leaf = 2`, `max_features = 'sqrt'`, `random_state = 42` |

### Output

| Output | Description |
|---|---|
| `Predicted MEDV` ($\hat{\mathbf{y}} \in \mathbb{R}^{N}$) | Giá trị nhà dự đoán bằng trung bình cộng dự đoán của 100 cây quyết định |
| `models/random_forest.pkl` | Tệp lưu trữ quần thể `100` cây quyết định (`estimators_`) và vector `feature_importances_` |

### Bảng Mức độ Quan trọng Đặc trưng Trung bình trên 100 Cây (Khớp 100% với Cell 15 `lab04.ipynb`)

| Hạng | Feature | Feature_Importance | Tỷ trọng Đóng góp (%) | Sự khác biệt so với Cây Quyết định Đơn lẻ (`Decision Tree`) |
|---:|---|---:|---:|---|
| 1 | **RM** | **`0.2955`** | `29.55%` | Không bị phụ thuộc độc tôn (62.78% ở DT), chia sẻ vai trò cho các biến khác |
| 2 | **LSTAT** | **`0.2324`** | `23.24%` | Đóng góp ổn định trên toàn bộ 100 cây quyết định |
| 3 | **NOX** | **`0.0888`** | `8.88%` | Khai thác hiệu quả tác động của mức độ ô nhiễm môi trường (ở DT chỉ 0.58%) |
| 4 | **INDUS** | **`0.0754`** | `7.54%` | Khai thác thông tin quy hoạch khu công nghiệp |
| 5 | **PTRATIO** | **`0.0673`** | `6.73%` | Phản ánh chất lượng giáo dục khu vực (tăng từ 1.59% ở DT lên 6.73%) |
| 6 | **CRIM** | **`0.0658`** | `6.58%` | Phản ánh mức độ an ninh đô thị |
| 7 | **DIS** | `0.0588` | `5.88%` | Khoảng cách tới các trung tâm việc làm |
| 8 | **TAX** | `0.0353` | `3.53%` | Thuế suất bất động sản |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Random Forest Regression workflow

```text
          X_train_scaled (404, 13) + y_train (404,)
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
   Bootstrap Sample 1   Bootstrap Sample b   Bootstrap Sample 100
    (404 rows w/ rep)    (404 rows w/ rep)    (404 rows w/ rep)
         |                    |                    |
         v                    v                    v
    Grow Tree T_1(x)     Grow Tree T_b(x)    Grow Tree T_100(x)
   (max_depth = 10,     (max_depth = 10,     (max_depth = 10,
    sqrt(13)=3 feats)    sqrt(13)=3 feats)    sqrt(13)=3 feats)
         |                    |                    |
         +--------------------+--------------------+
                              |
                              v
             Save Ensemble -> models/random_forest.pkl
                              |
                              v
             Aggregate Predictions on Test Set (102):
             y_hat(x) = (1 / 100) * sum_{b=1}^{100} T_b(x)
```

### Các bước thực hiện

**Step 1 — Lấy mẫu Bootstrap (Bootstrap Resampling)**

Từ tập huấn luyện gốc $\mathcal{D}_{\text{train}}$ gồm $N = 404$ mẫu, thuật toán rút ngẫu nhiên có hoàn lại (*sampling with replacement*) $B = 100$ tập dữ liệu con $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_{100}$, mỗi tập có kích thước $404$ mẫu.

**Step 2 — Xây dựng Cây với Không gian Đặc trưng Ngẫu nhiên (Random Subspace Method)**

Trên mỗi tập $\mathcal{D}_b$, xây dựng một cây quyết định hồi quy $T_b(\mathbf{x})$ với độ sâu tối đa `max_depth = 10`. Tại mỗi nút phân nhánh, thay vì xét cả 13 đặc trưng, thuật toán chỉ chọn ngẫu nhiên $m_{\text{try}} = \lfloor\sqrt{13}\rfloor = 3$ đặc trưng để tìm ngưỡng cắt tốt nhất. Cơ chế này làm giảm hệ số tương quan $\rho$ giữa các cây trong rừng.

**Step 3 — Tổ hợp Dự đoán (Ensemble Averaging & Variance Reduction)**

Dự đoán cuối cùng của `Random Forest` là trung bình cộng dự đoán của cả $B = 100$ cây:

$$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})$$

Về mặt toán học, nếu mỗi cây có phương sai $\sigma^2$ và tương quan trung bình giữa hai cây là $\rho$, thì phương sai của rừng giảm xuống còn:

$$\text{Var}\left(\hat{f}_{\text{RF}}(\mathbf{x})\right) = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2$$

Nhờ $B = 100$ lớn và $\rho$ nhỏ (do `max_features = 'sqrt'`), sai số tuyệt đối trung bình (`MAE`) và sai số phần trăm (`MAPE`) giảm mạnh.


## 4. Evaluation metrics
| Metric | Giá trị trên Tập Test (`102` mẫu) | Giá trị trên Tập Train (`404` mẫu) | Nhận xét & Đối chiếu với `MLP Regression` |
| :--- | :---: | :---: | :--- |
| **MSE (↓)** | **`11.5205`** | `4.1200` | Cạnh tranh sát sao với `MLP Regression` (`12.5605` ở cấu hình chuẩn và `10.0503` ở `MLP_Deep`) |
| **RMSE (↓)** | **`$3.3942k`** | `$2.0298k` | Sai số toàn phương trung bình căn bậc hai thấp và ổn định |
| **MAE (↓)** | **`$2.0994k` (`~$2,099 USD` — Best)** | `$1.4162k` | Đạt sai số tuyệt đối trung bình thấp nhất trên tập Hold-out Test (kế tiếp là MLP với `$2.2868k`) |
| **MAPE (↓)** | **`10.8235%` (`10.82%` — Best)** | `6.6300%` | Đạt sai số phần trăm trung bình thấp nhất trên tập Hold-out Test (kế tiếp là MLP với `11.71%`) |
| **$R^2$ Score (↑)** | **`0.8429` (`84.29%`)** | `0.9526` | Giải thích `84.29%` phương sai giá nhà |
| **5-Fold Cross-Validation** | `CV MSE = 12.5518 ± 1.9864` | `CV R² = 0.8459 ± 0.0395` | Đứng Hạng #2 trên 5-Fold CV, sau **`MLP Regression` (`CV R² = 0.8607 ± 0.0459`)** |
| **Within $\pm\$3,000	ext{ USD}$ (↑)** | **`81.37%` (Best)** | — | Tỷ lệ dự đoán chính xác trong biên độ $\pm\$3,000$ đạt `81.37%` |
| **Training Time** | `0.1289s` | — | Nhanh hơn mạng MLP nhưng chậm hơn `Linear` và `Decision Tree` khoảng 30–80 lần |

## 5. Used data
| Tập dữ liệu | Số mẫu | Kích thước | Vai trò đối với `Random Forest Regression` |
| :--- | :---: | :---: | :--- |
| `X_train_scaled`, `y_train` | `404` | `(404, 13)` | Rút 100 mẫu Bootstrap để huấn luyện 100 cây quyết định độc lập và đánh giá 5-Fold CV |
| `X_test_scaled`, `y_test` | `102` | `(102, 13)` | Đánh giá độc lập 5 chỉ số hồi quy và phân tích phần dư trên dữ liệu kiểm thử |

## Code References

| File | Responsibility |
|---|---|
| `src/model.py` (`build_baseline_models`, `RandomForestRegressorCompat`) | Khởi tạo `RandomForestRegressor` với `n_estimators = 100` từ `config.yaml` |
| `src/train.py` | Huấn luyện rừng ngẫu nhiên và lưu `models/random_forest.pkl` |
| `src/evaluate.py` | Đánh giá 5 chỉ số của `Random Forest` trên tập Train và tập Test |
| `notebooks/lab04.ipynb` (Mục 15, 16, 17, 19) | Trích xuất `feature_importances_` của 100 cây và đối sánh với `MLP Regression` |
