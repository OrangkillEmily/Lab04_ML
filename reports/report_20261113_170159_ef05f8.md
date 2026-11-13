# BÁO CÁO THỰC HÀNH LAB 04: DỰ ĐOÁN GIÁ NHÀ VỚI MẠNG NƠ-RON ĐA TẦNG (MULTILAYER PERCEPTRON REGRESSION)

- **Học phần:** Học máy (Machine Learning) – Trường Đại học Giao thông Vận tải TP.HCM (UTH)
- **Chủ đề:** Chương 5 – Mạng Nơ-ron Nhân tạo (Artificial Neural Networks – Practice Exercise 4: Predicting House Prices)
- **Phiên thực nghiệm (Run ID):** `20261113_170159_ef05f8` (Trạng thái: **SUCCESS**)
- **Bộ dữ liệu chuẩn:** Boston Housing Dataset (`data/raw/boston_housing.csv` – 506 mẫu, 13 đặc trưng, biến mục tiêu `MEDV` đơn vị `$1,000 USD`)

## 1. Mục tiêu bài thực hành
1. **Nắm vững cơ sở lý thuyết Chương 5** về Mạng Nơ-ron Nhân tạo (Artificial Neural Networks), nguyên lý hoạt động của nơ-ron toán học, cơ chế lan truyền thuận (Forward Propagation), hàm kích hoạt (`ReLU`, `Linear`) và thuật toán lan truyền ngược (Backpropagation) kết hợp bộ tối ưu `Adam`.
2. **Thiết kế và triển khai pipeline học máy hoàn chỉnh** cho bài toán hồi quy dự đoán giá nhà theo nguyên tắc **Clean Architecture** (tách biệt rõ cấu hình `config.yaml`, dữ liệu `data/`, mô-đun lõi `src/`, mô hình `models/` và hệ thống báo cáo `reports/`).
3. **Đảm bảo tính toàn vẹn dữ liệu và chống rò rỉ dữ liệu (Anti-Data-Leakage):** Nạp trực tiếp tệp dữ liệu chuẩn `data/raw/boston_housing.csv` (thay thế hoàn toàn hàm `load_boston` đã bị loại bỏ khỏi `scikit-learn >= 1.2`), chỉ `fit` bộ chuẩn hóa `StandardScaler` trên tập `Train` (`404` mẫu) và tách tập `Validation` (`81` mẫu) độc lập hoàn toàn với tập `Test` (`102` mẫu).
4. **Đối sánh thực nghiệm đa mô hình trên 5 chỉ số chuẩn:** Huấn luyện mạng `MLP Regression` (`13 → 32 → 16 → 1`, `993` tham số) và đặt lên bàn cân so sánh với `Linear Regression`, `Decision Tree` và `Random Forest` thông qua bộ 5 thước đo định lượng: **MSE**, **RMSE**, **MAE**, **MAPE (%)** và **$R^2$ Score**.

## 2. Bảng lộ trình thực thi chuẩn hóa (Execution Roadmap)

| Step | Giai đoạn / Tên mô-đun | Chức năng (What it does) | Đường dẫn Import (Import path) |
| :---: | :--- | :--- | :--- |
| **1** | **Setup & Run Tracking** | Nạp `config.yaml`, cố định `random_state = 42`, khởi tạo `Run_ID`, Logger và tiến độ JSON. | `src/utils.py`, `src/tracking.py` |
| **2** | **Direct Data Loader** | Đọc trực tiếp `data/raw/boston_housing.csv`, kiểm định 506 dòng × 14 cột, tách `X` (13 biến) và `y` (`MEDV`). | `src/data_loader.py` |
| **3** | **Exploratory Data Analysis** | Tính thống kê mô tả, ma trận tương quan Pearson, phát hiện ngưỡng kịch trần `MEDV = 50.0` và vẽ 4 biểu đồ EDA. | `src/eda.py` |
| **4** | **Leakage-Free Preprocessing** | Chia `Train` (404) / `Test` (102), `fit` `StandardScaler` chỉ trên `Train`, tách `Train_sub` (323) và `Validation` (81). | `src/preprocess.py` |
| **5** | **MLP & Baseline Builder** | Khởi tạo kiến trúc mạng `Sequential_MLP_Regressor` (`13-32-16-1`) và 3 mô hình Baseline chuẩn hóa. | `src/model.py` |
| **6** | **Model Training & EarlyStop** | Huấn luyện MLP với `Adam (lr=0.005)`, `batch_size=32`, `100` epochs kèm `EarlyStopping` và lưu trọng số tốt nhất. | `src/train.py` |
| **7** | **5-Metric Evaluator & Plots** | Tính toán `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` trên tập Train/Test và kết xuất 16 biểu đồ vào `reports/figures/`. | `src/evaluate.py`, `src/visualize.py` |
| **8** | **Automated Report Engine** | Tổng hợp số liệu thực nghiệm, sinh báo cáo Markdown và báo cáo HTML tương tác tự chứa (`LAB_04_REPORT.html`). | `src/report.py`, `render_report.py` |

## 3. Kiến trúc hệ thống (System Architecture)

```mermaid
graph LR
    subgraph Data Governance & Anti-Leakage Preprocessing
        D1[boston_housing.csv 506x14] --> D2[Data Integrity Audit]
        D2 --> D3[EDA & Correlation Analysis]
        D2 --> D4[Train/Test Split 80/20]
        D4 -->|Train: 404 mẫu| D5[StandardScaler.fit_transform]
        D4 -->|Test: 102 mẫu| D6[StandardScaler.transform ONLY]
        D5 --> D7[Train_sub: 323 mẫu]
        D5 --> D8[Validation: 81 mẫu]
    end

    subgraph Model Training, 5-Metric Evaluation & Reporting
        D7 --> M1[MLP Regressor 13-32-16-1]
        D8 -->|Monitor val_loss| M1
        D5 --> M2[3 Baselines: Linear / DT / RF]
        M1 --> E1[5-Metric Evaluator: MSE, RMSE, MAE, MAPE, R2]
        M2 --> E1
        D6 --> E1
        E1 --> R1[Single Figure Hub: reports/figures/]
        R1 --> R2[Interactive HTML & Markdown Reports]
    end
```

## 4. Cơ sở lý thuyết & Phân tích toán học

### 4.1. Lan truyền thuận (Forward Propagation) và Kiến trúc Mạng MLP `13-32-16-1`
Mạng **Multilayer Perceptron (MLP)** xếp chồng hai tầng ẩn kết nối đầy đủ (*Dense Layers*) sử dụng hàm kích hoạt phi tuyến **ReLU** ($\text{ReLU}(z) = \max(0, z)$) và tầng đầu ra dùng hàm **Linear** ($f(z) = z$):

$$\mathbf{z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \qquad \mathbf{a}^{(1)} = \text{ReLU}\bigl(\mathbf{z}^{(1)}\bigr) \in \mathbb{R}^{N \times 32}$$
$$\mathbf{z}^{(2)} = \mathbf{a}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}, \qquad \mathbf{a}^{(2)} = \text{ReLU}\bigl(\mathbf{z}^{(2)}\bigr) \in \mathbb{R}^{N \times 16}$$
$$\hat{\mathbf{y}} = \mathbf{z}^{(3)} = \mathbf{a}^{(2)}\mathbf{W}^{(3)} + \mathbf{b}^{(3)} \in \mathbb{R}^{N \times 1}$$

- **Tổng số tham số huấn luyện:** $(13 \times 32 + 32) + (32 \times 16 + 16) + (16 \times 1 + 1) = 448 + 528 + 17 = 993$ tham số.

### 4.2. Chuẩn hóa Z-Score Chống Data Leakage & Hệ thống 5 Chỉ số Đánh giá
$$\mathbf{x}_{\text{train}}^{\text{scaled}} = \frac{\mathbf{x}_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \qquad \mathbf{x}_{\text{test}}^{\text{scaled}} = \frac{\mathbf{x}_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}, \qquad \text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$
$$\text{MAPE (\%)} = \frac{100\%}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|, \qquad R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}$$

## 5. Nội dung chi tiết và Kết quả thực nghiệm

### 5.1. Quản lý dữ liệu chuẩn & Phân tích khám phá (Data Loading & EDA)
- Bộ dữ liệu `data/raw/boston_housing.csv` gồm `506` mẫu × `14` cột, `0` giá trị thiếu, `0` bản ghi trùng lặp.
- Biến mục tiêu `MEDV` có trung bình `22.5328` (`$22,533 USD`), độ lệch chuẩn `9.1971`, trung vị `21.2000` và `16` mẫu chạm ngưỡng trần `50.0`.
- Đặc trưng tương quan âm mạnh nhất là `LSTAT` ($r = -0.7377$) và tương quan dương mạnh nhất là `RM` ($r = +0.6954$).

### 5.2. Tiền xử lý nghiêm ngặt chống Data Leakage (Leakage-Free Preprocessing)
- Tập `Train` (`404` mẫu) và `Test` (`102` mẫu) được tách độc lập với `random_state = 42`.
- Bộ `StandardScaler` chỉ `fit` trên `404` mẫu `Train` (`leakage_free = True`), sau đó tách `X_train_sub` (`323` mẫu) và `X_val_scaled` (`81` mẫu).

### 5.3. Huấn luyện MLP Regression & Hội tụ EarlyStopping
- Đã chạy `85` epochs (`batch_size = 32`, `Adam lr = 0.005`).
- **Train MSE (Loss):** Giảm từ `590.3776` (Epoch 1) xuống `4.6031` (Epoch 85).
- **Epoch kiểm định tốt nhất (Best Validation Epoch):** Epoch `70` (`Best Val MSE = 14.0107`, `Best Val MAE = 2.7173`).
- **Nhận xét Train vs Validation:** Tại Epoch 70, Train MSE = 5.2718 còn Val MSE = 14.0107 (chênh lệch 2.7 lần) — có khoảng cách tổng quát hóa giữa Train và Validation; EarlyStopping giữ lại trọng số tại cực tiểu kiểm định (Epoch 70).

### 5.4. Đối sánh thực nghiệm 4 mô hình trên Tập Kiểm tra (`Test Set` — 102 mẫu)

| Mô hình (Model) | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test R² (↑) | Train MSE | Train MAPE | Train R² | Thời gian Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Regression** | 11.5628 | 3.4004 | 2.3160 | 12.22% | 0.8423 | 6.8245 | 9.62% | 0.9214 | 15.7892s |
| **Linear Regression** | 24.2911 | 4.9286 | 3.1891 | 16.87% | 0.6688 | 21.6414 | 16.57% | 0.7509 | 0.0083s |
| **Decision Tree** | 10.5431 | 3.2470 | 2.5192 | 13.88% | 0.8562 | 6.6944 | 9.60% | 0.9229 | 0.0071s |
| **Random Forest** | 11.5205 | 3.3942 | 2.0994 | 10.82% | 0.8429 | 4.1200 | 6.63% | 0.9526 | 0.1577s |

- **Mô hình tốt nhất theo Test R²:** `Decision Tree` (`R² = 0.8562`, `MSE = 10.5431`, `MAPE = 13.88%`).

### 5.5. Danh mục Biểu đồ Thực nghiệm Hợp nhất (`reports/figures/`)
- Phân phối biến mục tiêu & Tương quan: `figures/target_distribution.png`, `figures/correlation_heatmap.png`, `figures/feature_distributions.png`, `figures/feature_target_relationships.png`
- Huấn luyện MLP & Phân chia dữ liệu: `figures/train_val_test_split.png`, `figures/mlp_training_loss_curve.png`, `figures/mlp_train_vs_val_loss.png`, `figures/training_time_comparison.png`
- Đánh giá & Đối sánh 4 mô hình: `figures/actual_vs_predicted_mlp.png`, `figures/actual_vs_predicted_all.png`, `figures/residual_plot_all.png`, `figures/residual_distribution_all.png`, `figures/model_comparison_mse.png`, `figures/model_comparison_mae.png`, `figures/model_comparison_mape.png`, `figures/model_comparison_r2.png`

## 6. Kiểm thử và kiểm toán chất lượng (Quality Assurance)
1. **Clean Architecture & Chuẩn Docstring 5W1H:** Toàn bộ mô-đun trong `src/` tuân thủ PEP-8 và tách biệt cấu hình vào `config.yaml`.
2. **Kiểm định Chống Rò rỉ Dữ liệu (`Anti-Data-Leakage Verification`):** Cờ `leakage_free = True` xác nhận `StandardScaler` chỉ học trên `404` mẫu `Train`.
3. **Kiến trúc Lưu trữ Biểu đồ Đơn nhất (`Single Source of Truth Figures`):** 16 biểu đồ (`300 DPI`) lưu đồng thời tại `reports/figures/` và `reports/figures/<run_id>/`.
4. **Kiểm thử Thực thi Tự động (`End-to-End Pipeline Audit`):** Chạy `python run_lab04.py` và `render_report.py` đạt trạng thái `SUCCESS` 100%.

## 7. Kết luận
Hệ thống Lab 04 đã hoàn thành các mục tiêu của Chương 5. Sự kết hợp giữa chuẩn hóa Z-Score chống rò rỉ dữ liệu, kiến trúc MLP `13-32-16-1` với hàm kích hoạt `ReLU`/`Linear`, thuật toán tối ưu `Adam` và `EarlyStopping` đạt `Test R² = 0.8423`, `Test MAPE = 12.22%`; MLP không vượt đồng thời hồi quy tuyến tính và cây quyết định trên cả 5 chỉ số, xem bảng 5.4 để đối chiếu.

## 8. Tài liệu tham khảo
1. *Bài giảng Chương 5: Introduction to Artificial Neural Networks (ANNs)* & *Practice Exercise 4*, Trường Đại học Giao thông Vận tải TP.HCM (UTH).
2. Harrison, D., & Rubinfeld, D. L. (1978). *Hedonic housing prices and the demand for clean air*. Journal of Environmental Economics and Management, 5(1), 81–102.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
