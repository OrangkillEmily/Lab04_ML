# BÁO CÁO THỰC HÀNH LAB 04: DỰ ĐOÁN GIÁ NHÀ VỚI MẠNG NƠ-RON ĐA TẦNG (MULTILAYER PERCEPTRON REGRESSION)

- **Học phần:** Học máy (Machine Learning) – Trường Đại học Giao thông Vận tải TP.HCM (UTH)
- **Chủ đề:** Chương 5 – Mạng Nơ-ron Nhân tạo (Artificial Neural Networks – Practice Exercise 4: Predicting House Prices)
- **Phiên thực nghiệm (Run ID):** `20261119_001229_43dcff` (Trạng thái: **RUNNING**)
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
- Đã chạy `100` epochs (`batch_size = 32`, `Adam lr = 0.005`).
- **Train MSE (Loss):** Giảm từ `578.2240` (Epoch 1) xuống `12.1656` (Epoch 100).
- **Epoch kiểm định tốt nhất (Best Validation Epoch):** Epoch `91` (`Best Val MSE = 14.0724`, `Best Val MAE = 2.6805`).
- **Nhận xét Train vs Validation:** Hai đường Train và Validation bám sát nhau (tại Epoch 91: Train MSE = 11.7010, Val MSE = 14.0724), không có dấu hiệu Overfitting rõ rệt.

### 5.4. Đối sánh thực nghiệm 4 mô hình trên Tập Kiểm tra (`Test Set` — 102 mẫu)

| Mô hình (Model) | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test R² (↑) | Train MSE | Train MAPE | Train R² | Thời gian Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Regression** | 12.5605 | 3.5441 | 2.2868 | 11.71% | 0.8287 | 7.1522 | 9.53% | 0.9177 | 14.3336s |
| **Linear Regression** | 24.2911 | 4.9286 | 3.1891 | 16.87% | 0.6688 | 21.6414 | 16.57% | 0.7509 | 0.0015s |
| **Decision Tree** | 10.5431 | 3.2470 | 2.5192 | 13.88% | 0.8562 | 6.6944 | 9.60% | 0.9229 | 0.0037s |
| **Random Forest** | 11.5205 | 3.3942 | 2.0994 | 10.82% | 0.8429 | 4.1200 | 6.63% | 0.9526 | 0.1361s |

- **Mô hình tốt nhất theo Test R²:** `Decision Tree` (`R² = 0.8562`, `MSE = 10.5431`, `MAPE = 13.88%`).
- **Regularization MLP:** `l2_lambda = 0.001`, `dropout_rate = 0.05`.

#### Kiểm toán Giá trị Ngoại lai (EDA Outlier Analysis)

| Feature | Q1 | Q3 | IQR | Lower_Bound | Upper_Bound | IQR_Outlier_Count | IQR_Outlier_Pct | ZScore_gt3_Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CRIM | 0.082 | 3.6771 | 3.595 | -5.3105 | 9.0696 | 66 | 13.04 | 8 |
| ZN | 0.0 | 12.5 | 12.5 | -18.75 | 31.25 | 68 | 13.44 | 14 |
| INDUS | 5.19 | 18.1 | 12.91 | -14.175 | 37.465 | 0 | 0.0 | 0 |
| CHAS | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 35 | 6.92 | 35 |
| NOX | 0.449 | 0.624 | 0.175 | 0.1865 | 0.8865 | 0 | 0.0 | 0 |
| RM | 5.8855 | 6.6235 | 0.738 | 4.7785 | 7.7305 | 30 | 5.93 | 8 |
| AGE | 45.025 | 94.075 | 49.05 | -28.55 | 167.65 | 0 | 0.0 | 0 |
| DIS | 2.1002 | 5.1884 | 3.0883 | -2.5322 | 9.8208 | 5 | 0.99 | 5 |
| RAD | 4.0 | 24.0 | 20.0 | -26.0 | 54.0 | 0 | 0.0 | 0 |
| TAX | 279.0 | 666.0 | 387.0 | -301.5 | 1246.5 | 0 | 0.0 | 0 |
| PTRATIO | 17.4 | 20.2 | 2.8 | 13.2 | 24.4 | 15 | 2.96 | 0 |
| B | 375.3775 | 396.225 | 20.8475 | 344.1062 | 427.4963 | 77 | 15.22 | 25 |
| LSTAT | 6.95 | 16.955 | 10.005 | -8.0575 | 31.9625 | 7 | 1.38 | 5 |
| MEDV | 17.025 | 25.0 | 7.975 | 5.0625 | 36.9625 | 40 | 7.91 | 0 |

#### Kết quả 5-Fold Cross-Validation

| Model | K_Folds | CV_MSE_Mean | CV_MSE_Std | CV_RMSE_Mean | CV_MAE_Mean | CV_MAPE_Mean | CV_R2_Mean | CV_R2_Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MLP Regression | 5 | 11.47 | 3.8966 | 3.3514 | 2.3577 | 12.005 | 0.8607 | 0.0459 |
| Random Forest | 5 | 12.5518 | 1.9864 | 3.5339 | 2.3361 | 11.6827 | 0.8459 | 0.0395 |
| Decision Tree | 5 | 18.3829 | 11.6161 | 4.1529 | 2.8782 | 14.5435 | 0.7832 | 0.1212 |
| Linear Regression | 5 | 23.4886 | 2.0601 | 4.8428 | 3.3885 | 17.0105 | 0.7152 | 0.0419 |

#### Đối sánh Tuning Kiến trúc MLP

| Architecture | Hidden_Units | Total_Params | Train_MSE | Val_MSE | Test_MSE | Test_RMSE | Test_MAE | Test_MAPE | Test_R2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MLP_Small (13->16->8->1) | [16, 8] | 369 | 7.0677 | 16.7186 | 12.0885 | 3.4769 | 2.3476 | 12.4294 | 0.8352 |
| MLP_Baseline (13->32->16->1) | [32, 16] | 993 | 5.019 | 12.3076 | 11.6812 | 3.4178 | 2.3684 | 12.5317 | 0.8407 |
| MLP_Wide (13->64->32->1) | [64, 32] | 3009 | 6.0023 | 13.6515 | 12.2085 | 3.4941 | 2.3181 | 12.0351 | 0.8335 |
| MLP_Deep (13->64->32->16->1) | [64, 32, 16] | 3521 | 3.1183 | 11.904 | 10.0503 | 3.1702 | 2.1584 | 10.5635 | 0.863 |

#### Chẩn đoán Thống kê (Model Summary)

| Run_ID | Model | No_Observations_Train | No_Parameters | Log_Likelihood | AIC | BIC | HQIC | F_Statistic | Prob_F | Train_MSE | Train_RMSE | Train_MAE | Train_MAPE | Train_R2 | Adj_R2 | Test_MSE | Test_RMSE | Test_MAE | Test_MAPE | Test_R2 | Durbin_Watson | Ljung_Box_Q | Prob_Q | Jarque_Bera_JB | Prob_JB | Heteroskedasticity_H | Prob_H | Residual_Skew | Residual_Kurtosis | Residual_Variance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 20261119_001229_43dcff | MLP Regression | 404 | 993 | -970.6703 | 1969.3405 | 2025.3603 | 1991.5164 | 334.3912 | 0.0 | 7.1522 | 2.6744 | 1.9519 | 9.5258 | 0.9177 | 0.9149 | 12.5605 | 3.5441 | 2.2868 | 11.7096 | 0.8287 | 2.054 | 0.331 | 0.5651 | 482.6773 | 0.0 | 1.1294 | 0.3891 | 1.0384 | 7.9357 | 7.409 |
| 20261119_001229_43dcff | Linear Regression | 404 | 14 | -1194.3221 | 2416.6443 | 2472.6641 | 2438.8201 | 90.4266 | 0.0 | 21.6414 | 4.652 | 3.3148 | 16.5749 | 0.7509 | 0.7426 | 24.2911 | 4.9286 | 3.1891 | 16.8664 | 0.6688 | 2.1141 | 1.3288 | 0.249 | 579.8166 | 0.0 | 0.9651 | 0.8013 | 1.3788 | 8.1807 | 22.4183 |
| 20261119_001229_43dcff | Decision Tree | 404 | 14 | -957.3095 | 1942.6189 | 1998.6387 | 1964.7948 | 359.3079 | 0.0 | 6.6944 | 2.5874 | 1.8806 | 9.604 | 0.9229 | 0.9204 | 10.5431 | 3.247 | 2.5192 | 13.8809 | 0.8562 | 1.9605 | 0.1175 | 0.7318 | 457.5302 | 0.0 | 0.9124 | 0.5166 | -0.8344 | 7.9391 | 6.9348 |
| 20261119_001229_43dcff | Random Forest | 404 | 14 | -859.2524 | 1746.5047 | 1802.5245 | 1768.6806 | 602.5769 | 0.0 | 4.12 | 2.0298 | 1.3261 | 6.6295 | 0.9526 | 0.951 | 11.5205 | 3.3942 | 2.0994 | 10.8235 | 0.8429 | 2.0693 | 0.5048 | 0.4774 | 2170.2623 | 0.0 | 0.9785 | 0.8775 | 1.864 | 13.7251 | 4.2679 |

### 5.5. Danh mục Biểu đồ Thực nghiệm Hợp nhất (`reports/figures/`)
- Phân phối biến mục tiêu & Tương quan: `../reports/figures/target_distribution.png`, `../reports/figures/correlation_heatmap.png`, `../reports/figures/feature_distributions.png`, `../reports/figures/feature_target_relationships.png`
- Huấn luyện MLP & Phân chia dữ liệu: `../reports/figures/train_val_test_split.png`, `../reports/figures/mlp_training_loss_curve.png`, `../reports/figures/mlp_train_vs_val_loss.png`, `../reports/figures/training_time_comparison.png`
- Đánh giá & Đối sánh 4 mô hình: `../reports/figures/actual_vs_predicted_mlp.png`, `../reports/figures/actual_vs_predicted_all.png`, `../reports/figures/residual_plot_all.png`, `../reports/figures/residual_distribution_all.png`, `../reports/figures/model_comparison_mse.png`, `../reports/figures/model_comparison_mae.png`, `../reports/figures/model_comparison_mape.png`, `../reports/figures/model_comparison_r2.png`

## 6. Kiểm thử và kiểm toán chất lượng (Quality Assurance)
1. **Clean Architecture & Chuẩn Docstring 5W1H:** Toàn bộ mô-đun trong `src/` tuân thủ PEP-8 và tách biệt cấu hình vào `config.yaml`.
2. **Kiểm định Chống Rò rỉ Dữ liệu (`Anti-Data-Leakage Verification`):** Cờ `leakage_free = True` xác nhận `StandardScaler` chỉ học trên `404` mẫu `Train`.
3. **Kiến trúc Lưu trữ Biểu đồ Đơn nhất (`Single Source of Truth Figures`):** 16 biểu đồ (`300 DPI`) lưu đồng thời tại `reports/figures/` và `reports/figures/<run_id>/`.
4. **Kiểm thử Thực thi Tự động (`End-to-End Pipeline Audit`):** Chạy `python run_lab04.py` và `render_report.py` đạt trạng thái `SUCCESS` 100%.

## 7. Kết luận
Hệ thống Lab 04 đã hoàn thành các mục tiêu của Chương 5. Sự kết hợp giữa chuẩn hóa Z-Score chống rò rỉ dữ liệu, kiến trúc MLP `13-32-16-1` với hàm kích hoạt `ReLU`/`Linear`, thuật toán tối ưu `Adam` và `EarlyStopping` đạt `Test R² = 0.8287`, `Test MAPE = 11.71%`; MLP không vượt đồng thời hồi quy tuyến tính và cây quyết định trên cả 5 chỉ số, xem bảng 5.4 để đối chiếu.

## 8. Tài liệu tham khảo
1. *Bài giảng Chương 5: Introduction to Artificial Neural Networks (ANNs)* & *Practice Exercise 4*, Trường Đại học Giao thông Vận tải TP.HCM (UTH).
2. Harrison, D., & Rubinfeld, D. L. (1978). *Hedonic housing prices and the demand for clean air*. Journal of Environmental Economics and Management, 5(1), 81–102.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
