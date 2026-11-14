# BÀI THỰC HÀNH LAB 04: DỰ ĐOÁN GIÁ NHÀ VỚI MẠNG NƠ-RON ĐA TẦNG (MULTILAYER PERCEPTRON REGRESSION)

> **Học phần:** Học máy (Machine Learning) – Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Lab:** 04 · **Chương:** 5 (Introduction to Artificial Neural Networks – ANNs)  
> **Chủ đề:** Multilayer Perceptron Regression — Predicting House Prices on Boston Housing Dataset  
> **Tài liệu gốc:** `Chapter5_Artificial Neural_Networks.pdf` & `Practice exercise 4.pdf`

## 1. Giới thiệu chung
Bài thực hành số 04 tập trung vào việc thiết kế, huấn luyện và đánh giá mô hình **Mạng Nơ-ron Đa tầng (Multilayer Perceptron – MLP)** cho bài toán **Hồi quy (Regression)** dự đoán giá trị nhà trung vị (`MEDV`, đơn vị `$1,000 USD`) trên bộ dữ liệu chuẩn **Boston Housing Dataset** (`506` mẫu, `13` đặc trưng kinh tế - xã hội và quy hoạch đô thị) theo nguyên lý **Clean Architecture**:
- **Quản trị dữ liệu & Chống rò rỉ dữ liệu (Anti-Data-Leakage):** Đọc trực tiếp tệp dữ liệu chuẩn `data/raw/boston_housing.csv` (thay thế hoàn toàn hàm `sklearn.datasets.load_boston()` đã bị gỡ bỏ), phân tách độc lập tập `Train` (`404` mẫu) và `Test` (`102` mẫu) trước khi `fit` bộ chuẩn hóa `StandardScaler` chỉ trên tập `Train`.
- **Mạng Nơ-ron Đa tầng (`Sequential_MLP_Regressor`):** Xây dựng cấu trúc `Input(13)` $\rightarrow$ `Dense(32, ReLU)` $\rightarrow$ `Dense(16, ReLU)` $\rightarrow$ `Dense(1, Linear)` với `993` tham số huấn luyện, tối ưu hóa bằng thuật toán `Adam` ($\eta = 0.005$) kết hợp cơ chế dừng sớm `EarlyStopping` trên tập `Validation` (`81` mẫu, dừng tại Epoch `85/100` và khôi phục trọng số tốt nhất tại **Epoch `70`**).
- **Đối sánh đa mô hình & Hệ thống 5 chỉ số đánh giá:** Đặt `MLP Regression` lên bàn cân so sánh trực tiếp với 3 mô hình học máy cổ điển (`Linear Regression`, `Decision Tree`, `Random Forest`) qua bộ 5 chỉ số chuẩn xác của Chương 5: **MSE**, **RMSE**, **MAE**, **MAPE (%)** và **$R^2$ Score**.

## 2. Yêu cầu kỹ thuật cốt lõi
1. **Nạp và xác thực dữ liệu chuẩn (`src/data_loader.py`):**
   - Đọc trực tiếp từ tệp dữ liệu chuẩn `data/raw/boston_housing.csv` (`506` dòng $\times$ `14` cột).
   - Kiểm tra tự động tính toàn vẹn: đủ 13 biến độc lập (`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`) và biến mục tiêu `MEDV`, không có giá trị khuyết (`0` missing) và không có dòng trùng lặp (`0` duplicate).
2. **Tiền xử lý nghiêm ngặt chống Data Leakage (`src/preprocess.py`):**
   - Chia dữ liệu gốc theo tỷ lệ `80/20` (`random_state = 42`): Tập `Train` (`404` mẫu, `Mean MEDV = $22.80k`) và Tập `Test` (`102` mẫu, `Mean MEDV = $21.49k`).
   - Chuẩn hóa Z-Score ($z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$): Chỉ gọi `scaler.fit(X_train)` trên `404` mẫu huấn luyện (`test_scaled_mean_l2_norm = 0.460298`), tuyệt đối không để thông tin phân phối của `X_test` rò rỉ vào quá trình chuẩn hóa.
   - Tách tiếp `X_train_scaled` thành `X_train_sub` (`323` mẫu) và `X_val_scaled` (`81` mẫu) để giám sát `val_loss` mà không chạm vào `X_test`.
3. **Thiết kế kiến trúc MLP & Huấn luyện (`src/model.py`, `src/train.py`):**
   - Tầng ẩn 1: `32` nơ-ron, hàm kích hoạt `ReLU` ($f(z) = \max(0, z)$) giúp khắc phục hiện tượng triệt tiêu đạo hàm (*Vanishing Gradient*).
   - Tầng ẩn 2: `16` nơ-ron, hàm kích hoạt `ReLU` trích xuất tổ hợp đặc trưng phi tuyến bậc cao.
   - Tầng đầu ra: `1` nơ-ron, hàm kích hoạt `Linear` ($f(z) = z$) cho phép dự đoán giá trị thực dương liên tục `MEDV` $\in [5.0, 50.0]$.
4. **Đánh giá toàn diện 5 chỉ số & Phân tích phần dư (`src/evaluate.py`):**
   - Tính toán đầy đủ **MSE**, **RMSE**, **MAE**, **MAPE (%)** (theo Slide 10 Chương 5) và **$R^2$** trên tập `Test` (`102` mẫu):
     $$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2, \quad \text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|, \quad \text{MAPE} = \frac{100\%}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|, \quad R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
5. **Quản trị biểu đồ hợp nhất & Báo cáo tương tác (`src/visualize.py`, `src/report.py`, `render_report.py`):**
   - Lưu trữ toàn bộ 16 biểu đồ chất lượng cao (`300 DPI`) tại một nguồn duy nhất `reports/figures/` (kèm bản chụp lịch sử theo từng `reports/figures/<run_id>/`).
   - Tự động kết xuất báo cáo Markdown (`reports/experiment_summary.md`, `docs/LAB_04_REPORT.md`) và báo cáo HTML tương tác nhúng ảnh Base64 (`reports/LAB_04_REPORT.html`, `reports/report_<run_id>.html`).

## 3. Cấu trúc tệp tin bàn giao

```text
LAB04_MLP_Regression/
├── config.yaml                        # Cấu hình trung tâm (data, scaler, MLP 13-32-16-1, baselines, paths)
├── run_lab04.py                       # Script CLI điều phối thực thi toàn trình 1 lệnh duy nhất
├── render_report.py                   # Script kết xuất báo cáo HTML tương tác tự chứa (Base64 embedded)
├── requirements.txt                   # Danh mục thư viện phụ thuộc (numpy, pandas, matplotlib, scikit-learn, tensorflow)
│
├── data/
│   ├── raw/boston_housing.csv         # Bộ dữ liệu gốc Boston Housing (506 mẫu × 14 cột, có sẵn 100% offline)
│   └── processed/                     # Các ma trận X_train_sub (323), X_val (81), X_test (102) đã chuẩn hóa
│
├── src/
│   ├── utils.py                       # Tiện ích nạp YAML, cố định random seed (42), đọc/ghi JSON & Pickle
│   ├── tracking.py                    # Quản lý Run_ID, ghi nhật ký logs/ và tiến độ progress_results/
│   ├── data_loader.py                 # Đọc trực tiếp boston_housing.csv và kiểm định toàn vẹn 14 cột
│   ├── eda.py                         # Phân tích thống kê mô tả, tương quan Pearson và xuất 4 biểu đồ EDA
│   ├── preprocess.py                  # Chia Train/Val/Test và chuẩn hóa StandardScaler chống Data Leakage
│   ├── model.py                       # Khởi tạo kiến trúc MLP Regression (keras.Sequential) & 3 Baselines
│   ├── train.py                       # Huấn luyện MLP (EarlyStopping) và Linear, Decision Tree, Random Forest
│   ├── evaluate.py                    # Tính toán MSE, RMSE, MAE, MAPE (%), R² và xuất bảng dự đoán
│   ├── visualize.py                   # Xuất 16 biểu đồ Dual-Storage vào reports/figures/ và reports/figures/<run_id>/
│   ├── report.py                      # Tự động sinh báo cáo Markdown và báo cáo HTML tương tác theo Run_ID
│   └── pipeline.py                    # Master pipeline kết nối 6 bước thực nghiệm tuần tự
│
├── notebooks/
│   └── lab04.ipynb                    # Jupyter Notebook thực hành từng bước (20 mục từ lý thuyết đến thực nghiệm)
│
├── docs/
│   ├── LAB_04_REPORT.md               # Báo cáo kỹ thuật tổng hợp chuẩn học thuật UTH
│   ├── LAB_04_DESCRIPTION.md          # Đặc tả đề bài, kiến trúc hệ thống và tiêu chí nghiệm thu AC01-AC12
│   ├── Data_Analysis.md               # Tài liệu phân tích chuyên sâu EDA & phân phối dữ liệu
│   ├── Preprocessing_Phase.md         # Tài liệu quy trình chống Data Leakage & StandardScaler
│   ├── Training_Phase.md              # Tài liệu quá trình huấn luyện MLP & hội tụ hàm mất mát
│   ├── Evaluation_Phase.md            # Tài liệu đánh giá 5 chỉ số & phân tích phần dư
│   └── Model_Comparison.md            # Tài liệu đối sánh học thuật giữa MLP và 3 mô hình cổ điển
│
└── reports/
    ├── LAB_04_REPORT.html             # Báo cáo HTML tương tác theo sơ đồ luồng (Flow Map + 07 Stages)
    ├── experiment_summary.md          # Báo cáo tổng hợp thực nghiệm của phiên chạy mới nhất
    ├── figures/                       # Kho lưu trữ tập trung 16 biểu đồ chuẩn (Single Source of Truth)
    ├── metrics/                       # Các tệp JSON & CSV lưu số liệu đánh giá (evaluation_latest.json)
    └── predictions/                   # Kết quả dự đoán chi tiết trên 102 mẫu kiểm tra
```

## 4. Tiêu chí Nghiệm thu (Acceptance Criteria)

| ID | Tiêu chí | Điều kiện nghiệm thu | Trạng thái |
| :--- | :--- | :--- | :---: |
| **AC01** | Clean Architecture | Phân tách độc lập `config.yaml`, `src/`, `data/`, `models/`, `reports/`, `docs/`, `notebooks/` | **PASS** |
| **AC02** | Direct Local Dataset | Đọc trực tiếp `data/raw/boston_housing.csv` (`506 × 14`), không dùng `load_boston()` đã bị gỡ bỏ | **PASS** |
| **AC03** | Data Integrity Audit | Kiểm định tự động đủ 13 đặc trưng + `MEDV`, `0` giá trị thiếu, `0` bản ghi trùng lặp | **PASS** |
| **AC04** | Anti-Data-Leakage | `StandardScaler` chỉ `fit` trên `X_train` (`404` mẫu), `leakage_free_verified = True` | **PASS** |
| **AC05** | Validation Isolation | Tách `X_train_sub` (`323`) và `X_val_scaled` (`81`) từ tập Train; tuyệt đối không dùng `X_test` khi train | **PASS** |
| **AC06** | MLP Architecture | Mạng `13 → 32 (ReLU) → 16 (ReLU) → 1 (Linear)` đúng `993` tham số, tối ưu bằng `Adam (lr=0.005)` | **PASS** |
| **AC07** | Convergence & EarlyStop | Chạy `85/100` epochs, đạt đỉnh kiểm định tại `best_epoch = 70` (`Best Val MSE = 14.0107`, `Test R² = 0.8423`, `Test MAPE = 12.22%`) | **PASS** |
| **AC08** | 4-Model Benchmarking | Đối sánh công bằng `MLP Regression` với `Linear Regression`, `Decision Tree` và `Random Forest` | **PASS** |
| **AC09** | 5-Metric Evaluation | Đánh giá đầy đủ 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)` (Slide 10 Chương 5) và `R² Score` | **PASS** |
| **AC10** | Single Figure Storage | Hợp nhất 16 biểu đồ tại `reports/figures/` kèm thư mục lịch sử `reports/figures/<run_id>/` | **PASS** |
| **AC11** | End-to-End Automation | Thực thi toàn trình không lỗi chỉ với 1 lệnh `python run_lab04.py` và `notebooks/lab04.ipynb` | **PASS** |
| **AC12** | Interactive HTML Report | Báo cáo HTML tương tác (`LAB_04_REPORT.html`) có SVG Flow Map, 07 Stages và ảnh Base64 tự chứa | **PASS** |

## 5. Hướng dẫn Thực thi (Quickstart)

### 5.1. Chạy Pipeline Thực nghiệm Tự động Toàn trình
```bash
# Chạy trọn vẹn 6 bước (Data Loading -> EDA -> Preprocessing -> Training -> Evaluation -> Reporting)
python run_lab04.py

# Chỉ chạy lại bước Đánh giá và Xuất báo cáo trên các mô hình đã huấn luyện
python run_lab04.py --skip-eda --skip-preprocess --skip-train
```

### 5.2. Kết xuất Báo cáo HTML Tương tác & Báo cáo Markdown
```bash
# Chạy trực tiếp script kết xuất báo cáo giao diện UTH
python render_report.py

# Hoặc gọi qua mô-đun báo cáo của hệ thống
python -m src.report
```
