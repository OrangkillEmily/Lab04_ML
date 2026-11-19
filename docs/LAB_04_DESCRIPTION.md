# BÀI THỰC HÀNH LAB 04: DỰ ĐOÁN GIÁ NHÀ VỚI MẠNG NƠ-RON ĐA TẦNG (MULTILAYER PERCEPTRON REGRESSION)

> **Học phần:** Học máy (Machine Learning) – Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Lab:** 04 · **Chương:** 5 (Introduction to Artificial Neural Networks – ANNs)  
> **Chủ đề:** Multilayer Perceptron Regression — Predicting House Prices on Boston Housing Dataset  
> **Tài liệu gốc:** `Chapter5_Artificial Neural_Networks.pdf` & `Practice exercise 4.pdf`

## 1. Giới thiệu chung
Bài thực hành số 04 tập trung vào việc thiết kế, huấn luyện, điều chuẩn và đánh giá mô hình **Mạng Nơ-ron Đa tầng (Multilayer Perceptron – MLP)** cho bài toán **Hồi quy (Regression)** dự đoán giá trị nhà trung vị (`MEDV`, đơn vị `$1,000 USD`) trên bộ dữ liệu chuẩn **Boston Housing Dataset** (`506` mẫu, `13` đặc trưng kinh tế - xã hội và quy hoạch đô thị) theo nguyên lý **Clean Architecture**:
- **Quản trị dữ liệu, Kiểm toán Ngoại lai & Chống rò rỉ dữ liệu (Anti-Data-Leakage):** Đọc trực tiếp tệp dữ liệu chuẩn `data/raw/boston_housing.csv` (thay thế hoàn toàn hàm `sklearn.datasets.load_boston()` đã bị gỡ bỏ), kiểm toán ngoại lai theo **IQR & Z-Score > 3** (`eda_outlier_analysis.csv`), phân tách độc lập tập `Train` (`404` mẫu) và `Test` (`102` mẫu) trước khi `fit` bộ chuẩn hóa `StandardScaler` chỉ trên tập `Train`.
- **Mạng Nơ-ron Đa tầng (`Sequential_MLP_Regressor`) có Điều chuẩn $L_2$ & Dropout:** Xây dựng cấu trúc `Input(13)` $
ightarrow$ `Dense(32, ReLU)` $
ightarrow$ `Dropout(0.05)` $
ightarrow$ `Dense(16, ReLU)` $
ightarrow$ `Dropout(0.05)` $
ightarrow$ `Dense(1, Linear)` với `993` tham số huấn luyện, điều chuẩn `l2_lambda = 0.001`, tối ưu hóa bằng thuật toán `Adam` ($\eta = 0.005$) kết hợp `EarlyStopping` trên tập `Validation` (`81` mẫu, đạt cực tiểu kiểm định tại **Epoch `91`** với `Best Val MSE = 14.0724`).
- **Đối sánh đa mô hình, 5-Fold Cross-Validation, Tuning Kiến trúc & Model Summary:** Đặt `MLP Regression` lên bàn cân so sánh với `Linear Regression`, `Decision Tree`, `Random Forest` qua bộ 5 chỉ số chuẩn (**MSE**, **RMSE**, **MAE**, **MAPE %**, **$R^2$ Score**), kiểm định **5-Fold Cross-Validation**, đối sánh **4 kiến trúc MLP** và trích xuất chẩn đoán thống kê **`model_summary.csv`** (`Log-Likelihood`, `AIC`, `BIC`, `F-statistic`, `Durbin-Watson`, `Jarque-Bera`).

## 2. Yêu cầu kỹ thuật cốt lõi
1. **Nạp dữ liệu chuẩn & Kiểm toán Ngoại lai (`src/data_loader.py`, `src/eda.py`):**
   - Đọc trực tiếp `data/raw/boston_housing.csv` (`506` dòng $	imes$ `14` cột, `0` missing, `0` duplicate).
   - Thống kê mô tả, ma trận tương quan Pearson và kiểm toán ngoại lai IQR / Z-Score (`reports/metrics/eda_outlier_analysis.csv`) mà không xóa dòng dữ liệu gốc.
2. **Tiền xử lý nghiêm ngặt chống Data Leakage (`src/preprocess.py`):**
   - Chia dữ liệu gốc theo tỷ lệ `80/20` (`random_state = 42`): Tập `Train` (`404` mẫu) và Tập `Test` (`102` mẫu).
   - Chỉ gọi `scaler.fit(X_train)` trên `404` mẫu huấn luyện (`leakage_free_verified = True`, `scaler_samples_seen = 404`).
   - Tách tiếp `X_train_scaled` thành `X_train_sub` (`323` mẫu) và `X_val_scaled` (`81` mẫu).
3. **Thiết kế kiến trúc MLP, Điều chuẩn & Huấn luyện (`src/model.py`, `src/train.py`):**
   - Tích hợp `l2_lambda = 0.001` và `dropout_rate = 0.05` từ `config.yaml` để chống Overfitting.
   - Sửa chuẩn công thức `max_features = "sqrt"` ($\lfloor\sqrt{13}
floor = 3$) trong `RandomForestRegressorCompat`.
4. **Đánh giá 5 chỉ số, 5-Fold CV, Tuning & Model Summary (`src/evaluate.py`, `src/model_summary.py`):**
   - Tính toán đầy đủ **MSE**, **RMSE**, **MAE**, **MAPE (%)** và **$R^2$** trên tập `Test` (`102` mẫu) và **5-Fold Cross-Validation**.
   - Xuất báo cáo chẩn đoán thống kê `reports/metrics/model_summary.csv`, `model_summary.txt`, `model_summary.json` và `docs/Model_Summary.md`.
5. **Kiểm thử Đơn vị Tự động (`tests/test_lab04.py`):**
   - Bộ 8 bài kiểm thử đơn vị (`pytest tests/test_lab04.py`) kiểm định từ dữ liệu, chống rò rỉ, toán học OLS/MLP/RF cho tới K-Fold CV và Regularization.

## 3. Cấu trúc tệp tin bàn giao

```text
LAB04_MLP_Regression/
├── config.yaml                        # Cấu hình trung tâm (data, scaler, MLP + L2/Dropout, baselines, tuning_and_cv)
├── run_lab04.py                       # Script CLI điều phối thực thi toàn trình 1 lệnh duy nhất
├── render_report.py                   # Script kết xuất báo cáo HTML tương tác tự chứa (Base64 embedded)
├── requirements.txt                   # Danh mục thư viện phụ thuộc (numpy, pandas, matplotlib, scikit-learn, tensorflow, pytest)
│
├── data/
│   ├── raw/boston_housing.csv         # Bộ dữ liệu gốc Boston Housing (506 mẫu × 14 cột, 100% offline)
│   └── processed/                     # X_train_scaled (404), X_train_sub (323), X_val (81), X_test (102), scaler_statistics.json
│
├── src/
│   ├── utils.py                       # Tiện ích nạp YAML, cố định random seed (42), đọc/ghi JSON & Pickle
│   ├── tracking.py                    # Quản lý Run_ID, ghi nhật ký logs/ và tiến độ progress_results/
│   ├── data_loader.py                 # Đọc trực tiếp boston_housing.csv và kiểm định toàn vẹn 14 cột
│   ├── eda.py                         # Thống kê mô tả, tương quan Pearson, kiểm toán Outlier IQR/Z-Score & 4 biểu đồ EDA
│   ├── preprocess.py                  # Chia Train/Val/Test và chuẩn hóa StandardScaler chống Data Leakage
│   ├── model.py                       # Khởi tạo MLP Regression (L2 + Dropout) & 3 Baselines (RF max_features='sqrt')
│   ├── train.py                       # Huấn luyện MLP (EarlyStopping) và Linear, Decision Tree, Random Forest
│   ├── evaluate.py                    # Tính toán MSE, RMSE, MAE, MAPE (%), R² và kích hoạt model_summary
│   ├── model_summary.py               # Trích xuất Model Summary (AIC, BIC, LogL, DW, JB), 5-Fold CV & MLP Architecture Tuning
│   ├── visualize.py                   # Xuất 16 biểu đồ Dual-Storage vào reports/figures/ và reports/figures/<run_id>/
│   ├── report.py                      # Tự động sinh báo cáo Markdown đồng bộ với render_report.py
│   └── pipeline.py                    # Master pipeline kết nối 6 bước thực nghiệm tuần tự (gọi finish_run đúng 1 lần)
│
├── tests/
│   └── test_lab04.py                  # Bộ 8 Unit Tests tự động kiểm định toàn diện pipeline (pytest / unittest)
│
├── notebooks/
│   └── lab04.ipynb                    # Jupyter Notebook thực hành từng bước (đồng bộ 100% với kết quả thực nghiệm)
│
├── docs/
│   ├── LAB_04_REPORT.md               # Báo cáo kỹ thuật tổng hợp chuẩn học thuật UTH
│   ├── LAB_04_DESCRIPTION.md          # Đặc tả đề bài, kiến trúc hệ thống và 14 tiêu chí nghiệm thu AC01-AC14
│   ├── EXECUTION_GUIDE.md             # Hướng dẫn cài đặt, chạy pipeline, chạy pytest và mở báo cáo từng bước
│   ├── FILE_GUIDE.md                  # Giải thích chi tiết kiến trúc từng tệp theo khung 5W1H & Code Walkthrough
│   ├── Eda_Phase.md                   # Tài liệu giai đoạn EDA & Kiểm toán ngoại lai (5 mục chuẩn)
│   ├── Preprocessing_Phase.md         # Tài liệu giai đoạn Tiền xử lý chống Data Leakage (5 mục chuẩn)
│   ├── Feature_Engineering_Phase.md   # Tài liệu giai đoạn Biểu diễn đặc trưng & Không gian ẩn (5 mục chuẩn)
│   ├── Training_Phase.md              # Tài liệu giai đoạn Huấn luyện MLP, L2/Dropout & Tuning (5 mục chuẩn)
│   ├── Evaluation_Phase.md            # Tài liệu giai đoạn Đánh giá 5 chỉ số, 5-Fold CV & Model Summary (5 mục chuẩn)
│   ├── Inference_Phase.md             # Tài liệu giai đoạn Suy luận dự đoán giá nhà thực tế (5 mục chuẩn)
│   ├── Tests_Phase.md                 # Tài liệu giai đoạn Kiểm thử đơn vị tự động 8 tests (5 mục chuẩn)
│   ├── MLP Regression.md              # Hồ sơ chi tiết mô hình MLP Regression (5 mục chuẩn)
│   ├── LinearRegression.md            # Hồ sơ chi tiết mô hình Linear Regression (5 mục chuẩn)
│   ├── DecisionTreeRegression.md      # Hồ sơ chi tiết mô hình Decision Tree Regression (5 mục chuẩn)
│   ├── RandomForestRegression.md      # Hồ sơ chi tiết mô hình Random Forest Regression (5 mục chuẩn)
│   ├── Model_Comparison.md            # Tài liệu đối sánh học thuật 4 mô hình, 5-Fold CV & Tuning kiến trúc
│   └── Model_Summary.md               # Báo cáo trích xuất tham số và chẩn đoán phần dư chi tiết của 4 mô hình
│
└── reports/
    ├── LAB_04_REPORT.html             # Báo cáo HTML tương tác theo sơ đồ luồng (Flow Map + 07 Stages)
    ├── experiment_summary.md          # Báo cáo tổng hợp thực nghiệm của phiên chạy mới nhất
    ├── figures/                       # Kho lưu trữ tập trung 18 biểu đồ chuẩn (Single Source of Truth)
    ├── metrics/                       # Các tệp JSON & CSV (test_metrics, eda_outlier, kfold_cv, mlp_tuning, model_summary)
    └── predictions/                   # Kết quả dự đoán chi tiết trên 102 mẫu kiểm tra
```

## 4. Tiêu chí Nghiệm thu (Acceptance Criteria — `AC01` đến `AC14`)

| ID | Tiêu chí | Điều kiện nghiệm thu & Minh chứng kỹ thuật | Trạng thái |
| :--- | :--- | :--- | :---: |
| **AC01** | Clean Architecture | Phân tách độc lập `config.yaml`, `src/`, `data/`, `models/`, `reports/`, `docs/`, `notebooks/`, `tests/` | **PASS** |
| **AC02** | Direct Local Dataset | Đọc trực tiếp `data/raw/boston_housing.csv` (`506 × 14`), không dùng `load_boston()` đã bị gỡ bỏ | **PASS** |
| **AC03** | Data Integrity Audit | Kiểm định tự động đủ 13 đặc trưng + `MEDV`, `0` giá trị thiếu, `0` bản ghi trùng lặp | **PASS** |
| **AC04** | Anti-Data-Leakage | `StandardScaler` chỉ `fit` trên `X_train` (`404` mẫu), `leakage_free_verified = True` | **PASS** |
| **AC05** | Validation Isolation | Tách `X_train_sub` (`323`) và `X_val_scaled` (`81`) từ tập Train; khóa kín `X_test` (`102`) tới bước đánh giá cuối | **PASS** |
| **AC06** | MLP Architecture & Reg | Mạng `13 → 32 (ReLU) → 16 (ReLU) → 1 (Linear)` đúng `993` tham số, `Adam (lr=0.005)`, `l2_lambda=0.001`, `dropout=0.05` | **PASS** |
| **AC07** | Convergence & EarlyStop | Hội tụ ổn định tại `Best Epoch = 91` (`Best Val MSE = 14.0724`, khôi phục trọng số tốt nhất) | **PASS** |
| **AC08** | 4-Model Benchmarking | Đối sánh công bằng `MLP Regression` với `Linear Regression`, `Decision Tree` và `Random Forest` | **PASS** |
| **AC09** | 5-Metric Evaluation | Đánh giá đầy đủ 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)` (Slide 10 Chương 5) và `R² Score` | **PASS** |
| **AC10** | Single Figure Storage | Hợp nhất 16 biểu đồ tại `reports/figures/` kèm thư mục lịch sử `reports/figures/<run_id>/` | **PASS** |
| **AC11** | End-to-End Automation | Thực thi toàn trình không lỗi chỉ với 1 lệnh `python run_lab04.py` và `notebooks/lab04.ipynb` | **PASS** |
| **AC12** | Interactive HTML Report | Tự động sinh `LAB_04_REPORT.html`, `LAB_04_REPORT.md` và `LAB_04_DESCRIPTION.md` chuẩn giao diện UTH | **PASS** |
| **AC13** | Automated Unit Tests | Chạy `pytest tests/test_lab04.py` đạt `8/8 PASS` (Data, Anti-Leakage, OLS, MLP, RF `sqrt`, Metrics, Outlier, K-Fold) | **PASS** |
| **AC14** | K-Fold, Outlier & Summary | Hoàn tất 5-Fold CV (`MLP` Top 1: `CV R² = 0.8607`), Tuning 4 kiến trúc MLP (`MLP_Deep R² = 0.8630`), Outlier Audit & `model_summary.csv` | **PASS** |

## 5. Hướng dẫn Thực thi Nhanh (Quickstart)
```bash
# 1. Chạy toàn bộ Pipeline (Data Loading -> EDA -> Preprocessing -> Training -> Evaluation + Model Summary -> Reporting)
python run_lab04.py

# 2. Chạy riêng mô-đun trích xuất Model Summary, 5-Fold CV & Architecture Tuning
python -m src.model_summary

# 3. Chạy bộ 8 bài kiểm thử đơn vị tự động
pytest tests/test_lab04.py -v

# 4. Kết xuất lại báo cáo HTML & Markdown tương tác
python render_report.py
```
