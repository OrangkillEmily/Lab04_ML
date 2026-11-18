# HƯỚNG DẪN KIẾN TRÚC TỆP TIN TOÀN DIỆN (`docs/FILE_GUIDE.md` — 5W1H & CODE WALKTHROUGH)

Tài liệu này giải thích chi tiết từng tệp tin và thư mục trong dự án `LAB04_MLP_Regression` theo khung phân tích **7 phần (WHAT — WHY — WHERE — WHICH — WHEN — HOW — CODE WALKTHROUGH)**, đã cập nhật đầy đủ các mô-đun nâng cấp (`src/model_summary.py`, `tests/test_lab04.py`, `render_report.py` và bộ 7 tài liệu `docs/*_Phase.md`).

## MỤC LỤC HỆ THỐNG TỆP TIN
1. **Nhóm Tệp Gốc (Root Files)**: `config.yaml`, `requirements.txt`, `run_lab04.py`, `render_report.py`, `README.md`.
2. **Nhóm Mã nguồn Cốt lõi (`src/`)**: `__init__.py`, `utils.py`, `tracking.py`, `data_loader.py`, `eda.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `model_summary.py`, `visualize.py`, `report.py`, `pipeline.py`.
3. **Nhóm Kiểm thử Tự động (`tests/`)**: `tests/test_lab04.py` (`8` Unit Tests).
4. **Nhóm Dữ liệu (`data/`)**: `data/raw/boston_housing.csv`, `data/processed/*`.
5. **Nhóm Notebook & Tài liệu (`notebooks/`, `docs/`)**: `notebooks/lab04.ipynb`, 7 tệp `*_Phase.md`, 4 tệp mô hình, `Model_Comparison.md`, `Model_Summary.md`, `LAB_04_DESCRIPTION.md`, `LAB_04_REPORT.md`.

## PHẦN I — NHÓM TỆP GỐC (ROOT FILES)

### 1. `config.yaml`
1. **WHAT**: Tệp cấu hình trung tâm định dạng YAML lưu trữ toàn bộ siêu tham số, tỷ lệ chia dữ liệu, kiến trúc MLP, tham số điều chuẩn (`l2_lambda: 0.001`, `dropout_rate: 0.05`), cấu hình `tuning_and_cv` (`k_folds: 5`, 4 kiến trúc MLP) và đường dẫn thư mục.
2. **WHY**: Tách biệt hoàn toàn tham số cấu hình khỏi mã nguồn Python (Rule 6 — Reproducibility).
3. **WHERE**: `LAB04_MLP_Regression/config.yaml`, được nạp bởi `src/utils.py -> load_config()`.
4. **WHICH**: Cung cấp cấu hình cho toàn bộ các mô-đun trong `src/`, `tests/test_lab04.py` và `notebooks/lab04.ipynb`.
5. **WHEN**: Được đọc ngay khi khởi chạy bất kỳ mô-đun nào.
6. **HOW**: Tổ chức theo các khối khóa: `project`, `data`, `preprocessing`, `mlp_model`, `baseline_models`, `tuning_and_cv`, `paths`, `logging`.
7. **CODE WALKTHROUGH**: Cho phép bật/tắt điều chuẩn $L_2$ và `Dropout` hoặc thay đổi số Folds kiểm định chéo chỉ bằng cách sửa giá trị số trong YAML mà không cần can thiệp vào code.

### 2. `run_lab04.py` & `render_report.py`
- **`run_lab04.py`**: Điểm khởi chạy CLI duy nhất (`python run_lab04.py`) điều phối tuần tự 6 bước từ nạp dữ liệu đến xuất báo cáo thông qua `src.pipeline.main()`.
- **`render_report.py`**: Engine kết xuất báo cáo HTML tương tác tự chứa (`LAB_04_REPORT.html`) nhúng 16 biểu đồ Base64 và nạp động các bảng CSV (`eda_outlier_analysis.csv`, `test_metrics_comparison.csv`, `kfold_cross_validation.csv`, `mlp_architecture_tuning.csv`, `model_summary.csv`) cùng bảng nghiệm thu `AC01–AC14`.

## PHẦN II — NHÓM MÃ NGUỒN CỐT LÕI (`src/`)

### 3. `src/utils.py` & `src/tracking.py`
- **`src/utils.py`**: Cung cấp các hàm hạ tầng dùng chung (`get_project_root`, `ensure_dir`, `generate_run_id`, `set_global_seeds(42)`, `load_config`, `save_object`, `load_object`, `save_json`, `load_json`).
- **`src/tracking.py`**: Quản lý vòng đời thực nghiệm theo `Run_ID`, ghi log thời gian thực vào `logs/pipeline_<run_id>.log`, lưu tiến độ từng bước vào `progress_results/progress_<run_id>.json` và nối lịch sử vào `docs/experiment_logs.csv`.

### 4. `src/data_loader.py` & `src/eda.py`
- **`src/data_loader.py`**: Đọc trực tiếp `data/raw/boston_housing.csv` (`506 × 14`), kiểm định đầy đủ 13 đặc trưng + `MEDV`, xác nhận `0` giá trị thiếu và `0` bản ghi trùng lặp.
- **`src/eda.py`**: Tính thống kê mô tả (`eda_descriptive_statistics.csv`), ma trận tương quan Pearson (`eda_correlation_matrix.csv`), phát hiện `16` mẫu kịch trần `MEDV = 50.0`, kiểm toán ngoại lai theo **IQR & Z-Score > 3** (`compute_outlier_summary` $
ightarrow$ `eda_outlier_analysis.csv`) và gọi 4 biểu đồ EDA.

### 5. `src/preprocess.py`
- **WHAT**: Mô-đun chia tách `Train (404)` / `Test (102)`, chuẩn hóa Z-Score bằng `StandardScaler` và tách tiếp `Train_sub (323)` / `Validation (81)`.
- **WHY**: Đảm bảo 13 đặc trưng có cùng thang đo ($\mu = 0, \sigma = 1$) và tuân thủ nghiêm ngặt **Rule 7 (Chống rò rỉ dữ liệu)**.
- **HOW**: Chỉ gọi `scaler.fit_transform(X_train)` trên `404` mẫu `Train`, sau đó chạy `verify_no_data_leakage()` để lưu bằng chứng toán học `leakage_free_verified = True` và `scaler_samples_seen = 404` vào `data/processed/scaler_statistics.json`.

### 6. `src/model.py`
- **WHAT**: Định nghĩa kiến trúc mạng `Sequential_MLP_Regressor` (`build_mlp_model`) và 3 mô hình đối chứng (`build_baseline_models`), kèm các lớp tương thích NumPy (`NumPySequentialMLP`, `LinearRegressionCompat`, `DecisionTreeRegressorCompat`, `RandomForestRegressorCompat`).
- **HOW & WALKTHROUGH**:
  - Hỗ trợ điều chuẩn `l2_lambda` (`keras.regularizers.l2`) và tầng `Dropout(dropout_rate)` cho cả backend Keras lẫn `NumPySequentialMLP`.
  - Sửa chuẩn xác công thức chọn số đặc trưng ngẫu nhiên `max_features == "sqrt"` thành `max(1, int(np.sqrt(n_features)))` ($\lfloor\sqrt{13}
floor = 3$).

### 7. `src/train.py` & `src/evaluate.py`
- **`src/train.py`**: Huấn luyện mạng MLP trên `X_train_sub` (`323` mẫu) với `EarlyStopping` giám sát `X_val_scaled` (`81` mẫu), lưu trọng số tốt nhất tại `Best Epoch = 91` (`models/mlp_regression.keras`), huấn luyện 3 mô hình Baseline trên `X_train_scaled` (`404` mẫu) và xuất các biểu đồ đường cong hội tụ.
- **`src/evaluate.py`**: Tính toán bộ 5 chỉ số hồi quy (`MSE`, `RMSE`, `MAE`, `MAPE %`, `R²`) trên tập Train/Test, xuất bảng dự đoán `test_predictions_latest.csv`, vẽ 8 biểu đồ đối sánh và tự động kích hoạt `src.model_summary.main()`.

### 8. `src/model_summary.py` (Mô-đun Mới)
1. **WHAT**: Mô-đun độc lập trích xuất báo cáo thống kê chuyên sâu kiểu `statsmodels OLS Summary` cho cả 4 mô hình, đồng thời thực thi **5-Fold Cross-Validation** và **Tuning 4 kiến trúc MLP**.
2. **WHY**: Bổ sung các chỉ số kiểm định thống kê phần dư (`Log-Likelihood`, `AIC`, `BIC`, `HQIC`, `F-statistic`, `Durbin-Watson`, `Jarque-Bera`) và khắc phục hạn chế của phép chia Hold-out đơn lẻ.
3. **WHERE**: `LAB04_MLP_Regression/src/model_summary.py`.
4. **OUTPUT**:
   - `reports/metrics/model_summary.csv`, `model_summary.txt`, `model_summary.json` và `docs/Model_Summary.md`.
   - `reports/metrics/kfold_cross_validation.csv` và `reports/metrics/mlp_architecture_tuning.csv`.

### 9. `src/visualize.py`, `src/report.py` & `src/pipeline.py`
- **`src/visualize.py`**: Kho hàm đồ họa xuất 16 biểu đồ `300 DPI` vào `reports/figures/` và `reports/figures/<run_id>/`.
- **`src/report.py`**: Sinh báo cáo Markdown `reports/experiment_summary.md` và `docs/LAB_04_REPORT.md` tích hợp tự động 4 bảng CSV mới.
- **`src/pipeline.py`**: Điều phối viên cấp cao thực thi tuần tự 6 bước và gọi `finish_run()` đúng 1 lần duy nhất sau khi bước Reporting hoàn tất.

## PHẦN III — NHÓM KIỂM THỬ (`tests/`) & TÀI LIỆU (`docs/`)

### 10. `tests/test_lab04.py` (Mô-đun Mới)
- **WHAT**: Bộ kiểm thử đơn vị tự động gồm `8` bài test (`test_01` $
ightarrow$ `test_08`) kiểm định toàn bộ chuỗi xử lý từ nạp dữ liệu, chống rò rỉ, toán học OLS/MLP/RF cho tới Outlier Audit, Regularization và K-Fold CV.
- **HOW**: Chạy trực tiếp bằng lệnh `pytest tests/test_lab04.py -v` (đạt `8/8 PASS`).

### 11. Hệ thống Tài liệu Chuẩn hóa trong `docs/`
- **7 Tài liệu Giai đoạn (`*_Phase.md`)**: `Eda_Phase.md`, `Preprocessing_Phase.md`, `Feature_Engineering_Phase.md`, `Training_Phase.md`, `Evaluation_Phase.md`, `Inference_Phase.md`, `Tests_Phase.md` — mỗi tệp tuân thủ nghiêm ngặt bố cục 5 phần: `1. Purpose`, `2. Input-Output`, `3. How to do`, `4. Evaluation metrics`, `5. Used data`.
- **4 Tài liệu Mô hình**: `MLP Regression.md`, `LinearRegression.md`, `DecisionTreeRegression.md`, `RandomForestRegression.md`.
- **Tài liệu Tổng hợp**: `Model_Comparison.md`, `Model_Summary.md`, `LAB_04_DESCRIPTION.md`, `LAB_04_REPORT.md`, `EXECUTION_GUIDE.md`, `FILE_GUIDE.md`.
