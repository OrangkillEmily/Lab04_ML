# HƯỚNG DẪN KIẾN TRÚC TỆP TIN TOÀN DIỆN (FILE_GUIDE.MD — 5W1H & CODE WALKTHROUGH)

Tài liệu này giải thích chi tiết từng tệp tin và thư mục trong dự án `LAB04_MLP_Regression` theo khung phân tích **7 phần (WHAT — WHY — WHERE — WHICH — WHEN — HOW — CODE WALKTHROUGH)** theo đúng yêu cầu của **PART 9**.

---

## MỤC LỤC HỆ THỐNG TỆP TIN
1. **Nhóm Tệp Gốc (Root Files)**: `config.yaml`, `requirements.txt`, `run_lab04.py`, `README.md`, `.gitignore`.
2. **Nhóm Quản trị Quy tắc (`agents/`)**: `LAB.md`, `RULES.md`, `stepByStep.md`.
3. **Nhóm Mã nguồn Cốt lõi (`src/`)**: `__init__.py`, `utils.py`, `tracking.py`, `data_loader.py`, `eda.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `report.py`, `pipeline.py`.
4. **Nhóm Dữ liệu (`data/`)**: `data/raw/boston_housing.csv`, `data/processed/*`.
5. **Nhóm Notebook & Tài liệu (`notebooks/`, `docs/`)**: `notebooks/lab04.ipynb`, các tệp `.md` (dùng chung `../reports/figures/`).
6. **Nhóm Artifacts Thực nghiệm Tự động (`models/`, `logs/`, `progress_results/`, `reports/`)**.

---

## PHẦN I — NHÓM TỆP GỐC (ROOT FILES)

### 1. `config.yaml`
1. **WHAT — File này là gì?**
   Tệp cấu hình trung tâm định dạng YAML lưu trữ toàn bộ siêu tham số (hyperparameters), tỷ lệ chia dữ liệu, kiến trúc các tầng ẩn MLP và đường dẫn thư mục của dự án.
2. **WHY — Tại sao cần file này?**
   Tách biệt hoàn toàn tham số cấu hình khỏi logic mã nguồn Python (Rule 6 — Reproducibility). Khi sinh viên muốn thử nghiệm thay đổi `learning_rate`, `epochs`, `batch_size` hoặc số nơ-ron tầng ẩn, chỉ cần chỉnh sửa tại một nơi duy nhất mà không phải sửa từng tệp `.py`.
3. **WHERE — File nằm ở đâu?**
   Nằm tại thư mục gốc `LAB04_MLP_Regression/config.yaml`, được đọc bởi `src/utils.py -> load_config()`.
4. **WHICH — File này sử dụng những thành phần nào?**
   Được解析 (parse) bằng thư viện `PyYAML` (`yaml.safe_load`) và cung cấp dữ liệu đầu vào cho `data_loader.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `pipeline.py` và `notebooks/lab04.ipynb`.
5. **WHEN — Khi nào file được thực thi/đọc?**
   Được nạp ngay ở bước khởi đầu của Pipeline (`pipeline.py`) hoặc đầu mỗi module độc lập.
6. **HOW — File hoạt động như thế nào?**
   Tổ chức dữ liệu phân cấp theo 6 khối khóa chính: `project`, `data`, `preprocessing`, `mlp_model`, `baseline_models`, `paths`, `logging`.
7. **CODE WALKTHROUGH**:
   - Tại sao dùng YAML thay vì JSON hay `.ini`? Vì YAML hỗ trợ viết chú thích (`# comment`), cú pháp phân cấp trực quan và hỗ trợ kiểu dữ liệu danh sách/số thực tự nhiên.
   - Nếu bỏ file này thì điều gì xảy ra? Các module sẽ báo lỗi `FileNotFoundError` hoặc phải rơi về các giá trị mặc định rời rạc, làm mất tính nhất quán giữa các bước thực nghiệm.

---

### 2. `run_lab04.py`
1. **WHAT — File này là gì?**
   Điểm khởi chạy chính (Main Entry Point) của toàn bộ dự án LAB04.
2. **WHY — Tại sao cần file này?**
   Giúp người dùng chạy toàn bộ quy trình từ tải dữ liệu đến xuất báo cáo chỉ bằng lệnh đơn giản `python run_lab04.py` tại thư mục gốc mà không cần lo lắng về lỗi đường dẫn `PYTHONPATH`.
3. **WHERE — File nằm ở đâu?**
   Tại thư mục gốc `LAB04_MLP_Regression/run_lab04.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   Sử dụng `sys`, `pathlib.Path` và gọi hàm `main()` từ `src.pipeline`.
5. **WHEN — Khi nào file được thực thi?**
   Khi người dùng gõ `python run_lab04.py` trên Terminal / PowerShell.
6. **HOW — File hoạt động như thế nào?**
   Tự động thêm thư mục gốc dự án vào `sys.path` rồi ủy quyền điều phối cho `src.pipeline.main()`.
7. **CODE WALKTHROUGH**:
   - Việc chèn `sys.path.insert(0, str(ROOT_DIR))` đảm bảo câu lệnh `from src import ...` luôn hoạt động trên cả Windows, macOS và Linux bất kể cấu hình IDE.

---

### 3. `requirements.txt`
1. **WHAT — File này là gì?**
   Danh sách khai báo các thư viện Python và ràng buộc phiên bản tương thích của dự án.
2. **WHY — Tại sao cần file này?**
   Đảm bảo môi trường ảo `.venv` trên máy tính cá nhân của giảng viên/sinh viên cài đặt đúng bộ thư viện cần thiết chỉ bằng một lệnh `pip install -r requirements.txt`.
3. **WHERE — File nằm ở đâu?**
   Tại thư mục gốc `LAB04_MLP_Regression/requirements.txt`.
4. **WHICH — File này sử dụng những thành phần nào?**
   Khai báo: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `pyyaml`, `joblib`, `scipy`, `jupyter`, `ipykernel`, `nbformat`.
5. **WHEN — Khi nào file được sử dụng?**
   Tại **Step 5** trong quá trình thiết lập môi trường trước khi chạy code.
6. **HOW — File hoạt động như thế nào?**
   Trình quản lý gói `pip` đọc từng dòng và tải các gói tương ứng.
7. **CODE WALKTHROUGH**:
   - Sử dụng dải phiên bản linh hoạt (`>=...,<...`) thay vì ghim cứng một bản dựng duy nhất giúp tương thích trên cả Python 3.10, 3.11 và 3.12 trên Windows/macOS/Linux.

---

## PHẦN II — NHÓM MÃ NGUỒN CỐT LÕI (`src/`)

### 4. `src/__init__.py`
1. **WHAT**: Tệp khởi tạo gói mã nguồn Python (`src` package).
2. **WHY**: Đánh dấu thư mục `src/` là một Python package hợp lệ để các câu lệnh `from src.preprocess import ...` hoặc `python -m src.train` hoạt động chuẩn xác. *(Lưu ý: Trong sơ đồ cây mục tiêu Part 5 có ghi `init.py`, chuẩn cú pháp của Python là `__init__.py`)*.
3. **WHERE**: `LAB04_MLP_Regression/src/__init__.py`.
4. **WHICH**: Khai báo biến siêu dữ liệu `__version__ = "1.0.0"` và `__author__`.
5. **WHEN**: Được Python tự động nạp mỗi khi bất kỳ module con nào trong `src/` được import.
6. **HOW & WALKTHROUGH**: Không chứa logic tính toán nặng để tránh hiện tượng circular import (nhập vòng tròn).

---

### 5. `src/utils.py`
1. **WHAT — File này là gì?**
   Module cung cấp các hàm tiện ích hạ tầng dùng chung trên toàn dự án.
2. **WHY — Tại sao cần file này?**
   Tránh lặp lại code tạo thư mục, sinh dấu thời gian, thiết lập random seed và đọc/ghi tệp `.pkl`/`.json`/`.yaml` ở tất cả các module khác (Rule 3 — DRY: Don't Repeat Yourself).
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/utils.py`, nằm ở tầng thấp nhất của kiến trúc (không import bất kỳ module nào khác trong `src/`).
4. **WHICH — File này sử dụng những thành phần nào?**
   - Thư viện: `os`, `sys`, `json`, `pickle`, `random`, `uuid`, `datetime`, `pathlib`, `numpy`, `yaml`, và tùy chọn `joblib`, `tensorflow`.
   - Các hàm chính: `get_project_root()`, `ensure_dir()`, `get_current_timestamp()`, `generate_run_id()`, `set_global_seeds()`, `load_config()`, `save_object()`, `load_object()`, `save_json()`, `load_json()`, `configure_console_utf8()`.
5. **WHEN — Khi nào file được thực thi?**
   Được gọi liên tục bởi tất cả các module khác trong suốt vòng đời của Pipeline.
6. **HOW — File hoạt động như thế nào?**
   Mỗi hàm thực hiện một tác vụ nguyên tử (atomic utility): ví dụ `set_global_seeds(42)` đồng bộ hóa bộ sinh số ngẫu nhiên của Python, NumPy và TensorFlow; `save_object()` tự động dùng `joblib.dump` (nếu có) hoặc chuyển sang `pickle.dump`.
7. **CODE WALKTHROUGH**:
   - **Tại sao `save_object` và `load_object` hỗ trợ cả `joblib` và `pickle`?** Vì `joblib` tối ưu cho mảng NumPy lớn, nhưng trong các môi trường tối giản chưa cài `joblib`, `pickle` trong thư viện chuẩn của Python vẫn lưu/đọc được 100% đối tượng mà không gây lỗi dừng chương trình.
   - **Nếu bỏ `set_global_seeds()` thì điều gì xảy ra?** Mỗi lần chạy `run_lab04.py` sẽ chia tập Train/Test hoặc khởi tạo trọng số MLP khác nhau, vi phạm **Rule 6 (Reproducibility)**.

---

### 6. `src/tracking.py`
1. **WHAT — File này là gì?**
   Module quản lý theo dõi thực nghiệm (Experiment Tracking) và ghi nhật ký hoạt động (Logging).
2. **WHY — Tại sao cần file này?**
   Đảm bảo mỗi lần thực thi Pipeline đều được gắn một mã `Run_ID` riêng biệt (ví dụ `20261112_101925_15f561`), lưu lại toàn bộ thông số đầu vào, đầu ra, thời gian chạy và trạng thái của từng bước mà không làm mất lịch sử của các lần chạy trước (Part 10).
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/tracking.py`, phụ thuộc vào `src/utils.py` và được gọi bởi `data_loader.py`, `eda.py`, `preprocess.py`, `train.py`, `evaluate.py`, `report.py`, `pipeline.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Thư viện: `csv`, `json`, `pathlib.Path`.
   - Các hàm chính: `create_run()`, `write_log()`, `update_progress()`, `finish_run()`, `append_experiment_log()`.
   - Tệp đầu ra: `logs/pipeline_<run_id>.log`, `progress_results/progress_<run_id>.json`, `docs/experiment_logs.csv`, `reports/metrics/experiment_history.csv`.
5. **WHEN — Khi nào file được thực thi?**
   Được gọi ở đầu và cuối mỗi bước trong Pipeline.
6. **HOW — File hoạt động như thế nào?**
   - `create_run()` khởi tạo tệp JSON trạng thái `RUNNING`.
   - `update_progress()` nối thêm kết quả từng bước vào mảng `steps` trong tệp JSON.
   - `write_log()` ghi từng dòng log kèm timestamp chuẩn ISO-8601.
   - `append_experiment_log()` nối thêm kết quả của từng mô hình vào bảng CSV lịch sử.
7. **CODE WALKTHROUGH**:
   - Việc mở tệp CSV với `encoding="utf-8-sig"` giúp Microsoft Excel trên Windows mở trực tiếp tệp `experiment_logs.csv` mà không bị lỗi hiển thị tiếng Việt.

---

### 7. `src/data_loader.py`
1. **WHAT — File này là gì?**
   Module chuyên trách tải, xác thực tính toàn vẹn và trích xuất bộ dữ liệu dự đoán giá nhà.
2. **WHY — Tại sao cần file này?**
   Giải quyết triệt để vấn đề hàm `sklearn.datasets.load_boston()` đã bị gỡ bỏ từ `scikit-learn 1.2` (Part 4.1), đồng thời tách biệt hoàn toàn việc đọc dữ liệu thô khỏi bước phân tích và tiền xử lý.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/data_loader.py`, đọc dữ liệu từ `data/raw/boston_housing.csv`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Thư viện: `numpy`, `pandas`, `time`, `pathlib.Path`.
   - Các hàm chính: `load_housing_dataframe()`, `inspect_dataset()`, `extract_features_and_target()`, `main()`.
5. **WHEN — Khi nào file được thực thi?**
   Là **Bước 1 (`DATA_LOADING`)** trong Pipeline, và được gọi lại bởi `eda.py` cũng như `preprocess.py`.
6. **HOW — File hoạt động như thế nào?**
   Đọc trực tiếp tệp dữ liệu có sẵn `data/raw/boston_housing.csv` bằng `pd.read_csv()`, xác thực đầy đủ 14 cột bắt buộc (`CRIM..LSTAT` và `MEDV`), sau đó hàm `inspect_dataset()` kiểm tra số dòng (`506`), số cột (`14`), giá trị thiếu (`0`), trùng lặp (`0`) và tương quan với `MEDV`.
7. **CODE WALKTHROUGH**:
   - **Tại sao không dùng `load_boston()`?** Vì trên `scikit-learn >= 1.2`, gọi `from sklearn.datasets import load_boston` sẽ lập tức ném ngoại lệ `ImportError`.
   - **Nếu bỏ bước `inspect_dataset()` thì điều gì xảy ra?** Dữ liệu bị khuyết hoặc sai tên cột có thể lọt vào tầng huấn luyện gây lỗi `ValueError` hoặc `NaN` trong đạo hàm của mạng MLP.

---

### 8. `src/eda.py`
1. **WHAT — File này là gì?**
   Module thực hiện Phân tích Khám phá Dữ liệu (Exploratory Data Analysis).
2. **WHY — Tại sao cần file này?**
   Giúp định lượng các đặc tính thống kê của dữ liệu (độ lệch phân phối, hiện tượng cắt ngưỡng tại `MEDV = 50.0`, đa cộng tuyến giữa `RAD` và `TAX`, quan hệ phi tuyến giữa `LSTAT`/`RM` và `MEDV`) trước khi thiết kế mô hình.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/eda.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Gọi `load_housing_dataframe()`, `inspect_dataset()` từ `src/data_loader.py`.
   - Gọi 4 hàm vẽ biểu đồ EDA từ `src/visualize.py`.
   - Xuất bảng biểu ra: `reports/metrics/eda_descriptive_statistics.csv`, `reports/metrics/eda_correlation_matrix.csv`, `reports/metrics/eda_summary.json`.
5. **WHEN — Khi nào file được thực thi?**
   Là **Bước 2 (`EDA`)** trong Pipeline (ngay sau `DATA_LOADING` và trước `PREPROCESSING`).
6. **HOW & CODE WALKTHROUGH**:
   - Tính toán thêm `skewness` (độ lệch) và `kurtosis` (độ nhọn) cho từng cột và đếm số mẫu chạm ngưỡng trần `50.0` (`censoring_count_at_50k = 16`).

---

### 9. `src/preprocess.py`
1. **WHAT — File này là gì?**
   Module tiền xử lý dữ liệu: tách `X, y`, chia tập `Train/Test/Validation` và chuẩn hóa đặc trưng bằng `StandardScaler`.
2. **WHY — Tại sao cần file này?**
   Đưa toàn bộ 13 đặc trưng đầu vào về cùng thang đo chuẩn ($\mu = 0, \sigma = 1$) để mạng nơ-ron MLP hội tụ nhanh và ổn định, đồng thời đảm bảo tuyệt đối **Rule 7 — Data Leakage Prevention**.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/preprocess.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Thư viện: `numpy`, `pandas`, `sklearn.model_selection.train_test_split`, `sklearn.preprocessing.StandardScaler` (cùng lớp tương thích toán học `StandardScalerCompat`).
   - Các hàm chính: `split_dataset()`, `create_scaler()`, `verify_no_data_leakage()`, `main()`.
   - Tệp đầu ra: `data/processed/X_train_scaled.pkl`, `X_test_scaled.pkl`, `y_train.pkl`, `y_test.pkl`, `X_train_sub.pkl`, `X_val_scaled.pkl`, `y_train_sub.pkl`, `y_val.pkl`, `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`, `scaler_statistics.json` và `models/standard_scaler.pkl`.
5. **WHEN — Khi nào file được thực thi?**
   Là **Bước 3 (`PREPROCESSING`)** trong Pipeline, chạy trước `src/train.py`.
6. **HOW — File hoạt động như thế nào?**
   1. Tách `X` `(506, 13)` và `y` `(506,)`.
   2. Chia `X_train_df` (`404` mẫu) và `X_test_df` (`102` mẫu) với `test_size = 0.2, random_state = 42`.
   3. Chỉ gọi `scaler.fit_transform(X_train_df)` trên tập Train và gọi `scaler.transform(X_test_df)` trên tập Test.
   4. Từ `X_train_scaled` (`404` mẫu), tách tiếp 20% (`81` mẫu) làm `X_val_scaled` để giám sát MLP, còn lại `323` mẫu `X_train_sub` để huấn luyện MLP.
   5. Chạy hàm kiểm toán `verify_no_data_leakage()` để xác minh $|\mu_{\text{scaler}} - \mu_{\text{train\_raw}}| = 0$.
7. **CODE WALKTHROUGH**:
   - **Nếu fit `StandardScaler` trên toàn bộ 506 mẫu trước khi chia Train/Test thì sao?** Khi đó trung bình và độ lệch chuẩn của 102 mẫu Test đã bị rò rỉ vào tập Train (Data Leakage), làm sai lệch tính khách quan của thực nghiệm.
   - **Nếu không dùng `StandardScaler` thì sao?** Khi đưa dữ liệu thô (`TAX` ~ 400, `B` ~ 390) vào tầng `Dense(32, relu)`, tổng trọng số $z$ sẽ rất lớn ngay từ Epoch 1, làm đạo hàm bùng nổ hoặc khiến các nơ-ron `ReLU` bị chết (*dead ReLU neurons*).

---

### 10. `src/model.py`
1. **WHAT — File này là gì?**
   Module định nghĩa kiến trúc của mô hình **Multilayer Perceptron (MLP) Regression** và 3 mô hình học máy đối chiếu (**Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**).
2. **WHY — Tại sao cần file này?**
   Tách biệt phần khai báo cấu trúc toán học của mô hình khỏi kịch bản điều phối huấn luyện trong `src/train.py`.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/model.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Thư viện: `tensorflow.keras` (`Sequential`, `Dense`, `Input`, `Adam`, `EarlyStopping`), `scikit-learn` (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`), `numpy`, `zipfile`, `json`.
   - Các hàm và lớp chính: `build_mlp_model()`, `load_mlp_model()`, `build_baseline_models()`, `get_backend_info()`, cùng các lớp toán học `NumPySequentialMLP`, `LinearRegressionCompat`, `DecisionTreeRegressorCompat`, `RandomForestRegressorCompat`.
5. **WHEN — Khi nào file được thực thi?**
   Được gọi bởi `src/train.py` khi khởi tạo mô hình và bởi `src/evaluate.py` / `notebooks/lab04.ipynb` khi tải lại mô hình đã lưu.
6. **HOW — File hoạt động như thế nào?**
   - Hàm `build_mlp_model(input_dim=13, config)` xây dựng mô hình tuần tự gồm 2 tầng ẩn `ReLU` (`32` và `16` nơ-ron) và tầng đầu ra `Linear` (`1` nơ-ron), biên dịch với bộ tối ưu `Adam` và hàm mất mát `mean_squared_error` kèm metric `mae`.
   - Hàm `build_baseline_models(config)` trả về từ điển chứa 3 mô hình đối chiếu với siêu tham số đọc từ `config.yaml`.
7. **CODE WALKTHROUGH**:
   - **Tại sao không dùng hàm kích hoạt `Softmax` ở tầng đầu ra?** Vì `Softmax` biến đổi vector đầu ra thành phân phối xác suất có tổng bằng 1 cho bài toán phân loại đa lớp; nếu gắn `Softmax` vào tầng đầu ra có 1 nơ-ron, đầu ra sẽ luôn bằng $\frac{e^z}{e^z} = 1.0$ với mọi căn nhà! Do đó, bài toán hồi quy bắt buộc phải dùng `activation='linear'`.
   - **Tại sao hỗ trợ song song `tensorflow.keras` / `scikit-learn` và engine toán học `NumPy`?** Khi người dùng chạy trên máy cá nhân đã cài `requirements.txt`, hệ thống tự động dùng `tensorflow.keras` và `scikit-learn`. Trong môi trường kiểm thử độc lập chưa cài TensorFlow, lớp `NumPySequentialMLP` thực thi chính xác công thức Forward Pass, MSE Loss, Backpropagation và cập nhật Adam theo đúng lý thuyết Chapter 5.

---

### 11. `src/train.py`
1. **WHAT — File này là gì?**
   Module thực thi quá trình huấn luyện cho toàn bộ 4 mô hình và lưu trữ các tệp mô hình đã học.
2. **WHY — Tại sao cần file này?**
   Tập trung toàn bộ quy trình `fit()`, giám sát `EarlyStopping`, đo thời gian huấn luyện và xuất các tệp trọng số vào một module duy nhất.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/train.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Đọc dữ liệu từ `data/processed/`.
   - Gọi `build_mlp_model()`, `build_baseline_models()` từ `src/model.py`.
   - Gọi các hàm vẽ biểu đồ huấn luyện từ `src/visualize.py`.
   - Lưu đầu ra vào `models/`: `mlp_regression.keras`, `mlp_regression_weights.npz`, `linear_regression.pkl`, `decision_tree.pkl`, `random_forest.pkl`, `mlp_training_history.csv`, `mlp_training_history.json`, `training_metadata.json`.
5. **WHEN — Khi nào file được thực thi?**
   Là **Bước 4 (`TRAINING`)** trong Pipeline (sau `PREPROCESSING` và trước `EVALUATION`).
6. **HOW & CODE WALKTHROUGH**:
   - Mạng MLP được huấn luyện trên `(X_train_sub, y_train_sub)` và truyền `validation_data=(X_val_scaled, y_val)` để theo dõi `val_loss` và `val_mae` qua 100 epochs mà không hề chạm tới `X_test_scaled`.

---

### 12. `src/evaluate.py`
1. **WHAT — File này là gì?**
   Module đánh giá hiệu năng dự đoán của cả 4 mô hình trên tập Kiểm tra (`Test Set`) và tập Huấn luyện (`Train Set`).
2. **WHY — Tại sao cần file này?**
   Cung cấp thước đo khách quan (`MSE`, `RMSE`, `MAE`, `R²`) trên dữ liệu chưa từng thấy và xác định mô hình tốt nhất theo từng tiêu chí.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/evaluate.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Hàm `compute_regression_metrics(y_true, y_pred)`, `load_all_trained_models()`, `save_metrics_history()`, `main()`.
   - Tệp đầu ra: `reports/metrics/metrics_summary.csv`, `reports/metrics/test_metrics_comparison.csv`, `reports/metrics/evaluation_latest.json`, `reports/predictions/test_predictions_latest.csv`.
5. **WHEN — Khi nào file được thực thi?**
   Là **Bước 5 (`EVALUATION`)** trong Pipeline.
6. **HOW & CODE WALKTHROUGH**:
   - Đánh giá cả 4 mô hình trên cùng ma trận `X_test_scaled` (`102` mẫu) và xuất bảng đối chiếu từng mẫu dự đoán kèm phần dư $e_i = y_i - \hat{y}_i$.

---

### 13. `src/visualize.py`
1. **WHAT — File này là gì?**
   Module chuyên trách sinh toàn bộ 11+ biểu đồ phân tích thực nghiệm.
2. **WHY — Tại sao cần file này?**
   Tách biệt hoàn toàn mã nguồn đồ họa (`matplotlib` / `seaborn`) khỏi logic tính toán của `eda.py`, `train.py` và `evaluate.py`.
3. **WHERE — File nằm ở đâu?**
   `LAB04_MLP_Regression/src/visualize.py`.
4. **WHICH — File này sử dụng những thành phần nào?**
   - Các hàm vẽ biểu đồ:
     1. `plot_target_distribution()` → `target_distribution.png`
     2. `plot_correlation_heatmap()` → `correlation_heatmap.png`
     3. `plot_feature_distributions()` → `feature_distributions.png`
     4. `plot_feature_target_relationships()` → `feature_target_relationships.png`
     5. `plot_train_val_test_split()` → `train_val_test_split.png`
     6. `plot_mlp_training_loss_curve()` → `mlp_training_loss_curve.png`
     7. `plot_training_vs_validation_loss()` → `mlp_train_vs_val_loss.png`
     8. `plot_training_time_comparison()` → `training_time_comparison.png`
     9. `plot_actual_vs_predicted()` → `actual_vs_predicted_all.png` & `actual_vs_predicted_mlp.png`
     10. `plot_residuals()` → `residual_plot_all.png`
     11. `plot_residual_distributions()` → `residual_distribution_all.png`
     12. `plot_metric_comparison_bars()` → `model_comparison_mse.png`, `model_comparison_mae.png`, `model_comparison_r2.png`
5. **WHEN**: Được gọi trong các bước `EDA`, `TRAINING` và `EVALUATION`.
6. **HOW & WALKTHROUGH**: Sử dụng backend không giao diện `matplotlib.use("Agg")` để chạy mượt mà trên cả máy chủ lẫn máy cá nhân mà không bị treo cửa sổ đồ họa.

---

### 14. `src/report.py` & `src/pipeline.py`
- **`src/report.py`**: Đọc dữ liệu thực tế từ `progress_results/`, `reports/metrics/` và `logs/` ở **Bước 6 (`REPORTING`)** để sinh tự động báo cáo tổng hợp `reports/experiment_summary.md`, `reports/report_<run_id>.md` và `reports/report_<run_id>.html`.
- **`src/pipeline.py`**: Điều phối viên cấp cao kết nối tuần tự 6 bước từ `DATA_LOADING` đến `REPORTING`, quản lý ngoại lệ và ghi nhận tổng thời gian thực thi.

---

## PHẦN III — NHÓM NOTEBOOK, DỮ LIỆU & ARTIFACTS

### 15. `notebooks/lab04.ipynb`
- **WHAT**: Báo cáo thực nghiệm tương tác Jupyter Notebook gồm đúng **20 mục** (từ *1. Project Introduction* đến *20. Discussion and Conclusion*).
- **WHY**: Giúp giảng viên và sinh viên theo dõi từng bước lý thuyết Chương 5 gắn liền với code gọi từ `src/`, bảng số liệu thực tế và biểu đồ nhúng trực tiếp.
- **HOW**: Toàn bộ 20 cell code đã được thực thi tuần tự từ đầu đến cuối và lưu sẵn kết quả đầu ra.

### 16. Các Thư mục Dữ liệu và Artifacts (`data/`, `models/`, `logs/`, `progress_results/`, `reports/`)
- `data/raw/boston_housing.csv`: Tệp dữ liệu gốc 506 dòng × 14 cột.
- `data/processed/`: Chứa các tệp `.pkl` và `.csv` của các tập `Train`, `Validation`, `Test` đã chuẩn hóa cùng `scaler_statistics.json`.
- `models/`: Chứa `mlp_regression.keras`, 3 mô hình `.pkl`, `standard_scaler.pkl`, lịch sử hội tụ `mlp_training_history.csv` và `training_metadata.json`.
- `logs/`: Chứa nhật ký hoạt động chi tiết `pipeline_<run_id>.log`.
- `progress_results/`: Chứa hồ sơ theo dõi JSON từng bước `progress_<run_id>.json`.
- `reports/`: Chứa `experiment_summary.md`, thư mục `figures/` (biểu đồ), `metrics/` (bảng chỉ số) và `predictions/` (kết quả dự đoán trên tập Test).
