# LAB04 — MULTILAYER PERCEPTRON REGRESSION: DỰ ĐOÁN GIÁ NHÀ (HOUSE PRICE PREDICTION)

Dự án thực hành **Machine Learning — Practice Exercise 4 (Chapter 5: Introduction to Artificial Neural Networks)**, xây dựng hệ thống dự đoán giá nhà trung vị (`MEDV`, đơn vị `$1,000 USD`) bằng mạng nơ-ron đa tầng **PyTorch Multilayer Perceptron (BostonMLPNet)** chạy tăng tốc **GPU CUDA / CPU**, đồng thời đối chiếu toàn diện với 3 mô hình học máy truyền thống (**Linear Regression**, **Decision Tree Regression**, **Random Forest Regression**) qua **Hold-out Test**, **5-Fold Cross-Validation** và **Architecture Tuning**.

---

## 1. Tổng quan Dự án & Phát biểu Bài toán (Project Overview & Problem Statement)

- **Bài toán**: Hồi quy giám sát (*Supervised Regression*) — Dự đoán giá trị nhà trung vị `MEDV` từ `13` đặc trưng kinh tế - xã hội, quy hoạch đô thị, môi trường và cấu trúc nhà ở tại vùng đô thị Boston (`506` mẫu).
- **Mục tiêu cốt lõi**:
  1. Thiết kế quy trình tiền xử lý chuẩn hóa Z-score (`StandardScaler`) tuân thủ nghiêm ngặt nguyên tắc **Chống Rò rỉ Dữ liệu (Anti-Data-Leakage — `leakage_free_verified = True`)** và cố định hạt giống ngẫu nhiên (`seed = 42` cho `random`, `NumPy`, `PyTorch`).
  2. Xây dựng và huấn luyện mạng **PyTorch MLP Regression** (`BostonHousingDataset`, `DataLoader(batch_size=32, shuffle=True)`, `BostonMLPNet`: `Linear(13,32) -> ReLU -> Dropout(0.05) -> Linear(32,16) -> ReLU -> Dropout(0.05) -> Linear(16,1)`) với `nn.MSELoss()`, `torch.optim.Adam(lr=0.005, weight_decay=0.001)` và `EarlyStopping(patience=15)`.
  3. Đánh giá và so sánh khách quan với `Linear Regression`, `Decision Tree` và `Random Forest` trên tập `Test` (`102` mẫu) và **5-Fold Cross-Validation** (`506` mẫu) theo bộ **5 chỉ số**: **MSE**, **RMSE**, **MAE**, **MAPE (%)** và **$R^2$**, kèm `Max_Error` và thời gian huấn luyện.
  4. Khảo sát đánh đổi **Bias–Variance** qua 4 kiến trúc MLP (`MLP_Small`, `MLP_Baseline`, `MLP_Wide`, `MLP_Deep`), trích xuất báo cáo thống kê **`Model Summary`** chuẩn `statsmodels` (`AIC`, `BIC`, `Log-Likelihood`, `F-statistic`, `Durbin-Watson`, `Jarque-Bera`) và kiểm thử tự động `8/8` unit tests (`pytest`).

---
## 2. Bộ dữ liệu (Dataset Description & Strategy)

Dự án sử dụng bộ dữ liệu **Boston Housing Dataset** (Harrison & Rubinfeld, 1978) gồm **506 mẫu** và **14 cột** (`13` đặc trưng đầu vào `X` và `1` biến mục tiêu `y = MEDV`):
- **Lưu ý kỹ thuật về `load_boston()`**: Từ `scikit-learn >= 1.2`, hàm `load_boston()` đã bị gỡ bỏ. Module `src/data_loader.py` đọc trực tiếp tệp gốc `data/raw/boston_housing.csv`.
- **Đặc điểm dữ liệu & Kiểm toán Ngoại lai**:
  - Số mẫu: `506` | Số đặc trưng: `13` (`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`) | Biến mục tiêu: `MEDV` (`Mean = $22.53k`, `Std = $9.20k`, `Median = $21.20k`, `Min = $5.00k`, `Max = $50.00k` với `16` mẫu chạm ngưỡng trần `$50k`).
  - Giá trị thiếu (`Missing values`): `0` | Bản ghi trùng lặp (`Duplicates`): `0`.
  - Tương quan mạnh nhất với `MEDV`: `LSTAT` ($r = -0.7377$) và `RM` ($r = +0.6954$).
- **Phân chia dữ liệu chống rò rỉ**:
  - **Train tổng (79.84%)**: `404` mẫu (`StandardScaler` chỉ `fit` trên tập này).
  - **Train_sub cho MLP (63.83%)**: `323` mẫu (cập nhật trọng số qua `DataLoader`).
  - **Validation cho MLP (16.01%)**: `81` mẫu (giám sát `val_loss`, `val_mae`, `val_mape` và `EarlyStopping`).
  - **Test Hold-out (20.16%)**: `102` mẫu (tách biệt hoàn toàn cho đánh giá cuối cùng).

---

## 3. Kiến trúc Thư mục Dự án (Project Architecture)

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

---

## 4. Hướng dẫn Cài đặt và Thực thi Nhanh (Installation & Execution)

### 4.1. Cài đặt trên Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2. Cài đặt trên macOS / Linux / Git Bash
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3. Chạy Toàn bộ Dự án bằng 1 Lệnh Duy nhất
```bash
python run_lab04.py
```
*(Xem hướng dẫn chi tiết từng bước từ Step 1 đến Step 10 tại `docs/EXECUTION_GUIDE.md` và giải thích chi tiết từng file tại `docs/FILE_GUIDE.md`).*

---
## 5. Kết quả Thực nghiệm Chính thức (Results Summary)

Kết quả đồng bộ từ `notebooks/lab04.ipynb` (Run ID `20261119_183635_839530`, backend **`PyTorch (torch.nn.Module | CUDA)`** + **`scikit-learn`**):

### 5.1. Hiệu suất trên Tập Hold-out Test (`102` mẫu) & 5-Fold Cross-Validation (`506` mẫu)

| Mô hình (Model) | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test $R^2$ (↑) | Train $R^2$ | 5-Fold CV MSE (Mean ± Std ↓) | 5-Fold CV $R^2$ (Mean ± Std ↑) | CV Max Error Mean (Max) | Thời gian Test / CV Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLP Regression (`13->32->16->1`)** | `13.5348` | `3.6790` | `2.2902` | `12.09%` | `0.8154` | `0.8857` | **`12.4888 ± 2.7870`** | **`0.8483 ± 0.0352`** | `17.6804` (`24.3206`) | `6.1319s` / `4.0215s` |
| **Linear Regression (OLS)** | `24.2911` | `4.9286` | `3.1891` | `16.87%` | `0.6688` | `0.7509` | `23.4886 ± 2.0601` | `0.7152 ± 0.0419` | `21.7948` (`27.6509`) | `0.0083s` / `0.0011s` |
| **Decision Tree (`max_depth=6`)** | **`10.5431`** | **`3.2470`** | `2.5192` | `13.88%` | **`0.8562`** | `0.9229` | `18.3829 ± 11.6161` | `0.7832 ± 0.1212` | **`15.8742`** (`30.7333`) | **`0.0073s`** / `0.0028s` |
| **Random Forest (`100 trees`)** | `11.5205` | `3.3942` | **`2.0994`** | **`10.82%`** | `0.8429` | `0.9526` | `12.5518 ± 1.9864` | `0.8459 ± 0.0395` | `18.1242` (`26.2513`) | `0.1655s` / `0.1523s` |

### 5.2. Đối sánh Tuning 4 Kiến trúc PyTorch MLP (`mlp_architecture_tuning.csv`)

| Kiến trúc MLP | Params | Best Epoch | Train MSE | Val MSE | Test MSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test $R^2$ (↑) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`MLP_Small` (`13->16->8->1`)** | `369` | `93/100` | `8.4158` | `14.6431` | **`12.4992`** | **`2.2884`** | **`11.84%`** | **`0.8296`** |
| **`MLP_Baseline` (`13->32->16->1`)** | `993` | `41/56` | `8.4487` | `15.8257` | `13.5348` | `2.2902` | `12.09%` | `0.8154` |
| **`MLP_Wide` (`13->64->32->1`)** | `3,009` | `34/49` | `8.3452` | `14.6707` | `13.7687` | `2.4422` | `12.92%` | `0.8122` |
| **`MLP_Deep` (`13->64->32->16->1`)** | `3,521` | `31/46` | **`7.4476`** | **`12.8325`** | `13.7648` | `2.6371` | `13.26%` | `0.8123` |

### Nhận xét:
- **Trên 5-Fold Cross-Validation (`506` mẫu)**: **`PyTorch MLP Regression`** đứng **Hạng #1** về độ chính xác tổng quát (`CV R² = 0.8483 ± 0.0352`, `CV MSE = 12.4888 ± 2.7870`), nhỉnh hơn nhẹ `Random Forest` (`0.8459 ± 0.0395`) và ổn định hơn hẳn `Decision Tree` đơn lẻ (`0.7832 ± 0.1212`).
- **Trên tập Hold-out Test (`102` mẫu)**: `PyTorch MLP Regression` (`13-32-16-1`, `993` tham số, Best Epoch `41/56`) **giảm 44.28% MSE** so với `Linear Regression` (`24.2911` $\rightarrow$ `13.5348`), đạt `Within ±$3k USD = 76.47%` và có phần dư ít chệch nhất (`Mean_Residual = +0.0121`, `Durbin-Watson = 2.0024`).
- **Bài học Bias–Variance từ Architecture Tuning**: Trên tập dữ liệu nhỏ (`323` mẫu `Train_sub`), kiến trúc gọn **`MLP_Small` (`369` tham số)** tổng quát hóa tốt nhất trên tập Test (`Test R² = 0.8296`, `Test MAPE = 11.84%`), trong khi `MLP_Deep` (`3,521` tham số) đạt `Train MSE` và `Val MSE` thấp nhất nhưng không cải thiện thêm được trên `Test`.

---

## 6. Danh mục Biểu đồ Trực quan hóa (`reports/figures/`)

1. `target_distribution.png`: Phân phối giá nhà trung vị `MEDV` (Histogram + KDE + Boxplot).
2. `correlation_heatmap.png`: Ma trận tương quan Pearson giữa 13 đặc trưng và `MEDV`.
3. `feature_distributions.png`: Phân phối của 13 biến đặc trưng đầu vào ở thang đo gốc.
4. `feature_target_relationships.png`: Quan hệ phi tuyến giữa 6 đặc trưng quan trọng nhất và `MEDV`.
5. `train_val_test_split.png`: Đối chiếu kích thước và phân phối giữa tập `Train`, `Validation` và `Test`.
6. `mlp_training_loss_curve.png` & `mlp_train_vs_val_loss.png`: Đường cong hội tụ Loss/MAE/MAPE của PyTorch MLP qua các epoch.
7. `training_time_comparison.png`: So sánh thời gian huấn luyện giữa 4 mô hình.
8. `actual_vs_predicted_mlp.png` & `actual_vs_predicted_all.png`: Đối chiếu Giá trị Thực tế vs Dự đoán trên tập Test.
9. `residual_plot_all.png` & `residual_distribution_all.png`: Phân tích phần dư $e = y - \hat{y}$ trên tập Test.
10. `model_comparison_mse.png`, `model_comparison_mae.png`, `model_comparison_mape.png`, `model_comparison_r2.png`: Đối chiếu 4 mô hình theo MSE, MAE, MAPE (%) và $R^2$.
11. `kfold_cv_comparison.png`: Trực quan hóa `Mean ± Std` của 5-Fold Cross-Validation.
12. `mlp_architecture_tuning.png`: So sánh `Train / Val / Test MSE` của 4 kiến trúc MLP.

---

## 7. Giới hạn Kỹ thuật và Lưu ý Thực nghiệm (Technical Limitations)

1. **Giới hạn của bộ dữ liệu Boston Housing**: Kích thước mẫu nhỏ (`506` bản ghi thu thập năm 1978), có `16` mẫu bị cắt ngưỡng trần tại `MEDV = 50.0` (`$50,000 USD` — nguồn gốc chính của các sai số cực đại) và chứa biến lịch sử `B` cần lưu ý về đạo đức dữ liệu.
2. **Tương thích Đa môi trường (Dual-Backend Compatibility)**:
   - Hệ thống mặc định sử dụng **PyTorch (`BostonMLPNet`, tự động chạy GPU CUDA nếu khả dụng, fallback sang CPU)** cho mạng MLP và **`scikit-learn`** cho 3 mô hình đối chiếu.
   - Trong môi trường không cài sẵn PyTorch, `src/model.py` tự động chuyển sang lớp **`NumPySequentialMLP`** cài đặt tường minh Forward Pass, Backpropagation và Adam Optimizer bằng NumPy thuần để đảm bảo pipeline vẫn chạy và hội tụ bình thường.
