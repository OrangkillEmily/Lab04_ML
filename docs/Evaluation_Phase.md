# Evaluation Phase — Đánh giá Mô hình, 5-Fold CV & Tuning (`docs/Evaluation_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Model Evaluation** (`src/evaluate.py` và `src/model_summary.py`) đánh giá khách quan 4 mô hình trên tập Kiểm tra độc lập (`102` mẫu) theo bộ **5 chỉ số hồi quy** (`MSE`, `RMSE`, `MAE`, `MAPE %`, `R²`) kèm `Max_Error`, đồng thời thực hiện **5-Fold Cross-Validation** (`506` mẫu, `StandardScaler` chống rò rỉ trong từng fold), **Tuning 4 kiến trúc PyTorch MLP** và trích xuất bảng thống kê **`Model Summary`** chuẩn `statsmodels`.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `models/mlp_regression.pth`, `models/*.pkl` | Các mô hình đã huấn luyện |
| `data/processed/X_test_scaled.pkl`, `y_test.pkl` | Tập Hold-out Test `(102, 13)` |
| `data/raw/boston_housing.csv` (`506` mẫu) | Dữ liệu đầu vào cho quy trình `5-Fold Cross-Validation` |

### Output

| Thành phần | Mô tả |
|---|---|
| `reports/metrics/test_metrics_comparison.csv` | Bảng so sánh 4 mô hình trên tập Test |
| `reports/metrics/kfold_cross_validation.csv` | Bảng kết quả 5-Fold CV (`Mean ± Std`, `Max_Error`, `Train_Time`) |
| `reports/metrics/mlp_architecture_tuning.csv` | Bảng đối sánh 4 kiến trúc PyTorch MLP (`Small`, `Baseline`, `Wide`, `Deep`) |
| `reports/metrics/model_summary.txt`, `.csv`, `.json` & `docs/Model_Summary.md` | Báo cáo thống kê chuẩn `statsmodels` cho 7 cấu hình mô hình |
| `reports/predictions/test_predictions_latest.csv` | Dự đoán, phần dư và `APE (%)` trên từng mẫu của `102` căn hộ Test |
| `reports/figures/*` | Các biểu đồ `model_comparison_*`, `kfold_cv_comparison.png`, `mlp_architecture_tuning.png`, `actual_vs_predicted_*`, `residual_*` |

## 3. How to do — Quy trình thực hiện

```text
  [1] Hold-out Test Evaluation (src/evaluate.py):
      Predict 102 Test samples -> MSE, RMSE, MAE, MAPE %, R2, Max_Error
                                  |
                                  v
  [2] 5-Fold Cross-Validation (src/model_summary.py):
      For fold in 1..5: fit StandardScaler on 4 folds -> train & eval on 1 fold
      Aggregate Mean, Std, Max_Error & Training_Time across 5 folds
                                  |
                                  v
  [3] MLP Architecture Tuning (Train_sub 323 / Val 81 / Test 102):
      Compare Small [16,8], Baseline [32,16], Wide [64,32], Deep [64,32,16]
                                  |
                                  v
  [4] Statsmodels-style Diagnostics & Residual Audit:
      Log-Likelihood, AIC, BIC, F-stat, Durbin-Watson, Jarque-Bera, Within +-3k
```

## 4. Evaluation metrics — Kết quả Đánh giá Tổng hợp (`lab04.ipynb` Cell 46, 49, 52, 55, 60, 63)

### 4.1. Tập Test (`102` mẫu) & 5-Fold Cross-Validation (`506` mẫu)

| Mô hình | Test MSE | Test RMSE | Test MAE | Test MAPE | Test $R^2$ | Within $\pm\$3k$ | 5-Fold CV MSE | 5-Fold CV $R^2$ | CV Max Error Mean (Max) | CV Train Time Mean (Total) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLP Regression** | `13.5348` | `3.6790` | `2.2902` | `12.09%` | `0.8154` | `76.47%` | **`12.4888 ± 2.7870`** | **`0.8483 ± 0.0352`** | `17.6804` (`24.3206`) | `4.0215s` (`20.1073s`) |
| **Linear Regression** | `24.2911` | `4.9286` | `3.1891` | `16.87%` | `0.6688` | `58.82%` | `23.4886 ± 2.0601` | `0.7152 ± 0.0419` | `21.7948` (`27.6509`) | `0.0011s` (`0.0053s`) |
| **Decision Tree** | **`10.5431`** | **`3.2470`** | `2.5192` | `13.88%` | **`0.8562`** | `68.63%` | `18.3829 ± 11.6161` | `0.7832 ± 0.1212` | **`15.8742`** (`30.7333`) | `0.0028s` (`0.0142s`) |
| **Random Forest** | `11.5205` | `3.3942` | **`2.0994`** | **`10.82%`** | `0.8429` | **`81.37%`** | `12.5518 ± 1.9864` | `0.8459 ± 0.0395` | `18.1242` (`26.2513`) | `0.1523s` (`0.7613s`) |

### 4.2. Tuning 4 Kiến trúc PyTorch MLP

- **`MLP_Small` (`13->16->8->1`, `369` params)**: `Val MSE = 14.6431`, **`Test MSE = 12.4992`**, **`Test MAPE = 11.84%`**, **`Test R² = 0.8296`**.
- **`MLP_Baseline` (`13->32->16->1`, `993` params)**: `Val MSE = 15.8257`, `Test MSE = 13.5348`, `Test MAPE = 12.09%`, `Test R² = 0.8154`.
- **`MLP_Wide` (`13->64->32->1`, `3,009` params)**: `Val MSE = 14.6707`, `Test MSE = 13.7687`, `Test MAPE = 12.92%`, `Test R² = 0.8122`.
- **`MLP_Deep` (`13->64->32->16->1`, `3,521` params)**: **`Val MSE = 12.8325`**, `Test MSE = 13.7648`, `Test MAPE = 13.26%`, `Test R² = 0.8123`.

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Số mẫu | Mục đích |
|---|---:|---|
| `data/processed/X_test_scaled.pkl`, `y_test.pkl` | `102` | Đánh giá Hold-out Test, phân tích phần dư và Tuning |
| `data/processed/X_train_scaled.pkl`, `y_train.pkl` | `404` | Tính các chỉ số chẩn đoán `Model Summary` (`AIC`, `BIC`, `Durbin-Watson`, `Jarque-Bera`) |
| `data/raw/boston_housing.csv` | `506` | Thực thi `5-Fold Cross-Validation` (chuẩn hóa riêng trong từng fold) |
