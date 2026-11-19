# ĐỐI CHIẾU VÀ PHÂN TÍCH CHUYÊN SÂU CÁC MÔ HÌNH HỒI QUY (`docs/Model_Comparison.md`)

## 1. Bảng Tổng hợp So sánh Đa tiêu chí trên Tập Kiểm tra Độc lập (`Test Set` — 102 mẫu, Run `20261119_183635_839530` — PyTorch Edition)

| Tiêu chí Đánh giá | MLP Regression (`13-32-16-1`) | Linear Regression (OLS) | Decision Tree (`max_depth=6`) | Random Forest (`100 trees`) | Mô hình Dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Test MSE (↓)** | `13.5348` | `24.2911` | **`10.5431`** | `11.5205` | **Decision Tree** (Hold-out) |
| **Test RMSE (↓)** | `$3.6790k` | `$4.9286k` | **`$3.2470k`** | `$3.3942k` | **Decision Tree** (Hold-out) |
| **Test MAE (↓)** | `$2.2902k` | `$3.1891k` | `$2.5192k` | **`$2.0994k`** | **Random Forest** |
| **Test MAPE (↓)** | `12.09%` | `16.87%` | `13.88%` | **`10.82%`** | **Random Forest** |
| **Test $R^2$ Score (↑)** | `0.8154` (`81.54%`) | `0.6688` (`66.88%`) | **`0.8562` (`85.62%`)** | `0.8429` (`84.29%`) | **Decision Tree** (Hold-out) |
| **5-Fold CV MSE (Mean ± Std ↓)** | **`12.4888 ± 2.7870`** | `23.4886 ± 2.0601` | `18.3829 ± 11.6161` | `12.5518 ± 1.9864` | **MLP Regression (#1 CV)** |
| **5-Fold CV $R^2$ (Mean ± Std ↑)** | **`0.8483 ± 0.0352`** | `0.7152 ± 0.0419` | `0.7832 ± 0.1212` | `0.8459 ± 0.0395` | **MLP Regression (#1 CV)** |
| **Mean Residual ($e = y - \hat{y}$)** | **`+0.0121`** (Ít chệch nhất) | `+0.2358` | `+0.3102` | `+0.2700` | **MLP Regression** |
| **Tỷ lệ mẫu sai số $\le \pm\$3,000$ (↑)** | `76.47%` | `58.82%` | `68.63%` | **`81.37%`** | **Random Forest** |
| **Log-Likelihood / AIC (↓)** | `-1036.91` / `2101.82` | `-1194.32` / `2416.64` | `-957.31` / `1942.62` | **`-859.25` / `1746.50`** | **Random Forest** |
| **Durbin-Watson ($\approx 2.0$)** | **`2.0024`** | `2.1141` | `1.9605` | `2.0693` | **MLP Regression** |
| **Thời gian Huấn luyện (↓)** | `6.1319s` (CUDA) | `0.0083s` | **`0.0073s`** | `0.1655s` | **Decision Tree** |

## 2. Đối sánh Mở rộng: Tuning 4 Kiến trúc Mạng Nơ-ron Đa tầng PyTorch (`mlp_architecture_tuning.csv`)

| Kiến trúc MLP | Cấu trúc Tầng ẩn | Số tham số | Train MSE | Val MSE | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test $R^2$ (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MLP_Small` (Best Test)** | **`13 -> 16 -> 8 -> 1`** | **`369`** | `8.4158` | `14.6431` | **`12.4992`** | **`$3.5354k`** | **`$2.2884k`** | **`11.84%`** | **`0.8296`** |
| **`MLP_Baseline`** | `13 -> 32 -> 16 -> 1` | `993` | `8.4487` | `15.8257` | `13.5348` | `$3.6790k` | `$2.2902k` | `12.09%` | `0.8154` |
| **`MLP_Wide`** | `13 -> 64 -> 32 -> 1` | `3,009` | `8.3452` | `14.6707` | `13.7687` | `$3.7106k` | `$2.4422k` | `12.92%` | `0.8122` |
| **`MLP_Deep` (Best Val)** | `13 -> 64 -> 32 -> 16 -> 1` | `3,521` | **`7.4476`** | **`12.8325`** | `13.7648` | `$3.7101k` | `$2.6371k` | `13.26%` | `0.8123` |

## 3. Trực quan hóa So sánh MSE, MAE, MAPE (%) và $R^2$

### 3.1. So sánh Sai số Bình phương Trung bình (MSE)
[../reports/figures/model_comparison_mse.png]

### 3.2. So sánh Sai số Tuyệt đối Trung bình (MAE)
[../reports/figures/model_comparison_mae.png]

### 3.3. So sánh Sai số Phần trăm Tuyệt đối Trung bình (MAPE %)
[../reports/figures/model_comparison_mape.png]

### 3.4. So sánh Hệ số Xác định ($R^2$ Score)
[../reports/figures/model_comparison_r2.png]

## 4. Phân tích Học thuật về Hiệu năng và Thứ hạng của 4 Mô hình

1. **`PyTorch MLP Regression` dẫn đầu trên 5-Fold Cross-Validation & Bài học Bias–Variance khi Tuning kiến trúc:**
   - So với `Linear Regression`, mạng `PyTorch MLP Regression (13-32-16-1)` với điều chuẩn `weight_decay = 0.001` và `nn.Dropout(0.05)` giúp **giảm 44.28% sai số MSE** (`24.2911` $\rightarrow$ `13.5348`), hạ `MAE` từ `$3.1891k` xuống `$2.2902k`, hạ `MAPE` từ `16.87%` xuống **`12.09%`** và nâng tỷ lệ dự đoán trong biên độ $\pm\$3,000\text{ USD}$ từ `58.82%` lên **`76.47%`**.
   - Khi đánh giá bằng **5-Fold Cross-Validation** (`506` mẫu) để loại bỏ sự phụ thuộc vào một phép chia `random_state = 42` duy nhất, **`MLP Regression` vươn lên đứng Hạng #1** với `CV R² = 0.8483 ± 0.0352` và `CV MSE = 12.4888 ± 2.7870`.
   - Trong thực nghiệm Tuning 4 kiến trúc PyTorch MLP trên tập dữ liệu nhỏ (`323` mẫu `Train_sub`): kiến trúc gọn nhẹ **`MLP_Small (13->16->8->1, 369 params)`** đạt hiệu suất tổng quát hóa tốt nhất trên tập Test (`Test MSE = 12.4992`, `Test MAPE = 11.84%`, `Test R² = 0.8296`), trong khi **`MLP_Deep` (`3,521 params`)** tuy đạt `Train MSE` (`7.4476`) và `Val MSE` (`12.8325`) thấp nhất nhưng chỉ đạt `Test MSE = 13.7648` (`Test R² = 0.8123`), minh họa rõ sự đánh đổi **Bias–Variance** khi tăng dung lượng mạng quá mức trên tập dữ liệu kích thước vừa và nhỏ.
2. **`Decision Tree Regression` (`max_depth = 6`):**
   - Đạt `Test R² = 0.8562` và `Test MSE = 10.5431` tốt nhất trên tập Hold-out `102` mẫu, nhưng có `MAE` (`$2.5192k`), `MAPE` (`13.88%`) và tỷ lệ `Within ±$3k` (`68.63%`) kém hơn `MLP Regression`. Trên 5-Fold CV, `Decision Tree` có độ lệch chuẩn sai số rất lớn (`CV MSE = 18.3829 ± 11.6161`, `CV R² = 0.7832 ± 0.1212`), phản ánh nhược điểm phương sai cao của cây đơn lẻ.
3. **`Random Forest Regression` (`100 trees`, `max_features = 'sqrt'`):**
   - Khắc phục tốt độ bất ổn của `Decision Tree` nhờ cơ chế Bagging 100 cây với $m_{\text{try}} = \lfloor\sqrt{13}\rfloor = 3$, dẫn đầu về `Test MAE` (`$2.0994k`), `Test MAPE` (`10.82%`), tỷ lệ `Within ±$3k` (`81.37%`) và đứng sát nút Hạng #2 trên 5-Fold CV (`CV R² = 0.8459 ± 0.0395`, `CV MSE = 12.5518 ± 1.9864`).
4. **`Linear Regression` (OLS):**
   - Đứng cuối bảng trên mọi thước đo (`Test R² = 0.6688`, `CV R² = 0.7152`) do hiện tượng **Underfitting** khi ép siêu phẳng tuyến tính lên dữ liệu giá nhà có nhiều quan hệ phi tuyến.
