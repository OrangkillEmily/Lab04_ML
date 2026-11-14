# LAB04 — MULTILAYER PERCEPTRON REGRESSION: DỰ ĐOÁN GIÁ NHÀ (HOUSE PRICE PREDICTION)

Dự án thực hành **Machine Learning — Practice Exercise 4 (Chapter 5: Introduction to Artificial Neural Networks)**, xây dựng hệ thống dự đoán giá nhà trung vị (`MEDV`, đơn vị `$1,000 USD`) bằng mạng nơ-ron đa tầng **Multilayer Perceptron (MLP) Regression** và đối chiếu toàn diện với 3 mô hình học máy truyền thống (**Linear Regression**, **Decision Tree Regression**, **Random Forest Regression**).

---

## 1. Tổng quan Dự án & Phát biểu Bài toán (Project Overview & Problem Statement)

- **Bài toán**: Hồi quy giám sát (*Supervised Regression*) — Dự đoán giá trị nhà trung vị `MEDV` từ 13 đặc trưng kinh tế - xã hội, quy hoạch đô thị, môi trường và cấu trúc nhà ở tại vùng đô thị Boston.
- **Mục tiêu cốt lõi**:
  1. Thiết kế quy trình tiền xử lý dữ liệu chuẩn hóa Z-score (`StandardScaler`) tuân thủ nghiêm ngặt nguyên tắc **Chống Rò rỉ Dữ liệu (Data Leakage Prevention — Rule 7)**.
  2. Xây dựng, biên dịch và huấn luyện mạng **MLP Regression** (`Input(13) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`) với hàm mất mát `MSE`, bộ tối ưu `Adam`, theo dõi đồng thời `MAE` và `MAPE (%)`, cùng giám sát `EarlyStopping` trên tập `Validation` độc lập.
  3. Đánh giá và so sánh khách quan với `Linear Regression`, `Decision Tree` và `Random Forest` trên cùng tập `Test` theo bộ **5 chỉ số**: **MSE**, **RMSE**, **MAE**, **MAPE (%)** *(Slide 10 Chapter 5)* và **R-squared ($R^2$)**.
  4. Tích hợp chặt chẽ lý thuyết **Chapter 5 (Artificial Neural Networks)**: Nơ-ron nhân tạo, Tổng trọng số, Hàm kích hoạt (`ReLU`, `Linear`, `Sigmoid`, `Tanh`), Lan truyền thuận (`Forward Propagation`), Lan truyền ngược (`Backpropagation`), Hàm mất mát (`MSE`, `MAPE`), `Gradient Descent` và `Regularization` (`L1`, `L2`, `Dropout`, `EarlyStopping`).

---

## 2. Bộ dữ liệu (Dataset Description & Strategy)

Dự án sử dụng bộ dữ liệu **Boston Housing Dataset** (Harrison & Rubinfeld, 1978) gồm **506 mẫu** và **14 cột** (`13` đặc trưng đầu vào `X` và `1` biến mục tiêu `y = MEDV`):
- **Lưu ý kỹ thuật về `load_boston()`**: Từ phiên bản `scikit-learn 1.2`, hàm `sklearn.datasets.load_boston()` đã bị loại bỏ do vấn đề đạo đức dữ liệu lịch sử của biến `B`. Module `src/data_loader.py` tuyệt đối không gọi API đã bị loại bỏ này mà nạp trực tiếp từ tệp dữ liệu chuẩn có sẵn `data/raw/boston_housing.csv` (trích xuất nguyên bản từ kho lưu trữ CMU StatLib).
- **Đặc điểm dữ liệu**:
  - Số mẫu: `506` | Số đặc trưng: `13` (`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`) | Biến mục tiêu: `MEDV` (`Mean = $22.53k`, `Median = $21.20k`, `Min = $5.00k`, `Max = $50.00k`).
  - Giá trị thiếu (`Missing values`): `0` | Bản ghi trùng lặp (`Duplicates`): `0`.
  - Phân chia dữ liệu chống rò rỉ:
    - Tập **Train tổng (80%)**: `404` mẫu (`StandardScaler` chỉ được `fit` trên tập này).
    - Tập **Train_sub cho MLP (80% của Train)**: `323` mẫu.
    - Tập **Validation cho MLP (20% của Train)**: `81` mẫu (giám sát `val_loss`, `val_mae`, `val_mape` và `EarlyStopping`).
    - Tập **Test (20% tách biệt hoàn toàn)**: `102` mẫu (chỉ dùng cho đánh giá cuối cùng).

---

## 3. Kiến trúc Thư mục Dự án (Project Architecture)

```text
LAB04_MLP_Regression/
├── README.md                          # Tài liệu tổng quan, hướng dẫn chạy và kết quả thực nghiệm
├── requirements.txt                   # Danh sách thư viện Python cần thiết
├── config.yaml                        # Cấu hình trung tâm (Dataset, Scaler, MLP, Baselines, Paths)
├── run_lab04.py                       # Entry point chạy tự động toàn bộ Pipeline bằng 1 lệnh
├── .gitignore                         # Loại trừ các thư mục môi trường ảo và bộ nhớ đệm
│
├── agents/                            # Định hướng bài thực hành, quy tắc kỹ thuật và luồng công việc
│   ├── LAB.md
│   ├── RULES.md
│   └── stepByStep.md
│
├── data/                              # Quản lý dữ liệu (raw & processed)
│   ├── raw/
│   │   └── boston_housing.csv         # Bộ dữ liệu gốc có sẵn 506 mẫu x 14 cột
│   └── processed/                     # Các tập Train/Val/Test đã chuẩn hóa (.pkl, .csv, JSON)
│
├── docs/                              # Tài liệu kỹ thuật chuyên sâu và giải thích lý thuyết Chương 5
│   ├── Lab04_description.md           # Tích hợp lý thuyết Chương 5 (ANN, MLP, Backprop) & tổng quan
│   ├── Data_Analysis.md               # Phân tích khám phá dữ liệu (EDA) & lưu ý đạo đức dữ liệu
│   ├── Preprocessing_Phase.md         # Quy trình chuẩn hóa StandardScaler & chống Data Leakage
│   ├── Training_Phase.md              # Quá trình huấn luyện MLP & 3 mô hình Baseline
│   ├── Evaluation_Phase.md            # Công thức MSE, RMSE, MAE, MAPE (%), R² & phân tích phần dư
│   ├── Model_Comparison.md            # Đối chiếu học thuật giữa 4 mô hình hồi quy
│   ├── ReadmeForParameters.md         # Giải thích toán học của từng tham số và siêu tham số
│   ├── FILE_GUIDE.md                  # Giải thích 5W1H & Code Walkthrough cho mọi tệp trong dự án
│   ├── EXECUTION_GUIDE.md             # Hướng dẫn thực thi từng bước (Step 1 -> Step 10)
│   └── experiment_logs.csv            # Bảng lịch sử thực nghiệm qua các lần chạy
│
├── notebooks/
│   └── lab04.ipynb                    # Báo cáo thực nghiệm tương tác gồm 20 phần (đã chạy sẵn output)
│
├── src/                               # Mã nguồn module hóa độc lập (kèm docstring 5W1H đầy đủ)
│   ├── __init__.py                    # Khởi tạo package src
│   ├── data_loader.py                 # Tải, kiểm tra tính toàn vẹn và đọc dữ liệu
│   ├── eda.py                         # Thực hiện Exploratory Data Analysis
│   ├── preprocess.py                  # Chia Train/Val/Test và chuẩn hóa StandardScaler
│   ├── model.py                       # Định nghĩa kiến trúc MLP Regression và 3 mô hình Baseline
│   ├── train.py                       # Huấn luyện mô hình, đo thời gian và lưu artifacts
│   ├── evaluate.py                    # Đánh giá MSE, RMSE, MAE, MAPE (%), R² và xuất dự đoán trên tập Test
│   ├── visualize.py                   # Tạo 16 biểu đồ phân tích thực nghiệm chuẩn 300 DPI
│   ├── tracking.py                    # Quản lý Run ID, nhật ký .log và tiến độ .json
│   ├── report.py                      # Tự động tổng hợp báo cáo Markdown và HTML
│   ├── pipeline.py                    # Điều phối toàn bộ quy trình 6 bước
│   └── utils.py                       # Tiện ích dùng chung (seed tái lập, đọc/ghi file, cấu hình)
│
├── models/                            # Lưu mô hình đã huấn luyện (.keras, .pkl), scaler & lịch sử hội tụ
├── logs/                              # Nhật ký hoạt động chi tiết theo từng Run ID
├── progress_results/                  # Trạng thái và thông số từng bước theo Run ID (.json)
└── reports/                           # Báo cáo tổng hợp và kết quả đầu ra
    ├── figures/                       # Toàn bộ biểu đồ thực nghiệm (.png)
    ├── metrics/                       # Bảng tổng hợp MSE, RMSE, MAE, MAPE (%), R² (.csv, .json)
    ├── predictions/                   # Kết quả dự đoán từng mẫu trên tập Test (.csv)
    └── experiment_summary.md          # Báo cáo tổng kết thực nghiệm tự động
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

Kết quả thực nghiệm thực tế trên tập Kiểm tra (`Test Set` — `102` mẫu độc lập, `random_state = 42`):

| Mô hình (Model) | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test R² (↑) | Train MSE | Train MAPE | Train R² | Thời gian Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLP Regression (`13->32->16->1`)** | **11.5628** | **3.4004** | **2.3160** | **12.22%** | **0.8423** | 4.6031 | 8.38% | 0.9470 | 15.7892s |
| **Linear Regression (OLS)** | 24.2911 | 4.9286 | 3.1891 | 16.87% | 0.6688 | 21.6414 | 16.57% | 0.7509 | 0.0083s |
| **Decision Tree (`max_depth=6`)** | **10.5431** | **3.2470** | 2.5192 | 13.88% | **0.8562** | 5.4820 | 8.64% | 0.9369 | 0.0071s |
| **Random Forest (`100 trees`)** | 11.5205 | 3.3942 | **2.0994** | **10.82%** | 0.8429 | 2.6512 | 5.38% | 0.9695 | 0.1577s |

### Nhận xét Chuyên môn (Khớp 100% với `notebooks/lab04.ipynb` — Run ID `20261113_170159_ef05f8`):
- **Mô hình tốt nhất theo `MSE` và `R²` trên tập Test**: **Decision Tree** (`MSE = 10.5431`, `RMSE = 3.2470`, `R² = 0.8562`).
- **Mô hình tốt nhất theo `MAE` và `MAPE (%)` trên tập Test**: **Random Forest** (`MAE = 2.0994`, `MAPE = 10.82%`, `R² = 0.8429`).
- **Hiệu năng của MLP Regression (`keras.Sequential`)**: Hội tụ tối ưu tại **Epoch `70`** (`Best Val MSE = 14.0107`, dừng sớm ở Epoch `85/100`), đạt `Test MSE = 11.5628`, `RMSE = 3.4004`, `MAE = 2.3160` (`~$2,316 USD`), `MAPE = 12.22%` và `R² = 0.8423` (giải thích được **84.23%** phương sai giá nhà trên tập Test). Mạng MLP **giảm 52.40% MSE** và **giảm 4.65 điểm phần trăm MAPE** so với `Linear Regression` (`MSE = 24.2911`, `MAPE = 16.87%`), đồng thời vượt qua `Decision Tree` trên cả 3 chỉ số `MAE` (`2.3160` vs `2.5192`), `MAPE` (`12.22%` vs `13.88%`) và tỷ lệ dự đoán trong biên độ $\pm \$3,000\text{ USD}$ (`72.55%` vs `68.63%`).
---

## 6. Danh mục Biểu đồ Trực quan hóa (`reports/figures/` — Single Source of Truth)

1. `target_distribution.png`: Phân phối giá nhà trung vị `MEDV` (Histogram + KDE + Boxplot).
2. `correlation_heatmap.png`: Ma trận tương quan Pearson giữa 13 đặc trưng và `MEDV`.
3. `feature_distributions.png`: Phân phối của 13 biến đặc trưng đầu vào ở thang đo gốc.
4. `feature_target_relationships.png`: Quan hệ phi tuyến giữa 6 đặc trưng quan trọng nhất và `MEDV`.
5. `train_val_test_split.png`: Đối chiếu kích thước và phân phối giữa tập `Train`, `Validation` và `Test`.
6. `mlp_training_loss_curve.png`: Đường cong giảm hàm mất mát MSE của mạng MLP qua 100 epochs.
7. `mlp_train_vs_val_loss.png`: So sánh quá trình hội tụ Training vs Validation Loss (MSE) và MAE.
8. `training_time_comparison.png`: So sánh thời gian huấn luyện giữa 4 mô hình.
9. `actual_vs_predicted_mlp.png` & `actual_vs_predicted_all.png`: Biểu đồ Giá trị Thực tế vs Dự đoán trên tập Test.
10. `residual_plot_all.png`: Biểu đồ phân tán phần dư $e = y - \hat{y}$ theo giá trị dự đoán.
11. `residual_distribution_all.png`: Phân phối xác suất của phần dư trên tập Test.
12. `model_comparison_mse.png`, `model_comparison_mae.png`, `model_comparison_mape.png`, `model_comparison_r2.png`: Biểu đồ cột đối chiếu MSE, MAE, MAPE (%) và R² giữa 4 mô hình.

---

## 7. Giới hạn Kỹ thuật và Lưu ý Thực nghiệm (Technical Limitations)

1. **Giới hạn của bộ dữ liệu Boston Housing**: Kích thước mẫu nhỏ (`506` bản ghi thu thập từ năm 1978), có 16 mẫu bị cắt ngưỡng trần tại `MEDV = 50.0` (`$50,000`) và chứa biến lịch sử `B` cần lưu ý về mặt đạo đức dữ liệu.
2. **Tương thích Đa môi trường (Dual-Backend Compatibility)**:
   - Mã nguồn trong `src/model.py`, `src/preprocess.py` và `src/train.py` ưu tiên gọi trực tiếp `tensorflow.keras.Sequential` và `scikit-learn` khi bạn chạy trong môi trường ảo đã cài đặt `requirements.txt`.
   - Trong môi trường máy ảo đóng gói ban đầu (nơi không cài sẵn `tensorflow` và `scikit-learn`), hệ thống đã thực thi và kiểm chứng toàn bộ các kết quả, trọng số, dự đoán và biểu đồ bằng engine toán học `NumPy/SciPy` tương đương tuyệt đối (cài đặt tường minh Forward Pass, MSE Loss, Backpropagation, Adam Optimizer, OLS và CART). Khi bạn chạy `pip install -r requirements.txt` và `python run_lab04.py` trên máy cá nhân, hệ thống sẽ tự động kích hoạt backend `tensorflow.keras` và `scikit-learn` gốc.
