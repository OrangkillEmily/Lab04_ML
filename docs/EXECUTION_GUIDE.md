# HƯỚNG DẪN CÀI ĐẶT VÀ THỰC THI CHI TIẾT TỪNG BƯỚC (`docs/EXECUTION_GUIDE.md`)

Tài liệu này hướng dẫn chi tiết **11 bước (Step 1 → Step 11)** để giải nén, thiết lập môi trường ảo, cài đặt thư viện, thực thi Pipeline, chạy kiểm thử đơn vị (`pytest`), chạy mô-đun `model_summary`, thực thi Jupyter Notebook và kiểm tra kết quả trên cả **Windows PowerShell / Command Prompt** và **macOS / Linux / Git Bash**.

## Step 1: Giải nén gói dự án `LAB04_MLP_Regression.zip`

### Trên Windows (File Explorer hoặc PowerShell):
```powershell
Expand-Archive -Path .\LAB04_MLP_Regression.zip -DestinationPath .\
cd .\LAB04_MLP_Regression
```

### Trên macOS / Linux / Git Bash:
```bash
unzip LAB04_MLP_Regression.zip
cd LAB04_MLP_Regression
```

## Step 2: Mở thư mục dự án bằng Visual Studio Code (VS Code)
```bash
code .
```
Nhấn tổ hợp phím `` Ctrl + ` `` để mở cửa sổ Terminal ngay tại thư mục gốc của dự án.

## Step 3: Tạo Môi trường Ảo (Virtual Environment)

### Trên Windows PowerShell / CMD:
```powershell
python -m venv .venv
```

### Trên macOS / Linux / Git Bash:
```bash
python3 -m venv .venv
```

## Step 4: Kích hoạt Môi trường Ảo (Activate Virtual Environment)

### Trên Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Trên Windows Command Prompt (CMD):
```bat
.venv\Scripts\activate.bat
```

### Trên macOS / Linux / Git Bash:
```bash
source .venv/bin/activate
```

## Step 5: Cài đặt các Thư viện Phụ thuộc (`requirements.txt`)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

## Step 6: Kiểm tra Bộ dữ liệu (`data/raw/boston_housing.csv`)
```powershell
python -m src.data_loader
```
**Kết quả kỳ vọng:** `506 mẫu x 14 cột (13 đặc trưng + 1 mục tiêu 'MEDV')`, `Giá trị thiếu: 0`, `Dòng trùng lặp: 0`.

## Step 7: Chạy Toàn bộ Quy trình Thực nghiệm (`run_lab04.py`)
Thực thi lệnh duy nhất dưới đây tại thư mục gốc để chạy tự động toàn bộ 6 giai đoạn (**Data Loading → EDA + Outlier Audit → Preprocessing → Training (L2 + Dropout) → Evaluation + Model Summary + 5-Fold CV → Reporting**):

```powershell
python run_lab04.py
```

Hoặc chạy từng mô-đun riêng lẻ:
```powershell
python -m src.eda             # Phân tích khám phá dữ liệu & kiểm toán Outlier IQR/Z-Score
python -m src.preprocess      # Chia Train/Val/Test và chuẩn hóa StandardScaler chống rò rỉ
python -m src.train           # Huấn luyện MLP Regression (L2 + Dropout) và 3 mô hình Baseline
python -m src.evaluate        # Đánh giá 5 chỉ số trên tập Test, xuất biểu đồ & gọi model_summary
python -m src.model_summary   # Trích xuất Model Summary (.csv/.txt/.json/.md), 5-Fold CV & MLP Tuning
python -m src.report          # Tổng hợp báo cáo Markdown
python render_report.py       # Kết xuất báo cáo HTML tương tác LAB_04_REPORT.html
```

## Step 8: Chạy Bộ Kiểm thử Đơn vị Tự động (`tests/test_lab04.py`)
Kiểm định 8 chốt chặn chất lượng (`AC13`) bằng `pytest`:
```powershell
pytest tests/test_lab04.py -v
```
**Kết quả kỳ vọng:** `8 passed` (`test_01_dataset_integrity` $
ightarrow$ `test_08_tuning_kfold_and_regularization`).

## Step 9: Mở và Thực thi Báo cáo Thực nghiệm (`notebooks/lab04.ipynb`)
1. Mở `notebooks/lab04.ipynb` trong VS Code, chọn Kernel `.venv` và nhấn **Run All**.
2. Hoặc chạy trên trình duyệt:
   ```powershell
   jupyter notebook notebooks/lab04.ipynb
   ```

## Step 10: Kiểm tra các Mô hình Đã Huấn luyện và Bảng Số liệu Mới
- **Thư mục `models/`**:
  - `models/mlp_regression.pth`: Mô hình mạng nơ-ron MLP (`993` tham số, `Best Epoch = 91`).
  - `models/linear_regression.pkl`, `decision_tree.pkl`, `random_forest.pkl`, `standard_scaler.pkl`.
  - `models/mlp_training_history.csv`: Bảng lịch sử `loss`, `mae`, `mape`, `val_loss`, `val_mae`, `val_mape` qua `100` epochs.
- **Thư mục `reports/metrics/`**:
  - `eda_outlier_analysis.csv`: Bảng kiểm toán ngoại lai IQR & Z-Score > 3 cho 14 biến.
  - `test_metrics_comparison.csv`: Bảng so sánh 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` trên tập Test (`102` mẫu).
  - `kfold_cross_validation.csv`: Bảng kết quả 5-Fold Cross-Validation (`MLP` đạt Top 1 với `CV R² = 0.8607 ± 0.0459`).
  - `mlp_architecture_tuning.csv`: Bảng đối sánh 4 kiến trúc MLP (`MLP_Small`, `MLP_Baseline`, `MLP_Wide`, `MLP_Deep`).
  - `model_summary.csv`, `model_summary.txt`, `model_summary.json`: Chẩn đoán thống kê `Log-Likelihood`, `AIC`, `BIC`, `F-statistic`, `Durbin-Watson`, `Jarque-Bera`.

## Step 11: Đọc Báo cáo Tổng hợp (`reports/LAB_04_REPORT.html` & `docs/LAB_04_REPORT.pdf`)
1. Mở `reports/LAB_04_REPORT.html` (hoặc `docs/LAB_04_REPORT.html`) trên trình duyệt để xem báo cáo tương tác 7 Stages kèm đủ các bảng `Bảng 2.1`, `Bảng 6.1–6.4` và `AC01–AC14`.
2. Mở `docs/Model_Summary.md` và các tệp `docs/*_Phase.md` để tra cứu tài liệu kỹ thuật từng giai đoạn.
