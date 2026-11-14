# HƯỚNG DẪN CÀI ĐẶT VÀ THỰC THI CHI TIẾT TỪNG BƯỚC (EXECUTION_GUIDE.MD)

Tài liệu này hướng dẫn chi tiết **10 bước (Step 1 → Step 10)** để giải nén, thiết lập môi trường ảo, cài đặt thư viện, thực thi Pipeline, chạy Jupyter Notebook và kiểm tra kết quả trên cả **Windows PowerShell / Command Prompt** và **macOS / Linux / Git Bash**.

---

## Step 1: Giải nén gói dự án `LAB04_MLP_Regression.zip`

### Trên Windows (File Explorer hoặc PowerShell):
- **Cách 1 (Giao diện)**: Nhấp chuột phải vào tệp `LAB04_MLP_Regression.zip` → chọn **Extract All...** → chọn thư mục đích (ví dụ `D:\MachineLearning\`).
- **Cách 2 (Windows PowerShell)**:
  ```powershell
  Expand-Archive -Path .\LAB04_MLP_Regression.zip -DestinationPath .\
  cd .\LAB04_MLP_Regression
  ```

### Trên macOS / Linux / Git Bash:
```bash
unzip LAB04_MLP_Regression.zip
cd LAB04_MLP_Regression
```

---

## Step 2: Mở thư mục `LAB04_MLP_Regression` bằng Visual Studio Code (VS Code)

Mở ứng dụng **VS Code**, chọn **File → Open Folder...** và trỏ tới thư mục `LAB04_MLP_Regression`, hoặc gõ trực tiếp trên Terminal:
```bash
code .
```
Sau khi mở VS Code, nhấn tổ hợp phím `` Ctrl + ` `` (hoặc chọn **Terminal → New Terminal**) để mở cửa sổ dòng lệnh ngay tại thư mục gốc của dự án.

---

## Step 3: Tạo Môi trường Ảo (Virtual Environment)

Tuyệt đối không cài đặt trực tiếp thư viện vào Python hệ thống để tránh xung đột phiên bản. Hãy tạo môi trường ảo `.venv` riêng cho dự án:

### Trên Windows PowerShell / CMD:
```powershell
python -m venv .venv
```

### Trên macOS / Linux / Git Bash:
```bash
python3 -m venv .venv
```

---

## Step 4: Kích hoạt Môi trường Ảo (Activate Virtual Environment)

### Trên Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Lưu ý: Nếu Windows PowerShell báo lỗi chính sách thực thi `ExecutionPolicy`, hãy chạy lệnh sau trong PowerShell rồi kích hoạt lại)*:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Trên Windows Command Prompt (CMD):
```bat
.venv\Scripts\activate.bat
```

### Trên Windows Git Bash:
```bash
source .venv/Scripts/activate
```

### Trên macOS / Linux:
```bash
source .venv/bin/activate
```
*(Dấu hiệu thành công: Đầu dòng nhắc lệnh xuất hiện tiền tố `(.venv)`).*

---

## Step 5: Cài đặt các Thư viện Phụ thuộc (`requirements.txt`)

Nâng cấp `pip` và cài đặt toàn bộ các thư viện cần thiết (`numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `pyyaml`, `joblib`, `jupyter`, `ipykernel`):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 6: Kiểm tra Bộ dữ liệu (`data/raw/boston_housing.csv`)

Chạy độc lập module `src/data_loader.py` để xác nhận tệp dữ liệu `data/raw/boston_housing.csv` (`506` bản ghi × `14` cột) hợp lệ và không có giá trị thiếu:

```powershell
python -m src.data_loader
```
**Kết quả kỳ vọng trên màn hình**:
- Tên bộ dữ liệu: `Boston Housing Dataset (Harrison & Rubinfeld, 1978)`
- Kích thước: `506 mẫu x 14 cột (13 đặc trưng + 1 mục tiêu 'MEDV')`
- Giá trị thiếu: `0` | Dòng trùng lặp: `0`

---

## Step 7: Chạy Toàn bộ Quy trình Thực nghiệm (`run_lab04.py`)

Thực thi lệnh duy nhất dưới đây tại thư mục gốc để chạy tự động toàn bộ 6 giai đoạn (**Data Loading → EDA → Preprocessing → Training → Evaluation → Reporting**):

```powershell
python run_lab04.py
```

Hoặc chạy từng module riêng lẻ nếu muốn kiểm tra từng pha:
```powershell
python -m src.eda          # Chỉ chạy phân tích khám phá dữ liệu
python -m src.preprocess   # Chỉ chạy chia Train/Val/Test và chuẩn hóa StandardScaler
python -m src.train        # Chỉ chạy huấn luyện MLP Regression và 3 mô hình Baseline
python -m src.evaluate     # Chỉ chạy đánh giá trên tập Test và xuất biểu đồ
python -m src.report       # Chỉ chạy tổng hợp báo cáo thực nghiệm
```

Bạn cũng có thể bỏ qua bước EDA hoặc Preprocessing khi chạy lại Pipeline:
```powershell
python run_lab04.py --skip-eda
python run_lab04.py --skip-eda --skip-preprocess
```

---

## Step 8: Mở và Thực thi Báo cáo Thực nghiệm (`notebooks/lab04.ipynb`)

### Cách 1: Mở trực tiếp trong VS Code
1. Mở tệp `notebooks/lab04.ipynb` trong VS Code.
2. Nhấp vào nút **Select Kernel** ở góc phải phía trên → chọn môi trường `.venv` vừa tạo.
3. Nhấp **Run All** để chạy tuần tự từ Mục 1 đến Mục 20.

### Cách 2: Mở bằng Jupyter Notebook trên trình duyệt
```powershell
jupyter notebook notebooks/lab04.ipynb
```

---

## Step 9: Kiểm tra các Mô hình Đã Huấn luyện và Kết quả Dự đoán

Sau khi chạy xong Step 7, kiểm tra các tệp đầu ra tại:
- **Thư mục `models/`**:
  - `models/mlp_regression.keras`: Mô hình mạng nơ-ron MLP đã huấn luyện.
  - `models/linear_regression.pkl`: Mô hình Hồi quy Tuyến tính.
  - `models/decision_tree.pkl`: Mô hình Cây Quyết định Hồi quy.
  - `models/random_forest.pkl`: Mô hình Rừng Ngẫu nhiên Hồi quy.
  - `models/standard_scaler.pkl`: Bộ chuẩn hóa Z-score đã khớp trên tập Train.
  - `models/mlp_training_history.csv`: Bảng lịch sử `loss`, `mae`, `val_loss`, `val_mae` qua 85/100 epochs (Best Epoch = 70).
- **Thư mục `reports/metrics/` & `reports/predictions/`**:
  - `reports/metrics/test_metrics_comparison.csv`: Bảng so sánh 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` của 4 mô hình trên tập Test.
  - `reports/predictions/test_predictions_latest.csv`: Giá trị thực tế `Actual_MEDV`, giá trị dự đoán và phần dư của từng mẫu trong 102 mẫu Test.
- **Thư mục `reports/figures/` (Single Source of Truth cho cả `reports/` và `docs/`)**:
  - Chứa đầy đủ 13 tệp ảnh biểu đồ `.png` độ phân giải 300 DPI.

---

## Step 10: Đọc Báo cáo Tổng hợp và Nhật ký Thực nghiệm (`reports/` & `logs/`)

1. **Báo cáo tổng hợp Markdown**: Mở tệp `reports/experiment_summary.md` (hoặc mở `reports/report_<run_id>.html` bằng trình duyệt web).
2. **Nhật ký hoạt động chi tiết**: Mở tệp `logs/pipeline_<run_id>.log` để xem dấu thời gian từng bước.
3. **Hồ sơ JSON từng bước**: Mở tệp `progress_results/progress_<run_id>.json` để kiểm tra toàn bộ input/output metadata của lần chạy.
4. **Bảng lịch sử các lần chạy**: Mở tệp `docs/experiment_logs.csv`.
