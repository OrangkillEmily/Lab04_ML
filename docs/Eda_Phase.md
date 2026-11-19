# EDA Phase — Phân tích Khám phá Dữ liệu (`docs/Eda_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Exploratory Data Analysis (EDA)** (`src/data_loader.py` và `src/eda.py`) nhằm kiểm tra tính toàn vẹn của bộ dữ liệu **Boston Housing**, khảo sát phân phối của biến mục tiêu `MEDV`, đo lường mức độ tương quan tuyến tính Pearson giữa 13 đặc trưng đầu vào với giá nhà, và **kiểm toán ngoại lai (Outlier Audit)** bằng cả **IQR** lẫn **Z-Score (`|z| > 3`)** mà vẫn bảo toàn nguyên vẹn toàn bộ `506` bản ghi gốc.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `data/raw/boston_housing.csv` | Tệp dữ liệu gốc (`506` bản ghi × `14` cột, trích xuất từ CMU StatLib) |
| `config.yaml -> data, paths` | Tên biến mục tiêu (`MEDV`), danh sách `13` cột đặc trưng và đường dẫn lưu báo cáo |

### Output

| Thành phần | Mô tả |
|---|---|
| `reports/metrics/eda_descriptive_statistics.csv` | Bảng thống kê mô tả `14` biến (`count`, `mean`, `std`, `min`, `50%`, `max`, `skewness`) |
| `reports/metrics/eda_outlier_analysis.csv` | Bảng kiểm toán ngoại lai theo quy tắc `IQR` (`[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`) và `|Z-Score| > 3` |
| `reports/figures/target_distribution.png` | Biểu đồ phân phối giá nhà `MEDV` (Histogram + KDE + Boxplot) |
| `reports/figures/correlation_heatmap.png` | Ma trận tương quan Pearson `14 × 14` |
| `reports/figures/feature_target_relationships.png` | Biểu đồ phân tán quan hệ phi tuyến giữa các đặc trưng trọng yếu và `MEDV` |

## 3. How to do — Quy trình thực hiện

```text
  data/raw/boston_housing.csv (506 x 14)
                    |
                    v
    [1] Integrity Check (src/data_loader.py):
        0 missing values | 0 duplicate rows
                    |
                    v
    [2] Descriptive Stats & Skewness (src/eda.py)
                    |
                    v
    [3] Pearson Correlation Matrix (X vs MEDV)
                    |
                    v
    [4] Outlier Audit (IQR & |Z-Score| > 3)
        -> Retain all 506 rows (No row deletion)
                    |
                    v
    Save CSVs (reports/metrics/) & PNGs (reports/figures/)
```

## 4. Evaluation metrics — Chỉ số & Kết quả Thực nghiệm (`lab04.ipynb` Cell 10, 13, 14)

- **Kiểm tra toàn vẹn**: `506` mẫu × `14` cột, `0` giá trị khuyết thiếu, `0` dòng trùng lặp.
- **Thống kê `MEDV`**: `Mean = $22.53k`, `Std = $9.20k`, `Median = $21.20k`, `Min = $5.00k` ($> 0$, đảm bảo công thức `MAPE` không chia cho `0`), `Max = $50.00k` (**`16` mẫu chạm trần `$50,000 USD`** — hiện tượng *right-censoring*).
- **Tương quan Pearson mạnh nhất với `MEDV`**:
  - Tương quan nghịch: `LSTAT` ($r = -0.7377$), `PTRATIO` ($r = -0.5078$), `INDUS` ($r = -0.4837$).
  - Tương quan thuận: `RM` ($r = +0.6954$), `ZN` ($r = +0.3604$), `B` ($r = +0.3335$).
- **Kiểm toán Ngoại lai (`eda_outlier_analysis.csv`)**: Biến có nhiều outlier theo IQR nhất là `B` (`77` mẫu — `15.22%`), `ZN` (`68` mẫu — `13.44%`), `CRIM` (`66` mẫu — `13.04%`), `MEDV` (`40` mẫu — `7.91%`), `CHAS` (`35` mẫu — `6.92%`), `RM` (`30` mẫu — `5.93%`).

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Số mẫu | Số cột | Ghi chú |
|---|---:|---:|---|
| `data/raw/boston_housing.csv` | `506` | `14` | Toàn bộ dữ liệu thô (`13` đặc trưng `CRIM..LSTAT` + biến mục tiêu `MEDV`), giữ nguyên `100%` số dòng |
