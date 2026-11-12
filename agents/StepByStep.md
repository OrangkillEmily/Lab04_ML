# HƯỚNG DẪN QUY TRÌNH TỪNG BƯỚC & BÀN GIAO DỰ ÁN (STEP-BY-STEP WORKFLOW)

## 1. Bảng Ánh xạ Kiến trúc từ LAB01 sang LAB04 (LAB01 vs. LAB04 Mapping)

| Thành phần | LAB01 (Spam Email Classification) | LAB04 (House Price Prediction — MLP Regression) | Lý do Chuyển đổi & Cải tiến Kỹ thuật |
|---|---|---|---|
| **Bài toán** | Phân loại nhị phân (`Spam=1`, `Ham=0`) | Hồi quy liên tục (`MEDV` tính bằng `$1,000 USD`) | Chuyển từ dự đoán nhãn rời rạc sang dự đoán giá trị thực liên tục |
| **Dữ liệu gốc** | `data/raw/SpamAssassin.csv` (6,046 email) | `data/raw/boston_housing.csv` (506 mẫu × 14 cột) | Tải dữ liệu Boston Housing từ kho lưu trữ chuẩn thay thế `load_boston()` |
| **Tiền xử lý** | Làm sạch HTML/URL + `TfidfVectorizer` | Chia `Train/Val/Test` + `StandardScaler` Z-score | Chuẩn hóa 13 đặc trưng số về $\mu=0, \sigma=1$ giúp Gradient Descent hội tụ |
| **Mô hình chính** | Không có mạng nơ-ron | **Multilayer Perceptron (MLP)**: `13 -> 32 (ReLU) -> 16 (ReLU) -> 1 (Linear)` | Áp dụng kiến trúc ANN/MLP và Backpropagation trong Chapter 5 |
| **Mô hình đối chiếu** | Logistic Regression, Naive Bayes, SVM | `Linear Regression`, `Decision Tree`, `Random Forest` | Đúng yêu cầu Task 6 của đề bài `Practice exercise 4.pdf` |
| **Chỉ số đánh giá** | Accuracy, Precision, Recall, F1, ROC-AUC | `MSE`, `RMSE`, `MAE`, `R² Score`, Training/Prediction Time | Các thước đo chuẩn cho bài toán Hồi quy (Regression) |
| **Biểu đồ** | Phân bố nhãn, Confusion Matrix, ROC Curve | 11+ biểu đồ: Phân phối `MEDV`, Heatmap, Loss/MAE Curves, Actual vs Predicted, Residuals, Bar Comparison | Phản ánh trực quan độ lệch dự đoán và quá trình hội tụ của MLP |
| **Điểm khởi chạy** | `python src/pipeline.py` | `python run_lab04.py` (hoặc `python -m src.pipeline`) | Cung cấp entry point chuẩn tại thư mục gốc |

## 2. Luồng Thực thi 6 Giai đoạn trong Pipeline (`run_lab04.py`)

```text
[1. DATA LOADING]    src/data_loader.py
       │             Đọc data/raw/boston_housing.csv (506 mẫu, 13 features + MEDV)
       ▼
[2. EDA]             src/eda.py
       │             Thống kê mô tả, kiểm tra missing/duplicates, vẽ Biểu đồ 1, 2, 3
       ▼
[3. PREPROCESSING]   src/preprocess.py
       │             Chia Train (404) / Test (102) -> Fit StandardScaler trên Train
       │             -> Transform Train & Test -> Tách Train_sub (323) & Val (81)
       ▼
[4. TRAINING]        src/train.py
       │             Huấn luyện MLP (323 mẫu Train_sub, giám sát trên 81 mẫu Val)
       │             Huấn luyện Linear Regression, Decision Tree, Random Forest
       │             Lưu models/*.keras, models/*.pkl, vẽ Biểu đồ 4 & 5
       ▼
[5. EVALUATION]      src/evaluate.py
       │             Đánh giá cả 4 mô hình trên tập Test (102 mẫu độc lập)
       │             Tính MSE, RMSE, MAE, R² -> Vẽ Biểu đồ 6, 7, 8, 9, 10, 11
       ▼
[6. REPORTING]       src/report.py
                     Xuất reports/experiment_summary.md & reports/report_<run_id>.html
```

## 3. Trách nhiệm Từng Tệp trong Dự án
- `run_lab04.py`: Điểm khởi chạy duy nhất để chạy toàn bộ Pipeline.
- `config.yaml`: Trung tâm lưu trữ siêu tham số của dữ liệu, `StandardScaler`, mạng `MLP` và 3 mô hình baseline.
- `src/data_loader.py`: Tải và kiểm tra dữ liệu nhà ở, hỗ trợ cache và fallback an toàn.
- `src/eda.py`: Phân tích thống kê mô tả, độ lệch và tương quan đặc trưng.
- `src/preprocess.py`: Chia tập dữ liệu và chuẩn hóa `StandardScaler` đảm bảo không rò rỉ dữ liệu.
- `src/model.py`: Định nghĩa kiến trúc `Sequential` MLP và khởi tạo 3 mô hình baseline.
- `src/train.py`: Thực thi huấn luyện, lưu trọng số mô hình và lịch sử hội tụ qua các epoch.
- `src/evaluate.py`: Tính toán `MSE`, `RMSE`, `MAE`, `R²` và xuất bảng dự đoán trên tập Test.
- `src/visualize.py`: Sinh toàn bộ 11+ biểu đồ phân tích ở độ phân giải 300 DPI.
- `src/tracking.py`: Quản lý mã `Run_ID`, tệp nhật ký `.log` và tiến trình `.json`.
- `src/report.py`: Tự động tổng hợp báo cáo thực nghiệm từ kết quả thực tế.
- `notebooks/lab04.ipynb`: Báo cáo thực nghiệm tương tác gồm 20 mục hoàn chỉnh.
