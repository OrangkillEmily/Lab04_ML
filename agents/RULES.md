# QUY TẮC LẬP TRÌNH, KIẾN TRÚC VÀ KIỂM ĐỊNH CHẤT LƯỢNG (LAB04 RULES)

## Quy tắc 1: Kế thừa Kiến trúc LAB01 có chọn lọc (Reference Architecture)
- **Được phép kế thừa**: Tư duy phân tách module trong `src/`, quản lý cấu hình tập trung qua `config.yaml`, hệ thống theo dõi thực nghiệm `Run_ID` (`logs/`, `progress_results/`, `reports/`), và phong cách viết tài liệu học thuật.
- **Tuyệt đối không sao chép**: Không giữ lại bất kỳ đoạn mã xử lý văn bản Spam Email (`TF-IDF`, `clean_text`), không giữ các thuật toán phân loại (`LogisticRegression`, `NaiveBayes`, `SVM`), không dùng các chỉ số phân loại (`Accuracy`, `F1`, `ROC-AUC`, `Confusion Matrix`) của LAB01. Toàn bộ dự án phục vụ duy nhất bài toán **House Price Prediction (Regression)**.

## Quy tắc 2: Bám sát Đề bài Practice Exercise 4 & Lý thuyết Chapter 5
- Triển khai đầy đủ **4 mô hình hồi quy**: `MLP Regression` (trọng tâm), `Linear Regression`, `Decision Tree Regression` và `Random Forest Regression`.
- Tính toán đầy đủ các chỉ số: `MSE`, `RMSE`, `MAE` và `R²` trên cùng một tập `Test`.
- Giải thích rõ bản chất toán học của nơ-ron nhân tạo ($z = \sum w_i x_i + b, a = f(z)$), hàm kích hoạt (`ReLU`, `Linear`, `Sigmoid`, `Tanh`), `Forward Propagation`, `Backpropagation`, `MSE Loss`, `Adam Optimizer` và `Regularization` (`L1`, `L2`, `Dropout`, `EarlyStopping`) theo đúng nội dung `Chapter5_Artificial Neural_Networks.pdf`.

## Quy tắc 3: Code Đơn giản, Tường minh và Dễ hiểu (Simple & Understandable Code)
- Phân tách trách nhiệm rõ ràng giữa các module (`data_loader.py`, `eda.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `tracking.py`, `report.py`, `pipeline.py`, `utils.py`).
- Đặt tên biến có ý nghĩa học thuật (`X_train_scaled`, `X_val_scaled`, `X_test_scaled`, `y_pred`).
- Không over-engineer và không viết trùng lặp logic giữa Notebook và `src/`.

## Quy tắc 4: Bắt buộc có Docstring 5W1H cho mọi Hàm (Documentation Is Mandatory)
- Mọi hàm trong `src/` phải có docstring giải thích đầy đủ 5 mục:
  - **What**: Hàm thực hiện nhiệm vụ gì?
  - **Why**: Tại sao cần hàm này?
  - **Input**: Các tham số đầu vào là gì?
  - **Output**: Kết quả trả về là gì?
  - **How**: Logic hoạt động bên trong như thế nào?

## Quy tắc 5: Trung thực Thực nghiệm — Không Bịa Kết quả (No Fake Results)
- Toàn bộ bảng số liệu `MSE`, `RMSE`, `MAE`, `R²`, lịch sử hội tụ `mlp_training_history.csv`, tệp dự đoán `test_predictions_latest.csv`, biểu đồ `.png` và nhật ký `.log` đều phải được sinh ra từ quá trình chạy code thực tế trên dữ liệu thật (`506` bản ghi Boston Housing).
- Ghi nhận minh bạch thông tin môi trường thực thi trong metadata và báo cáo.

## Quy tắc 6: Đảm bảo Khả năng Tái lập Kết quả (Reproducibility)
- Cố định hạt giống ngẫu nhiên `random_state = 42` trong `config.yaml` và áp dụng đồng bộ cho `random`, `numpy.random` và `tensorflow.random` thông qua hàm `set_global_seeds()` trong `src/utils.py`.

## Quy tắc 7: Chống Rò rỉ Dữ liệu Tuyệt đối (Data Leakage Prevention)
1. Chia dữ liệu gốc thành `Train` (80% = 404 mẫu) và `Test` (20% = 102 mẫu) **trước khi** chuẩn hóa.
2. Chỉ gọi `scaler.fit()` trên `X_train`, sau đó dùng cùng `scaler` đó để `transform(X_train)` và `transform(X_test)`.
3. Tách tập `Validation` (20% của `Train` = 81 mẫu) từ `X_train_scaled` để theo dõi quá trình huấn luyện MLP và `EarlyStopping`.
4. **Cấm tuyệt đối** truyền `validation_data=(X_test_scaled, y_test)` vào hàm `model.fit()`. Tập `Test` chỉ được mở ra duy nhất một lần ở giai đoạn `evaluate.py`.
