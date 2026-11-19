# Tests Phase — Kiểm thử Tự động & Đảm bảo Chất lượng (`docs/Tests_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Automated Testing & QA** (`tests/test_lab04.py`) cung cấp bộ **`8` bài kiểm thử đơn vị (Unit Tests)** chạy bằng `pytest` nhằm kiểm chứng tự động tính đúng đắn của toàn bộ hệ thống: từ nạp dữ liệu, chống rò rỉ dữ liệu (`StandardScaler`), cấu trúc và khả năng hội tụ của **PyTorch MLP (`BostonMLPNet`)** lẫn fallback `NumPySequentialMLP`, đối chiếu 3 mô hình Baseline với `scikit-learn`, đến giao diện hàm `5-Fold CV` và `Model Summary`.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `tests/test_lab04.py` | Tệp kịch bản kiểm thử gồm `8` unit tests |
| `config.yaml` & `src/*.py` | Cấu hình trung tâm và các mô-đun mã nguồn cần kiểm định |
| `data/raw/boston_housing.csv` | Bộ dữ liệu chuẩn phục vụ kiểm thử tích hợp |

### Output

| Thành phần | Mô tả |
|---|---|
| Kết quả thực thi `pytest -v tests/test_lab04.py` | Trạng thái `PASSED` (`8/8` tests đạt yêu cầu) |
| Xác nhận giao diện tương thích | Đảm bảo `run_kfold_and_architecture_tuning(config)` luôn trả về đúng bộ 3 giá trị `(cv_df, arch_df, ascii_block)` |

## 3. How to do — Quy trình thực hiện

```text
  Run CLI: pytest -v tests/test_lab04.py
                      |
                      v
  [1] Data & Leakage Tests:
      Verify (506, 14), 0 missing, StandardScaler fit ONLY on Train (404)
                      |
                      v
  [2] Model & Compatibility Tests:
      Verify BostonMLPNet (993 params), forward/backward, .pth save/load
      Verify Linear / Decision Tree / Random Forest behavior
                      |
                      v
  [3] Evaluation, CV & Summary Interface Tests:
      Verify compute_regression_metrics (MSE, RMSE, MAE, MAPE, R2, Max_Error)
      Verify run_kfold_and_architecture_tuning() & model_summary outputs
                      |
                      v
              Result: 8 / 8 PASSED
```

## 4. Evaluation metrics — Chỉ số Nghiệm thu Kiểm thử (`lab04.ipynb` Cell 0 & Cell 65)

- **Tỷ lệ vượt qua kiểm thử (Pass Rate)**: **`8 / 8` Unit Tests (`100% PASSED`)**.
- **Các tiêu chí kiểm định bắt buộc**:
  - **Data Integrity**: `df.shape == (506, 14)`, không có `NaN` hoặc dòng trùng lặp.
  - **Anti-Data-Leakage**: `scaler.n_samples_seen_ == 404`, `Mean(X_train_scaled) ≈ 0`, `Std(X_train_scaled) ≈ 1`.
  - **MLP Parameter & Convergence**: Kiến trúc chuẩn `13 -> 32 -> 16 -> 1` có đúng `993` tham số trainable (`448 + 528 + 17`), loss giảm qua các epoch.
  - **Metric Consistency**: Hàm `compute_regression_metrics` tính đúng cả `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` và `Max_Error`.

## 5. Used data — Dữ liệu sử dụng

| Dữ liệu | Nguồn | Mục đích trong kiểm thử |
|---|---|---|
| `data/raw/boston_housing.csv` | Dữ liệu thực (`506 × 14`) | Kiểm thử tải dữ liệu, tách tập và chạy thử `K-Fold CV` |
| Dữ liệu tổng hợp (`synthetic arrays`) | Sinh ngẫu nhiên với `seed = 42` | Kiểm thử đơn vị nhanh cho `NumPySequentialMLP`, `StandardScaler` và `compute_regression_metrics` |
