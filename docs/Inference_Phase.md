# Inference Phase — Suy luận Thực tế từ Checkpoint PyTorch (`docs/Inference_Phase.md`)

## 1. Purpose — Mục đích

Giai đoạn **Inference** (`lab04.ipynb` Cell 64 & `src/evaluate.py`) kiểm chứng khả năng **tái nạp độc lập** bộ chuẩn hóa `models/standard_scaler.pkl` và checkpoint trọng số PyTorch `models/mlp_regression.pth` để dự đoán giá nhà trên dữ liệu thô mới ở chế độ suy luận (`model.eval()` + `torch.no_grad()`, tự động tắt `Dropout`), đồng thời đo độ trễ dự đoán hàng loạt trên tập Test.

## 2. Input — Output

### Input

| Thành phần | Mô tả |
|---|---|
| `models/standard_scaler.pkl` | Bộ chuẩn hóa Z-Score đã `fit` trên `404` mẫu `Train` |
| `models/mlp_regression.pth` | Checkpoint PyTorch lưu `state_dict` tối ưu tại **Epoch 41** của `BostonMLPNet` |
| Dữ liệu thô đầu vào (`13` đặc trưng) | Ví dụ: Căn hộ mẫu `#0` (`CRIM=0.00632`, `ZN=18.0`, `INDUS=2.31`, `CHAS=0`, `NOX=0.538`, `RM=6.575`, `AGE=65.2`, `DIS=4.09`, `RAD=1`, `TAX=296.0`, `PTRATIO=15.3`, `B=396.9`, `LSTAT=4.98`) |

### Output

| Thành phần | Mô tả |
|---|---|
| Giá nhà dự đoán (`Predicted MEDV`) | Giá trị số thực liên tục theo đơn vị `$1,000 USD` |
| `reports/predictions/test_predictions_latest.csv` | Bảng dự đoán chi tiết, phần dư và sai số `APE (%)` cho toàn bộ `102` mẫu Test |

## 3. How to do — Quy trình thực hiện

```text
  Raw Input Sample (1, 13) [Unscaled]
                  |
                  v
  [1] Load models/standard_scaler.pkl -> scaler.transform(raw_input)
                  |
                  v
  [2] Load models/mlp_regression.pth (src/model.py -> load_mlp_model)
      Switch to model.eval() + torch.no_grad() [Disable Dropout]
                  |
                  v
  [3] Forward Pass on CUDA GPU / CPU -> Predicted MEDV ($1,000 USD)
```

## 4. Evaluation metrics — Kết quả Suy luận & Độ trễ (`lab04.ipynb` Cell 49 & Cell 64)

- **Kiểm chứng Suy luận Đơn mẫu (Căn hộ mẫu `#0` — Cell 64)**:
  - **Giá thực tế (`Actual MEDV`)**: `$24.00k` (`$24,000 USD`).
  - **Giá dự đoán (`PyTorch MLP`)**: `$27.46k` (`$27,465 USD`).
  - **Sai số tuyệt đối (`AE`)**: `$3.46k`.
  - **Sai số phần trăm (`APE`)**: `14.44%`.
- **Tốc độ Suy luận Hàng loạt (`102` mẫu Test — Cell 49)**:
  - `PyTorch MLP Regression (CUDA)`: **`0.002030s`** (nhanh hơn `5.38` lần so với `Random Forest` `0.010916s`).
  - `Linear Regression`: `0.000413s` | `Decision Tree`: `0.000513s`.

## 5. Used data — Dữ liệu sử dụng

| Dữ liệu | Kích thước | Mục đích |
|---|---|---|
| Căn hộ mẫu `#0` (`df_housing.iloc[[0]]`) | `(1, 13)` | Kiểm chứng quy trình suy luận end-to-end từ dữ liệu thô chưa chuẩn hóa |
| `data/processed/X_test_scaled.pkl` | `(102, 13)` | Đo thời gian suy luận hàng loạt và xuất `test_predictions_latest.csv` |
