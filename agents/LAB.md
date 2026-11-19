# THỰC HÀNH 4 (LAB04): HỒI QUY MẠNG NƠ-RON ĐA TẦNG — DỰ ĐOÁN GIÁ NHÀ
**(Multilayer Perceptron Regression Exercise: Predicting House Prices — Practise 4)**

## 1. Phát biểu Bài toán (Problem Statement)
Cho một tập dữ liệu gồm các khu vực nhà ở với các đặc trưng về diện tích quy hoạch, số lượng phòng trung bình, vị trí địa lý, môi trường và chỉ số kinh tế - xã hội, hãy xây dựng mô hình **Multilayer Perceptron (MLP) Regression** để dự đoán giá trị nhà trung vị (`MEDV`) và đối chiếu hiệu năng với các thuật toán hồi quy truyền thống.

## 2. Bộ dữ liệu (Dataset)
Dự án sử dụng bộ dữ liệu **Boston Housing Dataset** (Harrison & Rubinfeld, 1978), gồm **506 mẫu** và **14 biến** (13 biến đặc trưng đầu vào `X` và 1 biến mục tiêu liên tục `y = MEDV` tính bằng ngàn USD):

- `CRIM`: Tỷ lệ tội phạm bình quân đầu người theo thị trấn.
- `ZN`: Tỷ lệ đất ở quy hoạch cho các lô trên 25.000 sq.ft (%).
- `INDUS`: Tỷ lệ diện tích kinh doanh phi bán lẻ (công nghiệp) trên mỗi thị trấn (%).
- `CHAS`: Biến giả sông Charles (`1` nếu khu vực giáp sông, `0` nếu không).
- `NOX`: Nồng độ khí Nitơ Oxit (parts per 10 million).
- `RM`: Số lượng phòng trung bình trên mỗi căn hộ.
- `AGE`: Tỷ lệ căn hộ có chủ sở hữu xây dựng trước năm 1940 (%).
- `DIS`: Khoảng cách có trọng số tới 5 trung tâm việc làm lớn tại Boston.
- `RAD`: Chỉ số khả năng tiếp cận đường cao tốc hướng tâm.
- `TAX`: Thuế suất bất động sản toàn phần trên mỗi `$10,000`.
- `PTRATIO`: Tỷ lệ học sinh / giáo viên theo từng thị trấn.
- `B`: Chỉ số thống kê lịch sử $1000(B_k - 0.63)^2$ (được ghi nhận rõ vấn đề lịch sử & đạo đức dữ liệu khi giải thích lý do `load_boston()` bị loại bỏ từ `scikit-learn 1.2`).
- `LSTAT`: Tỷ lệ dân cư có địa vị kinh tế - xã hội thấp (%).
- `MEDV` (**Target**): Giá trị trung vị của nhà có chủ sở hữu (Đơn vị: `$1,000 USD`).

## 3. Các Mô hình Triển khai (Models Implemented)
1. **Multilayer Perceptron (MLP) Regression** (*Mô hình trọng tâm — Chương 5*):
   - Kiến trúc `Sequential`: `Input(13)` → `Dense(32, activation='relu')` → `Dense(16, activation='relu')` → `Dense(1, activation='linear')`.
   - Bộ tối ưu: `Adam(learning_rate=0.005)`.
   - Hàm mất mát: `Mean Squared Error (MSE)`, theo dõi kèm `Mean Absolute Error (MAE)`.
2. **Linear Regression** (*Mô hình đối chiếu 1*): Hồi quy tuyến tính bình phương tối thiểu (OLS).
3. **Decision Tree Regression** (*Mô hình đối chiếu 2*): Cây quyết định hồi quy CART (`max_depth=6`).
4. **Random Forest Regression** (*Mô hình đối chiếu 3*): Rừng ngẫu nhiên hồi quy (`n_estimators=100, max_depth=10`).

## 4. Quy trình Thực hiện Chuẩn (6 Nhiệm vụ của Practice Exercise 4)
1. **Import Necessary Libraries**: Nạp `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (`train_test_split`, `StandardScaler`, baseline regressors) và `pytorch`.
2. **Load and Preprocess the Data**: Đọc dữ liệu từ `data/raw/boston_housing.csv`, tách `X` và `y`, chia `Train/Test` (80/20), khớp `StandardScaler` chỉ trên tập `Train` và biến đổi `Train`/`Test`, sau đó tách `Validation` (20% của `Train`) để giám sát MLP.
3. **Create a Multilayer Perceptron Model**: Khởi tạo mô hình `Sequential` với 2 tầng ẩn `ReLU` (32 và 16 nơ-ron) và tầng đầu ra `Linear` (1 nơ-ron), biên dịch với `Adam` và `MSE`.
4. **Train the Model**: Huấn luyện MLP trên tập `Train_sub` kết hợp theo dõi `MSE` và `MAE` trên tập `Validation` cùng cơ chế `EarlyStopping`.
5. **Evaluate the Model**: Đánh giá trên tập `Test` độc lập, dự đoán `y_pred` và vẽ biểu đồ đối chiếu `Actual vs. Predicted House Prices`.
6. **Compare Performance Metrics and Analyze Results**: Huấn luyện và đánh giá `Linear Regression`, `Decision Tree`, `Random Forest` trên cùng tập `Test`, so sánh `MSE`, `RMSE`, `MAE`, `R²` với `MLP Regression`.
