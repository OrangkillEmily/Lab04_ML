# THỐNG KÊ CHI TIẾT TOÀN BỘ MÔ HÌNH (COMPREHENSIVE MODEL SUMMARY)

Tài liệu này được trích xuất tự động 100% theo **một bố cục thống nhất chuẩn `statsmodels`** từ trọng số, hệ số hồi quy, cấu trúc cây và phần dư thực tế của toàn bộ các mô hình (4 mô hình chủ chốt + các biến thể kiến trúc MLP PyTorch) thông qua mô-đun `src/model_summary.py`.

## 1. Multilayer Perceptron (MLP) Regression Summary
```text
Multilayer Perceptron (MLP 13-32-16-1) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Architecture: Input(13) -> Linear(32, ReLU) -> Linear(16, ReLU) -> Linear(1, Linear)
Optimizer: Adam (lr=0.005, weight_decay=0.001, dropout=0.05, batch_size=32)
Total Parameters: 993 | Backend: PyTorch (torch.nn.Module | CUDA)
==============================================================================
                           MLP(13-32-16-1) Results                            
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:                 MLP(13-32-16-1)   Log Likelihood:             -1036.909
Date:                 Thu, 19 Nov 2026   AIC:                         2101.817
Time:                         18:37:02   BIC:                         2157.837
Sample:                        0 - 404   HQIC:                        2123.993
Covariance Type:         Adam Backprop   Train R-squared:               0.8857
Test MSE / RMSE:       13.535/   3.679   Test R-squared:                0.8154
Test MAE / MAPE:        2.290/  12.09%   F-statistic:                  232.517
==============================================================================
Layer (type)             Shape     Act  Param #    W_mean     W_std   W_L2Norm
------------------------------------------------------------------------------
hidden_layer_1      (13 -> 32)    relu      448   -0.0539    0.4498     9.2408
dropout_1            (dropout)    none        0    0.0000    0.0000     0.0000
hidden_layer_2      (32 -> 16)    relu      528    0.0224    0.2890     6.5591
dropout_2            (dropout)    none        0    0.0000    0.0000     0.0000
output_layer         (16 -> 1)  linear       17    0.1482    0.4480     1.8875
------------------------------------------------------------------------------
Total Trainable Params: 993 | Best Epoch: 41/56 (Configured: 100)
==============================================================================
Ljung-Box (L1) (Q):               0.01   Jarque-Bera (JB):              613.89
Prob(Q):                          0.92   Prob(JB):                        0.00
Heteroskedasticity (H):           0.84   Skew:                            0.57
Prob(H) (two-sided):              0.20   Kurtosis:                        8.93
Durbin-Watson:                   2.002   Residual Variance:            10.2841
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 2. Linear Regression (OLS) Summary
```text
Linear Regression (OLS) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Estimator: Ordinary Least Squares (fit_intercept=True)
Number of parameters: 14 (1 intercept + 13 standardized features)
==============================================================================
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:                  OLS Regression   Log Likelihood:             -1194.322
Date:                 Thu, 19 Nov 2026   AIC:                         2416.644
Time:                         18:37:02   BIC:                         2472.664
Sample:                        0 - 404   HQIC:                        2438.820
Covariance Type:       nonrobust (OLS)   Train R-squared:               0.7509
Test MSE / RMSE:       24.291/   4.929   Test R-squared:                0.6688
Test MAE / MAPE:        3.189/  16.87%   F-statistic:                   90.427
==============================================================================
                    coef    std err         t    P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const            22.7965      0.236    96.774    0.000      22.333      23.260
CRIM             -1.0021      0.308    -3.250    0.001      -1.608      -0.396
ZN                0.6963      0.370     1.882    0.061      -0.031       1.423
INDUS             0.2781      0.464     0.599    0.549      -0.634       1.190
CHAS              0.7187      0.247     2.914    0.004       0.234       1.204
NOX              -2.0223      0.498    -4.061    0.000      -3.001      -1.043
RM                3.1452      0.329     9.567    0.000       2.499       3.792
AGE              -0.1760      0.407    -0.432    0.666      -0.977       0.625
DIS              -3.0819      0.481    -6.408    0.000      -4.027      -2.136
RAD               2.2514      0.652     3.454    0.001       0.970       3.533
TAX              -1.7670      0.704    -2.508    0.013      -3.152      -0.382
PTRATIO          -2.0378      0.321    -6.357    0.000      -2.668      -1.408
B                 1.1296      0.271     4.166    0.000       0.596       1.663
LSTAT            -3.6117      0.395    -9.133    0.000      -4.389      -2.834
sigma2           22.4183      1.605    13.964    0.000      19.577      25.930
==============================================================================
Ljung-Box (L1) (Q):               1.33   Jarque-Bera (JB):              579.82
Prob(Q):                          0.25   Prob(JB):                        0.00
Heteroskedasticity (H):           0.97   Skew:                            1.38
Prob(H) (two-sided):              0.80   Kurtosis:                        8.18
Durbin-Watson:                   2.114   Residual Variance:            22.4183
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 3. Decision Tree Regression (CART) Summary
```text
Decision Tree Regression (CART) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Hyperparameters: max_depth=6, min_samples_split=5, min_samples_leaf=2
Actual Tree Depth: 6 | Number of Leaf Nodes: 34
==============================================================================
                             DecisionTree Results                             
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:                    DecisionTree   Log Likelihood:              -957.309
Date:                 Thu, 19 Nov 2026   AIC:                         1942.619
Time:                         18:37:02   BIC:                         1998.639
Sample:                        0 - 404   HQIC:                        1964.795
Covariance Type:         CART Impurity   Train R-squared:               0.9229
Test MSE / RMSE:       10.543/   3.247   Test R-squared:                0.8562
Test MAE / MAPE:        2.519/  13.88%   F-statistic:                  359.308
==============================================================================
Feature       importance    std err         z    P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
CRIM              0.0574     0.0086     6.667    0.000      0.0405      0.0742
ZN                0.0000     0.0001     0.000    1.000      0.0000      0.0002
INDUS             0.0044     0.0007     6.667    0.000      0.0031      0.0057
CHAS              0.0000     0.0001     0.000    1.000      0.0000      0.0002
NOX               0.0058     0.0009     6.667    0.000      0.0041      0.0075
RM                0.6278     0.0942     6.667    0.000      0.4432      0.8124
AGE               0.0109     0.0016     6.667    0.000      0.0077      0.0141
DIS               0.0680     0.0102     6.667    0.000      0.0480      0.0880
RAD               0.0017     0.0003     6.667    0.000      0.0012      0.0022
TAX               0.0095     0.0014     6.667    0.000      0.0067      0.0123
PTRATIO           0.0159     0.0024     6.667    0.000      0.0112      0.0206
B                 0.0029     0.0004     6.667    0.000      0.0020      0.0037
LSTAT             0.1957     0.0294     6.667    0.000      0.1381      0.2532
==============================================================================
Ljung-Box (L1) (Q):               0.12   Jarque-Bera (JB):              457.53
Prob(Q):                          0.73   Prob(JB):                        0.00
Heteroskedasticity (H):           0.91   Skew:                           -0.83
Prob(H) (two-sided):              0.52   Kurtosis:                        7.94
Durbin-Watson:                   1.960   Residual Variance:             6.9348
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 4. Random Forest Regression (Ensemble) Summary
```text
Random Forest Regression (Ensemble) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Ensemble: n_estimators=100, max_depth=10, max_features=sqrt
Tree Depth (Mean / Max): 10.0 / 10 | Mean Leaves per Tree: 90.3
==============================================================================
                             RandomForest Results                             
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:                    RandomForest   Log Likelihood:              -859.252
Date:                 Thu, 19 Nov 2026   AIC:                         1746.505
Time:                         18:37:02   BIC:                         1802.525
Sample:                        0 - 404   HQIC:                        1768.681
Covariance Type:         CART Impurity   Train R-squared:               0.9526
Test MSE / RMSE:       11.521/   3.394   Test R-squared:                0.8429
Test MAE / MAPE:        2.099/  10.82%   F-statistic:                  602.577
==============================================================================
Feature       importance    std err         z    P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
CRIM              0.0658     0.0083     7.907    0.000      0.0054      0.3141
ZN                0.0087     0.0025     3.415    0.001      0.0000      0.0645
INDUS             0.0754     0.0112     6.758    0.000      0.0005      0.3612
CHAS              0.0112     0.0027     4.082    0.000      0.0000      0.0980
NOX               0.0888     0.0099     8.997    0.000      0.0042      0.3350
RM                0.2955     0.0209    14.104    0.000      0.0304      0.6848
AGE               0.0256     0.0025    10.234    0.000      0.0036      0.0886
DIS               0.0588     0.0048    12.255    0.000      0.0068      0.1724
RAD               0.0103     0.0022     4.781    0.000      0.0000      0.0477
TAX               0.0353     0.0050     7.070    0.000      0.0009      0.2013
PTRATIO           0.0673     0.0091     7.401    0.000      0.0025      0.3189
B                 0.0250     0.0022    11.493    0.000      0.0022      0.0900
LSTAT             0.2324     0.0176    13.212    0.000      0.0167      0.6459
==============================================================================
Ljung-Box (L1) (Q):               0.50   Jarque-Bera (JB):             2170.26
Prob(Q):                          0.48   Prob(JB):                        0.00
Heteroskedasticity (H):           0.98   Skew:                            1.86
Prob(H) (two-sided):              0.88   Kurtosis:                       13.73
Durbin-Watson:                   2.069   Residual Variance:             4.2679
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 5. MLP_Small (13->16->8->1) Summary
```text
MLP_Small (13->16->8->1) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Architecture: Input(13) -> Linear(16, ReLU) -> Linear(8, ReLU) -> Linear(1, Linear)
Optimizer: Adam (lr=0.005, weight_decay=0.001, dropout=0.05, batch_size=32)
Total Parameters: 369 | Backend: PyTorch (torch.nn.Module | CUDA)
==============================================================================
                         MLP_Small(13-16-8-1) Results                         
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:            MLP_Small(13-16-8-1)   Log Likelihood:             -1031.477
Date:                 Thu, 19 Nov 2026   AIC:                         2090.953
Time:                         18:37:27   BIC:                         2146.973
Sample:                        0 - 404   HQIC:                        2113.129
Covariance Type:         Adam Backprop   Train R-squared:               0.8888
Test MSE / RMSE:       12.499/   3.535   Test R-squared:                0.8296
Test MAE / MAPE:        2.288/  11.84%   F-statistic:                  239.672
==============================================================================
Layer (type)             Shape     Act  Param #    W_mean     W_std   W_L2Norm
------------------------------------------------------------------------------
hidden_layer_1      (13 -> 16)    relu      224   -0.0720    0.4641     6.7731
dropout_1            (dropout)    none        0    0.0000    0.0000     0.0000
hidden_layer_2       (16 -> 8)    relu      136    0.2132    0.4012     5.1402
dropout_2            (dropout)    none        0    0.0000    0.0000     0.0000
output_layer          (8 -> 1)  linear        9    0.3316    0.5208     1.7464
------------------------------------------------------------------------------
Total Trainable Params: 369 | Best Epoch: 93/100 (Configured: 100)
==============================================================================
Ljung-Box (L1) (Q):               0.18   Jarque-Bera (JB):              480.87
Prob(Q):                          0.68   Prob(JB):                        0.00
Heteroskedasticity (H):           0.87   Skew:                            0.97
Prob(H) (two-sided):              0.31   Kurtosis:                        7.98
Durbin-Watson:                   1.950   Residual Variance:            10.0113
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 6. MLP_Wide (13->64->32->1) Summary
```text
MLP_Wide (13->64->32->1) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Architecture: Input(13) -> Linear(64, ReLU) -> Linear(32, ReLU) -> Linear(1, Linear)
Optimizer: Adam (lr=0.005, weight_decay=0.001, dropout=0.05, batch_size=32)
Total Parameters: 3009 | Backend: PyTorch (torch.nn.Module | CUDA)
==============================================================================
                         MLP_Wide(13-64-32-1) Results                         
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:            MLP_Wide(13-64-32-1)   Log Likelihood:             -1030.410
Date:                 Thu, 19 Nov 2026   AIC:                         2088.820
Time:                         18:37:27   BIC:                         2144.840
Sample:                        0 - 404   HQIC:                        2110.996
Covariance Type:         Adam Backprop   Train R-squared:               0.8893
Test MSE / RMSE:       13.769/   3.711   Test R-squared:                0.8122
Test MAE / MAPE:        2.442/  12.92%   F-statistic:                  241.100
==============================================================================
Layer (type)             Shape     Act  Param #    W_mean     W_std   W_L2Norm
------------------------------------------------------------------------------
hidden_layer_1      (13 -> 64)    relu      896   -0.0376    0.4064    11.7718
dropout_1            (dropout)    none        0    0.0000    0.0000     0.0000
hidden_layer_2      (64 -> 32)    relu     2080    0.0233    0.2039     9.2868
dropout_2            (dropout)    none        0    0.0000    0.0000     0.0000
output_layer         (32 -> 1)  linear       33    0.1205    0.2709     1.6772
------------------------------------------------------------------------------
Total Trainable Params: 3009 | Best Epoch: 34/49 (Configured: 100)
==============================================================================
Ljung-Box (L1) (Q):               0.17   Jarque-Bera (JB):              267.62
Prob(Q):                          0.68   Prob(JB):                        0.00
Heteroskedasticity (H):           0.98   Skew:                            0.38
Prob(H) (two-sided):              0.90   Kurtosis:                        6.92
Durbin-Watson:                   2.039   Residual Variance:             9.9585
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 7. MLP_Deep (13->64->32->16->1) Summary
```text
MLP_Deep (13->64->32->16->1) Model Summary – House Price Prediction
==============================================================================
Data: Boston Housing Dataset (data/raw/boston_housing.csv)
Number of observations: 506 (Train: 404 | Test: 102)
Architecture: Input(13) -> Linear(64, ReLU) -> Linear(32, ReLU) -> Linear(16, ReLU) -> Linear(1, Linear)
Optimizer: Adam (lr=0.005, weight_decay=0.001, dropout=0.05, batch_size=32)
Total Parameters: 3521 | Backend: PyTorch (torch.nn.Module | CUDA)
==============================================================================
                       MLP_Deep(13-64-32-16-1) Results                        
==============================================================================
Dep. Variable:                    MEDV   No. Observations:                 404
Model:         MLP_Deep(13-64-32-16-1)   Log Likelihood:             -1006.192
Date:                 Thu, 19 Nov 2026   AIC:                         2040.383
Time:                         18:37:27   BIC:                         2096.403
Sample:                        0 - 404   HQIC:                        2062.559
Covariance Type:         Adam Backprop   Train R-squared:               0.9018
Test MSE / RMSE:       13.765/   3.710   Test R-squared:                0.8123
Test MAE / MAPE:        2.637/  13.26%   F-statistic:                  275.631
==============================================================================
Layer (type)             Shape     Act  Param #    W_mean     W_std   W_L2Norm
------------------------------------------------------------------------------
hidden_layer_1      (13 -> 64)    relu      896   -0.0207    0.3992    11.5290
dropout_1            (dropout)    none        0    0.0000    0.0000     0.0000
hidden_layer_2      (64 -> 32)    relu     2080    0.0004    0.1989     9.0026
dropout_2            (dropout)    none        0    0.0000    0.0000     0.0000
hidden_layer_3      (32 -> 16)    relu      528    0.0225    0.2471     5.6138
dropout_3            (dropout)    none        0    0.0000    0.0000     0.0000
output_layer         (16 -> 1)  linear       17    0.2385    0.3445     1.6761
------------------------------------------------------------------------------
Total Trainable Params: 3521 | Best Epoch: 31/46 (Configured: 100)
==============================================================================
Ljung-Box (L1) (Q):               0.49   Jarque-Bera (JB):               78.97
Prob(Q):                          0.48   Prob(JB):                        0.00
Heteroskedasticity (H):           0.91   Skew:                            0.44
Prob(H) (two-sided):              0.51   Kurtosis:                        4.98
Durbin-Watson:                   2.011   Residual Variance:             8.8334
==============================================================================

Notes:
[1] Toàn bộ tham số và chỉ số được trích xuất động 100% từ mô hình thực nghiệm.
------------------------------------------------------------------------------
```

## 8. K-Fold Cross-Validation & Architecture Comparison Summary
```text
K-Fold Cross-Validation & MLP Architecture Tuning Summary
==============================================================================
1. 5-FOLD CROSS-VALIDATION RESULTS (506 samples, leak-free per-fold scaling)
------------------------------------------------------------------------------
            Model  K_Folds  CV_MSE_Mean  CV_MSE_Std  CV_RMSE_Mean  CV_MAE_Mean  CV_MAPE_Mean  CV_R2_Mean  CV_R2_Std
   MLP Regression        5      12.4888      2.7870        3.5177       2.4428       11.9853      0.8483     0.0352
    Random Forest        5      12.5518      1.9864        3.5339       2.3361       11.6827      0.8459     0.0395
    Decision Tree        5      18.3829     11.6161        4.1529       2.8782       14.5435      0.7832     0.1212
Linear Regression        5      23.4886      2.0601        4.8428       3.3885       17.0105      0.7152     0.0419
==============================================================================
2. MLP ARCHITECTURE & HYPERPARAMETER TUNING COMPARISON (PyTorch)
------------------------------------------------------------------------------
                Architecture Hidden_Units  Total_Params  Train_MSE  Val_MSE  Test_MSE  Test_RMSE  Test_MAE  Test_MAPE  Test_R2
    MLP_Small (13->16->8->1)      [16, 8]           369     8.4158  14.6431   12.4992     3.5354    2.2884    11.8373   0.8296
MLP_Baseline (13->32->16->1)     [32, 16]           993     8.4487  15.8257   13.5348     3.6790    2.2902    12.0913   0.8154
    MLP_Wide (13->64->32->1)     [64, 32]          3009     8.3452  14.6707   13.7687     3.7106    2.4422    12.9155   0.8122
MLP_Deep (13->64->32->16->1) [64, 32, 16]          3521     7.4476  12.8325   13.7648     3.7101    2.6371    13.2612   0.8123
==============================================================================
```
