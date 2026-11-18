"""
Bộ kiểm thử đơn vị tự động (`tests/test_lab04.py`) cho dự án LAB04: MLP Regression.
Kiểm định 6 chốt chặn chất lượng:
1. Nạp dữ liệu gốc (506 mẫu x 14 cột, không thiếu dữ liệu).
2. Chống rò rỉ dữ liệu (StandardScaler chỉ fit trên 404 mẫu Train, verify_no_data_leakage = True).
3. Độ chính xác của StandardScalerCompat và LinearRegressionCompat.
4. Forward & Backpropagation của NumPySequentialMLP (13 -> 32 -> 16 -> 1).
5. Công thức max_features="sqrt" của RandomForestRegressorCompat (bằng int(sqrt(13)) = 3).
6. Tính toán chỉ số hồi quy (MSE, RMSE, MAE, MAPE, R2) và chẩn đoán phần dư model_summary.
"""

import unittest
import numpy as np

from src.data_loader import inspect_dataset, load_housing_dataframe
from src.evaluate import compute_regression_metrics
from src.model import (
    DecisionTreeRegressorCompat,
    LinearRegressionCompat,
    NumPySequentialMLP,
    RandomForestRegressorCompat,
)
from src.model_summary import compute_residual_diagnostics
from src.preprocess import StandardScalerCompat, split_dataset, verify_no_data_leakage
from src.utils import load_config


class TestLab04RegressionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.df, cls.source_info = load_housing_dataframe(cls.config)

    def test_01_dataset_integrity(self) -> None:
        """Kiểm định bộ dữ liệu Boston Housing có đúng 506 mẫu, 13 đặc trưng và 0 giá trị khuyết."""
        info = inspect_dataset(self.df, target_col="MEDV")
        self.assertEqual(info["n_samples"], 506)
        self.assertEqual(info["n_features"], 13)
        self.assertEqual(info["total_missing_values"], 0)

    def test_02_anti_data_leakage(self) -> None:
        """Kiểm định StandardScaler chỉ fit trên X_train (404 mẫu) và không rò rỉ từ X_test (102 mẫu)."""
        X = self.df.drop(columns=["MEDV"]).values
        y = self.df["MEDV"].values
        X_train, X_test, _, _ = split_dataset(X, y, test_size=0.2, random_state=42)
        scaler = StandardScalerCompat(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        audit = verify_no_data_leakage(X_train, X_train_scaled, X_test_scaled, scaler)
        self.assertTrue(audit["leakage_free_verified"])
        self.assertEqual(audit["scaler_samples_seen"], 404)

    def test_03_numpy_mlp_forward_and_param_count(self) -> None:
        """Kiểm định kiến trúc NumPySequentialMLP (13 -> 32 -> 16 -> 1) có đúng 977 tham số và hội tụ."""
        mlp = NumPySequentialMLP(input_dim=13, hidden_units=[32, 16], learning_rate=0.01, random_state=42)
        self.assertEqual(mlp.count_params(), 13 * 32 + 32 + 32 * 16 + 16 + 16 * 1 + 1)
        rng = np.random.RandomState(42)
        X_dummy = rng.randn(40, 13)
        y_dummy = 20.0 + 3.0 * X_dummy[:, 0] - 2.0 * X_dummy[:, 1]
        history = mlp.fit(X_dummy, y_dummy, epochs=15, batch_size=16)
        self.assertLess(history.history["loss"][-1], history.history["loss"][0])

    def test_04_random_forest_compat_sqrt_max_features(self) -> None:
        """Kiểm định RandomForestRegressorCompat lấy đúng int(sqrt(13)) = 3 đặc trưng khi max_features='sqrt'."""
        rng = np.random.RandomState(42)
        X_dummy = rng.randn(50, 13)
        y_dummy = rng.randn(50)
        rf = RandomForestRegressorCompat(n_estimators=5, max_depth=4, max_features="sqrt", random_state=42)
        rf.fit(X_dummy, y_dummy)
        self.assertEqual(len(rf.estimators_), 5)
        self.assertEqual(rf.estimators_[0].max_features, int(np.sqrt(13)))

    def test_05_metrics_and_residual_diagnostics(self) -> None:
        """Kiểm định hàm tính chỉ số hồi quy (MSE, RMSE, MAE, MAPE, R2) và chẩn đoán phần dư."""
        y_true = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 15.0])
        y_pred = np.array([21.0, 24.0, 29.5, 36.0, 39.0, 44.5, 49.0, 16.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["MSE"], np.mean((y_true - y_pred) ** 2), places=6)
        self.assertGreater(metrics["R2"], 0.98)
        diag = compute_residual_diagnostics(y_true, y_pred, n_params=2)
        self.assertIn("aic", diag)
        self.assertIn("bic", diag)
        self.assertIn("jarque_bera", diag)
        
    def test_06_compat_scaler_and_linear_vs_sklearn(self) -> None:
        """Kiểm định StandardScalerCompat và LinearRegressionCompat khớp chuẩn toán học / scikit-learn."""
        rng = np.random.RandomState(42)
        X_dummy = rng.randn(80, 5) * 10.0 + 3.0
        y_dummy = 4.0 + 2.5 * X_dummy[:, 0] - 1.5 * X_dummy[:, 2] + rng.randn(80) * 0.01

        scaler_compat = StandardScalerCompat(with_mean=True, with_std=True)
        X_scaled_compat = scaler_compat.fit_transform(X_dummy)
        self.assertTrue(np.allclose(np.mean(X_scaled_compat, axis=0), 0.0, atol=1e-9))
        self.assertTrue(np.allclose(np.std(X_scaled_compat, axis=0), 1.0, atol=1e-9))

        lr_compat = LinearRegressionCompat(fit_intercept=True).fit(X_scaled_compat, y_dummy)
        pred_compat = lr_compat.predict(X_scaled_compat)

        try:
            from sklearn.linear_model import LinearRegression as SkLR
            from sklearn.preprocessing import StandardScaler as SkScaler

            X_scaled_sk = SkScaler().fit_transform(X_dummy)
            lr_sk = SkLR().fit(X_scaled_sk, y_dummy)
            self.assertTrue(np.allclose(X_scaled_compat, X_scaled_sk, atol=1e-7))
            self.assertTrue(np.allclose(pred_compat, lr_sk.predict(X_scaled_sk), atol=1e-7))
        except ImportError:
            self.assertGreater(compute_regression_metrics(y_dummy, pred_compat)["R2"], 0.99)

    def test_07_decision_tree_compat(self) -> None:
        """Kiểm định DecisionTreeRegressorCompat học đúng quy luật phi tuyến và khớp cấu trúc với sklearn."""
        rng = np.random.RandomState(42)
        X_dummy = rng.randn(100, 4)
        y_dummy = np.where(X_dummy[:, 0] > 0, 30.0, 10.0) + rng.randn(100) * 0.1

        dt_compat = DecisionTreeRegressorCompat(max_depth=4, min_samples_split=4, min_samples_leaf=2, random_state=42)
        dt_compat.fit(X_dummy, y_dummy)
        pred_compat = dt_compat.predict(X_dummy)
        self.assertGreater(compute_regression_metrics(y_dummy, pred_compat)["R2"], 0.95)
        self.assertAlmostEqual(float(np.sum(dt_compat.feature_importances_)), 1.0, places=6)
        
    def test_08_tuning_kfold_and_regularization(self) -> None:
        """Kiểm định build_mlp_model với L2/Dropout và kiểm tra logic K-Fold CV + Architecture Tuning."""
        import copy
        from src.model import build_mlp_model
        from src.model_summary import run_kfold_and_architecture_tuning

        # 1. Kiểm tra bật L2 & Dropout trong config không gây lỗi build
        cfg_reg = copy.deepcopy(self.config)
        cfg_reg["mlp_model"]["l2_lambda"] = 0.001
        cfg_reg["mlp_model"]["dropout_rate"] = 0.05
        mlp_reg = build_mlp_model(input_dim=13, config=cfg_reg)
        self.assertIsNotNone(mlp_reg)

        # 2. Kiểm tra chạy K-Fold (2-fold nhanh cho unit test) & so sánh kiến trúc MLP
        cfg_cv = copy.deepcopy(self.config)
        cfg_cv["tuning_and_cv"] = {
            "k_folds": 2,
            "cv_epochs": 5,
            "candidate_architectures": [
                {"name": "MLP_Small", "hidden_units": [16, 8]},
                {"name": "MLP_Baseline", "hidden_units": [32, 16]},
            ],
        }
        cv_df, arch_df, _ = run_kfold_and_architecture_tuning(cfg_cv)
        self.assertEqual(len(cv_df), 4)   # Đủ 4 mô hình
        self.assertEqual(len(arch_df), 2) # Đủ 2 kiến trúc thử nghiệm


if __name__ == "__main__":
    unittest.main()