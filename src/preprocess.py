"""
Module tiền xử lý dữ liệu (`src/preprocess.py`) cho dự án LAB04: MLP Regression.

Trách nhiệm của module (Tuân thủ nghiêm ngặt Rule 7 — Chống Data Leakage):
1. Đọc bộ dữ liệu gốc thông qua `src/data_loader.py`.
2. Kiểm tra và xử lý giá trị thiếu hoặc bản ghi trùng lặp (nếu có).
3. Tách ma trận đặc trưng `X` (13 biến độc lập) và vector mục tiêu `y` (`MEDV`).
4. Chia dữ liệu thành tập Huấn luyện (`Train`: 80% = 404 mẫu) và tập Kiểm tra (`Test`: 20% = 102 mẫu)
   bằng `train_test_split` với `random_state=42`.
5. Khởi tạo `StandardScaler`:
   - CHỈ `fit` trên tập `X_train` để tính trung bình $\\mu_{\\text{train}}$ và độ lệch chuẩn $\\sigma_{\\text{train}}$.
   - Áp dụng `transform` trên cả `X_train` và `X_test` bằng cùng một bộ tham số $(\\mu_{\\text{train}}, \\sigma_{\\text{train}})$.
6. Tách tập `Validation` (20% của tập `Train` = 81 mẫu) từ chính tập `X_train_scaled`, giữ lại
   `X_train_sub` (323 mẫu) để phục vụ theo dõi quá trình huấn luyện MLP và `EarlyStopping`.
   Tuyệt đối KHÔNG sử dụng `X_test` làm `validation_data` trong quá trình huấn luyện!
7. Lưu toàn bộ các tập dữ liệu đã xử lý vào `data/processed/` và lưu `models/standard_scaler.pkl`.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split as sk_train_test_split
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.data_loader import extract_features_and_target, load_housing_dataframe
from src.tracking import create_run, finish_run, update_progress, write_log
from src.utils import (
    ensure_dir,
    get_project_root,
    load_config,
    save_json,
    save_object,
    set_global_seeds,
)

ROOT_DIR = get_project_root()
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"


class StandardScalerCompat:
    """
    What: Lớp chuẩn hóa đặc trưng Z-score ($z = (x - \\mu) / \\sigma$) tương thích hoàn toàn
          với `sklearn.preprocessing.StandardScaler`.
    Why: Đảm bảo chương trình hoạt động đồng nhất cả khi môi trường có cài đặt `scikit-learn`
         hoặc khi chạy trong môi trường kiểm thử độc lập chỉ có `NumPy`.
    Input:
        - with_mean (bool): Có trừ đi giá trị trung bình $\\mu$ hay không (mặc định `True`).
        - with_std (bool): Có chia cho độ lệch chuẩn $\\sigma$ hay không (mặc định `True`).
    Output: Đối tượng scaler chứa các thuộc tính `mean_`, `scale_`, `var_`, `n_features_in_`,
            `n_samples_seen_`.
    How: Tính trung bình và độ lệch chuẩn theo từng cột trên tập Train (`axis=0`), thay thế
         các cột có độ lệch chuẩn bằng 0 bằng 1.0 để tránh chia cho 0 khi `transform`.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_seen_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "StandardScalerCompat":
        """
        What: Học vector trung bình `mean_` và độ lệch chuẩn `scale_` từ dữ liệu huấn luyện `X`.
        Why: Thu thập thống kê phân phối chỉ trên tập Train nhằm ngăn chặn rò rỉ dữ liệu (Data Leakage).
        Input:
            - X (DataFrame hoặc ndarray): Ma trận đặc trưng của tập Train.
            - y (Any): Tham số giữ chỗ để tương thích giao diện scikit-learn.
        Output: Chính đối tượng `self` sau khi đã học tham số.
        How: Chuyển `X` sang mảng `float64`, tính `np.mean(X, axis=0)` và `np.std(X, axis=0, ddof=0)`.
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_samples_seen_, self.n_features_in_ = X_arr.shape
        self.mean_ = np.mean(X_arr, axis=0) if self.with_mean else np.zeros(self.n_features_in_)
        self.var_ = np.var(X_arr, axis=0, ddof=0) if self.with_std else np.ones(self.n_features_in_)
        scale = np.sqrt(self.var_)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale if self.with_std else np.ones(self.n_features_in_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        What: Chuẩn hóa ma trận `X` bằng `mean_` và `scale_` đã học từ bước `fit()`.
        Why: Đưa toàn bộ 13 đặc trưng (vốn có đơn vị rất khác nhau như `NOX` ~ 0.5 và `TAX` ~ 400)
             về cùng thang đo chuẩn $\\mu \\approx 0, \\sigma \\approx 1$ để Gradient Descent hội tụ ổn định.
        Input:
            - X (DataFrame hoặc ndarray): Ma trận đặc trưng cần chuẩn hóa.
        Output:
            - np.ndarray: Ma trận đặc trưng đã chuẩn hóa Z-score.
        How: Tính `(X_arr - self.mean_) / self.scale_`.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler chưa được fit! Hãy gọi fit(X_train) trước.")
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """
        What: Kết hợp `fit(X)` và `transform(X)` trên tập dữ liệu huấn luyện.
        Why: Tiện lợi khi áp dụng trên tập `X_train`.
        Input:
            - X (DataFrame hoặc ndarray): Ma trận đặc trưng tập Train.
            - y (Any): Tham số giữ chỗ.
        Output:
            - np.ndarray: Ma trận `X_train` đã được chuẩn hóa.
        How: Gọi tuần tự `self.fit(X, y)` rồi trả về `self.transform(X)`.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_scaled: Any) -> np.ndarray:
        """
        What: Khôi phục ma trận đặc trưng đã chuẩn hóa về đơn vị đo ban đầu.
        Why: Hỗ trợ kiểm tra và diễn giải giá trị thực tế của các đặc trưng sau khi chuẩn hóa.
        Input:
            - X_scaled (ndarray): Ma trận đặc trưng ở thang đo chuẩn hóa.
        Output:
            - np.ndarray: Ma trận đặc trưng ở thang đo gốc.
        How: Tính `X_scaled * self.scale_ + self.mean_`.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler chưa được fit!")
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        return X_arr * self.scale_ + self.mean_


def split_dataset(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, Any]:
    """
    What: Chia ngẫu nhiên ma trận đặc trưng `X` và biến mục tiêu `y` thành hai tập không giao nhau.
    Why: Tách riêng tập Kiểm tra (`Test Set`) ngay từ đầu để đánh giá khách quan khả năng
         tổng quát hóa của các mô hình, hoặc tách tập `Validation` từ tập `Train`.
    Input:
        - X (DataFrame hoặc ndarray): Ma trận đặc trưng đầu vào.
        - y (Series hoặc ndarray): Vector giá trị mục tiêu.
        - test_size (float): Tỷ lệ mẫu dành cho tập thứ hai (mặc định `0.2` tức 20%).
        - random_state (int): Hạt giống ngẫu nhiên cố định (mặc định `42`).
    Output:
        - Tuple[X_part1, X_part2, y_part1, y_part2]: Bốn tập dữ liệu sau khi chia.
    How: Sử dụng `sklearn.model_selection.train_test_split` nếu `scikit-learn` khả dụng;
         nếu không, dùng bộ sinh hoán vị `np.random.RandomState(random_state).permutation(n_samples)`
         tương đương về mặt toán học.
    """
    if HAS_SKLEARN:
        return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)

    n_samples = len(X)
    n_test = int(np.ceil(n_samples * float(test_size)))
    n_train = n_samples - n_test

    rng = np.random.RandomState(int(random_state))
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
    else:
        X_train = np.asarray(X)[train_idx].copy()
        X_test = np.asarray(X)[test_idx].copy()

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
    else:
        y_train = np.asarray(y)[train_idx].copy()
        y_test = np.asarray(y)[test_idx].copy()

    return X_train, X_test, y_train, y_test


def create_scaler(with_mean: bool = True, with_std: bool = True) -> Any:
    """
    What: Khởi tạo đối tượng `StandardScaler`.
    Why: Chuẩn hóa dữ liệu đầu vào trước khi đưa vào mạng nơ-ron MLP và các mô hình hồi quy.
    Input:
        - with_mean (bool): Căn chỉnh trung bình về 0.
        - with_std (bool): Co giãn độ lệch chuẩn về 1.
    Output: Đối tượng `SklearnStandardScaler` (nếu có scikit-learn) hoặc `StandardScalerCompat`.
    How: Kiểm tra cờ `HAS_SKLEARN` và trả về thực thể lớp tương ứng.
    """
    if HAS_SKLEARN:
        return SklearnStandardScaler(with_mean=with_mean, with_std=with_std)
    return StandardScalerCompat(with_mean=with_mean, with_std=with_std)


def verify_no_data_leakage(
    X_train_raw: np.ndarray,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    scaler: Any,
) -> Dict[str, Any]:
    """
    What: Kiểm chứng toán học rằng `StandardScaler` chỉ được `fit` trên tập `X_train` và
          không bị rò rỉ thông tin từ tập `X_test`.
    Why: Đáp ứng yêu cầu kiểm toán Rule 7 & Audit 3 (Data Leakage Audit).
    Input:
        - X_train_raw (np.ndarray): Ma trận `X_train` trước khi chuẩn hóa.
        - X_train_scaled (np.ndarray): Ma trận `X_train` sau khi chuẩn hóa.
        - X_test_scaled (np.ndarray): Ma trận `X_test` sau khi chuẩn hóa.
        - scaler (Any): Đối tượng `StandardScaler` đã khớp.
    Output:
        - Dict[str, Any]: Báo cáo xác nhận độ lệch giữa `scaler.mean_` và trung bình `X_train_raw`.
    How: So sánh `np.max(np.abs(scaler.mean_ - X_train_raw.mean(axis=0))) < 1e-10` và kiểm tra
         trung bình của `X_train_scaled` xấp xỉ 0, độ lệch chuẩn xấp xỉ 1, trong khi trung bình của
         `X_test_scaled` khác 0 (do dùng tham số của Train).
    """
    train_raw_mean = np.mean(np.asarray(X_train_raw, dtype=np.float64), axis=0)
    mean_diff = float(np.max(np.abs(scaler.mean_ - train_raw_mean)))
    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    train_scaled_std_error = float(np.max(np.abs(np.std(X_train_scaled, axis=0) - 1.0)))
    test_scaled_mean_norm = float(np.linalg.norm(np.mean(X_test_scaled, axis=0)))

    leakage_free = (mean_diff < 1e-9) and (train_scaled_max_abs_mean < 1e-9)
    return {
        "leakage_free_verified": bool(leakage_free),
        "scaler_samples_seen": int(getattr(scaler, "n_samples_seen_", len(X_train_raw))),
        "max_diff_scaler_mean_vs_train_raw_mean": mean_diff,
        "max_abs_train_scaled_mean": train_scaled_max_abs_mean,
        "max_abs_train_scaled_std_error": train_scaled_std_error,
        "test_scaled_mean_l2_norm": round(test_scaled_mean_norm, 6),
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ quy trình tiền xử lý dữ liệu từ đọc dữ liệu gốc, chia Train/Test,
          chuẩn hóa `StandardScaler`, tách Validation từ Train, đến lưu trữ artifacts.
    Why: Đảm bảo mọi mô hình (MLP, Linear Regression, Decision Tree, Random Forest) đều sử dụng
         cùng một bộ dữ liệu đã chuẩn hóa và kiểm chứng không có Data Leakage.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy hiện tại.
    Output:
        - Dict[str, Any]: Từ điển chứa thông tin kích thước các tập dữ liệu và kiểm chứng chống rò rỉ.
    How:
        1. Nạp cấu hình từ `config.yaml` và cố định random seed.
        2. Đọc dữ liệu qua `load_housing_dataframe()`, kiểm tra giá trị thiếu và trùng lặp.
        3. Tách `X` và `y`, chia `X_train, X_test, y_train, y_test` theo tỷ lệ 80/20.
        4. Khớp `StandardScaler` trên `X_train` và biến đổi `X_train_scaled`, `X_test_scaled`.
        5. Tách tiếp `X_train_sub, X_val_scaled, y_train_sub, y_val` từ `X_train_scaled` (tỷ lệ 80/20 của Train)
           để theo dõi quá trình huấn luyện MLP mà không chạm vào `X_test_scaled`.
        6. Lưu toàn bộ các tệp `.pkl`, `.csv` và `models/standard_scaler.pkl`.
    """
    own_run = run_id is None
    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()
    write_log(run_id, "PREPROCESSING", "START", "Bắt đầu quy trình tiền xử lý dữ liệu (Rule 7 - Chống Data Leakage).")

    try:
        config = load_config()
        data_cfg = config.get("data", {})
        prep_cfg = config.get("preprocessing", {})

        test_size = float(data_cfg.get("test_size", 0.2))
        val_split = float(data_cfg.get("validation_split", 0.2))
        random_state = int(data_cfg.get("random_state", 42))

        set_global_seeds(random_state)

        # Bước 1 & 2: Tải và kiểm tra dữ liệu
        df_raw, source_info = load_housing_dataframe(config)
        target_col = source_info["target_column"]
        original_rows = int(len(df_raw))

        df = df_raw.copy()
        missing_before = int(df.isnull().sum().sum())
        if missing_before > 0:
            df = df.fillna(df.median(numeric_only=True))

        duplicates_count = int(df.duplicated().sum())
        if prep_cfg.get("drop_duplicates", False) and duplicates_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)

        rows_after_cleaning = int(len(df))

        # Bước 3: Tách X và y, sau đó chia Train / Test (80% / 20%) TRƯỚC KHI scale
        X_df, y_series = extract_features_and_target(df, target_col=target_col)
        feature_names = list(X_df.columns)

        X_train_df, X_test_df, y_train_ser, y_test_ser = split_dataset(
            X_df, y_series, test_size=test_size, random_state=random_state
        )

        # Bước 4 & 5: Fit StandardScaler CHỈ trên X_train, sau đó transform X_train và X_test
        scaler = create_scaler(
            with_mean=bool(prep_cfg.get("with_mean", True)),
            with_std=bool(prep_cfg.get("with_std", True)),
        )
        X_train_scaled = scaler.fit_transform(X_train_df)
        X_test_scaled = scaler.transform(X_test_df)

        y_train = np.asarray(y_train_ser, dtype=np.float64).ravel()
        y_test = np.asarray(y_test_ser, dtype=np.float64).ravel()

        # Bước 6: Tách Validation từ Train (để theo dõi MLP & EarlyStopping mà KHÔNG dùng Test Set)
        X_train_sub, X_val_scaled, y_train_sub, y_val = split_dataset(
            X_train_scaled, y_train, test_size=val_split, random_state=random_state
        )

        leakage_audit = verify_no_data_leakage(
            X_train_raw=X_train_df.values,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            scaler=scaler,
        )

        # Bước 7: Lưu dữ liệu đã xử lý và StandardScaler
        ensure_dir(PROCESSED_DIR)
        ensure_dir(MODEL_DIR)

        artifacts_to_save = {
            "X_train_scaled.pkl": X_train_scaled,
            "X_test_scaled.pkl": X_test_scaled,
            "y_train.pkl": y_train,
            "y_test.pkl": y_test,
            "X_train_sub.pkl": X_train_sub,
            "X_val_scaled.pkl": X_val_scaled,
            "y_train_sub.pkl": y_train_sub,
            "y_val.pkl": y_val,
            "X_train_raw.pkl": X_train_df.values,
            "X_test_raw.pkl": X_test_df.values,
            "feature_names.pkl": feature_names,
        }
        for filename, obj in artifacts_to_save.items():
            save_object(obj, PROCESSED_DIR / filename)

        # Lưu thêm định dạng CSV để sinh viên/giảng viên dễ dàng mở xem trực tiếp
        train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
        train_scaled_df[target_col] = y_train
        train_scaled_df.to_csv(PROCESSED_DIR / "train_scaled.csv", index=False)

        test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
        test_scaled_df[target_col] = y_test
        test_scaled_df.to_csv(PROCESSED_DIR / "test_scaled.csv", index=False)

        val_scaled_df = pd.DataFrame(X_val_scaled, columns=feature_names)
        val_scaled_df[target_col] = y_val
        val_scaled_df.to_csv(PROCESSED_DIR / "val_scaled.csv", index=False)

        scaler_path = save_object(scaler, MODEL_DIR / "standard_scaler.pkl")

        scaler_stats = {
            "feature_names": feature_names,
            "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
            "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
            "leakage_audit": leakage_audit,
        }
        save_json(scaler_stats, PROCESSED_DIR / "scaler_statistics.json")

        duration = round(time.perf_counter() - start_time, 4)

        print("\n" + "=" * 65)
        print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU & CHỐNG RÒ RỈ DỮ LIỆU (PREPROCESSING)")
        print("=" * 65)
        print(f"[INFO] Tổng số mẫu ban đầu          : {original_rows}")
        print(f"[INFO] Tập Train (80% tổng dữ liệu) : {X_train_scaled.shape} (y_train: {y_train.shape})")
        print(f"[INFO]   ├── Train_sub cho MLP (80%): {X_train_sub.shape} (y_train_sub: {y_train_sub.shape})")
        print(f"[INFO]   └── Validation cho MLP(20%): {X_val_scaled.shape} (y_val: {y_val.shape})")
        print(f"[INFO] Tập Test  (20% tách biệt)    : {X_test_scaled.shape} (y_test: {y_test.shape})")
        print(f"[AUDIT] Kiểm tra chống Data Leakage : {'PASSED' if leakage_audit['leakage_free_verified'] else 'FAILED'}")
        print(f"[SAVE] Đã lưu StandardScaler tại    : {scaler_path.relative_to(ROOT_DIR)}")

        output_payload = {
            "original_rows": original_rows,
            "rows_after_cleaning": rows_after_cleaning,
            "missing_values_handled": missing_before,
            "duplicate_rows_detected": duplicates_count,
            "train_shape": list(X_train_scaled.shape),
            "train_sub_shape": list(X_train_sub.shape),
            "val_shape": list(X_val_scaled.shape),
            "test_shape": list(X_test_scaled.shape),
            "feature_names": feature_names,
            "target_column": target_col,
            "parameters": {
                "test_size": test_size,
                "validation_split": val_split,
                "random_state": random_state,
                "scaler": "StandardScaler",
            },
            "leakage_audit": leakage_audit,
            "saved_files": [
                "data/processed/X_train_scaled.pkl",
                "data/processed/X_test_scaled.pkl",
                "data/processed/y_train.pkl",
                "data/processed/y_test.pkl",
                "data/processed/X_train_sub.pkl",
                "data/processed/X_val_scaled.pkl",
                "data/processed/y_train_sub.pkl",
                "data/processed/y_val.pkl",
                "data/processed/train_scaled.csv",
                "data/processed/val_scaled.csv",
                "data/processed/test_scaled.csv",
                "data/processed/scaler_statistics.json",
                "models/standard_scaler.pkl",
            ],
            "duration_seconds": duration,
        }

        update_progress(
            run_id,
            "PREPROCESSING",
            "SUCCESS",
            input_data={
                "dataset": source_info["file_path"],
                "original_rows": original_rows,
            },
            output_data=output_payload,
        )
        write_log(
            run_id,
            "PREPROCESSING",
            "SUCCESS",
            f"Hoàn thành tiền xử lý (Train={X_train_scaled.shape[0]}, Val={X_val_scaled.shape[0]}, "
            f"Test={X_test_scaled.shape[0]}) trong {duration}s. Leakage Free={leakage_audit['leakage_free_verified']}.",
        )

        if own_run:
            finish_run(run_id, "SUCCESS")

        return output_payload

    except Exception as exc:
        update_progress(run_id, "PREPROCESSING", "FAILED", error=str(exc))
        write_log(run_id, "PREPROCESSING", "FAILED", str(exc))
        if own_run:
            finish_run(run_id, "FAILED")
        raise


if __name__ == "__main__":
    main()
