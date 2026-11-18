"""
Module đánh giá hiệu suất mô hình (`src/evaluate.py`) cho dự án LAB04: MLP Regression.

Trách nhiệm của module:
1. Tải toàn bộ 4 mô hình đã huấn luyện (`MLP Regression`, `Linear Regression`, `Decision Tree`,
   `Random Forest`) từ thư mục `models/` cùng dữ liệu `X_train_scaled`, `y_train`, `X_test_scaled`, `y_test`.
2. Tính toán chính xác các chỉ số đánh giá hồi quy bắt buộc trên cả tập Train và tập Test
   (bao gồm đầy đủ `MSE`, `RMSE`, `MAE`, `MAPE (%)` và `R²`)
   - Thời gian huấn luyện (`Training_Time_Sec`) và thời gian dự đoán (`Prediction_Time_Sec`).
3. Tạo bảng so sánh kết quả thực nghiệm (không tự điền giá trị giả - Rule 5), xác định mô hình
   tốt nhất theo từng chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`).
4. Xuất kết quả dự đoán chi tiết từng mẫu trên tập Test (kèm sai số phần trăm `APE_%`) vào
   `reports/predictions/` và lưu bảng metrics vào `reports/metrics/`.
5. Gọi `src/visualize.py` để tạo các biểu đồ đánh giá (Biểu đồ 6, 7, 8, 9, 10, 11 và 12 - MAPE).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        mean_absolute_error as sk_mae,
        mean_absolute_percentage_error as sk_mape,
        mean_squared_error as sk_mse,
        r2_score as sk_r2,
    )
    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False

from src.model import load_mlp_model
from src.tracking import (
    append_experiment_log,
    create_run,
    finish_run,
    update_progress,
    write_log,
)
from src.utils import (
    ensure_dir,
    get_project_root,
    load_config,
    load_json,
    load_object,
    save_json,
)
from src.visualize import (
    create_stage_figure_dir,
    plot_actual_vs_predicted,
    plot_metric_comparison_bars,
    plot_residual_distributions,
    plot_residuals,
)

ROOT_DIR = get_project_root()
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
PREDICTIONS_DIR = ROOT_DIR / "reports" / "predictions"


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    What: Tính các chỉ số đánh giá hồi quy `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R2` giữa giá trị thực tế và dự đoán.
    Why: Đảm bảo công thức toán học thống nhất cho cả MLP Regression và 3 mô hình Baseline
         theo yêu cầu của Task 5 & Task 6 trong `Practice exercise 4.pdf` VÀ Slide 10 của
         `Chapter5_Artificial Neural_Networks.pdf` (Mean Squared Error & Mean Absolute Percentage Error - MAPE).
    Input:
        - y_true (np.ndarray): Vector giá trị thực tế $y$ (Ground Truth).
        - y_pred (np.ndarray): Vector giá trị dự đoán $\\hat{y}$ (Predictions).
    Output:
        - Dict[str, float]: Từ điển chứa `MSE`, `RMSE`, `MAE`, `MAPE` (đơn vị %), `R2`, `Max_Error`.
    How:
        - $\\text{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$
        - $\\text{RMSE} = \\sqrt{\\text{MSE}}$
        - $\\text{MAE} = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|$
        - $\\text{MAPE (\\%)} = \\frac{100\\%}{N} \\sum_{i=1}^{N} \\frac{|y_i - \\hat{y}_i|}{\\max(|y_i|, \\epsilon)}$
          (Lưu ý: Trên Slide 10 Chapter 5, ký hiệu $\\hat{y}_i$ dưới mẫu số được chú thích là
          *"entries in the ground truth label"*, tương ứng với giá trị thực tế `y_true` trong chuẩn `scikit-learn`).
        - $R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_t.shape != y_p.shape:
        raise ValueError(f"Kích thước y_true {y_t.shape} và y_pred {y_p.shape} không khớp!")

    errors = y_t - y_p
    abs_errors = np.abs(errors)

    if HAS_SKLEARN_METRICS:
        mse = float(sk_mse(y_t, y_p))
        mae = float(sk_mae(y_t, y_p))
        mape = float(sk_mape(y_t, y_p) * 100.0)
        r2 = float(sk_r2(y_t, y_p))
    else:
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(abs_errors))
        denom = np.maximum(np.abs(y_t), 1e-8)
        mape = float(np.mean(abs_errors / denom) * 100.0)
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    rmse = float(np.sqrt(mse))
    max_err = float(np.max(abs_errors))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Max_Error": max_err,
    }


def load_all_trained_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    What: Nạp toàn bộ 4 mô hình đã huấn luyện từ thư mục `models/`.
    Why: Cung cấp các mô hình để đánh giá trên cùng một tập Kiểm tra (`Test Set`).
    Input:
        - config (Optional[Dict[str, Any]]): Cấu hình từ `config.yaml`.
    Output:
        - Dict[str, Any]: Từ điển chứa 4 mô hình theo thứ tự chuẩn.
    How: Gọi `load_mlp_model()` cho `mlp_regression.keras` và `load_object()` cho 3 tệp `.pkl`.
    """
    mlp_path = MODEL_DIR / "mlp_regression.keras"
    lr_path = MODEL_DIR / "linear_regression.pkl"
    dt_path = MODEL_DIR / "decision_tree.pkl"
    rf_path = MODEL_DIR / "random_forest.pkl"

    for path in (mlp_path, lr_path, dt_path, rf_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình '{path.name}' trong {MODEL_DIR}. "
                "Vui lòng chạy src/train.py trước."
            )

    return {
        "MLP Regression": load_mlp_model(mlp_path, config=config),
        "Linear Regression": load_object(lr_path),
        "Decision Tree": load_object(dt_path),
        "Random Forest": load_object(rf_path),
    }


def save_metrics_history(summary_df: pd.DataFrame, run_id: str) -> Path:
    """
    What: Lưu bảng kết quả đánh giá của lần chạy hiện tại vào `reports/metrics/metrics_summary.csv`
          mà không xóa lịch sử các lần chạy trước.
    Why: Đáp ứng yêu cầu trong Part 6.2 & Part 10 ("Không ghi đè kết quả của các lần chạy trước").
    Input:
        - summary_df (pd.DataFrame): Bảng kết quả đánh giá của lần chạy hiện tại.
        - run_id (str): Mã định danh lần chạy.
    Output:
        - Path: Đường dẫn tới `reports/metrics/metrics_summary.csv`.
    How: Chèn cột `Run_ID` vào đầu bảng, đọc dữ liệu lịch sử nếu tệp đã tồn tại, nối thêm
         bản ghi mới và ghi lại xuống đĩa.
    """
    ensure_dir(METRICS_DIR)
    history_path = METRICS_DIR / "metrics_summary.csv"

    df_to_save = summary_df.copy()
    df_to_save.insert(0, "Run_ID", str(run_id))

    if history_path.exists() and history_path.stat().st_size > 0:
        try:
            old_df = pd.read_csv(history_path, encoding="utf-8-sig")
            old_df = old_df[old_df["Run_ID"].astype(str) != str(run_id)]
            combined = pd.concat([old_df, df_to_save], ignore_index=True)
        except Exception:
            combined = df_to_save
    else:
        combined = df_to_save

    combined.to_csv(history_path, index=False, encoding="utf-8-sig")
    combined.to_csv(ROOT_DIR / "reports" / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    return history_path


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ quy trình đánh giá 4 mô hình trên tập Train và tập Test, xuất bảng
          so sánh (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`), tệp dự đoán và các biểu đồ đánh giá.
    Why: Đo lường khách quan khả năng tổng quát hóa trên dữ liệu chưa từng thấy (`Test Set`)
         theo đúng Yêu cầu 5 & 6 của `Practice exercise 4.pdf` và Slide 10 của `Chapter 5`.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy hiện tại.
    Output:
        - Dict[str, Any]: Kết quả đánh giá chi tiết, bảng so sánh trên tập Test và mô hình tốt nhất.
    How:
        1. Nạp dữ liệu `X_train_scaled`, `y_train`, `X_test_scaled`, `y_test`.
        2. Tải 4 mô hình đã huấn luyện và thông tin thời gian huấn luyện từ `training_metadata.json`.
        3. Dự đoán trên cả tập Train và tập Test, tính `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`.
        4. Lưu bảng kết quả, bảng dự đoán trên tập Test và tạo các biểu đồ số 6, 7, 8, 9, 10, 11, 12.
    """
    own_run = run_id is None
    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()
    write_log(run_id, "EVALUATION", "START", "Bắt đầu đánh giá hiệu suất 4 mô hình (MSE, RMSE, MAE, MAPE, R²) trên tập Train và Test.")

    try:
        config = load_config()
        ensure_dir(METRICS_DIR)
        ensure_dir(PREDICTIONS_DIR)
        stage_fig_dir = create_stage_figure_dir(run_id, "evaluation")

        X_train = np.asarray(load_object(PROCESSED_DIR / "X_train_scaled.pkl"), dtype=np.float64)
        y_train = np.asarray(load_object(PROCESSED_DIR / "y_train.pkl"), dtype=np.float64).ravel()
        X_test = np.asarray(load_object(PROCESSED_DIR / "X_test_scaled.pkl"), dtype=np.float64)
        y_test = np.asarray(load_object(PROCESSED_DIR / "y_test.pkl"), dtype=np.float64).ravel()

        models = load_all_trained_models(config)

        train_meta_path = MODEL_DIR / "training_metadata.json"
        training_times = {}
        if train_meta_path.exists():
            meta = load_json(train_meta_path)
            training_times = meta.get("training_times_seconds", {})

        summary_rows: List[Dict[str, Any]] = []
        test_metrics_by_model: Dict[str, Dict[str, float]] = {}
        train_metrics_by_model: Dict[str, Dict[str, float]] = {}
        test_predictions: Dict[str, np.ndarray] = {}

        datasets = {
            "Train": (X_train, y_train),
            "Test": (X_test, y_test),
        }

        print("\n" + "=" * 65)
        print("BƯỚC 5: ĐÁNH GIÁ VÀ SO SÁNH HIỆU SUẤT MÔ HÌNH (MODEL EVALUATION)")
        print("=" * 65)

        for model_name, model_obj in models.items():
            t_time = float(training_times.get(model_name, 0.0))

            for split_name, (X_split, y_split) in datasets.items():
                pred_start = time.perf_counter()
                y_pred = np.asarray(model_obj.predict(X_split), dtype=np.float64).ravel()
                pred_time = round(time.perf_counter() - pred_start, 6)

                metrics = compute_regression_metrics(y_split, y_pred)
                metrics["Training_Time_Sec"] = round(t_time, 4)
                metrics["Prediction_Time_Sec"] = pred_time

                row = {
                    "Model": model_name,
                    "Dataset": split_name,
                    "MSE": round(metrics["MSE"], 4),
                    "RMSE": round(metrics["RMSE"], 4),
                    "MAE": round(metrics["MAE"], 4),
                    "MAPE": round(metrics["MAPE"], 4),
                    "R2": round(metrics["R2"], 4),
                    "Max_Error": round(metrics["Max_Error"], 4),
                    "Training_Time_Sec": round(t_time, 4),
                    "Prediction_Time_Sec": pred_time,
                }
                summary_rows.append(row)

                if split_name == "Test":
                    test_metrics_by_model[model_name] = metrics
                    test_predictions[model_name] = y_pred

                    model_cfg = (
                        config.get("mlp_model", {})
                        if model_name == "MLP Regression"
                        else config.get("baseline_models", {}).get(model_name.lower().replace(" ", "_"), {})
                    )
                    append_experiment_log(
                        run_id=run_id,
                        dataset_name=config.get("data", {}).get("dataset_name", "Boston Housing Dataset"),
                        dataset_shape=[int(len(y_train) + len(y_test)), int(X_test.shape[1] + 1)],
                        model_name=model_name,
                        hyperparameters=model_cfg,
                        training_time_sec=t_time,
                        metrics=metrics,
                        status="SUCCESS",
                    )
                else:
                    train_metrics_by_model[model_name] = metrics

                update_progress(
                    run_id,
                    f"EVALUATION - {model_name} - {split_name}",
                    "SUCCESS",
                    input_data={
                        "model_name": model_name,
                        "dataset": split_name,
                        "n_samples": int(len(y_split)),
                    },
                    output_data=row,
                )

        summary_df = pd.DataFrame(summary_rows)
        test_df = summary_df[summary_df["Dataset"] == "Test"].copy().reset_index(drop=True)

        print("\n--- BẢNG SO SÁNH HIỆU SUẤT TRÊN TẬP TEST (102 MẪU) ---")
        print(test_df[["Model", "MSE", "RMSE", "MAE", "MAPE", "R2", "Training_Time_Sec"]].to_string(index=False))

        # Lưu bảng tổng hợp metrics
        history_csv_path = save_metrics_history(summary_df, run_id)
        test_comparison_csv = METRICS_DIR / "test_metrics_comparison.csv"
        test_df.to_csv(test_comparison_csv, index=False, encoding="utf-8-sig")

        # Lưu kết quả dự đoán chi tiết từng mẫu trên tập Test (kèm APE_%)
        pred_df = pd.DataFrame({
            "Sample_Index": np.arange(len(y_test)),
            "Actual_MEDV": np.round(y_test, 4),
        })
        for model_name, y_pred in test_predictions.items():
            col_prefix = model_name.replace(" ", "_")
            pred_df[f"Pred_{col_prefix}"] = np.round(y_pred, 4)
            pred_df[f"Residual_{col_prefix}"] = np.round(y_test - y_pred, 4)
            pred_df[f"AbsError_{col_prefix}"] = np.round(np.abs(y_test - y_pred), 4)
            pred_df[f"APE_Pct_{col_prefix}"] = np.round(
                np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8) * 100.0, 4
            )

        pred_run_path = PREDICTIONS_DIR / f"test_predictions_{run_id}.csv"
        pred_latest_path = PREDICTIONS_DIR / "test_predictions_latest.csv"
        pred_df.to_csv(pred_run_path, index=False, encoding="utf-8-sig")
        pred_df.to_csv(pred_latest_path, index=False, encoding="utf-8-sig")

        # Xác định mô hình tốt nhất theo từng tiêu chí trên tập Test
        best_by_mse_row = test_df.loc[test_df["MSE"].idxmin()]
        best_by_mae_row = test_df.loc[test_df["MAE"].idxmin()]
        best_by_mape_row = test_df.loc[test_df["MAPE"].idxmin()]
        best_by_r2_row = test_df.loc[test_df["R2"].idxmax()]

        best_models = {
            "best_by_mse": {
                "model": str(best_by_mse_row["Model"]),
                "value": float(best_by_mse_row["MSE"]),
            },
            "best_by_mae": {
                "model": str(best_by_mae_row["Model"]),
                "value": float(best_by_mae_row["MAE"]),
            },
            "best_by_mape": {
                "model": str(best_by_mape_row["Model"]),
                "value": float(best_by_mape_row["MAPE"]),
            },
            "best_by_r2": {
                "model": str(best_by_r2_row["Model"]),
                "value": float(best_by_r2_row["R2"]),
            },
        }

        print("\n--- MÔ HÌNH TỐT NHẤT THEO TỪNG CHỈ SỐ (TEST SET) ---")
        print(f"  * Thấp nhất theo MSE  : {best_models['best_by_mse']['model']} (MSE  = {best_models['best_by_mse']['value']:.4f})")
        print(f"  * Thấp nhất theo MAE  : {best_models['best_by_mae']['model']} (MAE  = {best_models['best_by_mae']['value']:.4f})")
        print(f"  * Thấp nhất theo MAPE : {best_models['best_by_mape']['model']} (MAPE = {best_models['best_by_mape']['value']:.2f}%)")
        print(f"  * Cao nhất theo R²    : {best_models['best_by_r2']['model']} (R²   = {best_models['best_by_r2']['value']:.4f})")

        # Vẽ các biểu đồ đánh giá (Biểu đồ 6, 7, 8, 9, 10, 11, 12)
        saved_figures: List[str] = []
        saved_figures.extend(
            plot_actual_vs_predicted(y_test, test_predictions, test_metrics_by_model, stage_dir=stage_fig_dir)
        )
        saved_figures.extend(plot_residuals(y_test, test_predictions, stage_dir=stage_fig_dir))
        saved_figures.extend(plot_residual_distributions(y_test, test_predictions, stage_dir=stage_fig_dir))
        saved_figures.extend(plot_metric_comparison_bars(test_metrics_by_model, stage_dir=stage_fig_dir))

        # Tự động trích xuất báo cáo thống kê chuyên sâu model_summary cho cả 4 mô hình
        from src import model_summary
        summary_files = model_summary.main(run_id=run_id)

        eval_payload = {
            "run_id": run_id,
            "test_samples": int(len(y_test)),
            "train_samples": int(len(y_train)),
            "test_metrics": {
                k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                for k, v in test_metrics_by_model.items()
            },
            "train_metrics": {
                k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                for k, v in train_metrics_by_model.items()
            },
            "best_models": best_models,
            "saved_files": [
                str(history_csv_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                str(test_comparison_csv.relative_to(ROOT_DIR)).replace("\\", "/"),
                str(pred_run_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                str(pred_latest_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            ],
            "saved_figures": saved_figures,
            "duration_seconds": round(time.perf_counter() - start_time, 4),
        }

        save_json(eval_payload, METRICS_DIR / f"evaluation_{run_id}.json")
        save_json(eval_payload, METRICS_DIR / "evaluation_latest.json")

        update_progress(
            run_id,
            "EVALUATION",
            "SUCCESS",
            input_data={
                "test_samples": int(len(y_test)),
                "models": list(models.keys()),
            },
            output_data=eval_payload,
        )
        write_log(
            run_id,
            "EVALUATION",
            "SUCCESS",
            f"Hoàn tất đánh giá mô hình trong {eval_payload['duration_seconds']}s. "
            f"Best MSE={best_models['best_by_mse']['model']} ({best_models['best_by_mse']['value']:.4f}), "
            f"Best MAPE={best_models['best_by_mape']['model']} ({best_models['best_by_mape']['value']:.2f}%).",
        )

        if own_run:
            finish_run(run_id, "SUCCESS")

        return eval_payload

    except Exception as exc:
        update_progress(run_id, "EVALUATION", "FAILED", error=str(exc))
        write_log(run_id, "EVALUATION", "FAILED", str(exc))
        if own_run:
            finish_run(run_id, "FAILED")
        raise


if __name__ == "__main__":
    main()
