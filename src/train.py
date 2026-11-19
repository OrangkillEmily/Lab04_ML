"""
Module huấn luyện mô hình (`src/train.py`) cho dự án LAB04: MLP Regression.

Trách nhiệm của module:
1. Tải các tập dữ liệu huấn luyện và kiểm định (`X_train_scaled`, `y_train`, `X_train_sub`,
   `X_val_scaled`, `y_train_sub`, `y_val`) đã được tiền xử lý từ `data/processed/`.
2. Khởi tạo và biên dịch mô hình **Multilayer Perceptron (MLP) Regression** (`13 -> 32 -> 16 -> 1`).
3. Huấn luyện MLP trên tập `X_train_sub` (323 mẫu) và theo dõi quá trình hội tụ MSE/MAE trên
   tập `X_val_scaled` (81 mẫu) kết hợp `EarlyStopping` (tuyệt đối KHÔNG sử dụng tập Test `X_test_scaled`
   trong suốt quá trình huấn luyện để đảm bảo Rule 7 - Chống Data Leakage).
4. Huấn luyện 3 mô hình học máy đối chiếu (Baseline Models) trên tập `X_train_scaled`:
   - `Linear Regression`
   - `Decision Tree Regressor`
   - `Random Forest Regressor`
5. Đo thời gian huấn luyện thực tế của từng mô hình và lưu toàn bộ artifacts:
   - `models/mlp_regression.pth` (cùng `models/mlp_regression_weights.npz`)
   - `models/linear_regression.pkl`
   - `models/decision_tree.pkl`
   - `models/random_forest.pkl`
   - `models/mlp_training_history.csv` & `models/mlp_training_history.json`
   - `models/training_metadata.json`
6. Tạo các biểu đồ quá trình huấn luyện (Biểu đồ 4, Biểu đồ 5, Biểu đồ thời gian huấn luyện
   và Biểu đồ phân vùng Train/Val/Test).
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.model import build_baseline_models, build_mlp_model, get_backend_info
from src.tracking import create_run, finish_run, update_progress, write_log
from src.utils import (
    ensure_dir,
    get_current_timestamp,
    get_project_root,
    load_config,
    load_object,
    save_json,
    save_object,
    set_global_seeds,
)
from src.visualize import (
    create_stage_figure_dir,
    plot_mlp_training_loss_curve,
    plot_train_val_test_split,
    plot_training_time_comparison,
    plot_training_vs_validation_loss,
)

ROOT_DIR = get_project_root()
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"


def load_training_splits() -> Dict[str, Any]:
    """
    What: Đọc các tập dữ liệu huấn luyện và kiểm định từ `data/processed/`.
    Why: Đảm bảo quá trình huấn luyện tách biệt hoàn toàn với bước tiền xử lý và không
         chạm vào tập dữ liệu kiểm tra `X_test_scaled` trong khi huấn luyện.
    Input: Không có tham số đầu vào.
    Output:
        - Dict[str, Any]: Từ điển chứa `X_train_scaled`, `y_train`, `X_train_sub`, `y_train_sub`,
          `X_val_scaled`, `y_val` và `y_test` (chỉ dùng `y_test` để vẽ biểu đồ kích thước phân vùng).
    How: Sử dụng `load_object()` đọc các tệp `.pkl` trong `data/processed/`.
    """
    required_files = [
        "X_train_scaled.pkl",
        "y_train.pkl",
        "X_train_sub.pkl",
        "y_train_sub.pkl",
        "X_val_scaled.pkl",
        "y_val.pkl",
        "y_test.pkl",
    ]
    for fname in required_files:
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Không tìm thấy tệp dữ liệu '{fname}' tại {PROCESSED_DIR}. "
                "Vui lòng chạy src/preprocess.py trước."
            )

    return {
        "X_train_scaled": np.asarray(load_object(PROCESSED_DIR / "X_train_scaled.pkl"), dtype=np.float64),
        "y_train": np.asarray(load_object(PROCESSED_DIR / "y_train.pkl"), dtype=np.float64).ravel(),
        "X_train_sub": np.asarray(load_object(PROCESSED_DIR / "X_train_sub.pkl"), dtype=np.float64),
        "y_train_sub": np.asarray(load_object(PROCESSED_DIR / "y_train_sub.pkl"), dtype=np.float64).ravel(),
        "X_val_scaled": np.asarray(load_object(PROCESSED_DIR / "X_val_scaled.pkl"), dtype=np.float64),
        "y_val": np.asarray(load_object(PROCESSED_DIR / "y_val.pkl"), dtype=np.float64).ravel(),
        "y_test": np.asarray(load_object(PROCESSED_DIR / "y_test.pkl"), dtype=np.float64).ravel(),
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ giai đoạn huấn luyện cho mô hình MLP Regression và 3 mô hình Baseline.
    Why: Đáp ứng Yêu cầu 3, 4 và 6 trong `Practice exercise 4`, lưu trữ đầy đủ mô hình,
         lịch sử huấn luyện và biểu đồ hội tụ.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy hiện tại.
    Output:
        - Dict[str, Any]: Thông tin thời gian huấn luyện, lịch sử loss/MAE của MLP và đường dẫn các mô hình đã lưu.
    How:
        1. Nạp cấu hình từ `config.yaml` và cố định random seed.
        2. Đọc các tập `X_train_sub`, `X_val_scaled`, `X_train_scaled`.
        3. Khởi tạo và huấn luyện mô hình MLP (`epochs=100`, `batch_size=32`, `EarlyStopping`).
        4. Lưu mô hình MLP vào `models/mlp_regression.pth` và lịch sử vào `models/mlp_training_history.csv`.
        5. Khởi tạo và huấn luyện `Linear Regression`, `Decision Tree`, `Random Forest` trên tập Train.
        6. Lưu các mô hình baseline (`.pkl`), vẽ biểu đồ huấn luyện và cập nhật tracking.
    """
    own_run = run_id is None
    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()
    write_log(run_id, "TRAINING", "START", "Bắt đầu huấn luyện mô hình MLP Regression và 3 mô hình Baseline.")

    try:
        config = load_config()
        random_state = int(config.get("data", {}).get("random_state", 42))
        set_global_seeds(random_state)

        splits = load_training_splits()
        X_train_scaled = splits["X_train_scaled"]
        y_train = splits["y_train"]
        X_train_sub = splits["X_train_sub"]
        y_train_sub = splits["y_train_sub"]
        X_val_scaled = splits["X_val_scaled"]
        y_val = splits["y_val"]
        y_test = splits["y_test"]

        ensure_dir(MODEL_DIR)
        stage_fig_dir = create_stage_figure_dir(run_id, "training")

        mlp_cfg = config.get("mlp_model", {})
        epochs = int(mlp_cfg.get("epochs", 100))
        batch_size = int(mlp_cfg.get("batch_size", 32))
        es_cfg = mlp_cfg.get("early_stopping", {"enabled": True, "patience": 15, "restore_best_weights": True})

        backend_info = get_backend_info()
        training_times: Dict[str, float] = {}

        print("\n" + "=" * 65)
        print("BƯỚC 4: HUẤN LUYỆN MÔ HÌNH MLP & BASELINE MODELS (MODEL TRAINING)")
        print("=" * 65)
        print(f"[INFO] Backend MLP      : {backend_info['mlp_backend']}")
        print(f"[INFO] Backend Baselines: {backend_info['baseline_backend']}")

        # ------------------------------------------------------------------
        # 1. HUẤN LUYỆN MULTILAYER PERCEPTRON (MLP) REGRESSION
        # ------------------------------------------------------------------
        input_dim = int(X_train_scaled.shape[1])
        mlp_model = build_mlp_model(input_dim=input_dim, config=config)

        mlp_start = time.perf_counter()
        history_obj = mlp_model.fit(
            X_train_sub,
            y_train_sub,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            early_stopping_cfg=es_cfg,
            verbose=0,
        )
        best_epoch = int(getattr(mlp_model, "best_epoch_", int(np.argmin(history_obj.history["val_loss"]) + 1)))

        mlp_elapsed = round(time.perf_counter() - mlp_start, 4)
        training_times["MLP Regression"] = mlp_elapsed

        mlp_save_path = MODEL_DIR / "mlp_regression.pth"
        mlp_model.save(mlp_save_path)

        history_dict = {
            k: [round(float(v), 6) for v in vals]
            for k, vals in history_obj.history.items()
        }
        epochs_ran = len(history_dict.get("loss", []))

        history_df = pd.DataFrame(history_dict)
        history_df.insert(0, "epoch", np.arange(1, epochs_ran + 1))
        history_df.to_csv(MODEL_DIR / "mlp_training_history.csv", index=False)
        save_json(
            {
                "run_id": run_id,
                "epochs_configured": epochs,
                "epochs_executed": epochs_ran,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
                "history": history_dict,
            },
            MODEL_DIR / "mlp_training_history.json",
        )

        print(
            f"[DONE] MLP Regression     : {mlp_elapsed:.4f}s | Epochs={epochs_ran}/{epochs} (Best Epoch={best_epoch}) "
            f"| Final Train MSE={history_dict['loss'][-1]:.4f}, Best Val MSE={min(history_dict['val_loss']):.4f}"
        )

        update_progress(
            run_id,
            "TRAINING - MLP Regression",
            "SUCCESS",
            input_data={
                "train_sub_shape": list(X_train_sub.shape),
                "val_shape": list(X_val_scaled.shape),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": mlp_cfg.get("learning_rate", 0.005),
            },
            output_data={
                "epochs_executed": epochs_ran,
                "best_epoch": best_epoch,
                "initial_train_mse": history_dict["loss"][0],
                "final_train_mse": history_dict["loss"][-1],
                "best_val_mse": float(min(history_dict["val_loss"])),
                "best_val_mae": float(min(history_dict["val_mae"])),
                "training_time_seconds": mlp_elapsed,
                "saved_model": "models/mlp_regression.pth",
            },
        )
        write_log(
            run_id,
            "TRAINING - MLP Regression",
            "SUCCESS",
            f"Huấn luyện MLP hoàn tất sau {epochs_ran} epochs ({mlp_elapsed}s), Best Val MSE={min(history_dict['val_loss']):.4f}.",
        )

        # ------------------------------------------------------------------
        # 2. HUẤN LUYỆN 3 MÔ HÌNH ĐỐI CHIẾU (BASELINE MODELS)
        # ------------------------------------------------------------------
        baseline_models = build_baseline_models(config)
        baseline_filenames = {
            "Linear Regression": "linear_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "Random Forest": "random_forest.pkl",
        }

        for model_name, model_inst in baseline_models.items():
            b_start = time.perf_counter()
            model_inst.fit(X_train_scaled, y_train)
            b_elapsed = round(time.perf_counter() - b_start, 4)
            training_times[model_name] = b_elapsed

            b_path = MODEL_DIR / baseline_filenames[model_name]
            save_object(model_inst, b_path)

            print(f"[DONE] {model_name:<18} : {b_elapsed:.4f}s | Đã lưu tại: {b_path.relative_to(ROOT_DIR)}")

            update_progress(
                run_id,
                f"TRAINING - {model_name}",
                "SUCCESS",
                input_data={
                    "train_shape": list(X_train_scaled.shape),
                    "parameters": config.get("baseline_models", {}).get(
                        model_name.lower().replace(" ", "_"), {}
                    ),
                },
                output_data={
                    "model_name": model_name,
                    "training_time_seconds": b_elapsed,
                    "saved_model": f"models/{baseline_filenames[model_name]}",
                },
            )
            write_log(
                run_id,
                f"TRAINING - {model_name}",
                "SUCCESS",
                f"Huấn luyện {model_name} hoàn tất trong {b_elapsed}s.",
            )

        # ------------------------------------------------------------------
        # 3. LƯU METADATA VÀ VẼ BIỂU ĐỒ QUÁ TRÌNH HUẤN LUYỆN
        # ------------------------------------------------------------------
        saved_figures = []
        saved_figures.extend(plot_train_val_test_split(y_train_sub, y_val, y_test, stage_dir=stage_fig_dir))
        saved_figures.extend(plot_mlp_training_loss_curve(history_dict, stage_dir=stage_fig_dir))
        saved_figures.extend(
            plot_training_vs_validation_loss(history_dict, best_epoch=best_epoch, stage_dir=stage_fig_dir)
        )
        saved_figures.extend(plot_training_time_comparison(training_times, stage_dir=stage_fig_dir))

        training_metadata = {
            "run_id": run_id,
            "timestamp": get_current_timestamp("iso"),
            "backend_info": backend_info,
            "data_shapes": {
                "X_train_scaled": list(X_train_scaled.shape),
                "X_train_sub": list(X_train_sub.shape),
                "X_val_scaled": list(X_val_scaled.shape),
            },
            "mlp_architecture": {
                "layers": [
                    {"name": "input_layer", "units": input_dim, "activation": "none"},
                    *mlp_cfg.get("hidden_layers", []),
                    mlp_cfg.get("output_layer", {"units": 1, "activation": "linear"}),
                ],
                "optimizer": mlp_cfg.get("optimizer", "adam"),
                "learning_rate": mlp_cfg.get("learning_rate", 0.005),
                "loss": mlp_cfg.get("loss", "mean_squared_error"),
                "epochs_configured": epochs,
                "epochs_executed": epochs_ran,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
            },
            "training_times_seconds": training_times,
            "saved_models": [
                "models/mlp_regression.pth",
                "models/linear_regression.pkl",
                "models/decision_tree.pkl",
                "models/random_forest.pkl",
                "models/standard_scaler.pkl",
            ],
        }
        save_json(training_metadata, MODEL_DIR / "training_metadata.json")

        total_duration = round(time.perf_counter() - start_time, 4)

        update_progress(
            run_id,
            "TRAINING",
            "SUCCESS",
            input_data={"models": list(training_times.keys())},
            output_data={
                "training_metadata": training_metadata,
                "saved_figures": saved_figures,
                "duration_seconds": total_duration,
            },
        )
        write_log(run_id, "TRAINING", "SUCCESS", f"Hoàn tất toàn bộ giai đoạn Training trong {total_duration}s.")

        if own_run:
            finish_run(run_id, "SUCCESS")

        return training_metadata

    except Exception as exc:
        update_progress(run_id, "TRAINING", "FAILED", error=str(exc))
        write_log(run_id, "TRAINING", "FAILED", str(exc))
        if own_run:
            finish_run(run_id, "FAILED")
        raise


if __name__ == "__main__":
    main()
